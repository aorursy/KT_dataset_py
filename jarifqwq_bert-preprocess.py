# We will use the official tokenization script created by the Google team

!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
# !pip install pandarallel
import pandas as pd

# from pandarallel import pandarallel

# pandarallel.initialize(nb_workers=4, progress_bar=True)

import numpy as np

from tqdm import tqdm

import tokenization

import gc



from sklearn.preprocessing import LabelEncoder



pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import tensorflow as tf

from keras import backend as K

from tensorflow.keras.layers import Dense, Input, concatenate, add, BatchNormalization, PReLU, Dropout

# from tensorflow.keras.layers import MaxoutDense

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow_hub as hub



from transformers import TFBertModel

from transformers.tokenization_bert import BertTokenizer



from kaggle_datasets import KaggleDatasets
# Detect hardware, return appropriate distribution strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



print("REPLICAS: ", strategy.num_replicas_in_sync)
train_df = pd.read_table('../input/mercari/train.tsv')

# test_df = pd.read_table('../input/test.tsv')

print(train_df.shape)
def find_missing_brand(line, all_brands):

    brand = line[0]

    name = line[1]

    namesplit = name.split(" ")

    if brand == "missing":

        for one_word_brand in namesplit:   # 一个词的品牌

            if one_word_brand in all_brands:

                return one_word_brand

        two_word_brands = [namesplit[i] + " " + namesplit[i+1] for i in range(len(namesplit)-1)]

        for two_word_brand in two_word_brands:   # 2个词的品牌

            if two_word_brand in all_brands:

                return two_word_brand

        three_word_brands = [namesplit[i] + " " + namesplit[i+1] + " " + namesplit[i+2] for i in range(len(namesplit)-2)]

        for three_word_brand in three_word_brands:   # 3个词的品牌

            if three_word_brand in all_brands:

                return three_word_brand

        four_word_brands = [namesplit[i] + " " + namesplit[i+1] + " " + namesplit[i+2] + " " + namesplit[i+3] for i in range(len(namesplit)-3)]

        for four_word_brand in four_word_brands:   # 4个词的品牌

            if four_word_brand in all_brands:

                return four_word_brand

    return brand  



def compose_full_text(line):

    name = line[0]

    category = line[1]

    brand = line[2]

    description = line[3]

    composed_full_text = "Item name: " + str(name) + ", " + "item category: " + str(category) + ", " + "item_brand: " + str(brand) + "." + "Item description: " + str(description)

    return composed_full_text



def bert_encode(texts, tokenizer, max_len):

    all_tokens = []

    all_masks = []

    all_segments = []

    

    for text in texts:

        text = tokenizer.tokenize(text)

            

        text = text[:max_len-2]

        input_sequence = ["[CLS]"] + text + ["[SEP]"]

        pad_len = max_len - len(input_sequence)

        

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)

        tokens += [0] * pad_len

        pad_masks = [1] * len(input_sequence) + [0] * pad_len

        segment_ids = [0] * max_len

        

        all_tokens.append(tokens)

        all_masks.append(pad_masks)

        all_segments.append(segment_ids)

    

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)



def split_cat(text):

    try: return text.split("/")

    except: return ("missing", "missing", "missing")



def rmse(y_true, y_pred):

    # Y and Y_red have already been in log scale.

    # assert y_true.shape == y_pred.shape

    return K.sqrt(K.mean(K.square(y_pred - y_true )))
class DataGenerator(tf.keras.utils.Sequence):

    'Generates data for Keras'

    def __init__(self, dataframe, batch_size, tokenizer, max_len):

        self.dataframe = dataframe

        self.batch_size = batch_size

        self.tokenizer = tokenizer

        self.max_len = max_len

    

    def __len__(self):

        'Denotes the number of batches per epoch'

        return int(np.ceil(len(self.dataframe) / self.batch_size))



    def __getitem__(self, idx):

        'Generate one batch of data'

        train_full_text = bert_encode(self.dataframe["full_text"][idx * self.batch_size: min((idx + 1) * self.batch_size, len(self.dataframe))].values, self.tokenizer, self.max_len)

        train_subcat_0 = self.dataframe['subcat_0'][idx * self.batch_size: min((idx + 1) * self.batch_size, len(self.dataframe))].values

        train_subcat_1 = self.dataframe['subcat_1'][idx * self.batch_size: min((idx + 1) * self.batch_size, len(self.dataframe))].values

        train_subcat_2 = self.dataframe['subcat_2'][idx * self.batch_size: min((idx + 1) * self.batch_size, len(self.dataframe))].values

        train_brand = train_df['brand'][idx * self.batch_size: min((idx + 1) * self.batch_size, len(self.dataframe))].values

        train_condition = train_df['item_condition_id'][idx * self.batch_size: min((idx + 1) * self.batch_size, len(self.dataframe))].values

        train_shipping = train_df['shipping'][idx * self.batch_size: min((idx + 1) * self.batch_size, len(self.dataframe))].values

        

        train_y = train_df['price'][idx * self.batch_size: min((idx + 1) * self.batch_size, len(self.dataframe))].values

        return [train_full_text[0], train_full_text[1], train_full_text[2], train_subcat_0, train_subcat_1, train_subcat_2, train_brand, train_condition, train_shipping], train_y
def preprocessing(df):

    

    # 去除 price=0 的值 0️⃣

    df = df.drop(df[(df.price == 0)].index)

    

    # price 放缩到 ln(price+1) 📉

    df["price"] = np.log1p(df.price)

    

    # 找回一些缺失的 brand 🍳

    all_brands = set(df['brand_name'].values)   # 1,2,3,4词的品牌都很多

    df["brand_name"].fillna(value="missing", inplace=True)

    premissing = len(df.loc[df['brand_name'] == 'missing'])

    tqdm.pandas(desc="找回缺失brand_name👜")

    df['brand_name'] = df[['brand_name', 'name']].progress_apply(lambda x: find_missing_brand(x, all_brands), axis = 1)

    postmissing = len(df.loc[df['brand_name'] == 'missing'])

    print("处理前缺失：{}，处理后缺失：{}，找到了：{}".format(premissing, postmissing, premissing-postmissing))

    

    # 将商品类别分开，并填充其缺失值 ⛏

    tqdm.pandas(desc="将商品类别分开✂")

    df['subcat_0'], df['subcat_1'], df['subcat_2'] = zip(*df['category_name'].progress_apply(lambda x: split_cat(x)))

    # df.item_description.replace('No description yet',"missing", inplace=True)

    

    # 组合成新文本 Item name: ..., item category: ..., item brand: ..., item description: .... 📕

    tqdm.pandas(desc="组合新文本🧩")

    df["full_text"] = df[['name', 'category_name', 'brand_name', 'item_description']].progress_apply(compose_full_text, axis = 1)

    

    # 处理类别变量 subcat0, subcat1, subcat2 和 brand_name 👔

    le = LabelEncoder()

    

    le.fit(df["subcat_0"])

    df["subcat_0"] = le.transform(df["subcat_0"])



    le.fit(df["subcat_1"])

    df["subcat_1"] = le.transform(df["subcat_1"])



    le.fit(df["subcat_2"])

    df["subcat_2"] = le.transform(df["subcat_2"])



    le.fit(df.brand_name)

    df['brand'] = le.transform(df.brand_name)

    

    del le

    gc.collect()

    

    # 删除一些列节约内存 😭

    df.drop(['category_name', 'brand_name', 'name', 'item_description'], axis=1, inplace=True) 

    return df
train_df = preprocessing(train_df)
# full_text 长度 ☹

sns.distplot(a=train_df["full_text"].apply(lambda x: len(x)), kde=False)
np.percentile(train_df["full_text"].apply(lambda x: len(x)), [25, 50, 75]), np.mean(train_df["full_text"].apply(lambda x: len(x)))
gc.collect()
train_df.to_csv("./train_preprocessed.csv")
# module_url = "https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/1"

# bert_layer = hub.KerasLayer(module_url, trainable=True)
# bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/1", trainable=True)

# # 文本序列 📃

# full_text_ids = Input(shape=(max_len,), dtype=tf.int32, name="full_text_ids")

# full_text_mask = Input(shape=(max_len,), dtype=tf.int32, name="full_text_mask")

# full_text_segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="full_text_segment_ids")



# # 分类变量 🎨

# subcat_0 = Input(shape=(1,), dtype=tf.float32, name='subcat_0')

# subcat_1 = Input(shape=(1,), dtype=tf.float32, name='subcat_1')

# subcat_2 = Input(shape=(1,), dtype=tf.float32, name='subcat_2')

# brand = Input(shape=(1,), dtype=tf.float32, name='brand')

# condition = Input(shape=(1,), dtype=tf.float32, name='condition')

# shipping = Input(shape=(1,), dtype=tf.float32, name='shipping')



# # BERT 处理文字序列 😺

# _, sequence_output = bert_layer([full_text_ids, full_text_mask, full_text_segment_ids])

# full_text_encoding = sequence_output[:, 0, :]



# # concat 合并 🧱

# con = concatenate([full_text_encoding, subcat_0, subcat_1, subcat_2, brand, condition, shipping])

# con = BatchNormalization()(con)



# # 全连接分支 1️⃣

# x1 = Dense(30, activation='sigmoid')(con)



# x2 = Dense(470)(con)

# x2 = PReLU()(x2)

# con = concatenate([x1,x2])

# con = Dropout(0.02)(con)



# # 全连接分支 2️⃣

# x1 = Dense(256, activation='sigmoid')(con)



# x2 = Dense(11, activation='linear')(con)



# x3 = Dense(11)(con)

# x3 = PReLU()(x3)

# con = concatenate([x1, x2, x3])

# con = Dropout(0.02)(con)



# # 全连接分支 3️⃣

# out1 = Dense(1, activation='linear')(con)

# out2 = Dense(1, activation='relu')(con)

# # out3 = MaxoutDense(1, 30)(con)

# output = add([out1,out2])





# model = Model(inputs=[full_text_ids, full_text_mask, full_text_segment_ids, subcat_0, subcat_1, subcat_2, brand, condition, shipping], outputs=output)

# model.compile(Adam(lr=2e-6), loss=rmse)
# model.summary()
# vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

# do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

# tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
# batch_size = 8 * strategy.num_replicas_in_sync

# epochs = 1



# autotune = tf.data.experimental.AUTOTUNE

# pretrained_weights = 'bert-base-uncased'

# tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
# with strategy.scope():

#     train_generator = DataGenerator(train_df[:1400000], bs, tokenizer, max_len=300)

#     valid_generator = DataGenerator(train_df[1400000:], bs, tokenizer, max_len=300)

# model.fit_generator(generator = train_generator,

#           epochs=1,

#           validation_data=valid_generator)