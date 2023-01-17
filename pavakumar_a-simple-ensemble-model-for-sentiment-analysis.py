import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import string

import re

from tqdm.notebook import tqdm

import emoji



from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit



import tensorflow as tf

import tensorflow.keras as keras

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam, SGD

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint



import transformers

from tokenizers import BertWordPieceTokenizer



AUTO = tf.data.experimental.AUTOTUNE
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
data_df = pd.read_csv('/kaggle/input/student-shopee-code-league-sentiment-analysis/train.csv', index_col='review_id')
extra_df = pd.read_csv('/kaggle/input/shopee-reviews/shopee_reviews.csv')

extra_df.columns = ['rating','review']

extra_df = extra_df.drop(extra_df.loc[extra_df['rating']=='label'].index.to_list())

extra_df.rating = extra_df.rating.astype('int64')

extra_df.review = extra_df.review.astype('str')

extra_df = extra_df[['review', 'rating']]

extra_df = extra_df.sample(frac=0.1)
data_df = pd.concat([data_df, extra_df])
def clean_emoji(review):

    

    emojis = set([char for word in review.split() for char in list(word) if char in emoji.UNICODE_EMOJI])

    emojis = emoji.demojize(' '.join([emoji for emoji in emojis]))

    emojis = emojis.replace(':', '')

    emojis = emojis.replace('_', ' ')

    

    review = review.encode('ascii', 'ignore').decode('ascii')

    review = review + ' ' + emojis

    

    return review.lower()



def clean_urls(review):

    review = review.split()

    review = ' '.join([word for word in review if not re.match('^http', word)])

    return review



def clean_smileys(review):

    

    review = re.sub(r'(:\)|: \)|\(\:|:-\)|: -\)|: - \)|:D|: D)', ' smile ', review)

    review = re.sub(r'(:\(|: \(|\)\:|:-\(|: -\(|: - \(|:\'\()', ' dislike ', review)

    review = re.sub(r'(<3)', ' heart ', review)

    review = re.sub(r'(:/)', ' dislike ', review)

    review = re.sub(r'(;\)|; \))', ' wink ', review)

    return ' '.join([word for word in review.split()])



def recover_shortened_words(text):

    

    text = re.sub(r'\bapaa\b', 'apa', text)

    text = re.sub(r'\bbsk\b', 'besok', text)

    text = re.sub(r'\bbrngnya\b', 'barangnya', text)

    text = re.sub(r'\bbrp\b', 'berapa', text)

    text = re.sub(r'\bbgt\b', 'banget', text)

    text = re.sub(r'\bbngt\b', 'banget', text)

    text = re.sub(r'\bgini\b', 'begini', text)

    text = re.sub(r'\bbrg\b', 'barang', text)

    text = re.sub(r'\bdtg\b', 'datang', text)

    text = re.sub(r'\bd\b', 'di', text)

    text = re.sub(r'\bsdh\b', 'sudah', text)

    text = re.sub(r'\bdri\b', 'dari', text)

    text = re.sub(r'\bdsni\b', 'disini', text)

    text = re.sub(r'\bgk\b', 'gak', text)

    text = re.sub(r'\bhrs\b', 'harus', text)

    text = re.sub(r'\bjd\b', 'jadi', text)

    text = re.sub(r'\bjg\b', 'juga', text)

    text = re.sub(r'\bjgn\b', 'jangan', text)

    text = re.sub(r'\blg\b', 'lagi', text)

    text = re.sub(r'\blgi\b', 'lagi', text)

    text = re.sub(r'\blbh\b', 'lebih', text)

    text = re.sub(r'\blbih\b', 'lebih', text)

    text = re.sub(r'\bmksh\b', 'makasih', text)

    text = re.sub(r'\bmna\b', 'mana', text)

    text = re.sub(r'\borg\b', 'orang', text)

    text = re.sub(r'\bpjg\b', 'panjang', text)

    text = re.sub(r'\bka\b', 'kakak', text)

    text = re.sub(r'\bkk\b', 'kakak', text)

    text = re.sub(r'\bklo\b', 'kalau', text)

    text = re.sub(r'\bkmrn\b', 'kemarin', text)

    text = re.sub(r'\bkmrin\b', 'kemarin', text)

    text = re.sub(r'\bknp\b', 'kenapa', text)

    text = re.sub(r'\bkcil\b', 'kecil', text)

    text = re.sub(r'\bgmn\b', 'gimana', text)

    text = re.sub(r'\bgmna\b', 'gimana', text)

    text = re.sub(r'\btp\b', 'tapi', text)

    text = re.sub(r'\btq\b', 'thanks', text)

    text = re.sub(r'\btks\b', 'thanks', text)

    text = re.sub(r'\btlg\b', 'tolong', text)

    text = re.sub(r'\bgk\b', 'tidak', text)

    text = re.sub(r'\bgak\b', 'tidak', text)

    text = re.sub(r'\bgpp\b', 'tidak apa apa', text)

    text = re.sub(r'\bgapapa\b', 'tidak apa apa', text)

    text = re.sub(r'\bga\b', 'tidak', text)

    text = re.sub(r'\btgl\b', 'tanggal', text)

    text = re.sub(r'\btggl\b', 'tanggal', text)

    text = re.sub(r'\bgamau\b', 'tidak mau', text)

    text = re.sub(r'\bsy\b', 'saya', text)

    text = re.sub(r'\bsis\b', 'sister', text)

    text = re.sub(r'\bsdgkan\b', 'sedangkan', text)

    text = re.sub(r'\bmdh2n\b', 'semoga', text)

    text = re.sub(r'\bsmoga\b', 'semoga', text)

    text = re.sub(r'\bsmpai\b', 'sampai', text)

    text = re.sub(r'\bnympe\b', 'sampai', text)

    text = re.sub(r'\bdah\b', 'sudah', text)

    text = re.sub(r'\bberkali2\b', 'repeated', text)

    text = re.sub(r'\byg\b', 'yang', text)

    

    return text



def delete_repeated_char(text):

    text = re.sub(r'(\w)\1{2,}', r'\1', text)

    return text



def remove_punc(review):

    review =  review.translate(str.maketrans('', '', string.punctuation))

    review = ' '.join([word for word in review.split()])

    review = review.lower()

    return review
data_df['review'] = data_df['review'].apply(clean_emoji).apply(clean_urls).apply(clean_smileys).apply(recover_shortened_words).apply(delete_repeated_char).apply(remove_punc)



data_df['count'] = data_df['review'].str.split().map(len)

drop_indexes = data_df.loc[data_df['count']==0].index.tolist()

data_df = data_df.drop(drop_indexes)



data_df = data_df.drop_duplicates(subset=['review'])



data_df = data_df.sample(frac=1, random_state=42)

data_df.rating-=1
test_df = pd.read_csv('/kaggle/input/student-shopee-code-league-sentiment-analysis/test.csv', index_col='review_id')

test_df['review'] = test_df['review'].apply(clean_emoji).apply(clean_urls).apply(clean_smileys).apply(recover_shortened_words).apply(delete_repeated_char).apply(remove_punc)
import sklearn



class_weights = dict(zip(np.unique(data_df.rating), sklearn.utils.class_weight.compute_class_weight('balanced', np.unique(data_df.rating), data_df.rating)))
def regular_encode(texts, tokenizer, maxlen=512):

    enc_di = tokenizer.batch_encode_plus(

             texts, 

             return_attention_masks=False, 

             return_token_type_ids=False,

             pad_to_max_length=True,

             max_length=maxlen)

    

    return np.array(enc_di['input_ids'])



def build_model(transformer, max_len=512):

    

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    sequence_output = transformer(input_word_ids)[0]

    cls_token = sequence_output[:, 0, :]

    out = Dense(5, activation='softmax')(cls_token)

    

    model = Model(inputs=input_word_ids, outputs=out)

    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(Adam(lr=1e-5), loss='categorical_crossentropy',

    metrics=['accuracy'])

    

    return model
MAX_LEN = 192

EPOCHS = 4

BATCH_SIZE = 16 * strategy.num_replicas_in_sync
MODEL_NAMES = ['bert-base-multilingual-uncased','jplu/tf-xlm-roberta-base']

models = [0 for i in range(len(MODEL_NAMES))]

test_probs = [0 for i in range(len(MODEL_NAMES))]

test_results = [0 for i in range(len(MODEL_NAMES))]



with strategy.scope():

    

    for i in range(len(MODEL_NAMES)):

        

        models[i] = transformers.TFAutoModel.from_pretrained(MODEL_NAMES[i])

        

        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAMES[i])

        reviews = regular_encode(data_df.review.astype(str), tokenizer, maxlen=MAX_LEN)

    

        sss = StratifiedShuffleSplit(n_splits=4, random_state=42, test_size=0.2)

        X, y = reviews, tf.keras.utils.to_categorical(data_df.rating.astype(int).values, num_classes=5)



        models[i] = build_model(models[i], max_len=MAX_LEN)

        models[i].summary()



        for train_index, valid_index in sss.split(X, y):



            X_train, X_valid = X[train_index], X[valid_index]

            y_train, y_valid = y[train_index], y[valid_index]



            train_dataset = (

                tf.data.Dataset

                .from_tensor_slices((X_train, y_train))

                .repeat()

                .shuffle(2048)

                .batch(BATCH_SIZE)

                .prefetch(AUTO)

            )



            valid_dataset = (

                tf.data.Dataset

                .from_tensor_slices((X_valid, y_valid))

                .batch(BATCH_SIZE)

                .cache()

                .prefetch(AUTO)

            )



            n_steps = X_train.shape[0] // BATCH_SIZE

            models[i].fit(

                train_dataset,

                steps_per_epoch=n_steps,

                validation_data=valid_dataset,

                epochs=EPOCHS,

                class_weight=class_weights

            )

            

        test_reviews = regular_encode(test_df.review.astype(str), tokenizer, maxlen=MAX_LEN)



        test_dataset = (

        tf.data.Dataset

        .from_tensor_slices((test_reviews))

        .batch(BATCH_SIZE)

        .prefetch(AUTO)

        )



        test_probs[i] = models[i].predict(test_dataset, verbose=1)
test_df['rating'] = np.argmax(((test_probs[0] + test_probs[1])/2), axis = 1)

test_df.rating +=1

test_df.rating.unique() #sanity check LOL
test_df = test_df.drop(['review'], axis=1)

test_df.to_csv('submission.csv')