# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from transformers import BertTokenizer, TFBertModel
from transformers import RobertaTokenizer, TFRobertaModel
from transformers import XLMRobertaTokenizer, TFXLMRobertaModel
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import tensorflow as tf
from dask import bag, diagnostics
from sklearn.utils import shuffle
!pip install --quiet googletrans
from googletrans import Translator
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError:
    strategy = tf.distribute.get_strategy() # for CPU and single GPU
    print('Number of replicas:', strategy.num_replicas_in_sync)
submission = pd.read_csv("/kaggle/input/output/submission (2).csv")
submission.head()
submission.to_csv("submission1.csv", index = False)

train = pd.read_csv("/kaggle/input/contradictory-my-dear-watson/train.csv")
train.head()
train['language'].value_counts(normalize = True).plot(kind = 'bar', alpha = 0.7)
plt.show()
# model_name = 'bert-base-multilingual-cased'
# tokenizer = BertTokenizer.from_pretrained(model_name)
#tokenizer = RobertaTokenizer.from_pretrained('roberta-large-mnli')
#tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
tokenizer = AutoTokenizer.from_pretrained('jplu/tf-xlm-roberta-large')
def encode_sentence(s):
    tokens = list(tokenizer.tokenize(s))
    tokens.append('[SEP]')
    return tokenizer.convert_tokens_to_ids(tokens)
    
s = "I love machine learning"
encode_sentence(s)
def data_translate(source_data,dest_language):
    translator = Translator()
    if dest_language == 'zh':
        dest_language = 'zh-cn'
    dest_data = translator.translate(source_data, dest = dest_language).text 
    return dest_data
def translation_augment(source_data, languages, fraction):
    
    new_df = pd.DataFrame()
    
    for lang in languages:
        print(lang)
        sampled_rows = source_data.sample(frac=fraction, replace = False)
        prem_bag = bag.from_sequence(sampled_rows['premise'].tolist()).map(data_translate, lang)
        hypothesis_bag = bag.from_sequence(sampled_rows['hypothesis'].tolist()).map(data_translate, lang)
        
        with diagnostics.ProgressBar():
            prems = prem_bag.compute()
            hyps = hypothesis_bag.compute()
            
        aug_df = pd.DataFrame({'id': pd.Series([None]*len(sampled_rows)),
                                'premise': pd.Series(prems),
                                'hypothesis': pd.Series(hyps),
                                'lang_abv': pd.Series([lang]*len(sampled_rows)),
                                'language': pd.Series([None]*len(sampled_rows)),
                                'label': pd.Series(sampled_rows['label'].values)                              
                              })
        new_df = new_df.append(aug_df)
    new_df = shuffle(new_df)
    return new_df

def data_augment(train_df, fraction):
    
    english_df = train.loc[train.lang_abv == 'en']
    languages = list(set(train.lang_abv.values))
    languages.remove('en')

#     languages = ['fr', 'th', 'tr', 'ur', 'ru', 'bg', 'de', 'ar', 'zh-cn', 'hi',
#                  'sw', 'vi', 'es', 'el']

    print(languages)    
    translated_df = translation_augment(english_df,languages, fraction)
    train_df = train_df.append(translated_df)
    train_df = shuffle(train_df)
    return train_df
train = pd.read_csv('/kaggle/input/augment-data20/augmented_data_20percent.csv')
train.head()
len(train)
# print("Length of training data before augmentation", len(train))
# train = data_augment(train, fraction = 0.6)
# print("Length of training data after augmentation", len(train))

# train['lang_abv'].value_counts(normalize = True).plot(kind = 'bar', alpha = 0.7)
# plt.show()
# train.to_csv('augmented_data_60_percent.csv', index=False)
# train.head()
def bert_encode(premises, hypotheses, tokenizer):
    num_examples = len(premises)
    sen1 = tf.ragged.constant([encode_sentence(s) for s in np.array(premises)])
    sen2 = tf.ragged.constant([encode_sentence(s) for s in np.array(hypotheses)])
    cls = [tokenizer.convert_tokens_to_ids(['CLS'])]*sen1.shape[0]
    
    input_word_ids = tf.concat([cls, sen1, sen2], axis = -1)
    input_mask = tf.ones_like(input_word_ids).to_tensor()
    
    type_cls = tf.zeros_like(cls)
    type_sen1 = tf.zeros_like(sen1)
    type_sen2 = tf.ones_like(sen2)
    input_type_ids = tf.concat([type_cls, type_sen1, type_sen2], axis = -1).to_tensor()
    
    inputs = {
        
        'input_word_ids' : input_word_ids.to_tensor(),
        'input_mask': input_mask,
        'input_type_ids': input_type_ids
        
    }
    
    return inputs
train_input = bert_encode(train.premise.values, train.hypothesis.values, tokenizer)
max_len = 80
def build_model():
    #bert_encoder = TFBertModel.from_pretrained(model_name)
    bert_encoder = TFRobertaModel.from_pretrained('jplu/tf-xlm-roberta-large')
    #bert_encoder = TFXLMRobertaModel.from_pretrained('xlm-mlm-100-1280')
    input_word_ids = tf.keras.Input(shape =(max_len, ), dtype =tf.int32, name = "input_word_ids")
    input_mask = tf.keras.Input(shape = (max_len, ), dtype= tf.int32, name = "input_mask")
    input_type_ids = tf.keras.Input(shape= (max_len, ), dtype= tf.int32, name="input_type_ids")
    embedding = bert_encoder([input_word_ids, input_mask, input_type_ids])[0]
    output = tf.keras.layers.Dense(3, activation = 'softmax')(embedding[:,0,:])
    
    model = tf.keras.Model(inputs= [input_word_ids, input_mask, input_type_ids], outputs=output)
    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='sparse_categorical_crossentropy', metrics = 'accuracy')
    
    return model   
    
with strategy.scope():
    model = build_model()
    model.summary()
model.fit(train_input, train.label.values, epochs=3, verbose=1, batch_size=16, validation_split=0.2)
model.save_weights('RoBertamodel_augmented_data_20_percent_adam_sparse_categorical_entropy.h5')
test = pd.read_csv('/kaggle/input/contradictory-my-dear-watson/test.csv')
test.head()
test_input = bert_encode(test.premise.values, test.hypothesis.values, tokenizer)
predictions = [np.argmax(i) for i in model.predict(test_input) ]
submission = test.id.copy().to_frame()
submission.head()

submission['prediction'] = predictions
submission.to_csv("submission.csv", index = False)
