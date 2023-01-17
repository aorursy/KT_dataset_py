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
os.environ["WANDB_API_KEY"] = "0" ## to silence warning
from transformers import BertTokenizer,TFBertModel

import tensorflow as tf

import matplotlib as mlp

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="darkgrid")

from random import random
try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

except ValueError:

    strategy = tf.distribute.get_strategy() # for CPU and single GPU

    print('Number of replicas:', strategy.num_replicas_in_sync)
MLP_XKCD_COLOR = mlp.colors.XKCD_COLORS

MLP_BASE_COLOR = mlp.colors.BASE_COLORS

MLP_CNAMES = mlp.colors.cnames

MLP_CSS4 = mlp.colors.CSS4_COLORS

MLP_HEX = mlp.colors.hexColorPattern

MLP_TABLEAU = mlp.colors.TABLEAU_COLORS

print('I like COLORS :>')

def random_color_generator(color_type=None):

    if color_type is None:

        colors = sorted(MLP_CNAMES.items(), key=lambda x: random())

    else:

        colors = sorted(color_type.items(), key=lambda x: random())

    return dict(colors)
path = '/kaggle/input/contradictory-my-dear-watson/'

train = pd.read_csv(path+'train.csv')

test = pd.read_csv(path+'test.csv')
train.head()
train.info()
colors = random_color_generator()

train.language.value_counts().plot(kind='bar',color=colors)

plt.show()
plt.figure(figsize=(12,8))

labels,freq = np.unique(train.language.values,return_counts=True)

print(labels,freq)

plt.pie(freq,labels=labels,autopct = '%1.1f%%')

plt.show()
colors = random_color_generator()

fig, axes = plt.subplots(ncols=1, figsize=(8, 5), dpi=100)

train.label.value_counts().sort_values(ascending=True).plot(kind='bar',color=colors)

axes.set_xticklabels(['neutral', 'contradiction','entailment'])

plt.show()
train.premise[100]
train.hypothesis[100]
train.label[100]
import nltk

from nltk.corpus import stopwords

import string

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

PUNCTUATIONS = string.punctuation

#print(PUNCTUATIONS)
from wordcloud import WordCloud

colors = random_color_generator()

def show_word_cloud(data,title=None):

    word_cloud = WordCloud(

        background_color = list(colors.keys())[1],

        max_words =100,

        width=800,

        height=400,

        stopwords=STOPWORDS,

        max_font_size = 40, 

        scale = 3,

        random_state = 42 ).generate(data)

    fig = plt.figure(1, figsize = (20, 20))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize = 20)

        fig.subplots_adjust(top = 2.3)



    plt.imshow(word_cloud)

    plt.show()
#Most Comman words in entailment Prases

entailment = " ".join(train[train.label==0]['premise'])

show_word_cloud(entailment,'TOP 100 Entailment Words')
#Most Comman words in Neutral Prases

neutral = " ".join(train[train.label==1]['premise'])

show_word_cloud(neutral,'TOP 100 Neutral Words')
#Most Comman words in Contradictory Prases

contradiction = " ".join(train[train.label==2]['premise'])

show_word_cloud(contradiction,'TOP 100 Contradiction Words')
model_name = 'bert-base-multilingual-cased'

tokenizer = BertTokenizer.from_pretrained(model_name)
def encode_sentence(sentence):

    tokens = list(tokenizer.tokenize(sentence))

    tokens.append('[SEP]')

    return tokenizer.convert_tokens_to_ids(tokens)
encode_sentence('Are you lost Baby Girl ?')
def bert_encode(premise,hypothesis,tokenizer):

    sentence_1 = tf.ragged.constant([encode_sentence(s) for s in np.array(premise)])

    sentence_2 = tf.ragged.constant([encode_sentence(s) for s in np.array(hypothesis)])

    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])] * sentence_1.shape[0]

    #print('CLS -- ',cls)

    input_word_ids = tf.concat([cls,sentence_1,sentence_2],axis=-1)

    #print('Input Word Ids --- ',input_word_ids)

    input_mask = tf.ones_like(input_word_ids).to_tensor()

    #print('Input Mask -- ',input_mask)

    type_cls = tf.zeros_like(cls)

    type_s1 = tf.zeros_like(sentence_1)

    type_s2 = tf.ones_like(sentence_2)

    input_type_ids = tf.concat([type_cls, type_s1, type_s2], axis=-1).to_tensor()

    inputs = {'input_word_ids':input_word_ids.to_tensor(),

              'input_mask':input_mask,

              'input_type_ids':input_type_ids}

    return inputs
train_input = bert_encode(train.premise.values, train.hypothesis.values, tokenizer)
def build_model(max_len=50):

    bert_encoder = TFBertModel.from_pretrained(model_name)

    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

    input_type_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_type_ids")

    

    embedding = bert_encoder([input_word_ids, input_mask, input_type_ids])[0]

    output = tf.keras.layers.Dense(3, activation='softmax')(embedding[:,0,:])

    

    model = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=output)

    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
with strategy.scope():

    model = build_model()

    model.summary()
history = model.fit(train_input,train.label.values, epochs = 10, verbose = 1, 

          batch_size = 64, validation_split = 0.2)
plt.figure(figsize=(10, 6))

plt.plot(history.history['accuracy'],label='Accuracy')

plt.plot(history.history['loss'],label='Loss')

plt.legend(loc='best')

plt.title('Model Accuracy Vs Model Loss')

plt.show()
test_input = bert_encode(test.premise.values, test.hypothesis.values, tokenizer)
predictions = model.predict(test_input)

predictions = [np.argmax(prob) for prob in predictions]

predictions[:10]
submission = test.id.copy().to_frame()

submission['prediction'] = predictions
submission.head()
submission.to_csv("submission.csv", index = False)