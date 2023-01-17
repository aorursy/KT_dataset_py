%matplotlib inline

from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers.recurrent import GRU,SimpleRNN,LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping

import sys
import os
import numpy as np
import pandas as pd
import IPython
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from wordcloud import WordCloud
valid = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation-processed-seqlen128.csv")
train = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train-processed-seqlen128.csv")
test = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test-processed-seqlen128.csv")
submit = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv")
train = train[['id', 'comment_text', 'input_word_ids', 'input_mask','all_segment_id', 'toxic']].iloc[:20000] #limit på tränings set
train.info()
train.tail(12)
valid.tail(12)
test.tail(12)
train.isnull().any(),test.isnull().any() # test om det finns null värden i vår träningsdata
train_distribution = train["toxic"].value_counts().values
valid_distribution = valid["toxic"].value_counts().values

non_toxic = [train_distribution[0] / sum(train_distribution) * 100, valid_distribution[0] / sum(valid_distribution) * 100]
toxic = [train_distribution[1] / sum(train_distribution) * 100, valid_distribution[1] / sum(valid_distribution) * 100]

plt.figure(figsize=(9,6))
plt.bar([0, 1], non_toxic, alpha=.4, color="r", width=0.35, label="non-toxic")
plt.bar([0.4, 1.4], toxic, alpha=.4, width=0.35, label="toxic")
plt.xlabel("Dataset")
plt.ylabel("Percentage")
plt.xticks([0.2, 1.2], ["train", "valid"])
plt.legend(loc="upper right")

plt.show()
print(f"Träningsdata: \nnon-toxic rate: {train_distribution[0] / sum(train_distribution) * 100: .2f} %\ntoxic rate: {train_distribution[1] / sum(train_distribution) * 100: .2f} %")
print(f"Valideringsdata: \nnon-toxic rate: {valid_distribution[0] / sum(valid_distribution) * 100: .2f} %\ntoxic rate: {valid_distribution[1] / sum(valid_distribution) * 100: .2f} %")
# möjliggör unndersökning av längden på kommentarerna i träningsdata
train['char_length'] = train['comment_text'].apply(lambda x: len(str(x)))
# visar plottad histogram för längd på kommentarer (antal tecken)
sns.set()
train['char_length'].hist()
plt.show()
def nonan(x):
    if type(x) == str:
        return x.replace("\n", "")
    else:
        return ""

text = ' '.join([nonan(abstract) for abstract in train["comment_text"]])
wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,
                      width=1200, height=1000).generate(text)
fig = px.imshow(wordcloud)
fig.update_layout(title_text='Frekventa ord i kommentarer')
subset = train.query("toxic == 0")
text = subset.comment_text.values
wc = WordCloud(background_color="black",max_words=1500)
wc.generate(" ".join(text))
plt.figure(figsize=(7.5, 7.5))
plt.axis("off")
plt.title("Frekventa ord i icke-hatfulla kommentarer", fontsize=16)
plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17))
plt.show()

subset = train.query("toxic == 1")
text = subset.comment_text.values
wc = WordCloud(background_color="black",max_words=1500)
wc.generate(" ".join(text))
plt.figure(figsize=(7.5, 7.5))
plt.axis("off")
plt.title("Frekventa ord i hatfulla kommentarer", fontsize=16)
plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17))
plt.show()
train = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')
validation = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')

train.drop(['severe_toxic','obscene','threat','insult','identity_hate'], axis=1, inplace=True) # droppar klassificeringar av hatfulla kommentarer
train = train.loc[:15000,:] #bestämmer delmäng
train.shape #hämtar form på tränings set
train['comment_text'].apply(lambda x:len(str(x).split())).max() # hämtar max antal tecken i kommentar
xtrain, xvalid, ytrain, yvalid = train_test_split(train.comment_text.values, train.toxic.values, 
                                                  stratify=train.toxic.values, 
                                                  random_state=42, 
                                                  test_size=0.2, shuffle=True)
def roc_auc(predictions,target):
    '''
    This methods returns the AUC Score when given the Predictions
    and Labels
    '''
    
    fpr, tpr, thresholds = metrics.roc_curve(target, predictions)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc
# använder keras tokenizer
token = text.Tokenizer(num_words=None)
max_len = 1500

token.fit_on_texts(list(xtrain) + list(xvalid))
xtrain_seq = token.texts_to_sequences(xtrain)
xvalid_seq = token.texts_to_sequences(xvalid)

#zero paddar sekvenserna
xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)

word_index = token.word_index
# Upptäck hårdvara, returnera lämplig distributionsstrategi
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)
# En enkel modell (SimpleRNN) med ett dense layer

with strategy.scope():
    
    model = Sequential() #sekventiellt nätverk
    model.add(Embedding(len(word_index) + 1, #konverterar till 300 dimensionell vektor
                     300,
                     input_length=max_len))
    model.add(SimpleRNN(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
model.summary()
history = model.fit(xtrain_pad, ytrain, nb_epoch=5, batch_size=64)
scores = model.predict(xvalid_pad)
print("AUC: %.2f%%" % (roc_auc(scores,yvalid)))
# load the GloVe vectors in a dictionary:

embeddings_index = {}
f = open('/kaggle/input/glove840b300dtxt/glove.840B.300d.txt','r',encoding='utf-8') #precis som vår data för kommentarer så lägger vi till GloVe i vår Kaggle-databas
for line in tqdm(f):
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray([float(val) for val in values[1:]])
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
# skapa en embedding matris för orden vi har i vår datauppsättning
embedding_matrix = np.zeros((len(word_index) + 1,300))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
%%time
with strategy.scope():
    
    model = Sequential() # sekventiellt nätverk
    model.add(Embedding(len(word_index) + 1, # embedding lager som nu använder GloVe istället
                     300,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))

    model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3,return_sequences=True)) # tre LSTM lager med 100 units, dropout har lagts till
    model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3,return_sequences=True))
    model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
model.summary()
model.fit(xtrain_pad, ytrain, nb_epoch=5, batch_size=64)
scores = model.predict(xvalid_pad)
print("Auc: %.2f%%" % (roc_auc(scores,yvalid)))