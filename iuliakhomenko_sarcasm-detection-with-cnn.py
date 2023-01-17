import pandas as pd

import numpy as np

import string 

import re

import statistics

from nltk.corpus import stopwords

import seaborn as sns

import nltk

import spacy

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, f1_score

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from scipy import sparse

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

nltk.download('stopwords')

nltk.download('punkt')

nltk.download('averaged_perceptron_tagger')

import os 

import tensorflow as tf

import keras.backend as K

from keras import regularizers

from keras.models import Model

from keras.layers import *

from keras.layers import Embedding
# !unzip news-headlines-dataset-for-sarcasm-detection.zip
print(os.listdir("../input"))

data = pd.read_json("../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json", lines=True)
# are all sarcastic headlines from Onion, and no-sarcastic headlines are from Huffington post 

def get_source(row):

    if 'theonion' in row:

        return 'Onion'

    else:

        return 'HP'

data['source'] = data['article_link'].apply(get_source)    
data.groupby(['source','is_sarcastic']).count()
pd.set_option('display.max_colwidth', -1)

data[(data['source']=='Onion')&(data['is_sarcastic']==0)]

# this looks like an incorrect data point, as after manual inspection this article is obviously sarcastic
# correct this mistake

data.at[19948,'is_sarcastic']=1
#Let's draw class distribution

sns.countplot(x = 'is_sarcastic', data = data)
# remove punctuation

data['headline']=data['headline'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

# remove stopwords

data['headline'] = data['headline'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords.words("english")]))



# lematization

# we will use spaCy library for text lemmatization because unlike nltk it determines the pos autamatically and 

# return the correct lemma, while nltk needs pos to be specified explicitely for each word

spacy_model=spacy.load('en')

data['headline'] = data['headline'].apply(lambda x: " ".join([token.lemma_ for token in spacy_model(x)]))
# prepare text for word clouds

sarcastic_txt = " ".join(headline for headline in data[data.is_sarcastic==1]['headline'])

non_sarcastic_txt = " ".join(headline for headline in data[data.is_sarcastic==0]['headline'])
# Word cloud for positive class

wordcloud = WordCloud().generate(sarcastic_txt)

plt.figure(figsize=(16,10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()

# wordcloud.to_file("img/first_review.png")
#Word cloud for negative class



wordcloud = WordCloud().generate(non_sarcastic_txt)

plt.figure(figsize=(16,10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
# number of words

data['words_num'] = data['headline'].apply(lambda x: len(x.split()))

data = data[data.words_num!=0]

# number of characters

data['chars_num'] = data['headline'].apply(lambda x: len(x.replace(" ","")))



# number of special characters

special = string.punctuation

data['special_chars_num'] = data['headline'].apply(lambda x: len([char for char in x if char in special]))



# average word length

data['word_len_avg'] = data['headline'].apply(lambda x: statistics.mean([len(word) for word in x.split()]))



# number of stopwords

sw=stopwords.words(['english'])

data['stopwords_num'] = data['headline'].apply(lambda x: len([word for word in x.split() if word in sw]))



# stopwords/ total words ratio

data['sw_ratio'] = data['stopwords_num']/data['words_num']
data.groupby('is_sarcastic')[['words_num', 'chars_num', 'special_chars_num', 'stopwords_num', 'sw_ratio']].agg(['min', 'max', 'mean'])
# POS analysis

pos_family = {

    'noun' : ['NN','NNS','NNP','NNPS'],

    'pron' : ['PRP','PRP$','WP','WP$'],

    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],

    'adj' :  ['JJ','JJR','JJS'],

    'adv' : ['RB','RBR','RBS','WRB']}



def pos_check(tokens, pos_flag):    

    pos = nltk.pos_tag(tokens)

    counter=0

    for pair in pos:

        if pair[1] in pos_family[pos_flag]:

            counter+=1

    return counter



data['tokens'] = data['headline'].apply(lambda x: nltk.word_tokenize(x))

data['noun_ratio'] = data['tokens'].apply(lambda x: pos_check(x, 'noun'))/data['words_num']

data['pron_ratio'] = data['tokens'].apply(lambda x: pos_check(x, 'pron'))/data['words_num']

data['verb_ratio'] = data['tokens'].apply(lambda x: pos_check(x, 'verb'))/data['words_num']

data['adj_ratio'] = data['tokens'].apply(lambda x: pos_check(x, 'adj'))/data['words_num']

data['adv_ratio'] = data['tokens'].apply(lambda x: pos_check(x, 'adv'))/data['words_num']
data.head()
# define vocabulary size

vocab = set()

len_max = 0

for idx,row in data.iterrows():

    sentence = row['headline'].split(' ')

    if len(sentence)>len_max:

        len_max=len(sentence)

    vocab.update(sentence)

vocab_size = len(vocab)

# define sequence size
# create embeddings using glove 

import numpy as np

embeddings_index = dict()

f = open('../input/glove-6b-50d/glove.6B.50d.txt')

for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()

print('Loaded %s word vectors.' % len(embeddings_index))
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



# prepare tokenizer

t = Tokenizer()

t.fit_on_texts(data['headline'])

vocab_size = len(t.word_index) + 1

# integer encode the documents

encoded_docs = t.texts_to_sequences(data['headline'])

# pad documents to a max length of 4 words

padded_docs = pad_sequences(encoded_docs, maxlen=len_max, padding='post')

embedding_matrix = np.zeros((vocab_size, 50))

for word, i in t.word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector
# split our padded sentences on train and test

x_train, x_test, y_train, y_test = train_test_split(padded_docs, data['is_sarcastic'], test_size = 0.2,random_state=42)
def f1(y_true, y_pred):

    y_pred = K.round(y_pred)

    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)

    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)

    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)

    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)



    p = tp / (tp + fp + K.epsilon())

    r = tp / (tp + fn + K.epsilon())



    f1 = 2*p*r / (p+r+K.epsilon())

    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)

    return K.mean(f1)



def f1_loss(y_true, y_pred):

    

    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)

    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)

    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)

    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)



    p = tp / (tp + fp + K.epsilon())

    r = tp / (tp + fn + K.epsilon())



    f1 = 2*p*r / (p+r+K.epsilon())

    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)

    return 1 - K.mean(f1)
inputs_text = Input(shape=(len_max,))

#inputs_nlp = Input(shape=(10,))

# first we need to create our embeddings layer

#embedding_layer = Embedding(vocab_size, 32, input_length= len_max)(inputs)

embedding_layer = Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=len_max, trainable = False)(inputs_text)

conv_1 = Conv1D(filters=64, kernel_size=1, strides=1, padding="valid", activation="relu")(embedding_layer)

conv_2 = Conv1D(filters=64, kernel_size=2,strides=1, padding="valid", activation="relu")(embedding_layer)

conv_3 = Conv1D(filters=64, kernel_size=3, strides=1, padding="valid", activation="relu")(embedding_layer)



maxpool_1 = MaxPooling1D(2)(conv_1)

dropout_1 = Dropout(0.4)(maxpool_1)



maxpool_2 = MaxPooling1D(2)(conv_2)

dropout_2 = Dropout(0.4)(maxpool_2)



maxpool_3 = MaxPooling1D(2)(conv_3)

dropout_3 = Dropout(0.4)(maxpool_3)



text_features = Concatenate(axis=1)([dropout_1, dropout_2, dropout_3])



flattened = Flatten()(text_features)

#all_features = Concatenate(axis=1)([flattened, inputs_nlp])



fc2 = Dense(units=256, activation='relu')(flattened)

dropout_4=Dropout(0.3)(fc2)

output_layer = Dense(units=1, activation="sigmoid")(dropout_4)

model = Model(inputs=inputs_text, outputs=output_layer)

model.summary()
from keras.optimizers import RMSprop, Adam, SGD

optimizer = RMSprop(lr=0.0001)

model.compile(optimizer=optimizer, 

              loss= f1_loss, #'binary_crossentropy', 

              metrics=['acc', f1])
from keras.callbacks import ModelCheckpoint, EarlyStopping



checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5',

                                verbose=1,

                                save_best_only=True,

                                monitor='val_f1', mode='max')
early_stop = EarlyStopping(monitor='val_f1', mode='max', verbose=1, 

                          patience=100)

history = model.fit(x_train, y_train, 

         batch_size=1000, 

         epochs=2000,

         validation_split=0.2,

         callbacks=[checkpointer, early_stop],

         verbose=2, shuffle=True

                   )
import matplotlib.pyplot as plt



# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='upper left')

plt.show()
plt.plot(history.history['f1'])

plt.plot(history.history['val_f1'])

plt.title('Model F1 score')

plt.ylabel('F1 score')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='upper left')

plt.show()
model.load_weights('model.weights.best.hdf5')
pred = model.predict(x_test)
pred_arr = []

for i in pred:

    pred_arr.append(round(i[0]))



from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(y_test, pred_arr).ravel()

from sklearn.metrics import confusion_matrix, recall_score, precision_score

tn, fp, fn, tp = confusion_matrix(y_test, pred_arr).ravel()

print('TN: %d, \nFP: %d, \nFN: %d, \nTP: %d' % (tn, fp,fn, tp))

print('Precision: ', round(precision_score(y_test, pred_arr),2))

print('Recall: ', round(recall_score(y_test, pred_arr),2))

print('F1 score: ', round(f1_score(y_test, pred_arr),2))

print('Accuracy score: ', round(accuracy_score(y_test, pred_arr),2))
#y_test_cat=to_categorical(y_test)

score = model.evaluate(x_test, y_test )
score