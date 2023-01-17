# Tensorflow

import tensorflow as tf

import tensorflow_hub as hub

from tensorflow.keras.utils import plot_model

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Dense, Input, InputLayer, BatchNormalization, Dropout, Concatenate, Layer

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from tensorflow.keras import regularizers



from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import average_precision_score, auc, classification_report, confusion_matrix, roc_curve, precision_recall_curve



from nltk.corpus import stopwords

from nltk.util import ngrams



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import nltk

from collections import defaultdict

from collections import  Counter

import re

import gensim

import string

from tqdm import tqdm

# try:

#     import tensorflow_addons.metrics.F1Score 

# except ImportError as e:

#     !pip install tensorflow-addons  # module doesn't exist, install it.

# import tensorflow_addons.metrics.F1Score
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
train_data = train.text.values

train_labels = train.target.values

test_data = test.text.values

train_data.shape
# !pip install pyspellchecker

# from spellchecker import SpellChecker



# def remove_urls(text):

#     url = re.compile(r'https?://\S+|www\.\S+')

#     return url.sub(r'',text).strip()



# def remove_html(text):

#     html=re.compile(r'<.*?>')

#     return html.sub(r'',text).strip()



# def remove_emoji(text):

#     emoji_pattern = re.compile("["

#                            u"\U0001F600-\U0001F64F"  # emoticons

#                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs

#                            u"\U0001F680-\U0001F6FF"  # transport & map symbols

#                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

#                            u"\U00002702-\U000027B0"

#                            u"\U000024C2-\U0001F251"

#                            "]+", flags=re.UNICODE)

#     return emoji_pattern.sub(r'', text).strip()



# spell = SpellChecker()

# def correct_spellings(text):

#     corrected_text = []

#     misspelled_words = spell.unknown(text.split())

#     for word in text.split():

#         if word in misspelled_words:

#             corrected_text.append(spell.correction(word))

#         else:

#             corrected_text.append(word)

#     return " ".join(corrected_text)



# def remove_punct(text):

#     text  = "".join([char for char in text if char not in string.punctuation])

#     text = re.sub('[0-9]+', '', text)

#     return text.strip()





# stopword = stopwords.words('english')

# def remove_stopwords(text):

#     words = re.split('\W+', text)

#     non_stop_words = [word for word in words if word not in stopword]

#     return " ".join(non_stop_words)



# ps = nltk.PorterStemmer()



# def stemming(text):

#     words = re.split('\W+', text)

#     text = [ps.stem(word) for word in words]

#     return text



# wn = nltk.WordNetLemmatizer()

# def lemmatizer(text):

#     words = re.split('\W+', text)

#     lemmatized_words = [wn.lemmatize(word) for word in words]

#     return  " ".join(lemmatized_words)        

!pip install contractions

!pip install beautifulsoup4



import contractions

from bs4 import BeautifulSoup

import unicodedata

import re



def strip_html_tags(text):

    soup = BeautifulSoup(text, "html.parser")

    [s.extract() for s in soup(['iframe', 'script'])]

    stripped_text = soup.get_text()

    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)

    return stripped_text



def remove_accented_chars(text):

    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    return text



def expand_contractions(text):

    return contractions.fix(text)



def remove_special_characters(text, remove_digits=False):

    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'

    text = re.sub(pattern, '', text)

    return text



def pre_process_document(document):

    # strip HTML

    document = strip_html_tags(document)

    # lower case

    document = document.lower()

    # remove extra newlines (often might be present in really noisy text)

    document = document.translate(document.maketrans("\n\t\r", "   "))

    # remove accented characters

    document = remove_accented_chars(document)

    # expand contractions    

    document = expand_contractions(document)  

    # remove special characters and\or digits    

    # insert spaces between special characters to isolate them    

    special_char_pattern = re.compile(r'([{.(-)!}])')

    document = special_char_pattern.sub(" \\1 ", document)

    document = remove_special_characters(document, remove_digits=True)  

    # remove extra whitespace

    document = re.sub(' +', ' ', document)

    document = document.strip()

    

    return document





pre_process_corpus = np.vectorize(pre_process_document)
%%time



pd.options.display.max_colwidth = 200





# temp_cleaned_array = pre_process_corpus(train.text)

# temp_cleaned = pd.DataFrame(temp_cleaned_array)

# temp_cleaned.head(10)

print(train[train['target']==1]['text'].values[:10])



print(train[train['target']==0]['text'].values[:10])



train_data = pre_process_corpus(train.text)

print("T Shape:" + str(train_data.shape))

# train_data = pd.DataFrame(train_data)

train_labels = train.target.values

test_data = pre_process_corpus(test.text)



# pd.DataFrame(train_data).head(10)

# temp_cleaned = train.text.apply(lambda x: remove_URL(x))

# temp_cleaned = temp_cleaned.apply(lambda x: remove_html(x))

# temp_cleaned = temp_cleaned.apply(lambda x: remove_emoji(x))

# # temp_cleaned = temp_cleaned.apply(lambda x: correct_spellings(x))

# # temp_cleaned = temp_cleaned.apply(lambda x: remove_punct(x))

# # temp_cleaned = temp_cleaned.apply(lambda x: remove_stopwords(x))

# temp_cleaned = temp_cleaned.apply(lambda x: stemming(x))

# # temp_cleaned = temp_cleaned.apply(lambda x: lemmatizer(x))

# temp_cleaned.head(10)
%%time

module_url = 'https://tfhub.dev/google/universal-sentence-encoder-large/4'

use_embedding_layer = hub.KerasLayer(module_url, trainable=True, name='USE_embedding')

# use_embedding_module = hub.Module(module_url, trainable=False, name='USE_embedding')


class USEEmbeddingLayer(Layer):

    def __init__(self, **kwargs):

        self.dimensions = 512

        self.trainable=False

        self.module_url = 'https://tfhub.dev/google/universal-sentence-encoder-large/4'

        super(USEEmbeddingLayer, self).__init__(**kwargs)



    def build(self, input_shape):

        self.use = hub.Module(

            self.module_url,

            trainable=self.trainable,

            name="{}_module".format(self.name))

        

        super(USEEmbeddingLayer, self).build(input_shape)



    def call(self, inputs, mask=None):

        result = self.use(tf.squeeze(tf.cast(inputs, tf.string), axis=[1]),

                      as_dict=True,

                      signature='default',

                      )['default']

        return result



    def compute_mask(self, inputs, mask=None):

        return tf.not_equal(inputs, '--PAD--')



    def compute_output_shape(self, input_shape):

        return (input_shape[0], self.dimensions)
with tf.device('/gpu:0'):

    def build_model(pre_trained_embedding_Layer, train_pre_trained_embedding_Layer):

        model = Sequential()

        model.add(InputLayer(input_shape=[], dtype=tf.string))

        

        pre_trained_embedding_Layer.trainable = train_pre_trained_embedding_Layer

        model.add(pre_trained_embedding_Layer)

        #   model.add(USEEmbeddingLayer())

#         model.add(Dropout(0.5))

#         model.add(BatchNormalization())       

#         model.add(Dense(512, activation='relu'))

#         model.add(Dropout(0.8))

#         model.add(BatchNormalization())

#         model.add(Dense(128,  kernel_regularizer=regularizers.l2(0.001), activation='relu'))        # kernel_regularizer=regularizers.l2(0.01)

#         model.add(Dropout(0.8))

#         model.add(BatchNormalization())

#         model.add(Dense(64,  kernel_regularizer=regularizers.l2(0.001), activation='relu'))  # 

#         model.add(Dropout(0.8))

#         model.add(BatchNormalization())

#         model.add(Dense(1, kernel_regularizer=regularizers.l2(0.001), activation='sigmoid')) # kernel_regularizer=regularizers.l2(0.01),

        model.add(Dense(256, activation='relu'))

        model.add(BatchNormalization())

        model.add(Dropout(0.5))

        model.add(Dense(128, activation='relu'))

        model.add(BatchNormalization())

        model.add(Dropout(0.5))

        model.add(Dense(64, activation='relu'))

        model.add(BatchNormalization())

        model.add(Dropout(0.5))

        model.add(Dense(32, activation='relu'))

        model.add(BatchNormalization())

        model.add(Dropout(0.5))

        model.add(Dense(1, activation='sigmoid'))

        

        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(

        0.0001,

        decay_steps=187*1000, # STEPS_PER_EPOCH = Samples(6000)/BatchSize(32)

        decay_rate=1,

        staircase=False)

        optimizer = Adam(lr = 0.0001)



        metrics = [

            'accuracy', 

            tf.keras.metrics.Recall(),

            tf.keras.metrics.Precision()

        ]

        model.compile(optimizer, loss='binary_crossentropy', metrics=metrics)



        return model
model = build_model(use_embedding_layer, train_pre_trained_embedding_Layer = False)

model.summary()
X_train,X_test,y_train,y_test=train_test_split(train_data,train_labels,test_size=0.20, random_state = 777, shuffle = True)

print('Shape of train',X_train.shape)

print("Shape of Validation ",X_test.shape)



checkpoint = ModelCheckpoint('model_with_low_val_loss.h5', monitor='val_loss', mode='min', modelsave_best_only=True)

early_stopping = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.001, patience=10)

callbacks = [early_stopping, checkpoint]



train_history = model.fit(

    X_train, y_train,

    validation_data=(X_test,y_test),

    epochs=50,

    callbacks=callbacks,

    batch_size=32

)
model_loss = pd.DataFrame(model.history.history)

# model_loss.head()

model_loss[['loss','val_loss']].plot(ylim=[0,1])

model_loss[['accuracy','val_accuracy']].plot(ylim=[0,1])

pd.DataFrame(model.history.history).filter(regex="precision", axis=1).plot(ylim=[0,1])

pd.DataFrame(model.history.history).filter(regex="recall", axis=1).plot(ylim=[0,1])

# predictions = model.predict_classes(X_test) 

# print(classification_report(y_test, predictions, target_names=["Real", "Not Real"]))
def findClasses(predictions):

    true_preds = []

    a=1

    b=0



    for i in predictions:

        if i >= 0.5:

            true_preds.append(a)

        else:

            true_preds.append(b)

    return true_preds

model.load_weights('model_with_low_val_loss.h5')

predictions = model.predict(X_test) 

classes = findClasses(predictions)

print(classification_report(y_test, classes, target_names=["Real", "Not Real"]))


test_pred = model.predict(test_data)
pred =  pd.DataFrame(test_pred, columns=['preds'])

pred.plot.hist()
# This for loop its for round predictions

submission['target'] = findClasses(test_pred)

submission.to_csv('submission.csv', index=False)