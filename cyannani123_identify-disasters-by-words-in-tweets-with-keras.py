import numpy as np 

import pandas as pd

import os

import re

import tensorflow as tf

import tensorflow_hub as hub

import keras.backend as K

import matplotlib.pyplot as plt



from nltk.tokenize  import TweetTokenizer

from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.feature_extraction.text import CountVectorizer



#MAX_FEATURES = 10000

NUM_FEATURES = 2500

MIN_NGRAM_RANGE = 1

MAX_NGRAM_RANGE = 1
main_dir = '/kaggle/input/nlp-getting-started'

train_filename = "train.csv"

test_filename = "test.csv"



train_df = pd.read_csv(os.path.join(main_dir,train_filename))

test_df = pd.read_csv(os.path.join(main_dir,test_filename))
train_df.head()
test_df.head()
tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)



def custom_preprocessor(doc):

    """Preprocess text before tokenization."""

    

    doc = re.sub(r"http\S+", "this_was_an_url", doc).lower()

    doc = doc.replace("#","")

    doc = doc.replace("%20"," ")

    return doc



def custom_tokenizer(doc):

    """Tokenize text into list of words."""

    

    tokenized_doc = tokenizer.tokenize(doc)

    return tokenized_doc
all_texts = train_df["text"].append(test_df["text"]).reset_index(drop=True)

vectorizer = CountVectorizer(analyzer='word', 

                             ngram_range=(MIN_NGRAM_RANGE, MAX_NGRAM_RANGE),

                             preprocessor=custom_preprocessor,

                             tokenizer=custom_tokenizer,

                             #max_features=MAX_FEATURES

                            )

all_texts_vectorized = vectorizer.fit_transform(all_texts)

train_texts = all_texts_vectorized[0:len(train_df)]

test_texts = all_texts_vectorized[len(train_df):]
corr_df = pd.DataFrame(train_texts.todense(), columns=vectorizer.get_feature_names())

corr = corr_df.corrwith(train_df["target"]).fillna(0).abs().sort_values()

words_with_highest_correlation = list(corr[-(NUM_FEATURES):].index)
corr[-20:].plot.barh()
X = pd.DataFrame(train_texts.todense(), columns=vectorizer.get_feature_names())[words_with_highest_correlation]

y = train_df["target"]

X_te = pd.DataFrame(test_texts.todense(), columns=vectorizer.get_feature_names())[words_with_highest_correlation]



X.head()
def f1(y_true, y_pred):

    """F1 score for Keras model.

    

    Copyright (c) 2018 Guglielmo Camporese.

    

    https://www.kaggle.com/guglielmocamporese/macro-f1-score-keras

    

    """

    

    y_pred = K.round(y_pred)

    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)

    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)

    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)



    p = tp / (tp + fp + K.epsilon())

    r = tp / (tp + fn + K.epsilon())



    f1 = 2*p*r / (p+r+K.epsilon())

    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)

    return K.mean(f1)
def build_model():

    """Build the model."""

    

    model = Sequential([

    Dense(64, input_shape=(NUM_FEATURES,),activation="relu"),

    Dense(1,activation="sigmoid")

    ])

    

    model.compile(optimizer="adam",loss='binary_crossentropy', metrics=[f1])

    

    return model



model = build_model()

model.summary()
checkpoint = ModelCheckpoint('model.h5', monitor='val_f1', mode="max", save_best_only=True)

earlystopping = EarlyStopping(monitor='val_f1', min_delta=0, patience=10, verbose=0, mode='max', baseline=None, restore_best_weights=False)

reducelronplateau = ReduceLROnPlateau(monitor='val_f1', factor=0.75, patience=5, verbose=0, mode='max', min_delta=0.0001, cooldown=0, min_lr=0)



train_history = model.fit(

    X, y,

    validation_split=0.2,

    epochs=50,

    callbacks=[checkpoint, 

               earlystopping,

               reducelronplateau],

    batch_size=1500,

    verbose=0

)

plt.plot(train_history.history["val_f1"])

plt.legend(["val_f1"])

plt.show()

plt.plot(train_history.history["val_loss"])

plt.legend(["val_loss"])

plt.show()



print("Epochs Trained:", len(train_history.history["val_f1"]))

print("Best F1-Score:", max(train_history.history["val_f1"]))

print("Best Loss:", min(train_history.history["val_loss"]))
model.load_weights('model.h5')



submission = pd.DataFrame()

submission["id"] = test_df["id"]

submission["target"] = [int(el) for el in list(model.predict(X_te).round())]

submission.to_csv("submission.csv",index=False)

print(pd.read_csv("submission.csv"))