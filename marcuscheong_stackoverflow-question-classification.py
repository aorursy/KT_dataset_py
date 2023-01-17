# ========================

# Dependencies

# ========================



import numpy as np #Working with Arrays

import pandas as pd #Working with DataFrames



# Data Visualization ##########

import matplotlib.pyplot as plt

import seaborn as sns

###############################



# NLP Preprocessing ########################################################

!pip install langdetect

import re

import string

from langdetect import detect_langs

from sklearn.preprocessing import OneHotEncoder

from bs4 import BeautifulSoup

from tqdm import tqdm

from sklearn.decomposition import PCA

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

############################################################################



# Machine Learning ################################

import tensorflow as tf

from tensorflow import keras

from keras.models import Sequential

from keras.layers import Dense,Dropout,Activation

###################################################
# ========================

# Get Data

# ========================



path="../input/60k-stack-overflow-questions-with-quality-rate/data.csv"



dataDF=pd.read_csv(path)



print(dataDF.info())

dataDF.head()
# ===================================

# Combining text from Title and Body

# ===================================

dataDF['Text'] = dataDF['Title'] + ' ' + dataDF['Body']
# ====================================================

# Drop Text that are not predominantly in English

# ====================================================

drop_index=[]

for i in tqdm(range(len(dataDF))):

    detected_lan = detect_langs(dataDF['Text'].iloc[i])

    for lan in detected_lan:

        if lan.lang == "en" and lan.prob < 0.4:

            drop_index.append(i)

            break
dataDF=dataDF.drop(drop_index,axis=0)

print("Number of features dropped:", len(drop_index))
# ==========================

# Exploratory Data Analysis

# ==========================



# Check if there are any Nan values in DataFrame

print("Nan value check")

print(dataDF.isnull().any(),'\n')



print("Check the total and unique text and labels")

print(dataDF[['Text','Y']].describe()[:2])



# Distribution of ratings

y = dataDF['Y'].values

plt.title("Distribution of Labels")

sns.countplot(y)
# ===========================

# Cleaning/Updating DataFrame

# ===========================



# Store Tags in a list

# def clean_tags(text):

#     clean=re.split('[<>]',text)

#     clean=[w for w in clean if w != '']

#     return clean

# dataDF['Tags']=dataDF['Tags'].progress_apply(lambda x : clean_tags(x))



def clean_text(text):

    # Convert text to lowercase

    text=text.lower()

    # Remove punctuations

    text=''.join(c for c in text if c not in string.punctuation)

    # Convert html to text

    soup=BeautifulSoup(dataDF['Body'].values[0],'lxml')

    s=soup.get_text('\n')

    s=s.replace('\n','')

    # Expand common contractions

    text = re.sub(r"n\'t", " not", text)

    text = re.sub(r"\'re", " are", text)

    text = re.sub(r"\'s", " is", text)

    text = re.sub(r"\'d", " would", text)

    text = re.sub(r"\'ll", " will", text)

    text = re.sub(r"\'t", " not", text)

    text = re.sub(r"\'ve", " have", text)

    text = re.sub(r"\'m", " am", text)

    text = ' '.join(text.split())

    return text

tqdm.pandas()

dataDF['Text']=dataDF['Text'].progress_apply(lambda x : clean_text(x))

# ====================

# OneHotEncode Labels 

# ====================

encoder=OneHotEncoder()

encoded_arr=encoder.fit_transform(dataDF[['Y']]).toarray()





X=dataDF['Text']

y=encoded_arr

dataDF=dataDF.drop(['Id','Tags','CreationDate'],axis=1)

dataDF.head()
# ===========================

# Clear up unnecessary memory

# ===========================

del dataDF
# # =========================================

# # Visualizing The Most Commonly Used Words

# # =========================================

# unigram_vec=CountVectorizer(stop_words='english')

# unigram_bow=unigram_vec.fit_transform(tqdm(X))



# bigram_vec=CountVectorizer(ngram_range=(2,2),stop_words='english')

# bigram_bow=bigram_vec.fit_transform(tqdm(X))



# trigram_vec=CountVectorizer(ngram_range=(3,3),stop_words='english')

# trigram_bow=trigram_vec.fit_transform(tqdm(X))
# ==============================

# Preparing train and test data

# ==============================

from sklearn.model_selection import train_test_split



xtrain,xtest,ytrain,ytest= train_test_split(X,y,test_size=0.2)



train_text=xtrain.values

test_text=xtest.values
# =====================================

# Converting text data for model input

# =====================================



vectorizer = TfidfVectorizer(stop_words='english',max_features=4000)

vectorizer.fit(tqdm(X))

train_vec=vectorizer.transform(tqdm(train_text)).toarray()

test_vec=vectorizer.transform(tqdm(test_text)).toarray()



VOCAB_SIZE=len(vectorizer.vocabulary_)

FEATURES=len(train_vec[0])
# ===========================

# Model

# ===========================



model=Sequential()

    

model.add(Dense(units=32, 

                input_shape=(FEATURES,), 

                kernel_regularizer=keras.regularizers.l2(0.001), 

                activation='relu'))

model.add(Dropout(0.2))



model.add(Dense(units=64, 

                kernel_regularizer=keras.regularizers.l2(0.001), 

                activation='relu'))

model.add(Dropout(0.3))



model.add(Dense(units=128,

                kernel_regularizer=keras.regularizers.l2(0.001), 

                activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(3,activation='softmax'))

    

model.compile(loss='categorical_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])

    

model.summary()

callback=keras.callbacks.EarlyStopping(monitor='loss',patience=2)



history=model.fit(train_vec,ytrain,epochs=20,validation_split=0.3,shuffle=True,callbacks=[callback])
#  "Accuracy"

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()

# "Loss"

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
loss,acc=model.evaluate(test_vec,ytest,verbose=False)

print("Loss:",loss)

print("Accuracy:",acc)
# ==============================

# Tuning Model with Keras Tuner

# ==============================

print("Starting Keras Tuner \n")



import kerastuner as kt

import IPython





def build_model(hp):



    model=Sequential()



    model.add(Dense(units=hp.Int('units_0', min_value=32, max_value=128, step=32), 

                    input_shape=(FEATURES,), 

                    kernel_regularizer=keras.regularizers.l2(0.001), 

                    activation='relu'))

    model.add(Dropout(0.2))



    model.add(Dense(units=hp.Int('units_1', min_value=32, max_value=128, step=32), 

                    kernel_regularizer=keras.regularizers.l2(0.001), 

                    activation='relu'))

    model.add(Dropout(0.3))



    model.add(Dense(units=hp.Int('units_2', min_value=32, max_value=128, step=32),

                    kernel_regularizer=keras.regularizers.l2(0.001), 

                    activation='relu'))

    model.add(Dropout(0.5))



    model.add(Dense(3,activation='softmax'))

    

    hp_learning_rate=hp.Choice('learning_rate',values=[1e-2,1e-3,1e-4])



    model.compile(loss='categorical_crossentropy',

                  optimizer=keras.optimizers.RMSprop(learning_rate=hp_learning_rate),

                  metrics=['accuracy'])



    model.summary()

    

    return model



tuner=kt.Hyperband(build_model,

                  objective='val_accuracy',

                  max_epochs=20,

                  factor=3,

                  directory='my_dir',

                  project_name='project')



class CallBack(keras.callbacks.Callback):

    def on_train_end(*args, **kwargs):

        IPython.display.clear_output(wait=True)



tuner.search(train_vec,

             ytrain, 

             epochs=20,

            validation_split=0.3,

            verbose=0)



best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]



print(f"""

The hyperparameter search is complete.\n 

The optimal number of units in the input densely-connected

layer is {best_hps.get('units_0')} \n

The optimal number of units in the first densely-connected

layer is {best_hps.get('units_1')} \n

The optimal number of units in the second densely-connected

layer is {best_hps.get('units_2')} \n

The optimal learning rate for the optimizer

is {best_hps.get('learning_rate')}.

""")

model_t = tuner.hypermodel.build(best_hps)

history_t=model_t.fit(train_vec,ytrain,epochs=20,validation_split=0.3,shuffle=True,callbacks=[callback])
#  "Accuracy"

plt.plot(history_t.history['accuracy'])

plt.plot(history_t.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()

# "Loss"

plt.plot(history_t.history['loss'])

plt.plot(history_t.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
loss,acc=model_t.evaluate(test_vec,ytest)

print(loss,acc)