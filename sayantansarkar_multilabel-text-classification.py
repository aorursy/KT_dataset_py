# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_colwidth', None)

import string
import keras

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/medicaltranscriptions/mtsamples.csv')
df.head(1)
tweet = "I am tired! I like fruit...and milk"
translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
tweet.translate(translator)
no_punc_translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
df['transcription_lower']=df['transcription'].apply(lambda x: ' '.join([i for i in str(x).lower().translate(no_punc_translator).split(' ') if i.isalpha()]))
df.head(1)
df['medical_specialty'].nunique()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
vectorizer=CountVectorizer(analyzer='word')
feature_space=vectorizer.fit_transform(list(df['transcription_lower']))
count_vect_df = pd.DataFrame(feature_space.todense(), columns=vectorizer.get_feature_names())
new_df=pd.concat([df, count_vect_df], axis=1)
new_df.columns
X=new_df.loc[:, 'aa':]


lb_make = LabelEncoder()
new_df["medical_specialty_code"] = lb_make.fit_transform(new_df["medical_specialty"])


Y=new_df['medical_specialty_code']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
y_train=keras.utils.to_categorical(y_train, df['medical_specialty'].nunique())
y_test=keras.utils.to_categorical(y_test, df['medical_specialty'].nunique())
print(X_train.shape)
# print(X_train[0])

print(y_train.shape)
print(y_train[0])
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint
def build_sequential(input_size, output_size):
    model=Sequential()
    model.add(Dense(512, input_shape=(input_size, )   ))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(output_size)))
    model.add(Activation('softmax'))
    return model
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[keras.metrics.CategoricalAccuracy()])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size=200
epochs=5

# checkpoint=ModelCheckpoint('model-{epoch:03d}.model', monitor='val_loss', verbose=0, save_best_only=False, mode='auto')

history=model.fit(np.array(X_train), np.array(y_train), batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1)
print (model.evaluate(np.array(X_test), np.array(y_test)))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', max_features=10000)
tfidf_feature_space = tfidf_vect.fit_transform(df['transcription_lower'])
tfidf_vect_df = pd.DataFrame(tfidf_feature_space.todense(), columns=tfidf_vect.get_feature_names())
tfidf_df=pd.concat([df, tfidf_vect_df], axis=1)

X_tfidf=tfidf_df.loc[:, 'abc':]
encoder = LabelEncoder()
tfidf_df["medical_specialty_code"] = encoder.fit_transform(tfidf_df["medical_specialty"])
Y_tfidf=tfidf_df['medical_specialty_code']

X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(X_tfidf, Y_tfidf, test_size=0.2, random_state=42)

y_train_tfidf=keras.utils.to_categorical(y_train_tfidf, df['medical_specialty'].nunique())
y_test_tfidf=keras.utils.to_categorical(y_test_tfidf, df['medical_specialty'].nunique())

model = build_sequential(X_train_tfidf.shape[1], encoder.classes_)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size=100
epochs=30

# checkpoint=ModelCheckpoint('model-{epoch:03d}.model', monitor='val_loss', verbose=0, save_best_only=False, mode='auto')

history=model.fit(np.array(X_train_tfidf), np.array(y_train_tfidf), batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.3)
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(X_tfidf, Y_tfidf, test_size=0.2, random_state=42)
# svm_classifier=svm.SVC()
# svm_classifier.fit(X_train_tfidf, y_train_tfidf)
# predictions = svm_classifier.predict(X_test_tfidf)
metrics.accuracy_score(predictions, y_test_tfidf)
# tfidf_df['medical_specialty'].unique()
tfidf_df[['medical_specialty','Unnamed: 0']].groupby('medical_specialty').count()
# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(trainDF['text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(trainDF['text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 