# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
#os.chdir('../input')

# Any results you write to the current directory are saved as output.
test_set = pd.read_csv('../input/test-gaip-stud/test_data.csv')
import re
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))

def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    lower_case = letters_only.lower()
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = tok.tokenize(lower_case)
    return (" ".join(words)).strip()
test_text = [tweet_cleaner(i) for i in test_set.text]

len(test_text)
from nltk import word_tokenize
test_text = [word_tokenize(i) for i in test_text]
train_set = pd.read_csv('../input/gaip-yelp-project/cleaned_processed_train.csv')
train_set.dropna(axis=0, inplace=True)
from nltk.stem import PorterStemmer
portstem = PorterStemmer()
ded = []
for i in test_text:
    k = [portstem.stem(_) for _ in i]
    ded.append(k)
test_text = [i for i in ded]
transform_corpus = [train_set['text'], pd.Series(test_text)]
transform_corpus = pd.concat(transform_corpus)
transform_corpus.dropna(axis=0, inplace=True)
transform_corpus.head()
ded = []
for sublist in transform_corpus:
    for item in sublist:
        ded.append(item)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
ded = vectorizer.fit_transform(ded)
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
tfidf.fit(ded)
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(tfidf.transform(vectorizer.transform(train_set.text)), train_set.label, test_size=0.25, random_state=42)
encoder = LabelEncoder()
encoder.fit(train_set.label)
encoded_Y = encoder.transform(train_set.label)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
k = tfidf.transform(vectorizer.transform(train_set.text)).toarray()
Train_transformed = [train_set['label'], ]
Train_transformed = pd.concat(Train_transformed)
transform_corpus.dropna(axis=0, inplace=True)
Train_transformed.to_csv('Train_transformed.csv',index=False)
Train_transformed = pd.read_csv('Train_transformed.csv')
model = Sequential()
model.add(Dense(32, input_dim=61293))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
model.fit(X_train, np_utils.to_categorical(encoder.transform(Y_train)),
          batch_size=100,
          epochs=10,
          validation_data=(X_test, np_utils.to_categorical(encoder.transform(Y_test))))

score = model.evaluate(X_test, np_utils.to_categorical(encoder.transform(Y_test)), verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
df = pd.read_csv('../input/test-gaip-stud/test_data.csv')
print(df.head())

numpy_array1 = df.as_matrix()
test_reviews = numpy_array1[:,1]
test_id = numpy_array1[:,0]
#print(test_reviews[0])
final_predict = model.predict(tfidf.transform(vectorizer.transform(test_set.text)))
final_predict = np.argmax(final_predict, axis=1)
print(final_predict[:100])

submission = pd.DataFrame({'ID':test_id,'label':final_predict})
print(submission.head())
os.chdir('..')
submission.to_csv('group_6_sub4.csv',index=False)




