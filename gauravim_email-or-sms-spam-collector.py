# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#write all imports here

import string

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import mean_absolute_error

from sklearn.naive_bayes import MultinomialNB

from nltk.corpus import stopwords
#import the data and analize it

data = pd.read_csv('../input/spam.csv', encoding='latin-1')

data.head()
data = data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)

data.shape
data.isnull().any()

#hence, no row having null value in the dataset.
data.groupby('v1').describe().transpose()

#Hence the data is not normalized. It is biased to positive direction.
def text_process(text):

    

    text = text.translate(str.maketrans('', '', string.punctuation))

    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]

    

    return " ".join(text)
X = data['v2']

Y = data['v1']

#remove stop words, these are useless to predict anything

#for X_i in range(1, len(X)):

 #   X[X_i] = [word for word in X[X_i] if word.lower() not in stopwords.words('english')]



X = X.apply(text_process)



train_X, val_X, train_Y, val_Y = train_test_split(X, Y, test_size = 0.3)
len(X)
#apply transformation 



encoder = LabelEncoder()

tfidfv = TfidfVectorizer('english')



train_Y = encoder.fit_transform(train_Y)

val_Y = encoder.transform(val_Y)



train_X = tfidfv.fit_transform(train_X)

val_X = tfidfv.transform(val_X)
model = MultinomialNB(alpha = 0.2)

model.fit(train_X, train_Y)

print(model.score(val_X, val_Y))