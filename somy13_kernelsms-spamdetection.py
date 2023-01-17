# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.naive_bayes import MultinomialNB
df = pd.read_csv("../input/spam.csv", encoding= 'ISO-8859-1')

df.head()

drop_col = [col for col in df.columns if df[col].isnull().any()]

#drop_col

#df.columns

df = df.drop(drop_col, axis =1)
from sklearn.preprocessing import LabelEncoder



lenc = LabelEncoder()

c_vec = CountVectorizer(decode_error='ignore')

X = df['v2']

Y = df['v1']



Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.2)

model = MultinomialNB()



Ytrain = lenc.fit_transform(Ytrain)

Ytest = lenc.transform(Ytest)
Xtrain = c_vec.fit_transform(Xtrain)

Xtest = c_vec.transform(Xtest)

model.fit(Xtrain, Ytrain)

model.score(Xtest, Ytest)