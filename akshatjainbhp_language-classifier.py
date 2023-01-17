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
import pandas as pd

df1 = pd.read_csv("../input/english_text.csv")

df1
df1["ID"]=0

df1

df2 = pd.read_csv("../input/hinglish_text.csv")

df2
df2["ID"]=1

df2
df1['text'] = df1['text'].str.replace(r'\W', ' ')

df1
df2['text'] = df2['text'].str.replace(r'\W', ' ')

df2
df1=df1.apply(lambda x: x.astype(str).str.lower())

df1.head(20)
df2=df2.apply(lambda x: x.astype(str).str.lower())

df2.head(20)
frames = [df1, df2]

result = pd.concat(frames)

result
from sklearn.utils import shuffle

result = shuffle(result)

result
result.shape
result.head(60)
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer



X=result['text']

y=result['ID']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# print(X_train.shape, y_train.shape)

# print(X_test.shape, y_test.shape)

cv=CountVectorizer(stop_words='english')



cv_train= cv.fit_transform(X_train)

cv_train

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
cv.get_feature_names()

d=cv_train.toarray()

d
d.shape
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
cv_train
y_train.head
model.fit(cv_train,y_train.astype("int"))
cv_train.shape
pred = model.predict(cv.transform(X_test))

pred
model.score(cv_train,y_train.astype("int"))