# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df= pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")

print(df)
df.head(5)
df.tail(5)
df.info()
df['class'].value_counts()
df['habitat'].value_counts()
objFeatures = df.select_dtypes(include="object").columns

print(objFeatures)
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

for feat in objFeatures:

    df[feat]= le.fit_transform(df[feat].astype(str))
df.info()
x=df.drop(['class'], axis=1)

y=df['class']
x.info()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)   
x_train.info()
#Model training

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(x_train, y_train)
#prediction

y_prediction= gnb.predict(x_test)



print(y_prediction)
print(y_test)
x_test.info()
temp_prediction = []



for i,val in enumerate(y_prediction):

    if val==int(0):

      print("e")



        

    else:

        print("p")

        

    

for d,c in zip(temp_prediction, y_prediction):

    print(d,c)
