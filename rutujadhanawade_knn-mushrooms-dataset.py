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
mushrooms=pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')

mushrooms.head()
mushrooms.describe()

#Check for null

mushrooms.info()
#Check for nulls

mushrooms.isnull().sum()
#Counts of each category

columns=mushrooms.columns

for i in columns:

    print('Feature : ',str(i),mushrooms[i].value_counts())
#Label Encoding

from sklearn.preprocessing import LabelEncoder

label_encoder=LabelEncoder()

mushrooms_encoded=mushrooms.apply(label_encoder.fit_transform)

mushrooms_encoded.head(10)
#Train Test Split

from sklearn.model_selection import train_test_split

X=mushrooms_encoded.drop('class',axis=1)

y=mushrooms_encoded['class']

#print(X)

#print(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)

X_train.head()
#Fit the model

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=30)

knn.fit(X_train,y_train)

knn.predict(X_test)
#Accuracy

knn.score(X_test,y_test)