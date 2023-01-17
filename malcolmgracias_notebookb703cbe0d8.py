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



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# read data file

data = pd.read_csv ("/kaggle/input/mushroom-classification/mushrooms.csv")

print (data)
data.tail(5)
data.info()
data['class'].value_counts()
data['bruises'].value_counts()
# label Encoding

from sklearn import preprocessing



le = preprocessing.LabelEncoder()

le.fit([1, 2, 2, 6])

le.transform([1, 1, 2, 6])
#Fetch features of type Object

objFeatures = data.select_dtypes(include="object").columns

print(objFeatures)
data.info()
# iterate a loop for features of type object

from sklearn import preprocessing

le = preprocessing.LabelEncoder()



for feat in objFeatures:

    data[feat] = le.fit_transform(data[feat].astype(str))
#X and Y

X = data.drop(['class'], axis=1)# features

y = data['class'] # Labels
X.info()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train.info()
X_test.info()
# Model Training

from sklearn.naive_bayes import GaussianNB # Gaussian Naive Bayes



gnb = GaussianNB()



gnb.fit(X_train, y_train)
#Predicting y value

y_prediction = gnb.predict(X_test)



print (y_prediction)