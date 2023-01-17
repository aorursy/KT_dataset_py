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
#Read data file

data = pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")

print(data)
data.head(5)
data.info()
#To Know the count of each class of mushrooms

data['class'].value_counts()
#Extracting the columns of 'object' type

obj_features = data.select_dtypes(include = "object").columns

obj_features
from sklearn import preprocessing

le = preprocessing.LabelEncoder()



for feat in obj_features:

    data[feat] = le.fit_transform(data[feat].astype(str))
#After converting to int dtype

data.info()
data.head(10)
#defining the features and labels

x = data.drop(['class'], axis = 1)

y = data['class']
#splitting train and test set

from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
#Model Training

from sklearn.naive_bayes import GaussianNB



gnb = GaussianNB()



gnb.fit(x_train, y_train) 

#Predicting Y value

y_pred = gnb.predict(x_test)



print(y_pred)