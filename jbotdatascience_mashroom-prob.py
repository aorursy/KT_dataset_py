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
df= pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')

df
# Read Top 5 Rows

df.head(5)
df.info()
df['class'].value_counts()
#Fetch Features of Object Type

objFeatures= df.select_dtypes(include='object').columns

objFeatures
from sklearn import preprocessing

le=preprocessing.LabelEncoder()

for f in objFeatures:

    df[f]=le.fit_transform(df[f].astype(str))



df.info()
df.head(5)
# X and Y

X=df.drop(['class'], axis=1)

Y=df['class']
X.info()
X.describe()
from sklearn.model_selection import train_test_split



X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
from sklearn.naive_bayes import GaussianNB

gnb=GaussianNB()

gnb.fit(X_train,Y_train)
y_prediction=gnb.predict(X_test)

y_prediction