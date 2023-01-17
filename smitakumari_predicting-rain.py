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
df= pd.read_csv("/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv")
df.info()
df= df.drop(columns=["Evaporation","Sunshine","Cloud9am","Cloud3pm"],axis=1)

df.shape
df = df.drop(columns=["Date","Location"],axis=1)

df.shape
df.isnull().sum()
df= df.dropna(how="any")

df.shape
from sklearn.preprocessing import LabelEncoder

labelencoder_df = LabelEncoder()

df['WindGustDir']= labelencoder_df.fit_transform(df['WindGustDir'])

df['WindDir9am']= labelencoder_df.fit_transform(df['WindDir9am'])

df['WindDir3pm']= labelencoder_df.fit_transform(df['WindDir3pm'])

df['RainToday']= labelencoder_df.fit_transform(df['RainToday'])

df['RainTomorrow']= labelencoder_df.fit_transform(df['RainTomorrow'])
df.head()

df.tail()
import matplotlib.pyplot as plt

import seaborn as sns

corr= df.corr()

plt.figure(figsize=(12,10))

sns.heatmap(corr,xticklabels= corr.columns.values,yticklabels= corr.columns.values,annot= True,fmt='.2f',linewidth=0.30)
x=df.iloc[:,0:17].values

y=df.iloc[:,-1].values
x.shape

y.shape
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler

sc_x= StandardScaler()

x_train = sc_x.fit_transform(x_train)

x_test = sc_x.fit_transform(x_test)
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators= 200, random_state=0)

classifier.fit(x_train,y_train)
y_pred= classifier.predict(x_test)
from sklearn.metrics import confusion_matrix,accuracy_score

cm= confusion_matrix(y_test,y_pred)

accuracy = accuracy_score(y_test,y_pred)

print("RandomForestClassification:")

print("Accuracy = ",accuracy)

print(cm)
from collections import Counter

Counter(y_train)
Counter(y_test)
Counter(y_pred)