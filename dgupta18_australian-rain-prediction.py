# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns

import missingno as msno

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix,classification_report



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv(os.path.join(dirname, filename))

df.head(10)
df.info()
df.describe().T
msno.bar(df)
def wind_encode(col):

    if col == 'N':

        return 1

    elif col == 'S':

        return 2

    elif col == 'E':

        return 3

    elif col == 'W':

        return 4

    elif col == 'NE':

        return 5

    elif col == 'NW':

        return 6

    elif col == 'SE':

        return 7

    elif col == 'SW':

        return 8

    elif col == 'NNW':

        return 9

    elif col == 'NNE':

        return 10

    elif col == 'SSW':

        return 11

    elif col == 'SSE':

        return 12

    elif col == 'ESE':

        return 13

    elif col == 'ENE':

        return 14

    elif col == 'WSW':

        return 15

    elif col == 'WNW':

        return 16
df['WindGustDir'] = df['WindGustDir'].apply(wind_encode)

df['WindDir9am'] = df['WindDir9am'].apply(wind_encode)

df['WindDir3pm'] = df['WindDir3pm'].apply(wind_encode)
df['RainTodayYes'] = pd.get_dummies(df['RainToday'],drop_first=True)

df['RainTomorrowYes'] = pd.get_dummies(df['RainTomorrow'],drop_first=True)
df
rain_df = df.drop(['Evaporation','Sunshine','Cloud3pm','Cloud9am','Date','Location','RainToday','RainTomorrow'],axis = 1)
rain_df[~((rain_df['WindGustDir'] >= 1.0) & (rain_df['WindGustDir'] <=16.0))]
rain_df = rain_df.fillna(round(rain_df.mean()))
X = rain_df.drop(['RainTomorrowYes'],axis = 1)

y = rain_df['RainTomorrowYes']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101)
logreg = LogisticRegression()

logreg.fit(X_train,y_train)

logspred = logreg.predict(X_test)

print(confusion_matrix(logspred,y_test))

print(classification_report(logspred,y_test))

print(logreg.score(X_test,y_test))
rtree = RandomForestClassifier(n_estimators=100)

rtree.fit(X_train,y_train)

prediction = rtree.predict(X_test)
print(confusion_matrix(prediction,y_test))

print(classification_report(prediction,y_test))

print(rtree.score(X_train,y_train))
knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(X_train,y_train)

knn_predict = knn.predict(X_test)
print(confusion_matrix(knn_predict,y_test))

print(classification_report(knn_predict,y_test))
print(knn.score(X_test,y_test))