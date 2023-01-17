# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#importing data into dataframe

df=pd.read_csv(r"/kaggle/input/did-it-rain-in-seattle-19482017/seattleWeather_1948-2017.csv")

df.head()
#shape of dataframe

df.shape
#Info of dataframe

df.info()
#Unique values in predictor column

df['RAIN'].unique()
#Printing rows having null values

df[df['RAIN'].isnull()]
#Dropping rows having null values

df=df.dropna()
#updated shape of dataframe

df.shape
#Converting RAIN column into 1 and 0 by mapping True as 1 and False as 0

df['RAIN']=df['RAIN'].astype('int')
#Checking imbalance of dataset

df['RAIN'].value_counts()
df=df.drop('DATE',axis=1)
#Heatmap for correlation between variables

import seaborn as sns

plt.figure(figsize=(8, 8))

sns.heatmap(df.corr(),annot=True)
#TMIN and TMAX are highly positively correlated to each other as observed from above correlation matrix.
#Splitting dependent and independent variables

y=df.pop('RAIN')

X=df
#Splitting train and test dataset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
#Default Random forest classifier

from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier()
#Fitting train data into model

clf.fit(X_train,y_train)
#Prediction on test data

pred=clf.predict(X_test)
# Printing confusion matrix

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,pred))
#Classification report

from sklearn import metrics

print(metrics.classification_report(y_test, pred))
#Feature Importance

clf.feature_importances_
#We can see that first feature in dataframe i.e. PRCP is the most influencing factor for rain prediction.

#Also we achieved 100% accuracy on test dataset using random forest classifier.