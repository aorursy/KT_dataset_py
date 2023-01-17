# import libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os

# load data
df = pd.read_csv('../input/weatherAUS.csv')
df.head()
# let's check data info
df.info()
# we can also plot a heatmap to visualize missing values
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# we will drop some columns which has 30-40% of the values missing such as 'Evaporation'
# we drop date column as well since its of no use in this kernel
# we will drop few wind attributes as well
df.drop(['Date','Evaporation','Sunshine','Cloud9am','Cloud3pm',
         'WindGustDir','WindDir9am','WindDir3pm'],axis=1,inplace=True)
df.head()
# let's drop some of the rows of missing values
df.dropna(subset=['Pressure3pm','Humidity3pm','Pressure9am','RainToday'],axis=0,inplace=True)
# replace rest of the nulls with respective means
fill_feat = ['Temp9am','Humidity9am','MinTemp','MaxTemp','WindSpeed3pm','WindSpeed9am','WindGustSpeed']
for i in fill_feat:
    df[i].fillna(np.mean(df[i]),inplace=True)

df.info()
# count target class
sns.countplot(x='RainTomorrow',data=df)
# study relation between rain today and tomorrow
sns.countplot(x='RainTomorrow',data=df,hue='RainToday')
# now time to encode categorical variables to numerics
# we use Label encoder for 'Location', and pandas' get_dummies for 'RainToday' and 'RainTomorrow'
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Location'] = le.fit_transform(df['Location'])
df['RainToday'] = pd.get_dummies(df['RainToday'],drop_first=True)
df['RainTomorrow'] = pd.get_dummies(df['RainTomorrow'],drop_first=True)
# lets see if data is ready for training
df.head()
from sklearn.model_selection import train_test_split
X = df.drop('RainTomorrow',axis=1)
y = df['RainTomorrow']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print('Confusion Matrix = \n',confusion_matrix(y_test,pred))
print('\n',classification_report(y_test,pred))
sns.heatmap(confusion_matrix(y_test,pred),annot=True,yticklabels=False,cbar=False,cmap='plasma')