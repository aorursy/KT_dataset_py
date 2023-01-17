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
df = pd.read_csv('/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv')
df
import seaborn as sns
sns.countplot(df.RainTomorrow)
df.isnull().sum()
df.MinTemp.value_counts()
df.MinTemp.fillna(9.6,inplace=True)
df.MaxTemp.value_counts()
df.MaxTemp.fillna(20.0,inplace=True)
df.Rainfall.value_counts()
df.Rainfall.fillna(0.0,inplace=True)
df.Evaporation.value_counts()
df.Evaporation.fillna(4.0,inplace=True)
df.Sunshine.value_counts()
df.Sunshine.fillna(0.0,inplace=True)
df.WindGustDir.value_counts()
df.WindGustDir.fillna('W',inplace=True)
df.WindGustSpeed.value_counts()
df.WindGustSpeed.fillna(35.0,inplace=True)
df.WindDir9am.value_counts()
df.WindDir9am.fillna('N',inplace=True)
df.WindDir3pm.value_counts()
df.WindDir3pm.fillna('SE',inplace=True)
df.WindSpeed9am.value_counts()
df.WindSpeed9am.fillna(9.0,inplace=True)
df.WindSpeed3pm.value_counts()
df.WindSpeed3pm.fillna(13.0,inplace=True)
df.Humidity9am.value_counts()
df.Humidity9am.fillna(99.0,inplace=True)
df.Humidity3pm.value_counts()
df.Humidity3pm.fillna(52.0,inplace=True)
df.Pressure9am.value_counts()
df.Pressure9am.fillna(1016.4,inplace=True)
df.Pressure3pm.value_counts()
df.Pressure3pm.fillna(1015.5,inplace=True)
df.Cloud9am.value_counts()
df.Cloud9am.fillna(7.0,inplace=True)
df.Cloud3pm.value_counts()
df.Cloud3pm.fillna(7.0,inplace=True)
df.Temp9am.value_counts()
df.Temp9am.fillna(17.0,inplace=True)
df.Temp3pm.value_counts()
df.Temp3pm.fillna(20.0,inplace=True)
df.RainToday.value_counts()
df.RainToday.fillna('No',inplace=True)
df.info()
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

df['RainTomorrow']=label.fit_transform(df.RainTomorrow)

df['RainToday']=label.fit_transform(df.RainToday)

df['WindDir9am']=label.fit_transform(df.WindDir9am)

df['WindDir3pm']=label.fit_transform(df.WindDir3pm)

df['WindGustDir']=label.fit_transform(df.WindGustDir)

df['Location']=label.fit_transform(df.Location)

df['Date']=label.fit_transform(df.Date)
from sklearn.model_selection import train_test_split

train, test=train_test_split(df, test_size=0.2, random_state=1)
def data_splitting(df):

    x=df.drop(['RainTomorrow'], axis=1)

    y=df['RainTomorrow']

    return x, y



x_train, y_train=data_splitting(train)

x_test, y_test=data_splitting(test)
from sklearn.utils import resample

import imblearn

from imblearn.over_sampling import SMOTE

sm = SMOTE()

x_train, y_train = sm.fit_sample(x_train, y_train)
x_train = pd.DataFrame(data=x_train)

x_train.columns = ['Date','Location','MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','WindGustDir','WindGustSpeed','WindDir9am','WindDir3pm','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm','RainToday','RISK_MM'] 

y_train = pd.DataFrame(data = y_train)

y_train.columns = ['RainTomorrow']
sns.countplot('RainTomorrow', data=y_train)
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression



log_model=LogisticRegression()

log_model.fit(x_train, y_train)

prediction=log_model.predict(x_test)

score= accuracy_score(y_test, prediction)

print(score)
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

smote = XGBClassifier()

smote.fit(x_train, y_train)



smote_pred = smote.predict(x_test)

accuracy = accuracy_score(y_test, smote_pred)

print("Test Accuracy is {:.2f}%".format(accuracy * 100.0))
from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor()

reg.fit(x_train , y_train)

reg_train = reg.score(x_train , y_train)

reg_test = reg.score(x_test , y_test)





print(reg_train*100)

print(reg_test*100)