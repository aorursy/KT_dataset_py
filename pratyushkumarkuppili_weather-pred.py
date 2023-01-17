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
data = pd.read_csv('/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv')
data.head()
data.info()
data.shape[0]
data_missing_percentage = round((data.isna().sum()/data.shape[0] *100),2)
data_missing_percentage
data.Location.unique()
data_Albury = data[data.Location == 'Albury']
data_Albury
data_BadgerysCreek = data[data.Location =='BadgerysCreek']
data_BadgerysCreek
data.shape
plt.scatter(data.MinTemp, data.MaxTemp)
plt.scatter(data.MinTemp, data.RainTomorrow)
sns.countplot(data.RainTomorrow)
sns.countplot(data.RainTomorrow,hue = data.RainToday)
plt.figure(figsize=(10,8))
plt.subplot(1,3,1)
plt.hist(data.MinTemp, bins = 10)
plt.subplot(1,3,2)
plt.hist(data.MaxTemp, bins = 10)
plt.subplot(1,3,3)
plt.hist(data.MaxTemp - data.MinTemp, bins = 10)
data['temp_range'] = data['MaxTemp']-data['MinTemp']
data.head()
data.temp_range.describe()
data = data[~data.temp_range.isna()]
data.head()
data.temp_range.isna().sum()
data_range_1 = data[(data.temp_range<=10) & (data.temp_range>=0)]
data_range_2 = data[(data.temp_range<=20) & (data.temp_range>10)]
data_range_3 = data[(data.temp_range<=data.temp_range.max()) & (data.temp_range>20)]
plt.figure(figsize = (10,6))
plt.subplot(1,3,1)
sns.countplot(data_range_1.RainTomorrow)
plt.subplot(1,3,2)
sns.countplot(data_range_2.RainTomorrow)
plt.subplot(1,3,3)
sns.countplot(data_range_3.RainTomorrow)
sns.boxplot(data.Rainfall)
data.Rainfall = data.Rainfall.fillna(0)
data
data.Rainfall.isna().sum()
data.Evaporation.value_counts()
data.isna().sum()/data.shape[0]*100
cols = ['Evaporation','Sunshine', 'Cloud9am', 'Cloud3pm', 'Date', 'Location','MinTemp', 'MaxTemp']
data.drop(cols,1,inplace = True)
data.head()
data.WindGustDir.value_counts()
data = data.dropna()
data = data.reset_index(drop = True)
data
data.isna().sum()
data.shape
data.columns
data.info()
num_vals = ['Rainfall', 'WindGustSpeed','WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm', 'RISK_MM', 'temp_range']
cat_vals = ['WindGustDir', 'WindDir9am','WindDir3pm', 'RainToday']
data_num = data[num_vals]
data_num.head()
data['windspeed_change']= data['WindSpeed3pm']- data['WindSpeed9am']
data['humidity_change'] = data['Humidity3pm']- data['Humidity9am']
data['pressure_change'] = data['Pressure3pm'] - data['Pressure9am']
data['temp_range'] = data['Temp3pm'] - data['Temp9am']
data.head()
data.drop(['WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm'],1,inplace = True)
data.head()
num_vals = ['Rainfall', 'WindGustSpeed', 'RISK_MM', 'temp_range', 'windspeed_change', 'humidity_change', 'pressure_change']
cat_vals = ['WindGustDir', 'WindDir9am','WindDir3pm', 'RainToday']
data[cat_vals]
dummmies= pd.get_dummies(data[cat_vals], prefix_sep = '_', drop_first = True)
dummies
dummies = dummies.reset_index(drop = True)
dummies
data[num_vals].head()
from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
minmax_data = minmax.fit_transform(data[num_vals])
scaled =pd.DataFrame(minmax_data, columns = num_vals)
print(scaled.shape)
print(dummies.shape)
merged_data = pd.concat([scaled, dummies],axis = 1)
X = pd.concat([merged_data, data['RainTomorrow']], axis = 1)
X
X['RainTomorrow']
X['RainTomorrow'] = X['RainTomorrow'].map({'Yes':1, 'No':0})
X['RainTomorrow']
X.head()
from sklearn.model_selection import train_test_split
y = X['RainTomorrow']
y.head()
X = X.drop('RainTomorrow',1)
X.head()
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size= 0.85, random_state = 42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
y_pred
y_test
sum(y_test != y_pred)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
print("The accurcay is {}". format(accuracy_score(y_test, y_pred)))
print("The f1 score is {}". format(f1_score(y_test, y_pred)))
print("The precision is {}". format(precision_score(y_test, y_pred)))
confusion_matrix(y_test, y_pred)
