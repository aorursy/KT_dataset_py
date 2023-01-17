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



import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



rain_data = pd.read_csv('/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv')
rain_data.info()

rain_data.head()
rain_data.describe()
rain_data.median()
plt.figure(figsize=(15,10))

sns.catplot(x='Location',y='Rainfall',data=rain_data,aspect=10)
plt.figure(figsize=(12,6))

sns.heatmap(rain_data.isnull(),yticklabels=False,cbar=False)
rain_data['Date'] = pd.to_datetime(rain_data['Date'])
rain_data['Year'] = rain_data['Date'].dt.year



rain_data['Month'] = rain_data['Date'].dt.month



rain_data['Day'] = rain_data['Date'].dt.day
rain_data.drop(['Date','Evaporation','Sunshine','Cloud9am',"RISK_MM",'Cloud3pm'],axis=1,inplace=True)
plt.figure(figsize=(15,10))

sns.heatmap(rain_data.corr(), linecolor='black', linewidth=1, annot=True)
rain_data.drop(["Temp9am","Temp3pm"],inplace=True,axis=1)
plt.figure(figsize=(12,6))

sns.heatmap(rain_data.isnull(),yticklabels=False,cbar=False)
rain_data.select_dtypes(include=['object'])
rain_data['RainTomorrow'].value_counts()
rain_data['RainToday'].value_counts()
match_yn = {"RainTomorrow":{"Yes":1, "No":0},

             

             "RainToday":{"Yes":1, "No":0}

                

            }
rain_data.replace(match_yn,inplace=True)
rain_data.head()
final_data = pd.concat([rain_data,

                     pd.get_dummies(rain_data['Location']), 

                     pd.get_dummies(rain_data['WindGustDir'],prefix='WindGustDir'),

                     pd.get_dummies(rain_data['WindDir9am'],prefix='WindDir9am'),

                     pd.get_dummies(rain_data['WindDir3pm'],prefix='WindDir3pm'),

                     

                       ], axis=1)
final_data.drop(['Location','WindGustDir','WindDir9am','WindDir3pm'],axis=1,inplace=True)
nan_cols = [i for i in final_data.columns if final_data[i].isnull().any()]

nan_cols
for cols in nan_cols:

    

    final_data[cols].fillna((final_data[cols].median()), inplace=True)
nan_cols = [i for i in final_data.columns if final_data[i].isnull().any()]

nan_cols
final_data.columns.values
from sklearn.linear_model import LogisticRegression



logm = LogisticRegression(random_state=101)
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

X = final_data.drop('RainTomorrow',axis=1)

y = final_data['RainTomorrow']





std_scaler = StandardScaler()



X_std = std_scaler.fit_transform(X)







X_train, X_test, y_train, y_test = train_test_split(X_std,y, test_size=0.30, random_state=101)
logm.fit(X_train, y_train)
pred = logm.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,pred))
conf = confusion_matrix(y_test,pred)

print(conf)

print('\n')



print('True Positives: ',conf[0,0])

print('False Positives: ',conf[0,1])

print('False Negatives: ',conf[1,0])

print('True Negatives: ',conf[1,1])
