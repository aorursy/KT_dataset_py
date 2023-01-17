#import required libraries

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

import numpy as np

import time

%matplotlib inline

import warnings

import os

warnings.filterwarnings('ignore')

os.listdir('../input/2018-chicago-crime-data')
df = pd.read_csv('../input/2018-chicago-crime-data/Crimes_-_2018.csv')

print(df.isnull().sum())

print("--------------------------")

print("this dataset has ",len(df)," observations")
df.dropna(inplace=True)

df.reset_index(drop=True, inplace=True)

df.head(2)
from sklearn.preprocessing import StandardScaler

sf = df[['Primary Type','Longitude','Latitude']]

scaler = StandardScaler()

scaler.fit(sf.drop('Primary Type',axis=1))

scaled_features = scaler.transform(sf.drop('Primary Type',axis=1))

sf_feat = pd.DataFrame(scaled_features,columns=sf.columns[1:])

sf_feat.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(scaled_features,sf['Primary Type'],

                                                    test_size=0.30)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)

pred = knn.predict(X_test)

print(classification_report(y_test,pred))
results = pd.DataFrame(classification_report(y_test,pred,output_dict=True))

results = results.swapaxes("index", "columns") 

results['categories'] = results.index

results = results.sort_values('f1-score',ascending=0)

results.drop(['accuracy','macro avg','weighted avg'],inplace=True)

results.insert(0,'K-Value','k=1')

results.reset_index(drop=True, inplace=True)

fig = px.bar(results,x='categories',y="f1-score",color_discrete_sequence=('#00A8E8','#003459'),

             opacity=.7,title='F1 Scores by Crime Type')

fig.show()
error_rate = []

for i in range(1,40):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))

error_rt = pd.DataFrame(error_rate,columns = ['error rate'])

error_rt['K-value'] = range(1,40)

fig = px.line(error_rt,x='K-value',y='error rate',color_discrete_sequence=('#1D3557','#00A8E8')

             ,title='Error Rate Using Different K-Values')

fig.show()
knn = KNeighborsClassifier(n_neighbors=35)

knn.fit(X_train,y_train)

pred = knn.predict(X_test)

optimized_results = pd.DataFrame(classification_report(y_test,pred,output_dict=True))

optimized_results = optimized_results.swapaxes("index", "columns") 

optimized_results['categories'] = optimized_results.index

optimized_results.drop(['accuracy','macro avg','weighted avg'],inplace=True)

optimized_results.insert(0,'K-Value','k=25')

optimized_results.reset_index(drop=True, inplace=True)

combined = results.append(optimized_results,ignore_index = True)

combined = combined.sort_values('f1-score',ascending=0)

fig = px.bar(combined,x='categories',y="f1-score",color='K-Value',barmode='group',color_discrete_sequence=('#003459','#00A8E8'),

             opacity=.7,title='F1 Scores Using Different K-Values')

fig.show()