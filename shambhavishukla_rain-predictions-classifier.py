# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
w=pd.read_csv('../input/weatherAUS.csv')
w
w.drop('Location',axis=1,inplace=True)

w.drop('Evaporation',axis=1,inplace=True)

w.drop('Sunshine',axis=1,inplace=True)

w.drop('Date',axis=1,inplace=True)

w.drop('WindDir9am',axis=1,inplace=True)

w.drop('WindGustDir',axis=1,inplace=True)

w.drop('WindDir3pm',axis=1,inplace=True)

w.drop('Cloud9am',axis=1,inplace=True)

w.drop('Cloud3pm',axis=1,inplace=True)



w
import seaborn as sns

import matplotlib.pyplot as plt

sns.heatmap(w.corr())
w
w.drop('Cloud9am',axis=1,inplace=True)

w.drop('Cloud3pm',axis=1,inplace=True)
w
w
sns.heatmap(w.corr())
sns.heatmap(w.isnull())

w['Rainfall'].dropna()

w['Rainfall'].fillna(value=1.0,inplace=True)
w.isnull()
sns.heatmap(w.isnull())
w['WindGustSpeed'].fillna(value=1.0,inplace=True)

w['WindSpeed9am'].fillna(value=1.0,inplace=True)

w['WindSpeed3pm'].fillna(value=1.0,inplace=True)

w['Humidity9am'].fillna(value=1.0,inplace=True)

w['Humidity3pm'].fillna(value=1.0,inplace=True)

w['Pressure9am'].fillna(value=1.0,inplace=True)

w['Pressure3pm'].fillna(value=1.0,inplace=True)

w['Temp9am'].fillna(value=1.0,inplace=True)

w['Temp3pm'].fillna(value=1.0,inplace=True)

sns.heatmap(w.isnull())
w['RainToday'].fillna(value='yes',inplace=True)
sns.heatmap(w.isnull())
sns.heatmap(w.corr())
# KNN classifier
w['RainToday'].replace(to_replace='Yes',value=1,inplace=True)



w['RainToday'].replace(to_replace='No',value=0,inplace=True)



w['RainTomorrow'].replace(to_replace='Yes',value=1,inplace=True)



w['RainTomorrow'].replace(to_replace='No',value=0,inplace=True)

w['RainToday'].replace(to_replace='yes',value=1,inplace=True)

w['RainTomorrow'].replace(to_replace='yes',value=1,inplace=True)

w
w.info()

from sklearn.preprocessing import StandardScaler
w['RainToday']
scaler.fit(w.drop('RainTomorrow',axis=1))
scaled_features = scaler.transform(w.drop('RainTomorrow',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=w.columns[-1])
