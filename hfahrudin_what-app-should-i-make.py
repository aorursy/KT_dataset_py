import warnings
warnings.filterwarnings('ignore')
import sys
import scipy
import numpy
# matplotlib
import matplotlib
import numpy as np # linear algebra
# pandas
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
%matplotlib inline
# scikit-learn
import sklearn
import os
%matplotlib inline
from sklearn.metrics import accuracy_score
# Importing metrics for evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from pandas import get_dummies
from re import sub
# dataset : https://www.kaggle.com/lava18/google-play-store-apps
data =  pd.read_csv('../input/googleplaystore.csv')
review =  pd.read_csv('../input/googleplaystore_user_reviews.csv')
data.head(5)
data.info()
data.Category.unique()
data[data['Category'] == '1.9']
error = data.iloc[10472]
error_name = error['App']
fix = error.shift(1)
fix['App'] = error_name
fix['Category'] = 'LIFESTYLE'
fix['Genres'] = 'Lifestyle'
data.iloc[10472] = fix
data.iloc[10472]
data['Category'].value_counts().plot(kind='bar', figsize=(10,5))
data["Rating"] = data["Rating"].astype(float)
data.Rating.unique()
data[data['Rating'].isnull() & (data["Reviews"] == "0")].shape == data[data['Rating'].isnull()].shape

miss_null = data[data['Rating'].isnull() & (data["Reviews"] == "0")]
miss_notnull = data[data['Rating'].isnull() & (data["Reviews"] != "0")]
miss_null['Rating'] = miss_null['Rating'].fillna(0)
miss_null.shape
data = data[data['Rating'].notnull()]
data['Category'].value_counts().plot(kind='bar', figsize=(10,5))
miss_notnull['Category'].value_counts().plot(kind='bar', figsize=(10,5))
miss_notnull.shape
avg = data[['Category','Rating']].groupby(['Category'], as_index=True).mean().to_dict()
miss_notnull['Rating'] = miss_notnull['Category'].replace(avg.get('Rating'))
miss_notnull.head()
data = data.append(miss_null, ignore_index=True)
data = data.append(miss_notnull, ignore_index=True)
data.shape
data['Rating'] = data.Rating.round(1)
data.Rating.unique()
data['Price'] = data["Price"].apply(lambda x: float(sub(r'[^\d\-.]', '', x)))
data.Price.unique()
for name, row in data.iterrows():
    genres_list = row["Installs"].split("+")
    s = genres_list[0].replace(',','')
    data.loc[name, ['Installs']] = s
       
data["Installs"] = data["Installs"].astype(int)
data.head(5)
data['Genres'][data['Installs'] > 10000000].value_counts().plot(kind='bar', figsize=(14,5))
data['Category'][data['Installs'] > 10000000].value_counts().plot(kind='bar', figsize=(14,5))
data[['Category','Rating']].groupby(['Category']).mean().plot(kind='bar', figsize=(12,7))
profit_app = data[(data['Price'] != 0.) & (data['Installs'] > 10000)]
profit_app[['Category','Price']].groupby(['Category']).mean().plot(kind = 'bar', figsize=(12,5))
profit_app[profit_app['Category'] == 'FINANCE']
profit_app[profit_app['Category'] == 'LIFESTYLE']
cat = profit_app.Category.unique()
for ind in range (0,len(cat)):
    if profit_app['Category'][profit_app['Category'] == cat[ind]].count() < 4:
        profit_app = profit_app[profit_app.Category != cat[ind]]
profit_app[['Category','Price']].groupby(['Category']).mean().plot(kind = 'bar', figsize=(12,5))