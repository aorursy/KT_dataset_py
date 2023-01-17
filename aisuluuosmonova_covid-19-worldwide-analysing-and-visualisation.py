# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        
        from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
# Any results you write to the current directory are saved as output.

cov_sub=pd.read_csv('../input/covid19-global-forecasting-week-4/submission.csv')
cov_train=pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')
cov_test=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
country_viz = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')
cov_sub.head()
cov_train.head()
cov_train.describe()
cov_train.info(), cov_test.info()
import pandas_profiling
profile=pandas_profiling.ProfileReport(cov_train)
profile
cov_train.isnull().any()
#Visualisation of Covid-19 cases


import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline




sns.heatmap(cov_train.corr(),cmap='coolwarm',annot=True)
plt.figure(figsize=(18,10))
plt.plot(cov_train.Id, cov_train.ConfirmedCases)
plt.title('Confirmed Cases')
print()
#Call required libraries
import time                   # To time processes


 
import matplotlib.pyplot as plt                   # For graphics
import seaborn as sns
#import plotly.plotly as py #For World Map
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

from sklearn.preprocessing import StandardScaler  # For scaling dataset
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation #For clustering
from sklearn.mixture import GaussianMixture #For GMM clustering

import os                     # For os related operations
import sys                    # For data size


data = [dict(type='choropleth',
             locations = cov_train['Country_Region'],
             locationmode = 'country names',
             z = cov_train['ConfirmedCases'],
             text = cov_train['Country_Region'],
             colorbar = {'title':'Cluster Group'})]
layout = dict(title='Clustering of Countries based on K-Means',
              geo=dict(showframe = False,
                       projection = {'type':'mercator'}))
map1 = go.Figure(data = data, layout=layout)
iplot(map1)
# Convert string data into numerical
from sklearn.preprocessing import LabelEncoder
x=['Country_Region']
for i in x:
    a=LabelEncoder()
    cov_train[i]=a.fit_transform(cov_train[i])
cov_train.head()
# Create Train & Test Data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(cov_train[['Id', 'Province_State', 'Country_Region', 'Date', 'ConfirmedCases']], cov_train['Fatalities'], 
                                                    test_size=0.3, random_state=3)
X_train.head()
y = cov_train['Fatalities']

cov_train = cov_train.drop('Fatalities', axis = 1)

print("Shape of y:", y.shape)
cov_train = pd.get_dummies(cov_train, drop_first=True)
cov_train.head()
cov_train.columns
cov_train.fillna(value=0.0, inplace=True)
import catboost as cb
model = cb.CatBoostRegressor()
# Fit model
model.fit(cov_train,y)
# Get predictions
preds = model.predict(cov_train)
