# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.offline as py

color = sns.color_palette()

import plotly.graph_objs as go

py.init_notebook_mode(connected=True)

import plotly.tools as tls

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import pandas as pd

import numpy as np



import warnings

warnings.filterwarnings('ignore')



SMALL_SIZE = 10

MEDIUM_SIZE = 12



plt.rc('font', size=SMALL_SIZE)

plt.rc('axes', titlesize=MEDIUM_SIZE)

plt.rc('axes', labelsize=MEDIUM_SIZE)

plt.rcParams['figure.dpi']=150

# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/Skyserver_SQL2_27_2018 6_51_39 PM.csv')

df.head()
df.describe()
df.info()
columns = df.columns

percent_missing = df.isnull().sum() * 100 / len(df)

missing_value_df = pd.DataFrame({'column_name': columns,

                                 'percent_missing': percent_missing})

missing_value_df.sort_values('percent_missing')
sns.heatmap(df.corr())

df.corr()
df.drop(['specobjid','fiberid'],axis=1,inplace=True)
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(16, 4))

ax = sns.distplot(df[df['class']=='STAR'].redshift, bins = 30, ax = axes[0], kde = False)

ax.set_title('Star')

ax = sns.distplot(df[df['class']=='GALAXY'].redshift, bins = 30, ax = axes[1], kde = False)

ax.set_title('Galaxy')

fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(16, 4))

ax = sns.lvplot(x=df['class'], y=df['dec'], palette='coolwarm')

ax.set_title('dec')
di={'STAR':1,'GALAXY':2,'QSO':3}

df.replace({'class':di},inplace=True)



y=df['class']

df.drop(['objid','class'],axis=1,inplace=True)
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

sdss = scaler.fit_transform(df)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(df, 

                                                    y, test_size=0.33)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

preds = knn.predict(X_test)

acc_knn = (preds == y_test).sum().astype(float) / len(preds)*100

print("Accuracy of KNN: ", acc_knn)
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV



param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 



grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)



grid.fit(X_train,y_train)
grid_predictions = grid.predict(X_test)
acc_gv_rbf = (grid_predictions == y_test).sum().astype(float) / len(grid_predictions)*100

print("Accuracy of SVM: ", acc_gv_rbf)
from sklearn.naive_bayes import GaussianNB



gnb=GaussianNB()

gnb.fit(X_train,y_train)

preds2=gnb.predict(X_test)

acc_gnb=(preds2==y_test).sum().astype(float)/len(preds)*100

print("Accuracy of Naive Bayes: ",acc_gnb)
from sklearn.ensemble import RandomForestClassifier



rf=RandomForestClassifier()

rf.fit(X_train,y_train)

preds3=rf.predict(X_test)

acc_rf=(preds3==y_test).sum().astype(float)/len(preds)*100

print("Accuracy of Random Forest Classifier: ",acc_rf)