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
import numpy as np

import pandas as pd

from pandas import DataFrame

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from pandas import Series,DataFrame

import scipy

from pylab import rcParams

import urllib

import sklearn

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.metrics import plot_confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

import plotly.express as px

import warnings

warnings.filterwarnings('ignore')

print ('Setup Complete')
ds=pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
ds.head()
ds.describe()
ds.info()
ds_quality=ds['quality'].value_counts()

print(ds_quality)

fig = px.histogram(ds['quality'],x="quality",nbins=20)

fig.show()

import plotly.express as px

df = px.data.tips()

fig = px.violin(ds, x="quality",y='fixed acidity',color="quality", box=True, points="all")

fig.show()
import plotly.express as px

df = px.data.tips()

fig = px.violin(ds, x="quality",y='volatile acidity',color="quality", box=True, points="all")

fig.show()
import plotly.express as px

df = px.data.tips()

fig = px.violin(ds, x="quality",y='citric acid',color="quality", box=True, points="all")

fig.show()
import plotly.express as px

df = px.data.tips()

fig = px.violin(ds, x="quality",y='residual sugar',color="quality", box=True, points="all")

fig.show()
import plotly.express as px

df = px.data.tips()

fig = px.violin(ds, x="quality",y='chlorides',color="quality", box=True, points="all")

fig.show()
import plotly.express as px

df = px.data.tips()

fig = px.violin(ds, x="quality",y='free sulfur dioxide',color="quality", box=True, points="all")

fig.show() 
import plotly.express as px

df = px.data.tips()

fig = px.violin(ds, x="quality",y='total sulfur dioxide',color="quality", box=True, points="all")

fig.show() 
import plotly.express as px

df = px.data.tips()

fig = px.violin(ds, x="quality",y='density',color="quality", box=True, points="all")

fig.show() 
import plotly.express as px

df = px.data.tips()

fig = px.violin(ds, x="quality",y='pH',color="quality", box=True, points="all")

fig.show() 
import plotly.express as px

df = px.data.tips()

fig = px.violin(ds, x="quality",y='sulphates',color="quality", box=True, points="all")

fig.show() 
import plotly.express as px

df = px.data.tips()

fig = px.violin(ds, x="quality",y='alcohol',color="quality", box=True, points="all")

fig.show() 
corr = ds.corr()

fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f")

plt.xticks(range(len(corr.columns)), corr.columns);

plt.yticks(range(len(corr.columns)), corr.columns)

plt.show()
bins = (2, 6.5, 8)

labels = ['bad', 'good']

ds['quality'] = pd.cut(x = ds['quality'], bins = bins, labels = labels)
ds['quality'].value_counts()
labelencoder_y = LabelEncoder()

ds['quality'] = labelencoder_y.fit_transform(ds['quality'])
ds.head()
x = ds.drop(['quality'], axis=1)

y=ds.loc[:,['quality']]
x.head()
y.head()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 50)
print("Shape of X_train: ",x_train.shape)

print("Shape of X_test: ", x_test.shape)

print("Shape of y_train: ",y_train.shape)

print("Shape of y_test",y_test.shape)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train_scaled = sc.fit_transform(x_train)

x_test_scaled = sc.transform(x_test)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, accuracy_score

lr = LogisticRegression()

lr.fit(x_train_scaled, y_train)

lr_predict = lr.predict(x_test_scaled)
lr_conf_matrix = confusion_matrix(y_test, lr_predict)

lr_acc_score = accuracy_score(y_test, lr_predict)

print(lr_conf_matrix)

print(lr_acc_score*100)
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train_scaled,y_train)

dt_predict = dt.predict(x_test_scaled)
dt_conf_matrix = confusion_matrix(y_test, dt_predict)

dt_acc_score = accuracy_score(y_test, dt_predict)

print(dt_conf_matrix)

print(dt_acc_score*100)
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train_scaled,y_train)

nb_predict=nb.predict(x_test_scaled)
nb_conf_matrix = confusion_matrix(y_test, nb_predict)

nb_acc_score = accuracy_score(y_test, nb_predict)

print(nb_conf_matrix)

print(nb_acc_score*100)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(x_train_scaled, y_train)

rf_predict=rf.predict(x_test_scaled)
rf_conf_matrix = confusion_matrix(y_test, rf_predict)

rf_acc_score = accuracy_score(y_test, rf_predict)

print(rf_conf_matrix)

print(rf_acc_score*100)
from sklearn.svm import SVC
lin_svc = SVC()

lin_svc.fit(x_train_scaled, y_train)

lin_svc=rf.predict(x_test_scaled)
lin_svc_conf_matrix = confusion_matrix(y_test, rf_predict)

lin_svc_acc_score = accuracy_score(y_test, rf_predict)

print(lin_svc_conf_matrix)

print(lin_svc_acc_score*100)