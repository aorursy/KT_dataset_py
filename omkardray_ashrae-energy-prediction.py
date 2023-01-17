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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

w = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv')
w =w.dropna()
w.head()
w.describe()
m= pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')
m =m.dropna()
m.head()
m.describe()
train= pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv',nrows=10000)
train =train.dropna()
train.head()
train.info()
train.describe()
building_weather = pd.merge(m, w, on='site_id')
building_weather.dropna()
building_weather.head()
building_weather.info()
X = pd.merge(building_weather,train , on='building_id')
X.head()
X.shape
X =X.dropna()
X.shape
X.describe()
X= X.drop(columns=['site_id','meter'])
X.head()
X= X.drop(columns=['timestamp_y'])
X.head()
X.describe()
X= X.drop(X[X['wind_speed']==0].index,axis=0)

X.head()
X= X.drop(X[X['meter_reading']==0].index,axis=0)
X.shape
X= X.drop(X[X['wind_direction']==0].index,axis=0)
X.shape
X.describe()
X.info()
import datetime 
X['Date'] = pd.to_datetime(X['timestamp_x'])
X.head()
X.info()
X["hour"] = X["Date"].dt.hour
X["weekend"] = X["Date"].dt.weekday
X["month"] = X["Date"].dt.month
X["dayofweek"] = X["Date"].dt.dayofweek
X.head()
X.primary_use.nunique()
X.primary_use.value_counts()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
X['primary_use'] = le.fit_transform(X['primary_use'])

X.head()
X=X.sort_values(by=['primary_use'])
X.head(10)
X=X.tail(1000000)
X.shape
X.primary_use.value_counts()
X.head()
import seaborn as sns
X = X.drop(columns=['timestamp_x','Date'])
X.head()
X.head()
X.describe()
X.shape
y=X['meter_reading']
X.head()
X.info()
corr=X.corr()

mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
for i in range(1, X.shape[1]):
    plt.subplot(5, 5, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    # f.axes.set_ylim([0, train.shape[0]])
    vals = np.size(X.iloc[:, i].unique())
    if vals < 10:
        bins = vals
    else:
        vals = 10
    plt.hist(X.iloc[:, i], bins=30, color='#3F5D7D')
plt.savefig("histogram-distribution.png")
fig, axes = plt.subplots(round(len(X.columns) / 3), 3, figsize=(12, 25))
for i, ax in enumerate(fig.axes):
    if i < len(X.columns):
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
        sns.countplot(x=X.columns[i], alpha=0.7, data=X, ax=ax)
X.describe()
X=X.drop(columns=['precip_depth_1_hr'])
X.head()
X.cloud_coverage.value_counts()
X=X.drop(X[X['cloud_coverage']==9].index, axis=0)
X.cloud_coverage.value_counts()

X.shape
X.head()
fig, axes = plt.subplots(round(len(X.columns) / 3), 3, figsize=(12, 25))
for i, ax in enumerate(fig.axes):
    if i < len(X.columns):
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
        sns.countplot(x=X.columns[i], alpha=0.7, data=X, ax=ax)
X.skew(axis = 0, skipna = True)
X.var()
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled= pd.DataFrame(sc.fit_transform(X), columns=X.columns)
X_scaled.var()
X_scaled.head()
y=X_scaled.meter_reading

y

y.values
np.shape(y)
X=X.drop(columns=['meter_reading'])
X.head()
x= X.values
type(x)
np.shape(x)
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
import xgboost as xgb

xgb_model = xgb.XGBRegressor(objective="reg:linear", random_state=42)

xgb_model.fit(x, y)

y_pred = xgb_model.predict(x)

mse=mean_squared_error(y, y_pred)

print(np.sqrt(mse))
xgb_model.score(x,y)
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=42)

xgb_model.fit(X_train, y_train)
xgb_model.score(X_test,y_test)