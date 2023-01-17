# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pylab
import calendar
import numpy as np
import pandas as pd
import seaborn as sn
from scipy import stats
import missingno as msno
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
import warnings
matplotlib.style.use('ggplot')
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")

%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

train2016 = pd.read_csv('../input/train_2016_v2.csv', parse_dates=["transactiondate"])
properties2016 = pd.read_csv('../input/properties_2016.csv', low_memory=False)
data2016 = pd.merge(train2016,properties2016,on="parcelid",how="left")
print(data2016.shape)
missingValues = data2016.drop(columns=['parcelid', 'logerror', 'transactiondate'])
from sklearn import model_selection, preprocessing
from datetime import datetime
import xgboost as xgb
import warnings
from datetime import datetime
import time

warnings.filterwarnings("ignore")

missingValues = missingValues.fillna(-999)

for f in missingValues.columns:
    if missingValues[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(missingValues[f].values)) 
        missingValues[f] = lbl.transform(list(missingValues[f].values))
        
train_y = data2016.logerror.values
train_X = missingValues

xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)
featureImportance = model.get_fscore()
features = pd.DataFrame()
features['features'] = featureImportance.keys()
features['importance'] = featureImportance.values()
features.sort_values(by=['importance'],ascending=False,inplace=True)
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sn.barplot(data=features, ax=ax, x="importance",y="features")
ax.set_title('Features Ranked by XGBoost Importance')
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sn.barplot(data=features.head(20), ax=ax, x="importance",y="features")
ax.set_title('Top 20 Features')
topFeatures = features["features"].tolist()
corrMatt = missingValues[topFeatures].corr()
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
ax.set_title('Correlation Matrix')
sn.heatmap(corrMatt, mask=mask, square=True, cmap = 'coolwarm', center = 0)
topFeatures = features["features"].tolist()[:20]
corrMatt = train_X[topFeatures].corr()
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sn.heatmap(corrMatt, mask=mask, ax = ax, center = 0, square=True, cmap='coolwarm')
ax.set_title('Correlation Matrix for Top 20 Features')
continuous = ['basementsqft', 'finishedfloor1squarefeet', 'calculatedfinishedsquarefeet', 
              'finishedsquarefeet12', 'finishedsquarefeet13', 'finishedsquarefeet15',
              'finishedsquarefeet50', 'finishedsquarefeet6', 'garagetotalsqft', 'latitude',
              'longitude', 'lotsizesquarefeet', 'poolsizesum',  'yardbuildingsqft17',
              'yardbuildingsqft26', 'yearbuilt', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt',
              'landtaxvaluedollarcnt', 'taxamount']

discrete = ['bathroomcnt', 'bedroomcnt', 'calculatedbathnbr', 'fireplacecnt', 'fullbathcnt',
            'garagecarcnt', 'poolcnt', 'roomcnt', 'threequarterbathnbr', 'unitcnt',
            'numberofstories', 'assessmentyear', 'taxdelinquencyyear']
for col in continuous:
    values = data2016[col].dropna()
    lower = np.percentile(values, 1)
    upper = np.percentile(values, 99)
    fig = plt.figure(figsize=(18,9));
    sn.boxplot(y=values);
    plt.suptitle(col, fontsize=16)