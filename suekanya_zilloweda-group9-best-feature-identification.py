import numpy as np

import pandas as pd # for data processing-read

%matplotlib inline

from sklearn import neighbors

import seaborn as sns # for visualization

import matplotlib.pyplot as plt

from matplotlib import pyplot as plt

import matplotlib.colors as mcolors

from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder
prop2016 = pd.read_csv("../input/prop2016.csv", low_memory="False")

train2016=pd.read_csv("../input/trans2016.csv", low_memory="False")
t2016=prop2016.merge(train2016, on = 'parcelid', how="outer")
t2016['propertytype'] = 'NR'

t2016.loc[t2016['propertyzoningdesc'].str.contains('r', na=False, case=False), 'propertytype'] = 'R'
t2016.drop(['propertyzoningdesc'], axis=1)
from sklearn import model_selection, preprocessing

import xgboost as xgb

import warnings

warnings.filterwarnings("ignore")

mergedFilterd = t2016.fillna(-999)

for f in mergedFilterd.columns: 

    if mergedFilterd[f].dtype=='object':

        lbl = preprocessing.LabelEncoder() # Encode categorical features using an ordinal encoding scheme

        lbl.fit(list(mergedFilterd[f].values)) 

        mergedFilterd[f] = lbl.transform(list(mergedFilterd[f].values))

train_y = mergedFilterd.logerror.values

train_X = mergedFilterd.drop(["parcelid", "transactiondate", "logerror"], axis=1)

# parameters for estimation

xgb_params = {

    'eta': 0.05,

    'max_depth': 8,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:squarederror', # gave error when used reg:linear

    'eval_metric': 'rmse',

    'silent': 1

}

dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=10) # 100 rounds





importantFeatures = model.get_fscore()

features = pd.DataFrame()

features['features'] = importantFeatures.keys()

features['importance'] = importantFeatures.values()

features.sort_values(by=['importance'],ascending=False,inplace=True)

fig,ax= plt.subplots()

fig.set_size_inches(12,12)

plt.xticks(rotation=90)

sns.barplot(data=features.head(15),x="importance",y="features",ax=ax,orient="h",color="darkred")
# selecting values based on the result

t2016sel=t2016[['taxamount', 'latitude','longitude','taxvaluedollarcnt',

                'landtaxvaluedollarcnt','lotsizesquarefeet','calculatedfinishedsquarefeet',

                'structuretaxvaluedollarcnt','yearbuilt', 'bedroomcnt', 'rawcensustractandblock',

                'regionidzip', 'propertytype', 'bathroomcnt', 'logerror']]

plt.figure(figsize=(12,12))

sns.heatmap(t2016sel.corr(), cmap="rocket_r", vmin=-1, vmax=1, square=True, annot=True)