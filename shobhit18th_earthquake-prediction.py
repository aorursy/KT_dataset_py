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
# Importing the suitable library

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from catboost import CatBoostRegressor,Pool

import os

import warnings

warnings.filterwarnings("ignore")

import lightgbm as lgb
sc=StandardScaler()
#loaing the datasets

train_data=pd.read_csv("../input/train.csv",nrows=20000000,dtype={"acoustic_data":np.int16,"time_to_failure":np.float64},low_memory=True)

test_data=os.listdir("../input/test")

#sample_submission_data=pd.read_csv("../input/sample_submission.csv",low_memory=True)
sample_submission_data=pd.read_csv("../input/sample_submission.csv",low_memory=True)
sample_submission_data[:5]
test_data[0]
test_data[1]

pd.read_csv("../input/test/"+test_data[1])
print(train_data.shape)

#print(sample_submission_data.shape)
train_data.head()
plt.figure(figsize=(25,5))

plt.subplot(1,2,1)

print(train_data["time_to_failure"].plot(color="b"))

plt.subplot(1,2,2)

print(train_data["acoustic_data"].plot(color="r"))
train_data.head()

y_train=train_data["time_to_failure"]
features = ['mean','max','variance','min', 'stdev','skew','kurtosis','Quantile-20%','Quantile-50%','Quantile-75%','max-min-diff','max-mean-diff','mean-change-abs','abs-min','abs-max','std-first-50000','std-last-50000','mean-first-50000','mean-last-50000','max-first-50000','max-last-50000','min-first-50000','min-last-50000']
rows = 150000

segments = int(np.floor(train_data.shape[0] / rows))
X = pd.DataFrame(index=range(segments), dtype=np.float64,

                       columns=features)

Y = pd.DataFrame(index=range(segments), dtype=np.float64,

                       columns=['time_to_failure'])
for segment in range(segments):

    seg = train_data.iloc[segment*rows:segment*rows+rows]

    x = seg['acoustic_data'].values

    y = seg['time_to_failure'].values[-1]
x
for segment in range(segments):

    seg = train_data.iloc[segment*rows:segment*rows+rows]

    x = seg['acoustic_data'].values

    y = seg['time_to_failure'].values[-1]

    

    Y.loc[segment, 'time_to_failure'] = y

    X.loc[segment, 'mean'] = x.mean()

    X.loc[segment, 'stdev'] = x.std()

    X.loc[segment,'variance'] = np.var(x)

    X.loc[segment, 'max'] = x.max()

    X.loc[segment, 'min'] = x.min()

    X.loc[segment,"skew"]= pd.Series(x).skew()

    X.loc[segment,"kurtosis"]= pd.Series(x).kurtosis()

    X.loc[segment,"Quantile-20%"]= pd.Series(x).quantile(0.25)

    X.loc[segment,"Quantile-50%"]= pd.Series(x).quantile(0.50)

    X.loc[segment,"Quantile-75%"]= pd.Series(x).quantile(0.75)

    X.loc[segment, 'max-min-diff'] = x.max()-x.min()

    X.loc[segment, 'max-mean-diff'] = x.max()-x.mean()

    X.loc[segment, 'mean-change-abs'] = np.mean(np.diff(x))

    X.loc[segment, 'abs-min'] = np.abs(x).min()

    X.loc[segment, 'abs-max'] = np.abs(x).max()

    X.loc[segment, 'std-first-50000'] = x[:50000].std()

    X.loc[segment, 'std-last-50000'] = x[-50000:].std()

    X.loc[segment, 'mean-first-50000'] = x[:50000].min()

    X.loc[segment, 'mean-last-50000'] = x[-50000:].mean()

    X.loc[segment, 'max-first-50000'] = x[:50000].max()

    X.loc[segment, 'max-last-50000'] = x[-50000:].max()

    X.loc[segment, 'min-first-50000'] = x[:50000].min()

    X.loc[segment, 'min-last-50000'] = x[-50000:].min()

    
sns.distplot(Y)

yy=np.log(Y)
yy
x_train=sc.fit_transform(X)
sns.distplot(Y)
m=CatBoostRegressor(iterations=500,loss_function="MAE",boosting_type="Ordered")
m.fit(x_train,yy,silent=True)
m.best_score_
X_test = pd.DataFrame(columns=features,index=sample_submission_data.index)
for i in range(len(sample_submission_data)):

    file=os.listdir("../input/test/")[i]

    data1=pd.read_csv("../input/test/"+file,low_memory=True,dtype={"acoustic_data":np.int16})

    m1=data1["acoustic_data"].values

    x=m1

    X_test.loc[i, 'mean'] = x.mean()

    X_test.loc[i, 'stdev'] = x.std()

    X_test.loc[i,'variance'] = np.var(x)

    X_test.loc[i, 'max'] = x.max()

    X_test.loc[i, 'min'] = x.min()

    X_test.loc[i,"skew"]= pd.Series(x).skew()

    X_test.loc[i,"kurtosis"]= pd.Series(x).kurtosis()

    X_test.loc[i,"Quantile-20%"]= pd.Series(x).quantile(0.25)

    X_test.loc[i,"Quantile-50%"]= pd.Series(x).quantile(0.50)

    X_test.loc[i,"Quantile-75%"]= pd.Series(x).quantile(0.75)

    X_test.loc[i, 'max-min-diff'] = x.max()-x.min()

    X_test.loc[i, 'max-mean-diff'] = x.max()-x.mean()

    X_test.loc[i, 'mean-change-abs'] = np.mean(np.diff(x))

    X_test.loc[i, 'abs-min'] = np.abs(x).min()

    X_test.loc[i, 'abs-max'] = np.abs(x).max()

    X_test.loc[i, 'std-first-50000'] = x[:50000].std()

    X_test.loc[i, 'std-last-50000'] = x[-50000:].std()

    X_test.loc[i, 'mean-first-50000'] = x[:50000].min()

    X_test.loc[i, 'mean-last-50000'] = x[-50000:].mean()

    X_test.loc[i, 'max-first-50000'] = x[:50000].max()

    X_test.loc[i, 'max-last-50000'] = x[-50000:].max()

    X_test.loc[i, 'min-first-50000'] = x[:50000].min()

    X_test.loc[i, 'min-last-50000'] = x[-50000:].min()
X_test
test1_data= sc.fit_transform(X_test)
y_pred=m.predict(test1_data)

y_pred
sample_submission_data['time_to_failure'] = np.exp(y_pred)

sample_submission_data.to_csv('submission.csv',index=False)
params = {'num_leaves': 51,

         'min_data_in_leaf': 10, 

         'objective':'regression',

         'max_depth': -1,

         'learning_rate': 0.001,

         "boosting": "gbdt",

         "feature_fraction": 0.91,

         "bagging_freq": 1,

         "bagging_fraction": 0.91,

         "bagging_seed": 42,

         "metric": 'mae',

         "lambda_l1": 0.1,

         "verbosity": -1,

         "nthread": -1,

         "random_state": 42}
# model = lgb.LGBMRegressor(**params, n_estimators = 20000, n_jobs = -1)
#model.fit((x_train,Y,eval_set=[(X_tr, y_teval_metric='mae',verbose=1000, early_stopping_rounds=200)
#from IPython.display import YouTubeVideo

#YouTubeVideo("TffGdSsWKlA")
from sklearn.ensemble import GradientBoostingRegressor
GBoost = GradientBoostingRegressor(n_estimators=900, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

GBoost.fit(x_train,Y)
GBoost.score(x_train,Y)
y_pred=GBoost.predict(test1_data)
y_pred
sample_submission_data['time_to_failure'] = y_pred

sample_submission_data.to_csv('submission111.csv',index=False)