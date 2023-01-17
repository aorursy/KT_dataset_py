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
import pickle 

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

path1 = r"ad_operation.dat"

path2 = r"ad_static_feature.out"

path3 = r"test_sample.dat"

path4 = r"totalExposureLog.out"

path5 = r"user_data"



DataFeature = pickle.load(open('../input/tx-expore/DataFeatureWithOpt','rb'))

DataExposure= pickle.load(open('../input/tx-expore/DataExposure','rb'))
test_data=pd.read_csv(r'../input/tencent-ad/test_sample.dat',sep='\t',names=["样本id","广告id","创建时间","素材尺寸","广告行业id","商品类型","商品id","广告账户id","投放时间","人群定向","出价"])

def add_Feature(DataFeature, DataExposure):

    Data_train = []

    targets    = []

    for x, t in zip(DataFeature,DataExposure):

        if x["人群定向"] != "" and x["广告时段"] != 0 and x["出价"] != 0:

            x["age"] = []

            x["gender"] = []

            x["area"] = []

            x["status"] = []

            x["education"] = []

            x["consuptionAbility"] = []

            x["device"] = []

            x["work"] = []

            x["connectionType"] = []

            x["behavior"] = []

            x["os"]      = []

            Data_train.append(x)

            targets.append(t)

    for i in Data_train:

        i["广告行业id"] = [int(j) for j in str(i["广告行业id"]).strip('[]').split(',')]

        i["素材尺寸"]   = [float(j) for j in str(i["素材尺寸"]).strip('[]').split(',')]

        i["出价"]      = int(str(i["出价"]).strip('[]'))

        i["广告时段"]   = [int(j) for j in str(i["广告时段"]).strip('[]').split(',')]

        if i["人群定向"] == 'all':

            continue

        for j in i["人群定向"].split('|'):

            key,val = j.split(':')

            i[key] = val

    return Data_train, targets

Data_train, targets = add_Feature(DataFeature, DataExposure)

import collections

c = collections.Counter([i["广告id"] for i in Data_train])

c = dict(c)

ad_id = []

for k, v in c.items():

    if v >= 9:

        ad_id.append(k)

def Filter(Data_train, targets,ad_id):

    Data = []

    ta = []

    for x, t in zip(Data_train, targets):

        if x["广告id"] in ad_id:

            Data.append(x)

            ta.append(t)

    return Data, ta

Data_train, targets = Filter(Data_train, targets, ad_id)
def Filter(Data_train, targets):

    for d, t in zip(Data_train, targets):

        if len(d["广告行业id"]) != 1:

            d["广告行业id"] = d["广告行业id"][0] + 3000000

        else:

            d["广告行业id"] = d["广告行业id"][0]



Filter(Data_train, targets)
def Filter(Data_train, targets):#对bid做竞价处理

    data = []

    tg   = []

    for d, t in zip(Data_train, targets):

        if d["出价"] == d["竞价众数"]:

            data.append(d)

            tg.append(t)

    return data, tg

Data_train, targets = Filter(Data_train, targets)
X_train = [[i["广告行业id"], i["出价"]] for i in Data_train]

X_train = np.array(X_train)

y_train = np.array(targets)

X_train[X_train[:,0] < 0] = 0

from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()

enc.fit(X_train[:, [0]])

X_train = np.hstack((enc.transform(X_train[:, [0]]).toarray(), X_train[:,[1]]))

import lightgbm as lgbm

from sklearn import model_selection

np.random.seed(42)
def SMAPE_train(y_true, y_pred):

    grad = (2*y_pred-2*y_true)/(y_pred+y_true)-(y_pred-y_true)**2/(y_pred+y_true)**2

    hess = (2*(y_pred-y_true)**2)/(y_pred+y_true)**3-(2*(2*y_pred-2*y_true))/(y_pred+y_true)**2+2/(y_pred+y_true)

    return grad, hess



def SMAPE_valid(y_true, y_pred):

    loss = (y_true - y_pred)**2/(y_true + y_pred)

    return "SMAPE_eval", np.mean(loss), False
model = lgbm.LGBMRegressor(

    objective='regression',

    max_depth=1000,

    num_leaves=100,

    learning_rate=0.07,

    max_bin = 1000,

    n_estimators=10000,

    min_child_samples= 1,

    subsample= 1,

    colsample_bytree=1,

    reg_alpha=0,

    reg_lambda=0,

    min_data_in_leaf = 1

)

y_train[656]
n_splits = 5

cv = model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=42)

val_scores = [0] * n_splits

for i, (fit_idx, val_idx) in enumerate(cv.split(X_train, y_train)):

    X_fit = X_train[fit_idx]

    y_fit = y_train[fit_idx]

    X_val = X_train[val_idx]

    y_val = y_train[val_idx]

    model.fit(X_fit,y_fit,eval_set=[(X_fit, y_fit), (X_val, y_val)],eval_names=('fit', 'val'),eval_metric='l2',early_stopping_rounds=200,verbose=False)

model.fit(X_fit,y_fit)
print('Feature importances:', list(model.feature_importances_))