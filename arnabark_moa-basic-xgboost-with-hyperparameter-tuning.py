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
import catboost

import optuna

import imblearn

from catboost import CatBoostRegressor

from imblearn.under_sampling import RandomUnderSampler

import numpy as np

import pandas as pd

from catboost import *

import matplotlib.pyplot as plt

import seaborn as sns

from catboost import Pool

from datetime import datetime

from numpy import mean

from sklearn.datasets import make_classification

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RepeatedStratifiedKFold

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.linear_model import LinearRegression,RidgeCV

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

from scipy.stats import norm,skew

from scipy import stats

from sklearn.metrics import mean_squared_error,make_scorer

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from tqdm import tqdm

import pandas as pd

import nltk

import operator

import re

import sys

from scipy import stats

from nltk.corpus import stopwords

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer

# # from multiprocessing import Pool

# nltk.download("stopwords")

# nltk.download("punkt")

import statsmodels.api as sm

from statsmodels.formula.api import ols

import time
x_t=pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

x_test=pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

y_t_nonscored=pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')

y_t=pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')
print(x_t.shape)

print(x_test.shape)

print(y_t_nonscored.shape)

print(y_t.shape)
x_t.head()

x_t['cp_dose'].value_counts()
#log transform



#log transform skewed numeric features:

numeric_feats = x_t.dtypes[x_t.dtypes == ("float" or "int") ].index



skewed_feats = x_t[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index

print(skewed_feats)

x_t[skewed_feats] = np.log1p(x_t[skewed_feats])

x_test[skewed_feats] = np.log1p(x_test[skewed_feats])
x_t_m=x_t[[i for i in x_test.columns if i!='sig_id']]

x_test_m=x_test[[i for i in x_test.columns if i!='sig_id']]

# x_t_m=pd.get_dummies(x_t_m)

# x_test_m=pd.get_dummies(x_test_m)

x_t_m=x_t_m[x_test_m.columns]

y_t_m=y_t[[i for i in y_t.columns if i!='sig_id']]
sum(x_t_m.columns==x_test_m.columns)/len(x_t_m.columns==x_test_m.columns)==1
x_t_m
result=pd.DataFrame(x_test['sig_id'])

for j in [i for i in y_t_m.columns if sum(y_t_m[i]!=0)<=2]:

    y_t_m=y_t_m.drop([j],axis=1)

    result[j]=0
x_t_m=x_t_m.reset_index(drop=True)

y_t_m=y_t_m.reset_index(drop=True)

from sklearn.model_selection import StratifiedKFold
y_t_m
# categorical_features_indices= np.where(X_test.dtypes == np.object)[0]

# Train_set=Pool(X_train, y_train,cat_features=categorical_features_indices)



# Eval_set=Pool(X_valid, y_valid,cat_features=categorical_features_indices)

# def objective(trial):

#     param = {

#         'iterations':500,

#         'learning_rate':0.05,

#         'use_best_model':True,

#         'od_type' : "Iter",

#         'od_wait' : 100,

#         'task_type="GPU"'

# #         'random_seed': 240,

# #          "scale_pos_weight":trial.suggest_int("scale_pos_weight", 1, 10),

#         "depth": trial.suggest_int("max_depth", 2, 10),

#         "l2_leaf_reg": trial.suggest_loguniform("lambda", 1e-8, 100),

#           'eval_metric':trial.suggest_categorical("loss_function",['F1','Logloss','Accuracy'])

# #         'one_hot_max_size':1024

#         }



#     # Add a callback for pruning.

#     model=CatBoostClassifier(**param)

#     print(param)

#     model.fit(Train_set,eval_set=Eval_set,plot=False,verbose=False)

#     pred=model.predict(Pool(X_valid,cat_features= np.where(X_valid.dtypes == np.object)[0]))

#     acc=sklearn.metrics.accuracy_score(pred,y_valid)

    



#     return 1-acc
for column in y_t_m.columns:

    print(column)

    X_train,X_valid,y_train,y_valid=train_test_split(x_t_m,y_t_m[column],test_size=0.2,stratify=y_t_m[column])

    categorical_features_indices= np.where(X_train.dtypes == np.object)[0]

    Train_set=Pool(X_train, y_train,cat_features=categorical_features_indices)

    a=len(y_t_m[column]>0)/sum(y_t_m[column]>0)-1

    Eval_set=Pool(X_valid, y_valid,cat_features=categorical_features_indices)

    param={'use_best_model':True,

            'od_type' : "Iter",

            'od_wait' : 100,

            'task_type':"GPU",}

    param['loss_function']='Logloss'

    param['scale_pos_weight']=a

    model=CatBoostClassifier(**param)

    print(param)

    model.fit(Train_set,eval_set=Eval_set,plot=False,verbose=False)

    pred=model.predict(Pool(x_test_m,cat_features= np.where(x_test_m.dtypes == np.object)[0]))

    result[column]=pred
result.describe()
# def objective(trial):

#     param = {

#         "silent": 1,

# #            "scale_pos_weight":trial.suggest_int("scale_pos_weight", 1, 100),

#           "eval_metric": "logloss",

#         "booster": "gbtree",

#         "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),

#         "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),

#         'tree_method' : 'gpu_hist',

#         'nthread' : -1

        

#     }



#     if param["booster"] == "gbtree" or param["booster"] == "dart":

#         param["max_depth"] = trial.suggest_int("max_depth", 1, 9)

# #         param["eta"] = trial.suggest_loguniform("eta", 1e-8, 1.0)

# #         param["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)

# #         param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

#     X = x_t_m.values

#     y = y_t_m[column]

#     skf = StratifiedKFold(n_splits=5,shuffle=True)

#     arr=0

#     for train_index, test_index in skf.split(X, y):

#         x_train, x_valid = X[train_index], X[test_index]

#         y_train, y_valid = y[train_index], y[test_index]

#         dtrain = xgb.DMatrix(x_train, label=y_train)

#         dvalid = xgb.DMatrix(x_valid, label=y_valid)

# # Add a callback for pruning.

#         pruning_callback = optuna.integration.XGBoostPruningCallback(trial, str("validation-"+param["eval_metric"]))

#         bst = xgb.train(param, dtrain, evals=[(dvalid, "validation")],callbacks=[pruning_callback])

# #     preds = np.rint(bst.predict(dvalid))

#         y_pred = bst.predict(dvalid).astype(np.float64)

#         arr=arr+sklearn.metrics.log_loss(y_valid, y_pred)

#     return arr/5
# %%time

# import optuna

# import xgboost as xgb

# import sklearn

# for column in y_t_m.columns:

#     print(column)

# #     study = optuna.create_study()

# #     study.optimize(objective, n_trials=50)

# #     y_t_m[column]=y_t_m[column].astype('float64')

#     X_train,X_valid,Y_train,Y_valid=train_test_split(x_t_m,y_t_m,test_size=0.2,stratify=y_t_m[column])

#     dtrain = xgb.DMatrix(X_train, label=Y_train[column])

#     dvalid = xgb.DMatrix(X_valid, label=Y_valid[column])

#     dtest = xgb.DMatrix(x_test_m)

# #     temp=df_params[df_params['column']==column]['param'].reset_index(drop=True)[0]

#     temp={}

#     print(temp)

# #     arr.append(study.best_params)

#     temp['eval_metric']="logloss"

#     temp["booster"]='gbtree'

#     temp['tree_method']='gpu_hist'

#     temp['nthread']=-1

#     bst = xgb.train( temp,dtrain, evals=[(dvalid, "validation")])

#     result[column]=bst.predict(dtest)
# # arr=[{'scale_pos_weight': 2, 'lambda': 2.0609856845556447e-06, 'alpha': 2.617051622748541e-06, 'max_depth': 6}, {'scale_pos_weight': 2, 'lambda': 0.026482326432315916, 'alpha': 0.0008244672658155524, 'max_depth': 2}, {'scale_pos_weight': 4, 'lambda': 5.708135843484176e-08, 'alpha': 0.9192754239777133, 'max_depth': 7}, {'scale_pos_weight': 4, 'lambda': 0.008515265487129662, 'alpha': 0.420198671016534, 'max_depth': 4}, {'scale_pos_weight': 2, 'lambda': 0.06706642027786032, 'alpha': 3.43454902807587e-05, 'max_depth': 5}, {'scale_pos_weight': 2, 'lambda': 0.00018950811514951188, 'alpha': 0.011669553299950307, 'max_depth': 1}, {'scale_pos_weight': 8, 'lambda': 4.796901094462643e-07, 'alpha': 1.0179493347461164e-06, 'max_depth': 8}, {'scale_pos_weight': 4, 'lambda': 2.131420585381424e-06, 'alpha': 7.69690269908201e-08, 'max_depth': 7}, {'scale_pos_weight': 9, 'lambda': 2.799460620147439e-05, 'alpha': 0.29537204607292433, 'max_depth': 8}, {'scale_pos_weight': 5, 'lambda': 1.7743790322429437e-08, 'alpha': 0.11206545273367641, 'max_depth': 6}, {'scale_pos_weight': 6, 'lambda': 0.005060992979729095, 'alpha': 0.05049470212006941, 'max_depth': 2}, {'scale_pos_weight': 6, 'lambda': 0.007022969292655949, 'alpha': 8.480199058395783e-05, 'max_depth': 1}, {'scale_pos_weight': 7, 'lambda': 1.4242971452795586e-08, 'alpha': 0.01163388875442049, 'max_depth': 9}, {'scale_pos_weight': 9, 'lambda': 0.0009476397406443937, 'alpha': 0.0028332653168558685, 'max_depth': 8}, {'scale_pos_weight': 4, 'lambda': 0.007148951199801825, 'alpha': 0.008403079810124803, 'max_depth': 1}, {'scale_pos_weight': 8, 'lambda': 0.1175044270104437, 'alpha': 1.275818830486805e-07, 'max_depth': 2}, {'scale_pos_weight': 6, 'lambda': 8.447063082605443e-06, 'alpha': 0.15950065108238826, 'max_depth': 1}, {'scale_pos_weight': 4, 'lambda': 1.459687801997271e-08, 'alpha': 0.0070512409326262595, 'max_depth': 7}, {'scale_pos_weight': 10, 'lambda': 0.0014432848165521995, 'alpha': 0.0003744259821844032, 'max_depth': 4}, {'scale_pos_weight': 9, 'lambda': 3.9657509170496154e-08, 'alpha': 1.07198613262584e-08, 'max_depth': 7}, {'scale_pos_weight': 9, 'lambda': 0.012786573561955047, 'alpha': 9.653542083301758e-07, 'max_depth': 3}, {'scale_pos_weight': 7, 'lambda': 0.17605155036497785, 'alpha': 0.12461753974036073, 'max_depth': 2}, {'scale_pos_weight': 8, 'lambda': 1.854022726286052e-06, 'alpha': 4.6179861440176766e-05, 'max_depth': 5}, {'scale_pos_weight': 8, 'lambda': 0.00025398576690541794, 'alpha': 1.8096458513313066e-08, 'max_depth': 7}, {'scale_pos_weight': 3, 'lambda': 0.8785715512067258, 'alpha': 0.5119761444558435, 'max_depth': 3}, {'scale_pos_weight': 10, 'lambda': 0.14409479572071263, 'alpha': 5.015691793843948e-06, 'max_depth': 4}, {'scale_pos_weight': 3, 'lambda': 7.114494702539658e-06, 'alpha': 0.9023388918725609, 'max_depth': 6}, {'scale_pos_weight': 4, 'lambda': 3.6904169527088606e-05, 'alpha': 0.9267183191326552, 'max_depth': 1}, {'scale_pos_weight': 5, 'lambda': 0.00023593066709093642, 'alpha': 2.0595008299010454e-07, 'max_depth': 4}, {'scale_pos_weight': 7, 'lambda': 1.0709503470694408e-05, 'alpha': 0.00024153165178501828, 'max_depth': 7}, {'scale_pos_weight': 7, 'lambda': 0.0020018396326696632, 'alpha': 2.366160023353101e-06, 'max_depth': 8}, {'scale_pos_weight': 7, 'lambda': 8.869349344439546e-07, 'alpha': 2.2349754848726652e-06, 'max_depth': 1}, {'scale_pos_weight': 9, 'lambda': 0.09056660519032135, 'alpha': 0.4529662761918149, 'max_depth': 6}, {'scale_pos_weight': 4, 'lambda': 6.915533266016473e-07, 'alpha': 8.776703495074358e-08, 'max_depth': 1}, {'scale_pos_weight': 8, 'lambda': 6.867156089582059e-08, 'alpha': 2.1818490820084203e-06, 'max_depth': 7}, {'scale_pos_weight': 1, 'lambda': 0.0026005904684195283, 'alpha': 7.782773482990971e-08, 'max_depth': 3}, {'scale_pos_weight': 8, 'lambda': 1.332750590872826e-07, 'alpha': 0.002212866284901339, 'max_depth': 5}, {'scale_pos_weight': 4, 'lambda': 1.563880150374635e-07, 'alpha': 0.08156149338970485, 'max_depth': 2}, {'scale_pos_weight': 9, 'lambda': 0.0007807463413779823, 'alpha': 0.0003192133549983461, 'max_depth': 2}, {'scale_pos_weight': 1, 'lambda': 0.00014277119881093923, 'alpha': 1.0990914402069266e-07, 'max_depth': 3}, {'scale_pos_weight': 10, 'lambda': 0.0031799934025698562, 'alpha': 0.0027851022500338797, 'max_depth': 6}, {'scale_pos_weight': 1, 'lambda': 1.3502321513561877e-08, 'alpha': 1.348984772561901e-07, 'max_depth': 5}, {'scale_pos_weight': 3, 'lambda': 4.030194565138501e-07, 'alpha': 1.6056327612919764e-08, 'max_depth': 2}, {'scale_pos_weight': 4, 'lambda': 0.0028908544045502236, 'alpha': 0.18913350358061912, 'max_depth': 4}, {'scale_pos_weight': 3, 'lambda': 1.7122844389731093e-06, 'alpha': 0.04169548231540852, 'max_depth': 5}, {'scale_pos_weight': 1, 'lambda': 2.116090312251997e-08, 'alpha': 9.492075239439281e-05, 'max_depth': 3}, {'scale_pos_weight': 10, 'lambda': 0.00019464873661035272, 'alpha': 0.37875778089804363, 'max_depth': 9}, {'scale_pos_weight': 8, 'lambda': 2.943059378916662e-08, 'alpha': 2.098154636014842e-06, 'max_depth': 9}, {'scale_pos_weight': 2, 'lambda': 0.4985096359811047, 'alpha': 0.0001338251267432777, 'max_depth': 5}, {'scale_pos_weight': 4, 'lambda': 2.155687175506865e-07, 'alpha': 0.005609596721924019, 'max_depth': 4}, {'scale_pos_weight': 10, 'lambda': 1.553061400839337e-05, 'alpha': 2.2106805620025803e-08, 'max_depth': 4}, {'scale_pos_weight': 2, 'lambda': 5.38398902999234e-07, 'alpha': 2.086287996532942e-07, 'max_depth': 5}, {'scale_pos_weight': 8, 'lambda': 8.060404156393388e-08, 'alpha': 0.1685078798919148, 'max_depth': 2}, {'scale_pos_weight': 7, 'lambda': 9.307114874354866e-07, 'alpha': 3.3843744727294795e-06, 'max_depth': 5}, {'scale_pos_weight': 5, 'lambda': 4.240629857983314e-06, 'alpha': 0.014536146934599559, 'max_depth': 5}, {'scale_pos_weight': 3, 'lambda': 2.6177892011614326e-07, 'alpha': 0.005295131433160071, 'max_depth': 4}, {'scale_pos_weight': 10, 'lambda': 0.002102555201619389, 'alpha': 3.781748736314341e-07, 'max_depth': 9}, {'scale_pos_weight': 9, 'lambda': 1.7713588766578505e-08, 'alpha': 0.024133293802013595, 'max_depth': 6}, {'scale_pos_weight': 10, 'lambda': 0.0006946387694715314, 'alpha': 3.9687897105100246e-07, 'max_depth': 4}, {'scale_pos_weight': 1, 'lambda': 0.015890319582123388, 'alpha': 1.7012468846498273e-06, 'max_depth': 3}, {'scale_pos_weight': 10, 'lambda': 2.339200625395679e-08, 'alpha': 9.752307382316981e-07, 'max_depth': 4}, {'scale_pos_weight': 9, 'lambda': 1.2609034503478367e-06, 'alpha': 0.0008381828539226632, 'max_depth': 4}, {'scale_pos_weight': 3, 'lambda': 0.442409117962891, 'alpha': 6.631979264286477e-07, 'max_depth': 7}, {'scale_pos_weight': 1, 'lambda': 1.3737491303503302e-05, 'alpha': 0.014890402651445412, 'max_depth': 4}, {'scale_pos_weight': 2, 'lambda': 2.1677075506907536e-08, 'alpha': 0.0008288459856862685, 'max_depth': 4}, {'scale_pos_weight': 6, 'lambda': 1.000501236851631e-08, 'alpha': 1.0626096905122379e-06, 'max_depth': 2}, {'scale_pos_weight': 6, 'lambda': 5.168226927057098e-07, 'alpha': 0.0015002507884409315, 'max_depth': 6}, {'scale_pos_weight': 10, 'lambda': 3.8316665873577814e-07, 'alpha': 0.002724518810011536, 'max_depth': 2}, {'scale_pos_weight': 9, 'lambda': 0.08761251581154081, 'alpha': 3.071069183883837e-08, 'max_depth': 2}, {'scale_pos_weight': 6, 'lambda': 9.159656192632103e-05, 'alpha': 0.0013274585401529075, 'max_depth': 3}, {'scale_pos_weight': 3, 'lambda': 0.00010763474353960037, 'alpha': 4.015630233039448e-06, 'max_depth': 3}, {'scale_pos_weight': 9, 'lambda': 0.0001636390957368343, 'alpha': 1.7836571481864048e-06, 'max_depth': 6}, {'scale_pos_weight': 6, 'lambda': 0.07826494195822509, 'alpha': 7.088920054365115e-08, 'max_depth': 9}, {'scale_pos_weight': 10, 'lambda': 2.2581219454638268e-07, 'alpha': 0.4763241152153245, 'max_depth': 1}, {'scale_pos_weight': 6, 'lambda': 0.8090254732138108, 'alpha': 0.31858924277341466, 'max_depth': 9}, {'scale_pos_weight': 6, 'lambda': 6.134239246839323e-08, 'alpha': 0.0017312018332045769, 'max_depth': 4}, {'scale_pos_weight': 8, 'lambda': 1.594396975251976e-08, 'alpha': 0.3607473259702311, 'max_depth': 9}, {'scale_pos_weight': 7, 'lambda': 0.0025212172118355457, 'alpha': 4.471531129395384e-06, 'max_depth': 3}, {'scale_pos_weight': 6, 'lambda': 0.003686120269218026, 'alpha': 0.0007839485641604663, 'max_depth': 6}, {'scale_pos_weight': 8, 'lambda': 0.026934576082739654, 'alpha': 3.0937077434009896e-06, 'max_depth': 5}, {'scale_pos_weight': 8, 'lambda': 7.764575487290164e-06, 'alpha': 6.90412535438188e-07, 'max_depth': 2}, {'scale_pos_weight': 10, 'lambda': 0.00020460620731060558, 'alpha': 2.897274614681872e-05, 'max_depth': 3}, {'scale_pos_weight': 5, 'lambda': 0.0014706408320089447, 'alpha': 0.04050529494532535, 'max_depth': 7}, {'scale_pos_weight': 3, 'lambda': 4.1192823407803625e-08, 'alpha': 2.196697697923856e-08, 'max_depth': 9}, {'scale_pos_weight': 10, 'lambda': 0.00890988228679309, 'alpha': 7.741724632179337e-06, 'max_depth': 5}, {'scale_pos_weight': 3, 'lambda': 2.0624493821986145e-05, 'alpha': 0.0015412251700452095, 'max_depth': 3}, {'scale_pos_weight': 3, 'lambda': 0.004879988946174674, 'alpha': 2.440095489357154e-07, 'max_depth': 2}, {'scale_pos_weight': 4, 'lambda': 1.1525747645411504e-06, 'alpha': 2.7541485060371077e-05, 'max_depth': 8}, {'scale_pos_weight': 9, 'lambda': 0.02851652629768907, 'alpha': 9.577391546537329e-05, 'max_depth': 9}, {'scale_pos_weight': 2, 'lambda': 6.701378170082723e-08, 'alpha': 2.586025748457425e-08, 'max_depth': 1}, {'scale_pos_weight': 1, 'lambda': 0.1343706374939572, 'alpha': 5.883972006473122e-05, 'max_depth': 8}, {'scale_pos_weight': 7, 'lambda': 0.406013170156868, 'alpha': 7.762667806969227e-07, 'max_depth': 7}, {'scale_pos_weight': 6, 'lambda': 1.7473949645980335e-08, 'alpha': 0.4219017248279645, 'max_depth': 5}, {'scale_pos_weight': 2, 'lambda': 5.810179412637817e-05, 'alpha': 3.468761901913459e-07, 'max_depth': 2}, {'scale_pos_weight': 2, 'lambda': 1.0455196983994227e-06, 'alpha': 0.3930192032955631, 'max_depth': 1}, {'scale_pos_weight': 3, 'lambda': 6.2336647665456514e-06, 'alpha': 0.004877425248659075, 'max_depth': 5}, {'scale_pos_weight': 8, 'lambda': 2.2807840775153385e-07, 'alpha': 8.736480184451159e-08, 'max_depth': 3}, {'scale_pos_weight': 3, 'lambda': 0.00019238864726148184, 'alpha': 0.08466305607071972, 'max_depth': 7}, {'scale_pos_weight': 8, 'lambda': 0.5915929206345718, 'alpha': 0.0026665525836753294, 'max_depth': 9}, {'scale_pos_weight': 5, 'lambda': 1.8873917416361174e-08, 'alpha': 5.271465569613661e-06, 'max_depth': 3}, {'scale_pos_weight': 10, 'lambda': 0.0067711234661662675, 'alpha': 6.869377769700075e-05, 'max_depth': 2}, {'scale_pos_weight': 10, 'lambda': 1.7902404316168827e-07, 'alpha': 0.0005348395546882722, 'max_depth': 6}, {'scale_pos_weight': 5, 'lambda': 6.361516967742641e-08, 'alpha': 0.05108778324837577, 'max_depth': 2}, {'scale_pos_weight': 6, 'lambda': 0.1831536990408759, 'alpha': 0.009482769308151407, 'max_depth': 1}, {'scale_pos_weight': 2, 'lambda': 0.0067260415223701865, 'alpha': 6.316938511122413e-08, 'max_depth': 2}, {'scale_pos_weight': 9, 'lambda': 0.0495306765731755, 'alpha': 1.2545791046519735e-07, 'max_depth': 9}, {'scale_pos_weight': 6, 'lambda': 8.380563788201484e-08, 'alpha': 0.08055222763668403, 'max_depth': 4}, {'scale_pos_weight': 8, 'lambda': 6.547935500057434e-06, 'alpha': 0.001456372361159754, 'max_depth': 1}, {'scale_pos_weight': 2, 'lambda': 1.397093952140885e-06, 'alpha': 0.002177934080607095, 'max_depth': 5}, {'scale_pos_weight': 10, 'lambda': 8.307085819977164e-05, 'alpha': 0.24742423375455067, 'max_depth': 6}, {'scale_pos_weight': 7, 'lambda': 1.5670914900809343e-08, 'alpha': 1.4405447362765504e-07, 'max_depth': 4}, {'scale_pos_weight': 5, 'lambda': 8.665643642499485e-06, 'alpha': 0.08063787018497504, 'max_depth': 3}, {'scale_pos_weight': 3, 'lambda': 0.01620701794521586, 'alpha': 1.2059683360474958e-07, 'max_depth': 9}, {'scale_pos_weight': 1, 'lambda': 0.0003878064699113839, 'alpha': 0.00038018138912574275, 'max_depth': 1}, {'scale_pos_weight': 1, 'lambda': 0.0053472300706815055, 'alpha': 0.12871658529195806, 'max_depth': 9}, {'scale_pos_weight': 1, 'lambda': 0.7579116279416719, 'alpha': 2.204519945410827e-05, 'max_depth': 5}, {'scale_pos_weight': 8, 'lambda': 0.32727280175376844, 'alpha': 1.5537516732216264e-05, 'max_depth': 3}, {'scale_pos_weight': 6, 'lambda': 3.8144899354552e-06, 'alpha': 0.7344519092368686, 'max_depth': 5}, {'scale_pos_weight': 7, 'lambda': 1.6126173339205936e-06, 'alpha': 3.423950013655911e-07, 'max_depth': 5}, {'scale_pos_weight': 6, 'lambda': 8.329720144346348e-06, 'alpha': 0.0002767747406472048, 'max_depth': 7}, {'scale_pos_weight': 3, 'lambda': 0.0008837360514968716, 'alpha': 0.21940603127747765, 'max_depth': 6}, {'scale_pos_weight': 9, 'lambda': 4.647243693132857e-06, 'alpha': 0.0006996369015312986, 'max_depth': 4}, {'scale_pos_weight': 6, 'lambda': 4.296133446686113e-08, 'alpha': 0.5478401581144652, 'max_depth': 9}, {'scale_pos_weight': 9, 'lambda': 0.00017361737105119925, 'alpha': 0.0008931647881906599, 'max_depth': 1}, {'scale_pos_weight': 1, 'lambda': 0.19949694621860095, 'alpha': 0.00018171484543753284, 'max_depth': 1}, {'scale_pos_weight': 5, 'lambda': 1.2295028296912098e-08, 'alpha': 0.00028918163105858794, 'max_depth': 4}, {'scale_pos_weight': 2, 'lambda': 0.009733348389584425, 'alpha': 0.003143933683662404, 'max_depth': 4}, {'scale_pos_weight': 10, 'lambda': 0.034022264113786074, 'alpha': 7.822130302445441e-05, 'max_depth': 5}, {'scale_pos_weight': 9, 'lambda': 0.009384355548141009, 'alpha': 1.854484115813817e-08, 'max_depth': 4}, {'scale_pos_weight': 8, 'lambda': 3.81605201690939e-06, 'alpha': 1.2578507346516077e-06, 'max_depth': 3}, {'scale_pos_weight': 6, 'lambda': 4.636934953153242e-05, 'alpha': 0.007875433231481599, 'max_depth': 9}, {'scale_pos_weight': 7, 'lambda': 3.114200749276335e-08, 'alpha': 0.12378599012584848, 'max_depth': 2}, {'scale_pos_weight': 3, 'lambda': 2.9183699298272083e-08, 'alpha': 4.009238407535015e-06, 'max_depth': 2}, {'scale_pos_weight': 1, 'lambda': 4.573296592177708e-06, 'alpha': 3.770993892373514e-07, 'max_depth': 5}, {'scale_pos_weight': 5, 'lambda': 0.04759393615237392, 'alpha': 0.027268025169590676, 'max_depth': 7}, {'scale_pos_weight': 8, 'lambda': 0.05970087925221047, 'alpha': 8.154132200935809e-08, 'max_depth': 8}, {'scale_pos_weight': 6, 'lambda': 6.080981728259903e-06, 'alpha': 4.302967143747276e-07, 'max_depth': 1}, {'scale_pos_weight': 7, 'lambda': 0.03645166868056354, 'alpha': 0.0008109436690416099, 'max_depth': 7}, {'scale_pos_weight': 9, 'lambda': 1.547305438560754e-05, 'alpha': 0.0003638566553218744, 'max_depth': 2}, {'scale_pos_weight': 1, 'lambda': 0.5770908429032822, 'alpha': 1.6848439195953825e-06, 'max_depth': 5}, {'scale_pos_weight': 3, 'lambda': 0.03513360096192405, 'alpha': 4.259501815240785e-08, 'max_depth': 5}, {'scale_pos_weight': 4, 'lambda': 0.000174215341841354, 'alpha': 0.0029578460458470355, 'max_depth': 5}, {'scale_pos_weight': 1, 'lambda': 0.10803139556211583, 'alpha': 2.7521262805875223e-06, 'max_depth': 4}, {'scale_pos_weight': 6, 'lambda': 0.30923228014707155, 'alpha': 1.1018608165072334e-05, 'max_depth': 4}, {'scale_pos_weight': 9, 'lambda': 1.347378336468439e-05, 'alpha': 3.4100687112418005e-07, 'max_depth': 9}, {'scale_pos_weight': 7, 'lambda': 9.020999951899783e-06, 'alpha': 0.0011148625717211616, 'max_depth': 3}, {'scale_pos_weight': 5, 'lambda': 0.1643154336616746, 'alpha': 3.086582090693409e-05, 'max_depth': 8}, {'scale_pos_weight': 7, 'lambda': 0.1893605378654662, 'alpha': 0.18782143371503796, 'max_depth': 1}, {'scale_pos_weight': 6, 'lambda': 3.881978570660975e-05, 'alpha': 0.0003172254368764662, 'max_depth': 6}, {'scale_pos_weight': 3, 'lambda': 0.031551671750729586, 'alpha': 5.607237958754033e-06, 'max_depth': 5}, {'scale_pos_weight': 5, 'lambda': 7.009403747754234e-06, 'alpha': 0.011060337223004887, 'max_depth': 7}, {'scale_pos_weight': 3, 'lambda': 2.0349938934198008e-08, 'alpha': 0.5789524652885751, 'max_depth': 4}, {'scale_pos_weight': 2, 'lambda': 6.36113260895343e-07, 'alpha': 0.12905504180825839, 'max_depth': 3}, {'scale_pos_weight': 2, 'lambda': 0.012403827727049522, 'alpha': 0.036745179821728584, 'max_depth': 9}, {'scale_pos_weight': 4, 'lambda': 0.6127584583835702, 'alpha': 7.824960737197805e-06, 'max_depth': 2}, {'scale_pos_weight': 10, 'lambda': 7.082202834810185e-08, 'alpha': 6.362705693568493e-08, 'max_depth': 5}, {'scale_pos_weight': 4, 'lambda': 0.00013989969767746078, 'alpha': 0.003836649284335828, 'max_depth': 3}, {'scale_pos_weight': 2, 'lambda': 0.000729947559852818, 'alpha': 0.005205539405695293, 'max_depth': 7}, {'scale_pos_weight': 1, 'lambda': 0.0010379989481481985, 'alpha': 0.0003204410160787802, 'max_depth': 1}, {'scale_pos_weight': 8, 'lambda': 2.0645516682767937e-08, 'alpha': 3.135526832310544e-08, 'max_depth': 2}, {'scale_pos_weight': 8, 'lambda': 0.0022288619708509356, 'alpha': 0.010240773780721024, 'max_depth': 8}, {'scale_pos_weight': 7, 'lambda': 2.5767224194377875e-07, 'alpha': 5.105568945928655e-07, 'max_depth': 5}, {'scale_pos_weight': 7, 'lambda': 5.770925220788122e-07, 'alpha': 0.000575222901973266, 'max_depth': 5}, {'scale_pos_weight': 8, 'lambda': 5.717717152352693e-05, 'alpha': 0.5136548413884832, 'max_depth': 7}, {'scale_pos_weight': 1, 'lambda': 0.5444244425572936, 'alpha': 0.11765488526915957, 'max_depth': 5}, {'scale_pos_weight': 8, 'lambda': 0.047743564319931604, 'alpha': 0.04980211226237828, 'max_depth': 6}, {'scale_pos_weight': 4, 'lambda': 0.0008538309209078712, 'alpha': 0.0662311766140255, 'max_depth': 7}, {'scale_pos_weight': 3, 'lambda': 0.03559830338543013, 'alpha': 0.07052725252313127, 'max_depth': 1}, {'scale_pos_weight': 1, 'lambda': 8.230992060696519e-06, 'alpha': 1.7707288017771193e-07, 'max_depth': 9}, {'scale_pos_weight': 5, 'lambda': 1.0775425742393985e-08, 'alpha': 0.003331499252795676, 'max_depth': 7}, {'scale_pos_weight': 8, 'lambda': 0.0009081742920414149, 'alpha': 0.6257322054432402, 'max_depth': 2}, {'scale_pos_weight': 3, 'lambda': 1.2475390201021953e-08, 'alpha': 0.13646398296190404, 'max_depth': 7}, {'scale_pos_weight': 8, 'lambda': 3.545577497750592e-05, 'alpha': 8.699981939794504e-07, 'max_depth': 4}, {'scale_pos_weight': 4, 'lambda': 0.0005659799894330384, 'alpha': 0.00011236168399596936, 'max_depth': 7}, {'scale_pos_weight': 10, 'lambda': 6.775980540237443e-06, 'alpha': 0.09743566220492099, 'max_depth': 6}, {'scale_pos_weight': 1, 'lambda': 8.849022720980936e-06, 'alpha': 0.0007243845614907884, 'max_depth': 9}, {'scale_pos_weight': 4, 'lambda': 1.4830347940197162e-05, 'alpha': 5.150644013671719e-07, 'max_depth': 9}, {'scale_pos_weight': 4, 'lambda': 1.541078213205907e-08, 'alpha': 6.134731617596764e-06, 'max_depth': 3}, {'scale_pos_weight': 8, 'lambda': 1.331952860802075e-05, 'alpha': 0.1603997269376648, 'max_depth': 9}, {'scale_pos_weight': 6, 'lambda': 0.09341673668955365, 'alpha': 6.328178845971141e-05, 'max_depth': 9}, {'scale_pos_weight': 1, 'lambda': 0.0011282457719413853, 'alpha': 3.6115308203798084e-06, 'max_depth': 2}, {'scale_pos_weight': 4, 'lambda': 1.8852210570834984e-06, 'alpha': 1.0920481879588456e-08, 'max_depth': 5}, {'scale_pos_weight': 5, 'lambda': 5.176250788936423e-07, 'alpha': 0.023683064111957797, 'max_depth': 5}, {'scale_pos_weight': 4, 'lambda': 2.1814403499476966e-06, 'alpha': 1.346631634202945e-07, 'max_depth': 5}]

# df_params=pd.DataFrame()

# df_params['param']=arr

# df_params['column']=y_t_m.columns
# print(arr)
# for i in result.columns[1:]:

#     print(result[i])

#     result[i]=np.clip(result[i],0.025,0.975)
result.to_csv('submission.csv',index=False)
result.shape
# df_params.value_counts()