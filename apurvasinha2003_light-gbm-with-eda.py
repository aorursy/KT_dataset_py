# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

import numpy as np

import csv

import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix

import seaborn as sns; sns.set()

from sklearn.model_selection import train_test_split

from sklearn.externals.six import StringIO  

from sklearn.metrics import mean_squared_error as mse

%matplotlib inline 

from skopt import BayesSearchCV

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import StratifiedKFold,KFold

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

from tqdm import tqdm

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in tqdm(df.columns):

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
train = pd.read_csv(r'../input/train.csv',low_memory=True,index_col='ID_code')  

print(train.head(1))

train=reduce_mem_usage(train)
def get_percentage_missing(series):

    """ Calculates percentage of NaN values in DataFrame

    :param series: Pandas DataFrame object

    :return: float

    """

    num = series.isnull().sum()

    den = len(series)

    return round(num/den, 2)



# Only include columns that contain any NaN values

df_with_any_null_values = train[train.columns[train.isnull().any()].tolist()]

data =get_percentage_missing(df_with_any_null_values)

filter_df=data[data>0.5]

print(train.shape)

remove_cols=[]

for n,v in filter_df.iteritems():

    remove_cols.append(n)

remove_cols#no column is having more than 50% missing data
#Lets start by plotting a heatmap to determine if any variables are correlated

plt.figure(figsize = (12,8))

sns.heatmap(data=train.corr())

plt.show()

plt.gcf().clear()

#no variable is correlated
# Create correlation matrix

corr_matrix = train.corr().abs()

# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95

to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

for each in to_drop:

    remove_cols.append(each)

remove_cols
train.replace(np.nan,0,inplace=True)

#train.drop('ID_code',axis=1,inplace=True)
# # Seaborn visualization library

# #Lets start by plotting a heatmap to determine if any variables are correlated

# plt.figure(figsize = (12,8))

# # Create the default pairplot

# sns.pairplot(train)

# plt.show()
test = pd.read_csv(r'../input/test.csv',low_memory=True,index_col='ID_code')

test=reduce_mem_usage(test)
train = reduce_mem_usage(train)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=15) 

param = {   

        'boosting_type': 'gbdt',

        'objective': 'binary',

        'metric': 'auc',

        'nthread': 6,

        'learning_rate': 0.05,

        'max_depth': 5,

        'num_leaves': 40,

        'sub_feature': 0.9,

        'sub_row':0.9,

        'bagging_freq': 1,

        'lambda_l1': 0.1,

        'lambda_l2': 0.1,

        'random_state': 15,

        'tree_learner':'serial',

        'boost_from_average':'false'

        }
import lightgbm as lgb

from sklearn.metrics import roc_auc_score

# Create arrays and dataframes to store results

oof_preds = np.zeros(train.shape[0])

sub_preds = np.zeros(len(test))

feature_importance_df = pd.DataFrame()

feats = [f for f in train.columns if f not in ['target']]

    

for n_fold, (train_idx, valid_idx) in enumerate(kf.split(train[feats], train['target'])):

    print(n_fold)

    trn_data = lgb.Dataset(train.iloc[train_idx][feats], label=train['target'].iloc[train_idx])    

    val_data = lgb.Dataset(train.iloc[valid_idx][feats], label=train['target'].iloc[valid_idx])

        

    clf = lgb.train(param, trn_data,5120, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 200)   



    oof_preds[valid_idx] = clf.predict(train.iloc[valid_idx][feats], num_iteration=clf.best_iteration) 

    

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = feats

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = n_fold + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    

    # we perform predictions by chunks

    initial_idx = 0

    chunk_size = 100000

    current_pred = np.zeros(len(test))

    

    while initial_idx < test.shape[0]:

        final_idx = min(initial_idx + chunk_size, test.shape[0])

        idx = range(initial_idx, final_idx)

        current_pred[idx] = clf.predict(test.iloc[idx][feats], num_iteration=clf.best_iteration)

        initial_idx = final_idx

        

    sub_preds += current_pred / 5



print('Full AUC score %.6f' % roc_auc_score(train['target'], oof_preds))
cols = (feature_importance_df[["Feature", "importance"]]

        .groupby("Feature")

        .mean()

        .sort_values(by="importance", ascending=False)[:150].index)

best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

import seaborn as sns

plt.figure(figsize=(14,28))

sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))

plt.title('Features importance (averaged/folds)')

plt.tight_layout()
# x=train.loc[:,train.columns != 'target']

# y=train.loc[:,train.columns == 'target']

# test_data=test.copy()

# #test.drop('ID_code',axis=1,inplace=True)
del train

import gc

gc.collect()
# rf=RandomForestRegressor(random_state=23,verbose=0, warm_start=True,n_jobs=-1,n_estimators=100)

# rf.fit(x, y)

# imp=pd.DataFrame({'label':x.columns,'imp':rf.feature_importances_})

# feature_select=imp[imp['imp']>0.01]['label']

# feature_select
# test=test[feature_select]

# x=x[feature_select]
# from skopt  import BayesSearchCV

# import lightgbm as lgb 

# from sklearn.metrics import accuracy_score 

# from sklearn.model_selection import train_test_split 

# from sklearn.metrics import roc_auc_score
# #Now splitting our dataset into test and train 

# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3)
# train_data=lgb.Dataset(x_train,label=y_train)
# #setting parameters for lightgbm

# param = {'num_leaves':150, 'objective':'binary:logistic','max_depth':7,'learning_rate':.05,'max_bin':200}

# param['metric'] = ['auc', 'binary_logloss']
# #training our model using light gbm

# num_round=50

# lgbm=lgb.train(param,train_data,num_round)
# ypred2=lgbm.predict(x_test)

# ypred2

# #converting probabilities into 0 or 1

# for i in range(len(ypred2)):

#     if ypred2[i]>=.3:       # setting threshold to .5

#        ypred2[i]=1

#     else:  

#        ypred2[i]=0
# #calculating accuracy

# accuracy_lgbm = accuracy_score(ypred2,y_test)

# accuracy_lgbm

# #calculating roc_auc_score for light gbm. 

# auc_lgbm = roc_auc_score(y_test,ypred2)

# auc_lgbm
# print(test.head(5))

# ypred_final= lgbm.predict(test)
output_xgb=pd.DataFrame({'ID_code':test.index,'target':sub_preds})

output_xgb.to_csv(r'predictions.csv',index=False)