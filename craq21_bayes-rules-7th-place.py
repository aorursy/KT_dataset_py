import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.preprocessing import OneHotEncoder
data = pd.read_csv('/kaggle/input/infopulsehackathon/train.csv')

test = pd.read_csv('/kaggle/input/infopulsehackathon/test.csv')
ohe = OneHotEncoder(sparse=False)

ohe.fit(data.select_dtypes(object))

data = pd.concat((data,pd.DataFrame(ohe.transform(data.select_dtypes(object)))),axis=1)

test = pd.concat((test,pd.DataFrame(ohe.transform(test.select_dtypes(object)))),axis=1)



data.drop(data.select_dtypes(object).columns,axis=1,inplace=True)

test.drop(test.select_dtypes(object).columns,axis=1,inplace=True)
target = data['Energy_consumption']

data = data.drop(['Energy_consumption'],axis=1)
cols = data.columns
from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=-2,strategy="median")

data_imp = imp.fit_transform(data)

test_imp = imp.fit_transform(test)
def trip_featurize(df,feat1,feat2, feat3):

    df[f'{feat1}-{feat2}-{feat3}'] = df[feat1] - df[feat2] - df[feat3]

    df[f'{feat1}-{feat2}+{feat3}'] = df[feat1] - df[feat2] + df[feat3]

    df[f'{feat1}-{feat2}*{feat3}'] = df[feat1] - df[feat2] * df[feat3]

    df[f'{feat1}-{feat2}/{feat3}'] = df[feat1] - df[feat2] / df[feat3]

    

    df[f'{feat1}+{feat2}-{feat3}'] = df[feat1] + df[feat2] - df[feat3]

    df[f'{feat1}+{feat2}+{feat3}'] = df[feat1] + df[feat2] + df[feat3]

    df[f'{feat1}+{feat2}*{feat3}'] = df[feat1] + df[feat2] * df[feat3]

    df[f'{feat1}+{feat2}/{feat3}'] = df[feat1] + df[feat2] / df[feat3]

    

    df[f'{feat1}*{feat2}-{feat3}'] = df[feat1] * df[feat2] - df[feat3]

    df[f'{feat1}*{feat2}+{feat3}'] = df[feat1] * df[feat2] + df[feat3]

    df[f'{feat1}*{feat2}*{feat3}'] = df[feat1] * df[feat2] * df[feat3]

    df[f'{feat1}*{feat2}/{feat3}'] = df[feat1] * df[feat2] / df[feat3]

    

    df[f'{feat1}/{feat2}-{feat3}'] = df[feat1] / df[feat2] - df[feat3]

    df[f'{feat1}/{feat2}+{feat3}'] = df[feat1] / df[feat2] + df[feat3]

    df[f'{feat1}/{feat2}*{feat3}'] = df[feat1] / df[feat2] * df[feat3]

    df[f'{feat1}/{feat2}/{feat3}'] = df[feat1] / df[feat2] / df[feat3]

def double_featurize(df,feat1,feat2):

    df[f'{feat1}-{feat2}'] = df[feat1] - df[feat2]

    df[f'{feat1}+{feat2}'] = df[feat1] + df[feat2]

    df[f'{feat1}*{feat2}'] = df[feat1] * df[feat2]

    df[f'{feat1}/{feat2}'] = df[feat1] / df[feat2]
double_featurize(data,'feature_5','feature_122')

double_featurize(test,'feature_5','feature_122')



double_featurize(data,'feature_122','feature_33')

double_featurize(test,'feature_122','feature_33')



double_featurize(data,'feature_129','feature_5')

double_featurize(test,'feature_129','feature_5')



double_featurize(data,'feature_122','feature_248')

double_featurize(test,'feature_122','feature_248')



trip_featurize(data,'feature_5','feature_122', 'feature_33')

trip_featurize(test,'feature_5','feature_122', 'feature_33')



trip_featurize(data,'feature_122','feature_264', 'feature_33')

trip_featurize(test,'feature_122','feature_264', 'feature_33')



trip_featurize(data,'feature_122','feature_248', 'feature_5')

trip_featurize(test,'feature_122','feature_248', 'feature_5')



trip_featurize(data,'feature_229','feature_250', 'feature_5')

trip_featurize(test,'feature_229','feature_250', 'feature_5')
data_val = data.values

test_val = test.values
counts = []

for column in cols:

    counts.append(data[column].value_counts())
bins = np.zeros(data.shape)

for i in range(data.shape[0]):

    for j in range(len(cols)):

        bins[i][j] = counts[j][data_val[i][j]]
test_bins = np.zeros(test.shape)

for i in range(test.shape[0]):

    for j in range(len(cols)):

        try:

            test_bins[i][j] = counts[j][test_val[i][j]]

        except:

            test_bins[i][j] = 1
data_val = np.concatenate((data_imp,bins),axis=1)

test = np.concatenate((test_imp,test_bins),axis=1)
from sklearn.model_selection import KFold

import lightgbm as lgb
folds = KFold(shuffle=True,random_state=40,n_splits=5)
params = {

 'min_data_in_leaf': 47,

 'num_leaves': 61,

 'lr': 0.06172399472541107,

 'min_child_weight': 0.0070492703809497,

 'colsample_bytree': 0.06169,

 'bagging_fraction': 0.3809,

 'min_child_samples': 35,

 'subsample': 0.6077180918189186,

 'max_depth': 4,

 'objective': 'regression',

 'seed': 1337,

 'feature_fraction_seed': 1337,

 'bagging_seed': 1337,

 'drop_seed': 1337,

 'data_random_seed': 1337,

 'boosting_type': 'gbdt',

 'verbose': 1,

 'boost_from_average': True,

 'metric': 'mse',

 'cat_l2': 24.38,

 'cat_smooth': 18.49,

 'feature_fraction': 0.1045,

 'lambda_l1': 2.306,

 'lambda_l2':18.02,

 'max_cat_threshold':50,

 'min_gain_to_split':  0.2585,

 'min_sum_hessian_in_leaf':0.001923,         

}
target = target.values
from sklearn.metrics import mean_squared_error
pred = np.zeros((1,test.shape[0]))

scores = 0

features = range(data_val.shape[1])

for fold, (trn_idx, val_idx) in enumerate(folds.split(data_val)):

    x_trn, y_trn = data_val[trn_idx], target[trn_idx]

    x_val, y_val = data_val[val_idx], target[val_idx]

    

    train = lgb.Dataset(x_trn,label=y_trn)

    valid = lgb.Dataset(x_val,label=y_val)

    

    model = lgb.train(params,train,num_boost_round=10000,early_stopping_rounds=100,valid_sets=(train,valid),verbose_eval=False)

    

    valid_prediction = model.predict(x_val,num_iteration=model.best_iteration)

    test_prediction = model.predict(test,num_iteration=model.best_iteration)

    

    score = mean_squared_error(y_val, valid_prediction)

    scores += score

    pred += test_prediction

    

    print('Validation fold {} : '.format(fold),score)

    

print("SCORE",scores/5)
pred /= 5
sub = pd.read_csv('/kaggle/input/infopulsehackathon/sample_submission.csv')
sub['Energy_consumption'] = pred[0]

sub.to_csv('submission.csv',index=False)