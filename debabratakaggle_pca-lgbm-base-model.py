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
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
target = train['target']
# train = train.drop('target',axis=1)
train.head()
#find col which have constant value
d = []
for col in train.columns.drop(['ID','target']).tolist():

    p = int((train[col][train[col] == 0].value_counts()[0] * 100)/ train.shape[0])
    
    if (p ==100):
        d.append(col)
        
    else:
        continue
print(len(d))
#dropping constant cols 
train1 = train.drop(d,axis =1)
test1 = test.drop(d,axis =1)
train1.shape
train1 = train1.drop('ID', axis=1)
test1 = test1.drop('ID', axis=1)
test2 = test1
print(train1.shape)
print(test1.shape)
import numpy as np
import matplotlib.pyplot as plt

plt.hist(train1['target'])
plt.hist(np.log(1+train['target']))
target = np.log(1+train['target'])
print(train1.shape)
print(test1.shape)
# a = train1.corr(method='pearson')['target'].abs().sort_values()
# cols = a[a< 0.1].index.tolist()
train2=train1
# train1 = train1.drop(['target'], axis=1)
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(train1)
train1 = pd.DataFrame(np_scaled, columns=train1.columns)
np_scaled1 = min_max_scaler.fit_transform(test1)
test1 = pd.DataFrame(np_scaled1, columns=test1.columns)
# from sklearn.decomposition import FactorAnalysis
# model = FactorAnalysis(n_components=3500)
# model.fit(train1)
from sklearn.decomposition import TruncatedSVD
model = TruncatedSVD(n_components=3500)
model.fit(train1)
# from sklearn.decomposition import PCA
# model = PCA(n_components=3500)
# model.fit(train1)
# var = get_covariance()
#calculate the covariance % 
var = model.explained_variance_ratio_
var1=np.cumsum(np.round(var, decimals=4)*100)
var1[var1 > 95]
import matplotlib.pyplot as plt
plt.figure(figsize=(25,10))
plt.plot(var1)
plt.grid()
#selecting n = 1750, point of saturation
model_new = TruncatedSVD(n_components= 1750, n_iter=10,random_state=42)
train1 = model_new.fit_transform(train1)
test1  = model_new.fit_transform(test1)
train1 =  pd.DataFrame(data=train1)
test1 = pd.DataFrame(data=test1)
train=train1
test=test1
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV


oof_preds = np.zeros(train.shape[0])
sub_preds = np.zeros(test.shape[0])
feature_importance_df = pd.DataFrame()

folds = KFold(n_splits= 10, random_state=123, shuffle=True)

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train, target)):
    train_x, train_y = train.iloc[train_idx], target.iloc[train_idx]
    valid_x, valid_y = train.iloc[valid_idx], target.iloc[valid_idx]

    
    clf = LGBMRegressor(
                        nthread=4,
                        n_estimators=1000,
                        learning_rate=0.01,
                        num_leaves=200,
                        max_depth=8,
                        reg_alpha=0.3,
                        reg_lambda=0.1,
                        min_split_gain=2,
                        silent=-1,
                        verbose=-1,
                        bagging_fraction = 0.5,
                        bagging_freq = 4,
                        metrics = 'rmse',
                        boosting_type = 'gbdt',
                        min_child_split=10
                        )

            
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], eval_metric= 'rmse', verbose= 100, early_stopping_rounds= 100)



    oof_preds[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration_)
    sub_preds += clf.predict(test, num_iteration=clf.best_iteration_) / folds.n_splits

#     fold_importance_df = pd.DataFrame()
#     fold_importance_df["feature"] = train_x.columns.tolist()
#     fold_importance_df["importance"] = clf.feature_importances_
#     fold_importance_df["fold"] = n_fold + 1
#     feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    print('Fold %2d rmse_score : %.6f' % (n_fold + 1, np.sqrt(np.abs(mean_squared_error(valid_y, oof_preds[valid_idx])))))


print(f'Full r2_score {r2_score(target, oof_preds)}')
print(f'Full RMSE score {np.sqrt(np.abs(mean_squared_error(target,oof_preds)))}' ) 
from sklearn.feature_selection import SelectFromModel
model = SelectFromModel(clf, prefit=True)
train = model.transform(train)
x = model.get_support(indices=True)

feature_names = [train1.columns[idx]
                     for idx, _
                     in enumerate(train1.columns)
                     if idx
                     in x]
    

train2 = pd.DataFrame(data=train,columns=feature_names)
test2 = pd.DataFrame(data=test,columns=feature_names)
train2.shape
# feature_importance_df.sort_values(by='importance', ascending=False).head()
# cols = fold_importance_df.sort_values(by="importance", ascending=False)[:100].index
# best_features = fold_importance_df.loc[(cols)]
# best_features.sort_values(by='importance',ascending=False)
# best_features.set_index('feature').plot(kind='barh', figsize=(26,18) )
#fit
clf.fit(train1, target)
y = clf.predict(test1)
z = np.exp(y)-1
z
test1 = pd.read_csv('../input/test.csv')
t = test1
t['target'] = z
t1 = t[['ID','target']]
t1.head()
t1.to_csv('submission.csv',index=False)
