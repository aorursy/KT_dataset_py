import os

print((os.listdir('../input/')))
import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import roc_auc_score

import xgboost as xgb

from xgboost import XGBClassifier

from catboost import CatBoostClassifier
df_train = pd.read_csv('../input/webclubrecruitment2019/TRAIN_DATA.csv')

df_test = pd.read_csv('../input/webclubrecruitment2019/TEST_DATA.csv')

test_index_beg=df_test['Unnamed: 0'] #copying test index for later

df_train.head()
print(df_train)
df_train.isnull().values.any()
train_X = df_train.loc[:, 'V1':'V16']

train_y = df_train.loc[:, 'Class']

test_X = df_test.loc[:, 'V1':'V16']
cat_columns = ['V2', 'V3', 'V4', 'V5', 'V7', 'V8', 'V9', 'V11', 'V16']

cont_columns = ['V1', 'V6', 'V10', 'V12', 'V13', 'V14', 'V15']
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
ax = sns.countplot(x = train_y ,palette="Set2")

sns.set(font_scale=1.5)

ax.set_xlabel(' ')

ax.set_ylabel(' ')

fig = plt.gcf()

fig.set_size_inches(10,5)

ax.set_ylim(top=50000)

for p in ax.patches:

    ax.annotate('{:.2f}%'.format(100*p.get_height()/len(train_y)), (p.get_x()+ 0.3, p.get_height()+10000))



plt.title('Distribution of Targets')

plt.xlabel('Targets')

plt.ylabel('Frequency [%]')

plt.show()
sns.set(style="white")





# Compute the correlation matrix

corr = train_X.corr()





# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
rf = RandomForestClassifier(n_estimators=50, random_state=123)



# model = XGBClassifier(

#  learning_rate =0.1,

#  n_estimators=1000,

#  max_depth=7,

#  min_child_weight=1,

#  gamma=0.1,

#  subsample=0.8,

#  colsample_bytree=0.8,

#  objective= 'binary:logistic',

#  nthread=4,

#  scale_pos_weight=1,

#  seed=100) highest so far



# model = XGBClassifier(

#  learning_rate =0.1,

#  n_estimators=1000,

#  max_depth=7,

#  min_child_weight=1,

#  gamma=0.1,

#  subsample=0.8,

#  colsample_bytree=0.8,

#  objective= 'binary:logistic',

#  nthread=4,

#  scale_pos_weight=1,

#  seed=100)



# dtc = DecisionTreeClassifier(criterion= 'gini' , splitter= 'best', max_depth=None, min_samples_split=3, min_samples_leaf=1, 

#                              min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 

#                              min_impurity_split=None, class_weight=None, presort=True) 
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import KFold
# print(test_X)
K = 5

kf = KFold(n_splits = K, random_state = 3228, shuffle = True)
xgb_params = {'learning_rate' :0.1, 'n_estimators': 950,'max_depth':7,'min_child_weight':1,'gamma':0.1,'subsample':0.8,'colsample_bytree':0.8,'objective': 'binary:logistic',

 'nthread':4,'scale_pos_weight':9,'seed':123,'eval_metric': 'auc'}

xgb_preds = []
for train_index, test_index in kf.split(train_X,train_y):

    xtr, xvl = train_X.iloc[train_index], train_X.iloc[test_index]

    ytr, yvl = train_y.iloc[train_index], train_y.iloc[test_index]





    d_train = xgb.DMatrix(xtr, ytr)

    d_valid = xgb.DMatrix(xvl, yvl)

    d_test = xgb.DMatrix(test_X)

    

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    model = xgb.train(xgb_params, d_train, 5000,  watchlist, maximize=True, verbose_eval=50, early_stopping_rounds=125)

                        

    xgb_pred = model.predict(d_test)

    xgb_preds.append(list(xgb_pred))
preds=[]

for i in range(len(xgb_preds[0])):

    sum=0

    for j in range(K):

        sum+=xgb_preds[j][i]

    preds.append(sum / K)



# output = pd.DataFrame({'id': id_test, 'target': preds})
# import lightgbm as lgb

# from sklearn import metrics

# from sklearn import model_selection as ms

# from lightgbm import LGBMClassifier

# folds = KFold(n_splits=5, shuffle=True, random_state=123)

# oof_preds = np.zeros(train_X.shape[0])

# sub_preds = np.zeros(test_X.shape[0])

# for n_fold, (train_index, valid_index) in enumerate(folds.split(train_X,train_y)):

#     xtr, ytr = train_X.iloc[train_index], train_y.iloc[train_index]

#     xvl, yvl = train_X.iloc[valid_index], train_y.iloc[valid_index]

    

#     clf = LGBMClassifier(

#         n_estimators=1500,

#         learning_rate=0.01,

#         num_leaves=123,

#         colsample_bytree=.8,

#         subsample=.8,

#         max_depth=10,

#         reg_alpha=.05,

#         reg_lambda=.05,

#         min_split_gain=.01,

#         min_child_weight=2,

#         gamma = 5,

# #         categorical_feature= cat_columns

#     )

    

#     clf.fit(xtr, ytr,eval_set= [(xtr, ytr), (xvl, yvl)],eval_metric='auc', verbose=150, early_stopping_rounds=150)

    

#     oof_preds[valid_index] = clf.predict_proba(xvl, num_iteration=clf.best_iteration_)[:, 1]

#     sub_preds += clf.predict_proba(test_X, num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

    

#     print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(yvl, oof_preds[valid_index])))

#     del clf, xtr, ytr, xvl, yvl
# fpreds= np.zeros(test_X.shape[0])



# fpreds = (sub_preds + preds)/2
# df_test = df_test.loc[:, 'V1':'V16']

# pred = model.predict_proba(df_test)
# print(pred)
result=pd.DataFrame()

result['Id'] = test_index_beg

result['PredictedValue'] = pd.DataFrame(preds)

result.head()
print(result)
result.to_csv('output.csv', index=False)
check = pd.read_csv('output.csv')

print(check)