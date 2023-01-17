from sklearn.metrics import f1_score , precision_score , recall_score , accuracy_score

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split



from lightgbm.sklearn import LGBMRegressor

# import xgboost as xgb

import lightgbm as lgb 

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier

from sklearn.model_selection import RandomizedSearchCV , GridSearchCV

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.preprocessing import LabelBinarizer

# from xgboost import XGBClassifier

from sklearn.externals import joblib

from joblib import dump, load



import pickle, os

# import ciso8601

from datetime import datetime
path = '../input/tj19data/'

train_original = pd.read_csv(path+'train.csv')

test_original = pd.read_csv(path+'test.csv')

demo = pd.read_csv(path+'demo.csv')

txn = pd.read_csv(path+'txn.csv')

print(train_original.shape,test_original.shape,demo.shape,txn.shape)
demo_train = demo[demo['id'].isin(train_original['id'].values)]

train = pd.merge(train_original, demo_train, on='id', how='left')



demo_test = demo[demo['id'].isin(test_original['id'].values)]

test = pd.merge(test_original, demo_test, on='id', how='left')



train.shape, test.shape
label_df = pd.get_dummies(train['label'])

label_df.columns = [f'label_{str(i)}' for i in label_df.columns]

label_df.head()
train_label = pd.concat([train, label_df],axis=1)

train_label.head()
category_list = ['c0','c1','c2','c3','c4']



for j in label_df.columns:

  for i in (category_list):

    means = train_label.groupby(i)[j].mean()

    

    train[str(i)+'_mean_'+str(j)] = train_label[i].map(means)

    test[str(i)+'_mean_'+str(j)] = train_label[i].map(means)



train.shape, test.shape
c_id = ['id']

c = ['c5','c6','c7','old_cc_label']



train_plus = train.copy()

test_plus = test.copy()

for i in c:

  pivoted = txn[[i]+c_id].pivot_table(index='id', columns=[i],aggfunc=len, fill_value=0)

  pivoted.columns = [f'{str(i)}_{str(k)}' for k in pivoted.columns]



  pivot_ordered = pivoted.loc[train_plus.id.tolist()]

  train_plus = pd.concat([train_plus, pivot_ordered.reset_index(drop=True)],axis=1)



  pivot_ordered = pivoted.loc[test_plus.id.tolist()]

  test_plus = pd.concat([test_plus, pivot_ordered.reset_index(drop=True)],axis=1)
train_plus.head()
agg = ['mean','max','sum','min','median']

# num_columns



main_c = {

    'n3':agg,

    'n4':agg,

    'n5':agg,

    'n6':agg,

    'n7':agg,

}



# for i in num_columns:

#   main_c = {**main_c,**{i:agg}} 

  

pivoted_num = txn.groupby('id').agg(main_c)

pivoted_num.head()
train_plus_add = train_plus.copy()

test_plus_add = test_plus.copy()



pivot_ordered = pivoted_num.loc[train_plus_add.id.tolist()]

train_plus_add = pd.concat([train_plus_add, pivot_ordered.reset_index(drop=True)],axis=1)



pivot_ordered = pivoted_num.loc[test_plus_add.id.tolist()]

test_plus_add = pd.concat([test_plus_add, pivot_ordered.reset_index(drop=True)],axis=1)



train_plus_add.shape , test_plus_add.shape
columns = train_plus_add.drop(['label'],axis=1).columns

x = train_plus_add.drop(['label'],axis=1)

y = train_plus_add['label'].values

x_test = test_plus_add.copy()
import pandas as pd

test_fam_txt_all_word = pd.read_csv("../input/techjam-2019-fam-txt-all-word/test_fam_txt_all_word.csv")

train_fam_txt_all_word = pd.read_csv("../input/techjam-2019-fam-txt-all-word/train_fam_txt_all_word.csv")



print(train_fam_txt_all_word.shape,test_fam_txt_all_word.shape)
xx = pd.concat([x, train_fam_txt_all_word.drop(['label'],axis=1).set_index('id').loc[x.id].reset_index(drop=True)],axis=1)

xx_test = pd.concat([x_test, test_fam_txt_all_word.set_index('id').loc[x_test.id].reset_index(drop=True)],axis=1)
columns = xx.columns

x_train, x_val, y_train, y_val = train_test_split(xx, y, test_size=0.15, random_state=42)

print(x_train.shape, x_val.shape, y_train.shape, y_val.shape )
lgb_train = lgb.Dataset(x_train, label=y_train)

lgb_val = lgb.Dataset(x_val, label=y_val)
# import lightgbm as lgb 

# lgb_params = {

#               "objective" : "multiclass",

#               "metric" : "multi_logloss",

#               "num_leaves" : 150, "learning_rate" : 0.012, 

#               "bagging_fraction" : 0.75, "feature_fraction" : 0.5, "bagging_frequency" :2,

#               'reg_alpha':1,

#               'max_depth': 25,

#               'num_class': 13,

#              }



# model = lgb.train(lgb_params, lgb_train, 1500, valid_sets=[lgb_train,lgb_val], early_stopping_rounds=100, verbose_eval=20)
# import lightgbm as lgb 



# lgb_data = lgb.Dataset(xx, label=y)

# lgb_params = {

#               "objective" : "multiclass",

#               "metric" : "multi_logloss",

#               "num_leaves" : 150, "learning_rate" : 0.012, 

#               "bagging_fraction" : 0.75, "feature_fraction" : 0.5, "bagging_frequency" :2,

#               'reg_alpha':1,

#               'max_depth': 25,

#               'num_class': 13,

#              }

# # model.best_iteration

# model_ans = lgb.train(lgb_params, lgb_data, 1256, valid_sets=[lgb_data], verbose_eval=20)
y_pred = model_ans.predict(xx_test)
df_pred = pd.DataFrame(y_pred)

df_pred.columns = [f'class{i}' for i in df_pred.columns]

df_pred.head()
ans = pd.concat([test_original, df_pred],axis=1)

ans.head()
ans.to_csv('techjam_final_all_word_kaggle.csv',index=False)