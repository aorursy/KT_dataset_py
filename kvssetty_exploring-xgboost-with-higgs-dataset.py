import numpy as np

import pandas as pd

import scipy as sp

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import xgboost as xgb
import os

print(os.listdir("../input"))
higgs_df = pd.read_csv('../input/higgsb/training.csv')
higgs_df.head()
higgs_df.info()
pd.set_option('precision',2)

plt.figure(figsize=(10, 8))

sns.heatmap(higgs_df.drop(['Label','EventId'],axis=1).corr(), square=True)

plt.suptitle("Pearson Correlation Heatmap")

#plt.show()
higgs_df_binary_label = pd.get_dummies(higgs_df['Label'])
higgs_df['Label'] = higgs_df_binary_label


corr_with_label = higgs_df.corr()["Label"].sort_values(ascending=False)

plt.figure(figsize=(14,6))

corr_with_label.drop("Label").plot.bar()

plt.show()

higgs_df.info()
higgs_df = higgs_df.drop(['EventId',], axis=1)
X_higgs = higgs_df.drop(['Label','Weight'],axis=1)
y_higgs = higgs_df['Label']
w_higgs = higgs_df['Weight']
X_higgs.shape
y_higgs.shape
col_names = list(X_higgs.columns)

col_names
w_higgs.shape
y_higgs.value_counts()
xg_dmat = xgb.DMatrix(X_higgs, label=y_higgs, missing=-999.000, weight=w_higgs,

                     silent=False, feature_names=col_names, feature_types=None, nthread=-1)
xg_dmat.feature_names
xg_dmat.get_base_margin()
xg_dmat.get_label()
xg_dmat.get_weight()
xg_dmat.num_col()
xg_dmat.num_row()
xg_dmat.save_binary('xg_dmat_obj01')
xg_dmat_training = xg_dmat.slice(X_higgs.index[0:200000])

xg_dmat_validating = xg_dmat.slice(X_higgs.index[200000:2500000])

xg_dmat_training.num_row()
xg_dmat_validating.num_row()
xg_dmat_training.save_binary('xg_dmat_training_obj01')

xg_dmat_validating.save_binary('xg_dmat_validating_obj01')
param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic','nthread': 4,

        'eval_metric':['aucpr','auc']}
bst = xgb.train(param, xg_dmat_training, num_boost_round=2000,

                evals = [(xg_dmat_validating, 'validate'), (xg_dmat_training, 'train')],

               early_stopping_rounds=5, verbose_eval=20)
bst.save_model('final_model01.model')
bst.attributes()
bst.dump_model('final_model01.nice.txt', fmap='../input/feature-map/featmap.txt', with_stats=False)
bst.dump_model('final_model01.raw.txt')
bst.get_fscore(fmap='../input/feature-map/featmap.txt')
bst.get_score(fmap='../input/feature-map/featmap.txt', importance_type='gain')
predictions01 =bst.predict(xg_dmat_validating, output_margin=False, ntree_limit=0, 

                           pred_leaf=False, pred_contribs=False,approx_contribs=False, 

                           pred_interactions=False, validate_features=True)
predictions01
len(predictions01)
predictions02 =bst.predict(xg_dmat_validating, output_margin=True, ntree_limit=0, 

                           pred_leaf=False, pred_contribs=False,approx_contribs=False, 

                           pred_interactions=False, validate_features=True)

predictions02
predictions03 =bst.predict(xg_dmat_validating, output_margin=False, ntree_limit=0, 

                           pred_leaf=True, pred_contribs=False,approx_contribs=False, 

                           pred_interactions=False, validate_features=True)
predictions03[0:2]
predictions03.shape
predictions04 =bst.predict(xg_dmat_validating, output_margin=False, ntree_limit=0, 

                           pred_leaf=False, pred_contribs=True,approx_contribs=False, 

                           pred_interactions=False, validate_features=True)

predictions04
predictions04.shape
predictions05 =bst.predict(xg_dmat_validating, output_margin=False, ntree_limit=0, 

                           pred_leaf=False, pred_contribs=False,approx_contribs=True, 

                           pred_interactions=False, validate_features=True)

predictions05
predictions05.shape
predictions06 =bst.predict(xg_dmat_validating, output_margin=False, ntree_limit=10, 

                           pred_leaf=True, pred_contribs=False,approx_contribs=False, 

                           pred_interactions=False, validate_features=True)

predictions06
predictions06.shape
df_predictions06 = pd.DataFrame(predictions06)
df_predictions06.head()      # the values are the indicies of leafs in the first ten trees