import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor

import lightgbm as lgb

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

from sklearn.metrics import classification_report



# Drawing settings in Jupyter

plt.style.use('ggplot')

%config InlineBackend.figure_format = 'retina'
%%time



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.shape
y0 = train[train['target'] == 0]['target']

y1 = train[train['target'] == 1]['target']

print('target Ratio: ' + str(len(y1) / (len(y0) + len(y1))))



train['target'].value_counts().plot.bar()

plt.title('target(y)');
data = train.set_index('ID_code').copy()



Y = 'target'

selected_X = data.drop(Y, axis = 1)
# LGB



# To Evaluate the model

X_train, X_test, Y_train, Y_test = train_test_split(selected_X, data[Y], test_size = 0.5, random_state = 0)

lgb_train = lgb.Dataset(X_train,Y_train)



# To Submit

#lgb_train = lgb.Dataset(selected_X, data[Y])
# LightGBM parameters

params = {'boosting_type': 'gbdt',

          'max_depth' : 10,

          'objective': 'binary',

          'nthread': 3, # Updated from nthread

          'num_leaves': 1024,

          'learning_rate': 0.15,

          'max_bin': 512,

          'subsample_for_bin': 200,

          'subsample': 1,

          'subsample_freq': 1,

          'colsample_bytree': 0.8,

          'reg_alpha': 5,

          'reg_lambda': 10,

          'min_split_gain': 0.5,

          'min_child_weight': 1,

          'min_child_samples': 5,

          'scale_pos_weight': 1,

          'num_class' : 1,

          'metric' : 'binary_error'}



# train

gbm = lgb.train(params,

                lgb_train,

                num_boost_round = 50,

                #valid_sets = lgb_eval, 

                #early_stopping_rounds = 10

               )



# To Evalueate The Model

y_pred = gbm.predict(X_test, num_iteration = gbm.best_iteration)



# To Submit

#y_pred = gbm.predict(test.set_index('ID_code'), num_iteration = gbm.best_iteration)
y_pred_df = pd.DataFrame(y_pred).set_index(Y_test.index)[0]
logit_roc_auc = roc_auc_score(Y_test, y_pred_df)

fpr, tpr, thresholds = roc_curve(Y_test, y_pred_df)

plt.figure(figsize=(12, 6))

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % logit_roc_auc)

plt.plot([0, 1], [0, 1], 'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend(loc = 'lower right')

plt.show()
#y_pred_df = pd.DataFrame(y_pred).set_index(test['ID_code']).rename(columns = {0:'target'})['target']

#pd.DataFrame(y_pred_df).to_csv('\to_submit.csv', encoding = 'utf-8')