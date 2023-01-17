# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import recall_score

from sklearn.metrics import precision_score

from sklearn.metrics import f1_score

import operator



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/train_ctrUa4K.csv',low_memory = False)

data['Loan_Status'] = data['Loan_Status'].map({'Y': 1, 'N': 0})

data = pd.get_dummies(data=data, columns=['Gender', 'Married','Dependents','Education','Self_Employed','Property_Area'])

train, test = train_test_split(data, test_size=0.2,random_state=42)

print(train.dtypes)

print(test.dtypes)
train_X = train.drop(['Loan_Status','Loan_ID'],axis = 1)

train_Y = train[['Loan_Status']]

labels = np.array(train_Y['Loan_Status'])

features = np.array(train_X)

test_X = test.drop(['Loan_Status','Loan_ID'],axis = 1)

test_Y = test[['Loan_Status']]

labels_test = np.array(test_Y['Loan_Status'])

features_test = np.array(test_X)
#xGB Classifier with hyperparameter tuning

dtrain = xgb.DMatrix(train_X,label=labels)

dtest = xgb.DMatrix(test_X)

param = {'max_depth':10, 'eta':0.02, 'silent':1, 'objective':'binary:logistic' }

param['nthread'] = 4

param['eval_metric'] = 'auc'

param['subsample'] = 0.7

param['colsample_bytree']= 0.7

param['min_child_weight'] = 0

param['booster'] = "gbtree"

watchlist  = [(dtrain,'train')]

num_round = 1240

early_stopping_rounds=10



#Training

bst = xgb.train(param, dtrain, num_round, watchlist,early_stopping_rounds=early_stopping_rounds)

#Predict for validation set

pred_prob = bst.predict(dtest)

#pred_prob = model.predict(features_test2)

predictions = [round(value) for value in pred_prob]

predictions = [int(value) for value in predictions]

accuracy = accuracy_score(labels_test.astype(int),predictions)

print('Accuracy:', round(accuracy, 2), '%.')

print('Precision:',round(precision_score(labels_test.astype(int),predictions),2),'%.')

print ('Recall:',round(recall_score(labels_test.astype(int),predictions),2),'%.')

print('roc_auc_score:',round(roc_auc_score(labels_test.astype(int),predictions),2),'%.')

print('f1 score:',round(f1_score(labels_test.astype(int),predictions),2))
def get_xgb_feat_importances(clf):



    if isinstance(clf, xgb.XGBModel):

        # clf has been created by calling

        # xgb.XGBClassifier.fit() or xgb.XGBRegressor().fit()

        fscore = clf.booster().get_fscore()

    else:

        # clf has been created by calling xgb.train.

        # Thus, clf is an instance of xgb.Booster.

        fscore = clf.get_fscore()



    feat_importances = []

    for ft, score in fscore.items():

        feat_importances.append({'Feature': ft, 'Importance': score})

    feat_importances = pd.DataFrame(feat_importances)

    feat_importances = feat_importances.sort_values(

        by='Importance', ascending=False).reset_index(drop=True)

    # Divide the importances by the sum of all importances

    # to get relative importances. By using relative importances

    # the sum of all importances will equal to 1, i.e.,

    # np.sum(feat_importances['importance']) == 1

    feat_importances['Importance'] /= feat_importances['Importance'].sum()

    # Print the most important features and their importances

    print(feat_importances.head())

    return feat_importances
feat_importance = get_xgb_feat_importances(bst)
ax = feat_importance.plot.bar(x='Feature', y='Importance', rot=90)
test_data = pd.read_csv('/kaggle/input/test_lAUu6dG.csv',low_memory = False)

test_data = pd.get_dummies(data=test_data, columns=['Gender', 'Married','Dependents','Education','Self_Employed','Property_Area'])

test_df = test_data.drop(['Loan_ID'],axis = 1)

dftest = xgb.DMatrix(test_df)

pred_prob_1 = bst.predict(dftest)

pred_prob_0 = 1-pred_prob_1

pred_score = pd.DataFrame(

    {'pred_0': pred_prob_0,

     'pred_1': pred_prob_1

    })

pred_score.to_csv("pred_score_xgb.csv")