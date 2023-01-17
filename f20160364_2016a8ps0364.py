# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.preprocessing import RobustScaler

import xgboost

from xgboost import XGBClassifier
df = pd.read_csv('/kaggle/input/eval-lab-2-f464/train.csv')

test = pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')
plt.figure(figsize=(18,18))

sns.heatmap(df.corr(),vmin=-1,cmap='coolwarm',annot=True);
df.describe()
num_features = ['id','chem_1', 'chem_2', 'chem_4','chem_5','chem_6','attribute']

num_features1 = ['chem_1', 'chem_2', 'chem_4','chem_5','chem_6','attribute']

X_train = df[num_features]

y_train = df["class"]

X = X_train

y = y_train
# from sklearn.tree import DecisionTreeClassifier

# from sklearn.ensemble import RandomForestClassifier

# from sklearn.linear_model import LinearRegression

# from sklearn.neighbors import KNeighborsClassifier

# from sklearn.ensemble import ExtraTreesClassifier

# from numpy import loadtxt

# from xgboost import XGBClassifier

# from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# import xgboost as xgb

# from sklearn.metrics import accuracy_score  #Find out what is accuracy_score





# clf1 = DecisionTreeClassifier().fit(x_train,y_train)

# clf2 = ExtraTreesClassifier(n_estimators=3000,min_samples_split=2,min_samples_leaf=2,max_features='sqrt',max_depth=110,bootstrap=False).fit(x_train,y_train) # also best



# clf3 = DecisionTreeClassifier().fit(x_train_scaled,y_train)

# clf4 = ExtraTreesClassifier(n_estimators=3000,min_samples_split=2,min_samples_leaf=2,max_features='sqrt',max_depth=110,bootstrap=False).fit(x_train_scaled,y_train) 



# params = {"objective": "reg:linear",

#           "eta": 0.3,

#           "max_depth": 5,

#           "min_child_weight": 3,

#           "silent": 1,

#           "subsample": 0.7,

#           "colsample_bytree": 0.7,

#           "seed": 1}

# num_trees=250

# gbm = xgb.train(params, xgb.DMatrix(x_train[num_features],y_train), num_trees)

# y_pred = gbm.predict(xgb.DMatrix(x_val[num_features]))

# print(accuracy_score(y_val,[int(round(x)) for x in y_pred])



# clf9 = XGBClassifier(silent=True, 

#                       scale_pos_weight=1,

#                       learning_rate=0.03,  

#                       colsample_bytree = 1,

#                       subsample = 0.8,

#                       objective='multi:softprob', 

#                       n_estimators=10000, 

#                       reg_alpha = 0.3,

#                       max_depth=3, 

#                       gamma=1).fit(x_train[num_features],y_train)



# clf9 = XGBClassifier().fit(x_train_scaled[num_features],y_train)



# y_pred9 = clf9.predict(x_val[num_features])

# for i in range(len(y_pred9)):

#     y_pred9[i] = round(y_pred9[i])

# print(accuracy_score(y_train,clf9.predict(x_train[num_features])))

# print(accuracy_score(y_val,y_pred9))

# print("****************")



# Building the model 

# extra_tree_forest = ExtraTreesClassifier(n_estimators=1000, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=False, oob_score=False, random_state=42, verbose=0, warm_start=False, class_weight='balanced') 

  

# Training the model 

# extra_tree_forest.fit(X[num_features], y) 

  

# Computing the importance of each feature 

# feature_importance = extra_tree_forest.feature_importances_ 

  

# Normalizing the individual importances 

# feature_importance_normalized = np.std([tree.feature_importances_ for tree in extra_tree_forest.estimators_], axis = 0)

# y_pred10 = extra_tree_forest.predict(test[num_features]) 

# print(accuracy_score(y_train,extra_tree_forest.predict(X[num_features])))

# print(accuracy_score(y_val,y_pred10))



# y_pred_1 = clf1.predict(x_val)

# y_pred_2 = clf2.predict(x_val)



# y_pred_3 = clf3.predict(x_val_scaled)

# y_pred_4 = clf4.predict(x_val_scaled)



# y_pred_5 = clf5.predict(x_val)

# y_pred_6 = clf6.predict(x_val_scaled)



# y_pred_7 = clf7.predict(x_val)

# y_pred_8 = clf8.predict(x_val_scaled)



# clf1 = DecisionTreeClassifier().fit(x_train,y_train)

# clf2 = RandomForestClassifier(n_estimators=1000,min_samples_split=2,min_samples_leaf=2,max_features='sqrt',max_depth=30,bootstrap=False).fit(x_train,y_train) # also best



# y_pred_1 = clf1.predict(x_test)

# y_pred_2 = clf2.predict(x_test)



# print("**********")

# clf = RandomForestClassifier(n_estimators=1000).fit(X[num_features],y)

# y_pred = clf.predict(test[num_features])

# print(accuracy_score(y_val,y_pred))
# Submission 1

clf1 = XGBClassifier().fit(X[num_features],y)

y_pred1 = clf1.predict(test[num_features])



# Submission 2

clf2 = XGBClassifier().fit(X[num_features1],y)

y_pred2 = clf2.predict(test[num_features1])
answer = pd.DataFrame(data={'id':test['id'],'class':y_pred1})

answer.to_csv('final1.csv',index=False)



answer = pd.DataFrame(data={'id':test['id'],'class':y_pred2})

answer.to_csv('final2.csv',index=False)