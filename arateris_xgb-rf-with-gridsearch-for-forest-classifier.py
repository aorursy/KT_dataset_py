###### This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from collections import Counter



from sklearn.model_selection import cross_val_score,cross_validate, train_test_split, GridSearchCV, StratifiedKFold

from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score, make_scorer

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.linear_model import LogisticRegression



from lightgbm import LGBMClassifier

from xgboost import XGBClassifier, plot_importance



from tqdm import tqdm



import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/learn-together/train.csv", index_col='Id')

test = pd.read_csv("../input/learn-together/test.csv", index_col='Id')



y = train['Cover_Type'] # this is the target

X = train.drop('Cover_Type', axis = 1)

X_test = test.copy()



print('Train set shape : ', X.shape)

print('Test set shape : ', X_test.shape)
X.head()
X_test.head()
print('Missing Label? ', y.isnull().any())

print('Missing train data? ', X.isnull().any().any())

print('Missing test data? ', X_test.isnull().any().any())
print (X.dtypes.value_counts())

print (X_test.dtypes.value_counts())
X.describe()
X.nunique()
X.drop(['Soil_Type15', 'Soil_Type7'], axis=1, inplace = True)

X_test.drop(['Soil_Type15', 'Soil_Type7'], axis=1, inplace = True)
X_test.describe()
columns = X.columns
scaler = StandardScaler()

X.loc[:,:] = scaler.fit_transform(X)

X_test.loc[:,:] = scaler.transform(X_test)

X.describe()
X_train,  X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=1)

print(X_train.shape)

print(y_train.shape)

print(X_val.shape)

print(y_val.shape)
xgb= XGBClassifier( n_estimators=1000,  #todo : search for good parameters

                    learning_rate= 0.5,  #todo : search for good parameters

                    objective= 'binary:logistic', #this outputs probability,not one/zero. should we use binary:hinge? is it better for the learning phase?

                    random_state= 1,

                    n_jobs=-1)
param = {'n_estimators': [100, 500, 1000, 2000],

         'learning_rate': [0.001, 0.01, 0.1, 0.5, 1]}

grider = GridSearchCV(xgb, param, n_jobs=-1, cv=5, scoring='accuracy', verbose=True)

# res = grider.fit(X, y) #commented out as it takes some time, results shown below
rf = RandomForestClassifier(n_estimators = 100, n_jobs=-1, random_state=1)

param = {'n_estimators': [20, 100, 500, 1000, 2000]}

grider_rf = GridSearchCV(rf, param, n_jobs=-1, cv=5, scoring='accuracy', verbose=True)

# score_rf = grider_rf.fit(X, y)  #commented out, this takes quite some time. results shown below
xgb.fit(X=X_train, y=y_train,

        eval_metric='merror', # merror: Multiclass classification error rate. It is calculated as #(wrong cases)/#(all cases). 

        eval_set = [(X_train, y_train), (X_val, y_val)],

        early_stopping_rounds = 100,

        verbose = False

       )

xgb_val_pred = xgb.predict(X_val)

#print(xgb_val_pred)
rf = RandomForestClassifier(n_estimators = 1000, n_jobs=-1, random_state=1)

rf.fit(X=X_train, y=y_train)

rf_val_pred = rf.predict(X_val)
extc = ExtraTreesClassifier(n_estimators = 1000, n_jobs=-1, random_state=1)

extc.fit(X=X_train, y=y_train)

extc_pred = extc.predict(X_val)
lr = LogisticRegression( n_jobs=-1)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_val)
lbm_reg = LGBMClassifier() #to optimize param

lbm_reg.fit(X_train, y_train)

lbm_pred = lbm_reg.predict(X_val)
print(' random forest Val accuracy : ', accuracy_score(rf_val_pred, y_val))

print(' XGBR Val accuracy : ',accuracy_score(xgb_val_pred, y_val))

print(' LR Val accuracy : ',accuracy_score(lr_pred, y_val))

print(' LBM Val accuracy : ',accuracy_score(lbm_pred, y_val))

print(' EXTC Val accuracy : ',accuracy_score(extc_pred, y_val))

plt.subplots(2,2,figsize=(20,10))

ax=plt.subplot(1,2,1)

sns.heatmap(confusion_matrix(y_val, rf_val_pred), annot=True)

ax.set(ylabel='True label', xlabel='RF label')

ax2=plt.subplot(1,2,2)

sns.heatmap(confusion_matrix(y_val, xgb_val_pred), annot=True)

ax2.set(ylabel='True label', xlabel='XGB label')
plt.figure(figsize=(25,10))

sns.barplot(y=xgb.feature_importances_, x=columns)
extc.fit(X,y)

preds_test = extc.predict(X_test)

preds_test.shape
preds_test
count = { 1: 0.37062,

 2: 0.49657,

 3: 0.05947,

 4: 0.00106,

 5: 0.05623864624345282, #ongoing check...

 6: 0.04450142430004312,

 7: 0.05375584033702544}

weight = [count[x]/(sum(count.values())) for x in range(1,7+1)]

class_weight_lgbm = {i: v for i, v in enumerate(weight)}
def imbalanced_accuracy_score(y_true, y_pred):

    return accuracy_score(y_true, y_pred, sample_weight=[weight[x] for x in y_true-1])



imbalanced_accuracy_scorer = make_scorer(imbalanced_accuracy_score, greater_is_better=True)



def imbalanced_cross_validate(clf, X, y, cfg_args={}, fit_params={}, cv=5):

    return cross_validate(clf_inst, X, y, scorer= imbalanced_accuracy_scorer, cv=cv, n_jobs=-1, fit_params=fit_params )
# xgb= XGBClassifier( n_estimators=1000,  #todo : search for good parameters

#                     learning_rate= 0.5,  #todo : search for good parameters

#                     objective= 'binary:logistic', #this outputs probability,not one/zero. should we use binary:hinge? is it better for the learning phase?

#                     random_state= 1,

#                     n_jobs=-1)

# param = {'n_estimators': [500, 750, 1000],

#          'learning_rate': [0.1, 0.3, 0.5],

#          'max_depth': [6, 10, 25, 50]}

# xgb_grider = GridSearchCV(xgb, param, 

#                           n_jobs=-1, 

#                           cv=5, 

#                           scoring=imbalanced_accuracy_scorer, 

#                           verbose=2)

# res = xgb_grider.fit(X, y) 

# print(xgb_grider.best_score_)

# print(xgb_grider.best_params_)
# lgb= LGBMClassifier( n_estimators=1000,  #todo : search for good parameters

#                     learning_rate= 0.5,  #todo : search for good parameters

#                     objective= 'binary:logistic', #this outputs probability,not one/zero. should we use binary:hinge? is it better for the learning phase?

#                     random_state= 1,

#                     n_jobs=-1)

# param = {'n_estimators': [500, 750, 1000],

#          'learning_rate': [0.1, 0.3, 0.5],

#          'max_depth': [6, 10, 25, 50]}

# lgb_grider = GridSearchCV(lgb, param, 

#                           n_jobs=-1, 

#                           cv=5, 

#                           scoring=imbalanced_accuracy_scorer, 

#                           verbose=2)

# res = lgb_grider.fit(X, y) 

# print(lgb_grider.best_score_)

# print(lgb_grider.best_params_)
rf= RandomForestClassifier( n_estimators=1000, 

                            min_samples_split = 2, 

                            min_samples_leaf = 1,

                            bootstrap = False,

                            random_state=2019, 

                            class_weight=count)

param = {'n_estimators': [500, 750, 1000, 1500, 2000],

         'max_features': ['sqrt', 0.3, None],

         'max_depth': [100, 250, 500, None]}

rf_grider = GridSearchCV(rf, param, 

                          n_jobs=-1, 

                          cv=5, 

                          scoring=imbalanced_accuracy_scorer, 

                          verbose=2)

res = rf_grider.fit(X, y) #commented out as it takes some time, results shown below

print(rf_grider.best_score_)

print(rf_grider.best_params_)

ext= ExtraTreesClassifier( n_estimators=1000, 

                            min_samples_split = 2, 

                            min_samples_leaf = 1,

                            bootstrap = False,

                            random_state=2019, 

                            class_weight=count)

param = {'n_estimators': [500, 750, 1000, 1500, 2000],

         'max_features': ['sqrt', 0.3, None],

         'max_depth': [100, 250, 500, None]}

ext_grider = GridSearchCV(ext, param, 

                          n_jobs=-1, 

                          cv=5, 

                          scoring=imbalanced_accuracy_scorer, 

                          verbose=2)

res = ext_grider.fit(X, y) #commented out as it takes some time, results shown below

print(ext_grider.best_score_)

print(ext_grider.best_params_)

# Save test predictions to file

output = pd.DataFrame({'Id': X_test.index,

                       'Cover_Type': preds_test})

output.to_csv('submission.csv', index=False)

output.head()