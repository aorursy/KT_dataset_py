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



import warnings

warnings.filterwarnings("ignore")
specs = pd.read_csv('../input/data-science-bowl-2019/specs.csv')

specs.head()
tlabels = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')

tlabels.head()
train = pd.read_csv('../input/data-science-bowl-2019/train.csv', parse_dates=['timestamp'])

print(train.dtypes)

train.head()
test = pd.read_csv('../input/data-science-bowl-2019/test.csv', parse_dates = ['timestamp'])

print(test.dtypes)

test.head()
sample_submission = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')
print('train file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))

print('train_labels file have {} rows and {} columns'.format(tlabels.shape[0], tlabels.shape[1]))

print('test file have {} rows and {} columns'.format(test.shape[0], test.shape[1]))

print('specs file have {} rows and {} columns'.format(specs.shape[0], specs.shape[1]))

print('sample_submission file have {} rows and {} columns'.format(sample_submission.shape[0], sample_submission.shape[1]))
import seaborn as sns

x=tlabels['accuracy_group'].value_counts()

sns.barplot(x.index,x)
train2 = train.copy()

train2 = pd.get_dummies(train2, columns=['type', 'world'])



train2.head()
train2 = pd.merge(train2, tlabels, how = 'right', on = ['game_session', 'installation_id'])

train2.head()
train2 = train2.drop(['title_y'], axis = 1)
train2 = train2.rename(columns = {'title_x': 'title'})
train2.head()
map_label_traintitle = dict(zip(train2['title'].value_counts().sort_index().keys(),

                     range(1, len(train2['title'].value_counts())+1)))



train2['title'] = train2['title'].replace(map_label_traintitle)



train2.head()
train2 = train2.drop(['num_correct', 'num_incorrect', 'accuracy'], axis = 1)

train2.columns
map_label_testtitle = dict(zip(test['title'].value_counts().sort_index().keys(),

                     range(1, len(test['title'].value_counts())+1)))



test['title'] = test['title'].replace(map_label_testtitle)



test = pd.get_dummies(test, columns = ['type', 'world'])
train2['month'] = train2['timestamp'].dt.month

train2['day'] = train2['timestamp'].dt.weekday

train2['hour'] = train2['timestamp'].dt.hour
test['month'] = test['timestamp'].dt.month

test['day'] = test['timestamp'].dt.weekday

test['hour'] = test['timestamp'].dt.hour
x=train2['hour'].value_counts()

sns.barplot(x.index,x)
train2.isna().sum()
test.isna().sum()
train_data = train2.drop(['event_id', 'game_session', 'timestamp', 'event_data'], axis = 1)
test_data = test.drop(['event_id', 'game_session', 'timestamp', 'event_data'], axis = 1)
train_data.columns
test_data.columns
columnsTitles = ['installation_id', 'event_count', 'event_code', 'game_time', 'title','type_Activity', 'type_Assessment', 'type_Clip', 'type_Game',

       'world_CRYSTALCAVES', 'world_MAGMAPEAK', 'world_NONE','world_TREETOPCITY', 'month', 'day', 'hour', 'accuracy_group']



train_data = train_data.reindex(columns=columnsTitles)

train_data.head()
test_data.head()
X_train_data = train_data.copy().drop('accuracy_group', axis = 1)

y_train_data = train_data[['installation_id','accuracy_group']]
X_train_data.set_index('installation_id', inplace = True)

y_train_data.set_index('installation_id', inplace = True)

test_data.set_index('installation_id', inplace = True)
y_train_data.head()
# Logistic Regression



# from sklearn.linear_model import LogisticRegression



# logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs')



# Fit the model with data

# logreg.fit(X_train_data,y_train_data)
# y_pred_train_lr = logreg.predict(X_train_data)

# y_pred_test_lr = logreg.predict(test_data)
# y_pred_test_lr = pd.DataFrame(y_pred_test_lr, columns = ['accuracy_group'])

# y_pred_test_lr['installation_id'] = test['installation_id']
# columnsTitles = ['installation_id', 'accuracy_group']



# y_pred_test_lr = y_pred_test_lr.reindex(columns=columnsTitles)

# y_pred_test_lr.head()
# group_lr_pred = pd.DataFrame(y_pred_test_lr.groupby(['installation_id'])['accuracy_group'].mean())

# group_lr_pred = group_lr_pred.round().astype(int)

# group_lr_pred.head(10)
# finalsubmission = pd.DataFrame({'installation_id': group_lr_pred.index,'accuracy_group': group_lr_pred['accuracy_group']})

# finalsubmission.index = sample_submission.index

# finalsubmission.to_csv('submission.csv', index=False)
from sklearn import metrics

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, f1_score
# Decision Tree

# from sklearn import tree

# from sklearn.tree import DecisionTreeClassifier

# parameters={'max_depth':range(1,10)}

# clf=GridSearchCV(tree.DecisionTreeClassifier(),param_grid=parameters,n_jobs=-1,cv=10)

# clf.fit(X_train_data,y_train_data)

# print(clf.best_score_)

# print(clf.best_params_)
# dt_reg = clf.best_estimator_

# pred_train_dtree = dt_reg.predict(X_train_data)

# pred_test_dtree = dt_reg.predict(test_data)
# print(metrics.f1_score(y_train_data, pred_train_dtree, average = None).round(5))
# y_pred_test_dt = pd.DataFrame(pred_test_dtree, columns = ['accuracy_group'])

# y_pred_test_dt['installation_id'] = test['installation_id']
# columnsTitles = ['installation_id', 'accuracy_group']



# y_pred_test_dt = y_pred_test_dt.reindex(columns=columnsTitles)

# y_pred_test_dt.head()
# group_dt_pred = pd.DataFrame(y_pred_test_dt.groupby(['installation_id'])['accuracy_group'].mean())

# group_dt_pred = group_dt_pred.round().astype(int)

# group_dt_pred.head(10)
# finalsubmission_dt = pd.DataFrame({'installation_id': group_dt_pred.index,'accuracy_group': group_dt_pred['accuracy_group']})

# finalsubmission_dt.index = sample_submission.index

# finalsubmission_dt.to_csv('submission.csv', index=False)
# Random Forest

# from sklearn.ensemble import RandomForestClassifier

# rfc = RandomForestClassifier()

# rfc.fit(X = X_train_data,y = y_train_data)
# pred_train_rf = rfc.predict(X_train_data)

# pred_test_rf = rfc.predict(test_data)
# print(metrics.f1_score(y_train_data, pred_train_rf, average = None).round(5))
# y_pred_test_rf = pd.DataFrame(pred_test_rf, columns = ['accuracy_group'])

# y_pred_test_rf['installation_id'] = test['installation_id']
# columnsTitles = ['installation_id', 'accuracy_group']



# y_pred_test_rf = y_pred_test_rf.reindex(columns=columnsTitles)

# y_pred_test_rf.head()
# group_rf_pred = pd.DataFrame(y_pred_test_rf.groupby(['installation_id'])['accuracy_group'].mean())

# group_rf_pred = group_rf_pred.round().astype(int)

# group_rf_pred.head(10)
# finalsubmission_rf = pd.DataFrame({'installation_id': group_rf_pred.index,'accuracy_group': group_rf_pred['accuracy_group']})

# finalsubmission_rf.index = sample_submission.index

# finalsubmission_rf.to_csv('submission.csv', index=False)
# Adaboost

# from sklearn.ensemble import AdaBoostClassifier

# from sklearn.tree import DecisionTreeClassifier
# Create adaboost-decision tree classifer object

# Adaboost_model = AdaBoostClassifier(

  #  DecisionTreeClassifier(max_depth=5),

   # n_estimators = 600,

    #learning_rate = 1)
# Train model

#%time Adaboost_model.fit(X_train_data, y_train_data)
# Predict on Test 

# pred_train_ada = Adaboost_model.predict(X_train_data)

# pred_test_ada = Adaboost_model.predict(test_data)

# print("Accuracy on train is:",accuracy_score(y_train_data, pred_train_ada))

# print("f1 score - train:",metrics.f1_score(y_train_data, pred_train_ada, average = None).round(5))
# y_pred_test_ada = pd.DataFrame(pred_test_ada, columns = ['accuracy_group'])

# y_pred_test_ada['installation_id'] = test['installation_id']
# columnsTitles = ['installation_id', 'accuracy_group']

# y_pred_test_ada = y_pred_test_ada.reindex(columns=columnsTitles)

# y_pred_test_ada.head()
# group_ada_pred = pd.DataFrame(y_pred_test_ada.groupby(['installation_id'])['accuracy_group'].mean())

# group_ada_pred = group_ada_pred.round().astype(int)

# group_ada_pred.head(10)
# group_ada_pred_2 = pd.DataFrame(y_pred_test_ada.groupby(['installation_id'])['accuracy_group'].agg(lambda x:x.value_counts().index[0])) 

# group_ada_pred_2 = group_ada_pred_2.round().astype(int)

# group_ada_pred_2.head(10)
# submission

# finalsubmission_ada = pd.DataFrame({'installation_id': group_ada_pred_2.index,'accuracy_group': group_ada_pred_2['accuracy_group']})

# finalsubmission_ada.index = sample_submission.index

# finalsubmission_ada.to_csv('submission.csv', index=False)
# Gradient Boost

# from sklearn.ensemble import GradientBoostingClassifier

# GBM_model = GradientBoostingClassifier(n_estimators=50,

  #                                     learning_rate=0.3,

  #                                     subsample=0.8)
#### Train Gradient Boosting Classifer

# %time GBM_model.fit(X_train_data, y_train_data)
# Predict on Test 

# from sklearn.metrics import cohen_kappa_score

# pred_train_gb = GBM_model.predict(X_train_data)

# pred_test_gb = GBM_model.predict(test_data)

# print("Accuracy on train is:",accuracy_score(y_train_data, pred_train_gb))

# print("f1 score - train:",metrics.f1_score(y_train_data, pred_train_gb, average = None).round(5))
# y_pred_test_gb = pd.DataFrame(pred_test_gb, columns = ['accuracy_group'])

# y_pred_test_gb['installation_id'] = test['installation_id']
# columnsTitles = ['installation_id', 'accuracy_group']

# y_pred_test_gb = y_pred_test_gb.reindex(columns=columnsTitles)

# y_pred_test_gb.head()
# group_gb_pred = pd.DataFrame(y_pred_test_gb.groupby(['installation_id'])['accuracy_group'].agg(lambda x:x.value_counts().index[0])) 

# group_gb_pred = group_gb_pred.round().astype(int)

# group_gb_pred.head(10)
# submission...

# finalsubmission_gb = pd.DataFrame({'installation_id': group_gb_pred.index,'accuracy_group': group_gb_pred['accuracy_group']})

# finalsubmission_gb.index = sample_submission.index

# finalsubmission_gb.to_csv('submission.csv', index=False)
# Grid Search GB

from sklearn.ensemble import GradientBoostingClassifier

GBM = GradientBoostingClassifier()

# Use a grid over parameters of interest

param_grid = { 

           "n_estimators" : [50],

           "max_depth" : [8],

           "learning_rate" : [0.3, 0.8]}

 

CV_GBM = GridSearchCV(estimator=GBM, param_grid=param_grid, cv= 10)
%time CV_GBM.fit(X_train_data, y_train_data)
# Find best model

best_gbm_model = CV_GBM.best_estimator_

print (CV_GBM.best_score_, CV_GBM.best_params_)
pred_train_gsgb = best_gbm_model.predict(X_train_data)

pred_test_gsgb = best_gbm_model.predict(test_data)
y_pred_test_gsgb = pd.DataFrame(pred_test_gsgb, columns = ['accuracy_group'])

y_pred_test_gsgb['installation_id'] = test['installation_id']
columnsTitles = ['installation_id', 'accuracy_group']

y_pred_test_gsgb = y_pred_test_gsgb.reindex(columns=columnsTitles)

y_pred_test_gsgb.head()
group_gsgb_pred = pd.DataFrame(y_pred_test_gsgb.groupby(['installation_id'])['accuracy_group'].agg(lambda x:x.value_counts().index[0])) 

group_gsgb_pred = group_gsgb_pred.round().astype(int)

group_gsgb_pred.head(10)
finalsubmission_gsgb = pd.DataFrame({'installation_id': group_gsgb_pred.index,'accuracy_group': group_gsgb_pred['accuracy_group']})

finalsubmission_gsgb.index = sample_submission.index

finalsubmission_gsgb.to_csv('submission.csv', index=False)