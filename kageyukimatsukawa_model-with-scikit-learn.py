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
# Check train.csv roughly

df = pd.read_csv("/kaggle/input/titanic/train.csv")

df.head()
# Check there is no unexpected "y" field

print('count of null value of Survived column : {}'.format(df[df.Survived.isnull()].size))

print('-----------------------')

print('counts of unique values')

print(df.Survived.value_counts())
# Just check SibSp counts (Just satisfy my interest)

df.SibSp.value_counts()
# Just check Parch counts (Just satisfy my interest)

df.Parch.value_counts()
# Prepare for one-hot encoding for categorical variable

one_hot_case1 = {

    "Pclass":object,

    "Sex":object,

    "Embarked":object,

    "Cabin":object,

}
# Load train.csv again with one_hot_case1 setting

df_case1 = pd.read_csv("/kaggle/input/titanic/train.csv",

                dtype=one_hot_case1)
# Divide to y and x features

y_train = df_case1 .loc[:,["Survived"]]

X_case1 = df_case1 .iloc[:,2:]

# Drop unnecessary columns

X_case1 = X_case1.drop(["Name","Ticket"],axis=1)
# Check column including null

def null_check(df):

    for col in df.columns:

        if np.sum(df[col].isnull()) > 0:

            print(col)
null_check(X_case1)
# One-hot encoding with one_hot_case1 setting

X_case1 = pd.get_dummies(X_case1,

              dummy_na=True,

              columns=one_hot_case1.keys())
# Set average value against null

from sklearn.impute import SimpleImputer

imp_case1 = SimpleImputer()

imp_case1.fit(X_case1)

X_case1 = pd.DataFrame(imp_case1.transform(X_case1), columns=X_case1.columns)
null_check(X_case1)
# Check number of columns. I think there are too many columns. That's why, I will do feature selection in next step.

print(X_case1.shape[1])
# Feature selection

from sklearn.feature_selection import RFE

from sklearn.ensemble import RandomForestClassifier





selector = RFE(RandomForestClassifier(n_estimators=100, random_state=1),

               n_features_to_select=20,

               step=.05)



selector.fit(X_case1,y_train)



X_case1_selected = X_case1.loc[:, X_case1.columns[selector.support_]]

print('X_fin_case1 shape:(%i,%i)' % X_case1_selected.shape)

X_case1_selected.head()
# Prepare for one-hot encoding for categorical variable

one_hot_case2 = {

    "Pclass":object,

    "Sex":object,

}
# Load train.csv again with one_hot_case2 setting

df_case2 = pd.read_csv("/kaggle/input/titanic/train.csv",

                dtype=one_hot_case2)
# Drop unnecessary columns

X_case2 = df_case2.iloc[:,2:]

X_case2 = X_case2.drop(["Name","Ticket", "Embarked", "Cabin"],axis=1)
# One-hot encoding with one_hot_case2 setting

X_case2 = pd.get_dummies(X_case2,

              dummy_na=True,

              columns=one_hot_case2.keys())
# Set average value against null

imp_case2 = SimpleImputer()

imp_case2.fit(X_case2)

X_case2 = pd.DataFrame(imp_case2.transform(X_case2), columns=X_case2.columns)
null_check(X_case2)
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

import warnings

warnings.filterwarnings('ignore')



pl = Pipeline([('scl',StandardScaler()),

               ('pca',PCA(random_state=1)),

               ('est',LogisticRegression(solver='liblinear', random_state=1))])

param_grid = {'pca__n_components':[None,5,7,10],

             'est__C':[0.001,0.01,0.1,0.2,0.3,0.5,1.0,10.0,100.0],

             'est__penalty':['l1', 'l2']}



gs = GridSearchCV(estimator=pl,

                  param_grid=param_grid,

                  scoring='roc_auc',

                  cv=3,

                  return_train_score=False)

gs.fit(X_case1, y_train)

print('case 1 best score : {}'.format(gs.best_score_))

print('case 1 best param : %s' % gs.best_params_)

gs.fit(X_case1_selected, y_train)

print('case 1 selected best score : {}'.format(gs.best_score_))

print('case 1 selected best param : %s' % gs.best_params_)

gs.fit(X_case2, y_train)

print('case 2 best score : {}'.format(gs.best_score_))

print('case 2 best param : %s' % gs.best_params_)
from sklearn.svm import SVC



pl = Pipeline([('scl',StandardScaler()),

               ('pca',PCA(random_state=1)),

               ('est',SVC(random_state=1))])

param_grid = {'est__gamma':[0.001,0.002,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.5,1.0,10.0],

             'est__kernel':['linear','rbf'],

             'pca__n_components':[None,5,7,10],}



gs = GridSearchCV(estimator=pl,

                  param_grid=param_grid,

                  scoring='roc_auc',

                  cv=3,

                  return_train_score=False)

gs.fit(X_case1, y_train)

print('case 1 best score : {}'.format(gs.best_score_))

print('case 1 best param : %s' % gs.best_params_)

gs.fit(X_case1_selected, y_train)

print('case 1 selected best score : {}'.format(gs.best_score_))

print('case 1 selected best param : %s' % gs.best_params_)

gs.fit(X_case2, y_train)

print('case 2 best score : {}'.format(gs.best_score_))

print('case 2 best param : %s' % gs.best_params_)
from sklearn.ensemble import RandomForestClassifier



pl = Pipeline([('scl',StandardScaler()),

              # ('pca',PCA(random_state=1)),

               ('est',RandomForestClassifier(random_state=1))])

param_grid = {'est__max_depth':[4,5,6,7,10],

             'est__n_estimators':[1200,1300,1400]}

            # 'pca__n_components':[None,5,7,10],}



gs = GridSearchCV(estimator=pl,

                  param_grid=param_grid,

                  scoring='roc_auc',

                  cv=3,

                  return_train_score=False)

#gs.fit(X_case1, y_train)

#print('case 1 best score : {}'.format(gs.best_score_))

#print('case 1 best param : %s' % gs.best_params_)

gs.fit(X_case1_selected, y_train)

print('case 1 selected best score : {}'.format(gs.best_score_))

print('case 1 selected best param : %s' % gs.best_params_)

#gs.fit(X_case2, y_train)

#print('case 2 best score : {}'.format(gs.best_score_))

#print('case 2 best param : %s' % gs.best_params_)
from sklearn.ensemble import GradientBoostingClassifier



pl = Pipeline([('scl',StandardScaler()),

              # ('pca',PCA(random_state=1)),

               ('est',GradientBoostingClassifier(random_state=1))])

param_grid = {'est__max_depth':[1,2,3,4,5],

             'est__n_estimators':[50,60,70,80,90,100,200,250],

             'est__learning_rate':[0.09,0.1]}

            # 'pca__n_components':[None,10],}



gs = GridSearchCV(estimator=pl,

                  param_grid=param_grid,

                  scoring='roc_auc',

                  cv=3,

                  return_train_score=False)

gs.fit(X_case1, y_train)

print('case 1 best score : {}'.format(gs.best_score_))

print('case 1 best param : %s' % gs.best_params_)

gs.fit(X_case1_selected, y_train)

print('case 1 selected best score : {}'.format(gs.best_score_))

print('case 1 selected best param : %s' % gs.best_params_)

gs.fit(X_case2, y_train)

print('case 2 best score : {}'.format(gs.best_score_))

print('case 2 best param : %s' % gs.best_params_)
# Load test.csv

test_df = pd.read_csv("/kaggle/input/titanic/test.csv",

                     dtype=one_hot_case1)

ID = test_df.loc[:,["PassengerId"]]

# Remove unnecessary columns

X_test = test_df.drop(["Name","Ticket"],axis=1)
# One-hot encoding with one_hot_case1 setting

X_test = pd.get_dummies(X_test,

                       dummy_na=True,

                       columns=one_hot_case1.keys())
# Remove columns which doesn't exist in train data

X_train_none_df = pd.DataFrame(None,

                              columns=X_case1.columns,

                              dtype=float)

X_test_with_none_train = pd.concat([X_test, X_train_none_df])

set_columns_x_train = set(X_case1.columns)

set_columns_x_test = set(X_test.columns)

X_test_arrange = X_test_with_none_train.drop(list(set_columns_x_test - set_columns_x_train),axis=1)
# Set 0 in empty columns which doesn't exist in test data originally

X_test_arrange.loc[:,list(set_columns_x_train - set_columns_x_test)] = X_test_arrange.loc[:,list(set_columns_x_train - set_columns_x_test)].fillna(0, axis=1)
# Set average value against null

X_test_arrange = X_test_arrange.reindex(X_case1.columns,axis=1)

X_test_arrange = pd.DataFrame(imp_case1.transform(X_test_arrange), columns=X_test_arrange.columns)
# Feature selection by RFE using train data

X_test_fin = X_test_arrange.loc[:, X_test_arrange.columns[selector.support_]]
# Predict

gb = GradientBoostingClassifier( max_depth=2, n_estimators=90, random_state=1)

gb.fit(X_case1_selected, y_train)

predict_y = pd.DataFrame(gb.predict(X_test_fin), columns=["Survived"])

ID.join(predict_y).to_csv('/kaggle/working/matsukawa2_submission.csv', index=False)