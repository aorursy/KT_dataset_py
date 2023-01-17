# This Python 3 environment comes with many helpful analytics libraries installed



# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.





import matplotlib.pyplot as plt

import seaborn as sns



pd.options.display.max_columns = 100
# Starting with train data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')





train.info()

train.head()
# train['Cabin'][0] is np.NaN

train = train.drop({'Cabin'}, axis=1)
# Adding feature for each Pclass

for idx, f_name in enumerate(['Pclass1', 'Pclass2', 'Pclass3']):

    # print(idx + 1, f_name)

    train[f_name] = train.apply(lambda x: 1 if x['Pclass'] == (idx + 1) else 0, axis = 1)
# Gender feature 

train['Gender'] = (train['Sex'] == 'female').astype(int)
# Feature for each port of embarkment

for f_val in list(train['Embarked'].unique()):

    f_name = 'Embarked.{}'.format(f_val)

    train[f_name] = train.apply(lambda x: 1 if x['Embarked'] == f_val else 0, axis=1)
train['Age_isNaN'] = train['Age'].isnull().astype(int)
train['Child'] = (train['Age'] <= 18).astype(int)

train['Adult'] = (train['Age'] > 18).astype(int)
train['Fare'].hist(bins=100)
for_fittin = train.drop({'Pclass', 'Name', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Embarked', 'Age'}, axis=1)


for_fittin.info(), for_fittin.sample()
cr = for_fittin.corr()

cols = list(cr.columns)



plt.figure(figsize=(14, 12))

sns.heatmap(cr, 

            xticklabels=cols,

            yticklabels=cols,

            annot=True)

for_fittin = for_fittin.drop({'Embarked.nan'}, axis=1)
for_fittin.sample()
import sklearn

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression



import xgboost as xgb
train_f, cv_f = train_test_split(for_fittin, test_size=0.2, random_state=42)
X_train = train_f.drop({'Survived'}, axis=1)

y_train = train_f['Survived']



X_cv = cv_f.drop({'Survived'}, axis=1)

y_cv = cv_f['Survived']
lg_param_grid = {

   'C': [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.5, 1, 5, 10],

    'class_weight': [None, 'balanced'],

    'max_iter': [100, 200, 500, 1000]

}



optimizer = GridSearchCV(estimator=LogisticRegression(), param_grid=lg_param_grid)

optimizer.fit(X_train, y_train)



optimizer.best_estimator_, optimizer.best_params_, optimizer.best_score_
optimizer.best_estimator_.score(X_cv, y_cv)
xgb_model = xgb.XGBClassifier()

xgb_model.fit(X_train, y_train)

#xgb_model.score(X_cv, y_cv)
xgb_model.score(X_cv, y_cv)
test = pd.read_csv('../input/test.csv')
# train['Cabin'][0] is np.NaN

test = test.drop({'Cabin'}, axis=1)



# Adding feature for each Pclass

for idx, f_name in enumerate(['Pclass1', 'Pclass2', 'Pclass3']):

    # print(idx + 1, f_name)

    test[f_name] = test.apply(lambda x: 1 if x['Pclass'] == (idx + 1) else 0, axis = 1)

    

# Gender feature 

test['Gender'] = (test['Sex'] == 'female').astype(int) 







# Feature for each port of embarkment

#for f_val in reversed(list(test['Embarked'].unique())):

for f_val in ['Embarked.S', 'Embarked.C', 'Embarked.Q']:

    #f_name = 'Embarked.{}'.format(f_val)

    f_name = f_val

    test[f_name] = test.apply(lambda x: 1 if x['Embarked'] == f_val else 0, axis=1)

    

    

test['Age_isNaN'] = test['Age'].isnull().astype(int)

    

test['Child'] = (test['Age'] <= 18).astype(int)

test['Adult'] = (test['Age'] > 18).astype(int)



#test['Age_isNaN'] = test['Age'].isnull().astype(int)
test = test.drop({'Pclass', 'Name', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Embarked', 'Age'}, axis=1)
xgb_model.predict(test)
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': xgb_model.predict(test)})
submission
submission.to_csv('submission_1.csv', index=False)