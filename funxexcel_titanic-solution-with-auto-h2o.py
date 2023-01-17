import pandas as pd

import numpy as numpy

import h2o

#from h2o.estimators.gbm import H2OGradientBoostingEstimator

#from h2o.grid.grid_search import H2OGridSearch

from h2o.automl import H2OAutoML

from sklearn import metrics

from sklearn.metrics import roc_auc_score
h2o.init()
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")
train.isnull().sum()
test.isnull().sum()
all = pd.concat([train, test], sort = False)

all.info()
#Fill Missing numbers with median for Age and Fare

all['Age'] = all['Age'].fillna(value=all['Age'].median())

all['Fare'] = all['Fare'].fillna(value=all['Fare'].median())



#Treat Embarked

all['Embarked'] = all['Embarked'].fillna('S')



#Bin Age

#Age

all.loc[ all['Age'] <= 16, 'Age'] = 0

all.loc[(all['Age'] > 16) & (all['Age'] <= 32), 'Age'] = 1

all.loc[(all['Age'] > 32) & (all['Age'] <= 48), 'Age'] = 2

all.loc[(all['Age'] > 48) & (all['Age'] <= 64), 'Age'] = 3

all.loc[ all['Age'] > 64, 'Age'] = 4 



#Cabin

all['Cabin'] = all['Cabin'].fillna('Missing')

all['Cabin'] = all['Cabin'].str[0]



#Family Size & Alone 

all['Family_Size'] = all['SibSp'] + all['Parch'] + 1

all['IsAlone'] = 0

all.loc[all['Family_Size']==1, 'IsAlone'] = 1
#Title

import re

def get_title(name):

    title_search = re.search(' ([A-Za-z]+\.)', name)

    

    if title_search:

        return title_search.group(1)

    return ""
all['Title'] = all['Name'].apply(get_title)

all['Title'].value_counts()
all['Title'] = all['Title'].replace(['Capt.', 'Dr.', 'Major.', 'Rev.'], 'Officer.')

all['Title'] = all['Title'].replace(['Lady.', 'Countess.', 'Don.', 'Sir.', 'Jonkheer.', 'Dona.'], 'Royal.')

all['Title'] = all['Title'].replace(['Mlle.', 'Ms.'], 'Miss.')

all['Title'] = all['Title'].replace(['Mme.'], 'Mrs.')

all['Title'].value_counts()
#Drop unwanted variables

all_1 = all.drop(['Name', 'Ticket'], axis = 1)

all_1.head()
all_dummies = pd.get_dummies(all_1, drop_first = True)

all_dummies.head()
all_train = h2o.H2OFrame(all_dummies[all_dummies['Survived'].notna()])

all_test = h2o.H2OFrame(all_dummies[all_dummies['Survived'].isna()])
target = 'Survived'

features = [f for f in all_train.columns if f not in ['Survived','PassengerId']]
train_df, valid_df, test_df = all_train.split_frame(ratios=[0.7, 0.15], seed=2018)
train_df[target] = train_df[target].asfactor()

valid_df[target] = valid_df[target].asfactor()

test_df[target] = test_df[target].asfactor()
predictors = features



aml = H2OAutoML(max_models = 50, max_runtime_secs=5000, seed = 1)

aml.train(x=predictors, y=target, training_frame=train_df, validation_frame=valid_df)
lb = aml.leaderboard

lb
aml.leader.params.keys()
aml.leader.model_id
#metalearner = h2o.get_model(aml.metalearner()['name'])
pred_val = aml.predict(test_df[predictors])[0].as_data_frame()

pred_val
true_val = (test_df[target]).as_data_frame()

prediction_auc = roc_auc_score(pred_val, true_val)

prediction_auc
TestForPred = all_test.drop(['PassengerId', 'Survived'], axis = 1)
fin_pred = aml.predict(TestForPred[predictors])[0].as_data_frame()
PassengerId = all_test['PassengerId'].as_data_frame()
h2o_Sub = pd.DataFrame({'PassengerId': PassengerId['PassengerId'].tolist(), 'Survived':fin_pred['predict'].tolist() })

h2o_Sub.head()
h2o_Sub.to_csv("1_auto_h2o_50_Submission.csv", index = False)