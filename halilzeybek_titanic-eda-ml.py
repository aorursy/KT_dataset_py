# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

gender_submission_df = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
train_df.info()
train_df.head()
test_df.head()
train_df[train_df['Age'].isnull()]
train_median = train_df.groupby(['SibSp','Parch']).median()

train_age_median = train_median.reset_index()[['SibSp','Parch','Age']]

train_age_isnull = train_df[train_df['Age'].isnull()]
# Find the missing values of age and fill the values according to Sibsp and Parch columns.

index_nan_age = list(train_df["Age"][train_df["Age"].isnull()].index)

for each in index_nan_age:

    x = train_df['Age'][(train_df['SibSp'].iloc[each]==train_df['SibSp']) & (train_df['Parch'].iloc[each]==train_df['Parch'])].median()

    y = train_df['Age'].median()

    if np.isnan(x):

        train_df['Age'].iloc[each]=y

    else:

        train_df['Age'].iloc[each]=x

#Check if there are still nan values in age column

train_df[train_df['Age'].isnull()]

    
# Find the missing values of age and fill the values according to Sibsp and Parch columns.

index_nan_age = list(test_df["Age"][test_df["Age"].isnull()].index)

for each in index_nan_age:

    x = test_df['Age'][(test_df['SibSp'].iloc[each]==test_df['SibSp']) & (test_df['Parch'].iloc[each]==test_df['Parch'])].median()

    y = test_df['Age'].median()

    if np.isnan(x):

        test_df['Age'].iloc[each]=y

    else:

        test_df['Age'].iloc[each]=x

#Check if there are still nan values in age column

test_df[test_df['Age'].isnull()]
train_df.groupby(['Embarked']).median()
#Embarked column is most related with Fare column. As our rows in dataframe whose Embarked values are nan has high Fare values 

#their values of Embarked should be 'C'

train_df["Embarked"] = train_df["Embarked"].fillna("C")

train_df[train_df['Embarked'].isnull()]
test_df[test_df['Embarked'].isnull()]
train_df['cabin_first_letter']=np.nan

not_null_cabin_indexes =list(train_df[train_df['Cabin'].notnull()].index)

for each in not_null_cabin_indexes:

    train_df['cabin_first_letter'].iloc[each]=train_df['Cabin'].iloc[each][0]

train_df.groupby(['cabin_first_letter']).median()



test_df['cabin_first_letter']=np.nan

not_null_cabin_indexes =list(test_df[test_df['Cabin'].notnull()].index)

for each in not_null_cabin_indexes:

    test_df['cabin_first_letter'].iloc[each]=test_df['Cabin'].iloc[each][0]

test_df.groupby(['cabin_first_letter']).median()
train_df1 = train_df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
train_df1['Sex']=[0 if each=='male' else 1 for each in train_df1['Sex']]
train_df1['Sex']=train_df1['Sex'].astype("category")
train_df1['Pclass']=train_df1['Pclass'].astype("category")
train_df1['Age']=(train_df1['Age'] - train_df1['Age'].min()) / (train_df1['Age'].max() - train_df1['Age'].min())
train_df1['Fare']=(train_df1['Fare'] - train_df1['Fare'].min()) / (train_df1['Fare'].max() - train_df1['Fare'].min())
train_df1 = pd.get_dummies(train_df1,drop_first=True)
train_df1
test_df1 = test_df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

test_df1['Sex']=[0 if each=='male' else 1 for each in test_df1['Sex']]

test_df1['Sex']=test_df1['Sex'].astype("category")

test_df1['Pclass']=test_df1['Pclass'].astype("category")

test_df1['Age']=(test_df1['Age'] - test_df1['Age'].min()) / (test_df1['Age'].max() - test_df1['Age'].min())

test_df1['Fare']=(test_df1['Fare'] - test_df1['Fare'].min()) / (test_df1['Fare'].max() - test_df1['Fare'].min())

test_df1 = pd.get_dummies(test_df1,drop_first=True)

test_df1
test_df1["Fare"] = test_df1["Fare"].fillna(np.mean(test_df1["Fare"]))
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=0.1)

lr.fit(train_df1[['Age','Sex_1']],train_df['Survived'])
lr.get_params
y_pred = lr.predict(test_df1[['Age','Sex_1']])
from sklearn.metrics import confusion_matrix

confusion_matrix(y_pred=y_pred,y_true=gender_submission_df['Survived'].values)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier()

params_rf={'max_depth':[2,3,4,5],'max_features':['auto','sqrt','log2'],'n_estimators':[0,100,150,200]}

grid_dt = GridSearchCV(estimator=rf, param_grid=params_rf, scoring='accuracy', cv=10, n_jobs=-1)
grid_dt.fit(train_df1,train_df['Survived'])
from sklearn.metrics import confusion_matrix

classifier = grid_dt.best_estimator_

y_pred_grid = classifier.predict(test_df1)

confusion_matrix(y_pred=y_pred_grid,y_true=gender_submission_df['Survived'].values)
classifier.get_params
'''test_df['cabin_first_letter_T']=0

'''
'''from sklearn.svm import SVC

svc = SVC(C=1)

svc.fit(df1,train_df['Survived'])

svc.score(test_df,gender_submission_df['Survived'])

'''
#test_survived = pd.Series(svc.predict(test_df), name = "Survived").astype(int)

#results = pd.concat([gender_submission_df['PassengerId'], test_survived],axis = 1)

#results.to_csv("titanic.csv", index = False)
'''from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

rf.fit(df1,train_df['Survived'])

rf.score(test_df,gender_submission_df['Survived'])

'''
'''test_survived = pd.Series(rf.predict(test_df), name = "Survived").astype(int)

results = pd.concat([gender_submission_df['PassengerId'], test_survived],axis = 1)

results.to_csv("titanic.csv", index = False)

'''
results_3 = pd.concat([gender_submission_df['PassengerId'], pd.Series(y_pred,name="Survived").astype(int)],axis = 1)
results_3.to_csv("titanic3.csv", index = False)


# Import xgboost

import xgboost as xgb





# Instantiate the XGBClassifier: xg_cl

xg_cl = xgb.XGBClassifier(objective='binary:logistic')



# Fit the classifier to the training set

xg_cl.fit(train_df1[['Age','Sex_1']],train_df['Survived'])



# Predict the labels of the test set: preds

preds = xg_cl.predict(test_df1[['Age','Sex_1']])



confusion_matrix(y_pred=preds,y_true=gender_submission_df['Survived'].values)
train_df1.columns