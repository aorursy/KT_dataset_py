# Let's start with loading several helpful packages 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_predict, train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix
train_df = pd.read_csv('../input/train.csv')



# keep only the numerical columns

data = train_df['Survived Pclass Age SibSp Parch Fare'.split()].fillna(0)

X, y = data['Pclass Age SibSp Parch Fare'.split()], data['Survived']

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.4, random_state=1)



lr_model = LogisticRegression()

lr_model.fit(Xtrain, ytrain)

predictions = lr_model.predict(Xtest)



print('Accuracy: {:.2%}'.format(accuracy_score(predictions, ytest)))

print('Confusion matrix:')

print(confusion_matrix(predictions, ytest, labels=[0, 1]))
test_df = pd.read_csv('../input/test.csv')



# keep only the numerical columns and the PassengerId (we need it to create the submission file)

final = test_df['PassengerId Age Pclass SibSp Parch Fare'.split()].fillna(0)

pid, Xfinal = final['PassengerId'], final[['Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]



lr_model.fit(X, y)



final_predictions = lr_model.predict(Xfinal)



if not os.path.isdir('./output'):

    os.mkdir('output')

results = pd.DataFrame({'PassengerId': pid, 'Survived': final_predictions})

print(results.shape)

results.to_csv('./output/submission1a.csv', index=False)
train_df = pd.read_csv('../input/train.csv')





data = train_df['Survived Age Pclass Sex SibSp Parch Fare'.split()].fillna(0)

# encode sex as binary variable

data.Sex = data.Sex.apply(lambda x: 0 if x == 'male' else 1)



X, y = data['Age Pclass Sex SibSp Parch Fare'.split()], data['Survived']

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.4, random_state=1)



lr_model = LogisticRegression()

lr_model.fit(Xtrain, ytrain)

predictions = lr_model.predict(Xtest)



print('Accuracy: {:.2%}'.format(accuracy_score(predictions, ytest)))

print('Confusion matrix:')

print(confusion_matrix(predictions, ytest, labels=[0, 1]))
def transform_data(df, labeled_data=True):

    df.Age = df.Age.fillna(-0.5)

    df.Fare = df.Fare.fillna(0)

    df.Cabin = df.Cabin.fillna('N')



    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)

    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

    df.Age = pd.cut(df.Age, bins, labels=group_names)



    df.Sex = df.Sex.apply(lambda x: 0 if x == 'male' else 1)

    df.Cabin = df.Cabin.apply(lambda x: x[0])



    cat_vars=['Cabin', 'Age', 'Pclass'] # 'Fare']

    for v in cat_vars:

        dummies = pd.get_dummies(df[v], prefix=v)

        df = df.join(dummies)

    df.drop(cat_vars, axis=1, inplace=True)

    

    return df
train_df = pd.read_csv('../input/train.csv')



train_df.drop(['PassengerId', 'Name', 'Ticket', 'Embarked'], axis=1, inplace=True)

train_df = transform_data(train_df)



y = train_df['Survived']

X = train_df.drop('Survived', axis=1)



Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.4, random_state=1)



lr_model = LogisticRegression()

lr_model.fit(Xtrain, ytrain)

predictions = lr_model.predict(Xtest)



print('Accuracy: {:.2%}'.format(accuracy_score(predictions, ytest)))

print('Confusion matrix:')

print(confusion_matrix(predictions, ytest, labels=[0, 1]))
train_df = pd.read_csv('../input/train.csv')

final_df = pd.read_csv('../input/test.csv')

test_df = pd.read_csv('../input/test.csv')



y = train_df['Survived']

train_df.drop('Survived', axis=1, inplace=True)



df = pd.concat([train_df, test_df]).reset_index()

df.drop(['Name', 'Ticket', 'Embarked'], axis=1, inplace=True)



df = transform_data(df)

train_df = df[:len(train_df)]

final_df = df[len(train_df):]



pid = final_df['PassengerId']

Xfinal =  final_df.drop('PassengerId', axis=1)



X = train_df.drop('PassengerId', axis=1)



lr_model.fit(X, y)



final_predictions = lr_model.predict(Xfinal)



if not os.path.isdir('./output'):

    os.mkdir('output')

results = pd.DataFrame({'PassengerId': pid, 'Survived': final_predictions})

results.to_csv('./output/submission1b.csv', index=False)
import lightgbm as lgb
train_df = pd.read_csv('../input/train.csv')

final_df = pd.read_csv('../input/test.csv')

test_df = pd.read_csv('../input/test.csv')



y = train_df['Survived']

train_df.drop('Survived', axis=1, inplace=True)



df = pd.concat([train_df, test_df]).reset_index()

df.drop(['Name', 'Ticket', 'Embarked'], axis=1, inplace=True)



df = transform_data(df)

train_df = df[:len(train_df)]

final_df = df[len(train_df):]



pid = final_df['PassengerId']

Xfinal =  final_df.drop('PassengerId', axis=1)



X = train_df.drop('PassengerId', axis=1)

lgb_model = lgb.LGBMClassifier(

    n_estimators=100,

    objective='binary',

    n_jobs=4,

)



Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.4, random_state=1)



lgb_model.fit(Xtrain, ytrain)

predictions = lgb_model.predict(Xtest)



print('Accuracy: {:.2%}'.format(accuracy_score(predictions, ytest)))

print('Confusion matrix:')

print(confusion_matrix(predictions, ytest, labels=[0, 1]))
lgb_model = lgb.LGBMClassifier(

    n_estimators=300,

    max_depth=6,

    learning_rate=0.03,

    num_leaves=25,

    objective='binary',

    n_jobs=4,

)



Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.4, random_state=1)



predictions = cross_val_predict(lgb_model, Xtrain, ytrain, cv=10)



print('Accuracy: {:.2%}'.format(accuracy_score(predictions, ytrain)))

print('Confusion matrix:')

print(confusion_matrix(predictions, ytrain, labels=[0, 1]))
lgb_model.fit(Xtrain, ytrain)

predictions = lgb_model.predict(Xtest)



print('Accuracy: {:.2%}'.format(accuracy_score(predictions, ytest)))

print('Confusion matrix:')

print(confusion_matrix(predictions, ytest, labels=[0, 1]))
lgb_model.fit(X, y)



final_predictions = lgb_model.predict(Xfinal)



if not os.path.isdir('./output'):

    os.mkdir('output')

results = pd.DataFrame({'PassengerId': pid, 'Survived': final_predictions})

print(results.head(20))

results.to_csv('./output/submission2a.csv', index=False)
X.head()