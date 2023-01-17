## import library

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from IPython.display import clear_output

from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf

from sklearn import preprocessing



print("Tensorflow Version:- ",tf.__version__)







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/titanic/train.csv') # training data

test = pd.read_csv('/kaggle/input/titanic/test.csv') # testing data

submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv') # submission data
print("Training Data: {} & Test Data: {}".format(train.shape, test.shape))
train.head()
train.describe()
train.isnull().sum()
test.isnull().sum()
target = train['Survived']
## Histogram of Age

train.Age.hist(bins=20)
## Sex ratio

train.Sex.value_counts().plot(kind='barh')
## Class ratio

train['Pclass'].value_counts().plot(kind='barh')
## Survival ratio

train.groupby('Sex').Survived.mean().plot(kind='barh').set_xlabel('% Survive')
train.groupby('Parch').Survived.mean().plot(kind='barh').set_xlabel('% Survive')
train.groupby('SibSp').Survived.mean().plot(kind='barh').set_xlabel('% Survive')
train.groupby('Pclass').Survived.mean().plot(kind='barh').set_xlabel('% Survive')
train.groupby('Embarked').Survived.mean().plot(kind='barh').set_xlabel('% Survive')
train=train[list(test)]

total_data=pd.concat((train, test))  # all training + test data set

print(train.shape, test.shape, total_data.shape)

total_data.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
## Fill nan with 0 else with 1 value.

total_data['Cabin'] = [0 if str(x) == 'nan' else 1 for x in total_data['Cabin']]
## Fill nan with 0 else with 1 value.

total_data['Sex'] = [0 if str(x) == 'male' else 1 for x in total_data['Sex']]
## drop Embarked rows with 2 missing values

emb_dummies = pd.get_dummies(total_data['Embarked'], drop_first=True, prefix='Embarked')

total_data = pd.concat([total_data, emb_dummies], axis=1)

total_data.drop('Embarked', axis=1, inplace=True)
total_data.head(10)
!pip install impyute
total_data.isna().sum()
from impyute.imputation.cs import mice



imputed = mice(total_data.values)

mice_ages = imputed[:, 2]



total_data['Age']=mice_ages
total_data.isna().sum()
Xtrain=total_data[:len(train)]

Xtest=total_data[len(train):]
Xtest['Fare'] = Xtest['Fare'].fillna(0)
Xtest.isna().sum()       
import optuna

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

kf=StratifiedKFold(n_splits=5)



def test(trial):

    C=trial.suggest_loguniform('C', 10e-10, 10)

    model=LogisticRegression(C=C, class_weight='balanced',max_iter=10000, solver='lbfgs', n_jobs=-1)

    score=-cross_val_score(model, Xtrain, target, cv=kf, scoring='roc_auc').mean()

    return score

t=optuna.create_study()





t.optimize(test, n_trials=50)



print(t.best_params)



print(-t.best_value)

#params=t.best_params

from sklearn.linear_model import LogisticRegression

## Model Preration



model=LogisticRegression(C=0.2539894188596627, class_weight='balanced',max_iter=10000, solver='lbfgs', n_jobs=-1)

model.fit(Xtrain, target)

#predictions=model.predict_proba(Xtest)[:,1] # probability of getting 1 replace '1' with 0 to get probability of 0

predictions=model.predict(Xtest) # to get target column as either 0 or 1 value no probability

submission['Survived']=predictions

submission.to_csv('submit.csv', index=False)

submission.head()
