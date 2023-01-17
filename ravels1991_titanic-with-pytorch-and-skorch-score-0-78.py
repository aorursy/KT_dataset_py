!pip install skorch
import os

import random

import numpy as np

import pandas as pd



#PyTorch libraries

import torch

import torch.nn as nn

import torch.nn.functional as F



#Skorch

from skorch import NeuralNetBinaryClassifier



#Sklearn

from sklearn.model_selection import GridSearchCV



import warnings

warnings.filterwarnings('ignore')
def seed_everything(seed_value):

    random.seed(seed_value)

    np.random.seed(seed_value)

    torch.manual_seed(seed_value)

    os.environ['PYTHONHASHSEED'] = str(seed_value)

    

    if torch.cuda.is_available(): 

        torch.cuda.manual_seed(seed_value)

        torch.cuda.manual_seed_all(seed_value)

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = True



seed = 1234

seed_everything(seed)
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
train.head()
def title_extract(df):

    df['Title'] = df['Name'].str.extract('([a-zA-Z]+)\.')

    df['Title'] = df['Title'].apply(lambda x: 'Unknown' if x not in ['Miss','Master','Mr','Mrs', 'Dr', 'Rev'] else x)

    return df
train = title_extract(train)

test = title_extract(test)
def fill_nan_age(df):

    df['group_mean_age'] = round(df.groupby(['Sex', 'Title'])['Age'].transform('mean'))

    df['Age'].fillna(df['group_mean_age'], inplace=True)

    del df['group_mean_age']

    return df
train = fill_nan_age(train)

test = fill_nan_age(test)
def cat_age(df):

    age_bins = [0, 10, 18, 30, 55, 100]

    group_names = ['child', 'teenager', 'young adult', 'adult', 'elderly']

    df['age_cat'] = pd.cut(df['Age'], age_bins, right=False, labels=group_names)

    del df['Age']

    return df
train = cat_age(train)

test = cat_age(test)
def alone_family(df):

    df['Family'] = train['SibSp'] + train['Parch']

    df['Alone'] = pd.Series(np.where(df['Family'] == 0, 1, 0))

    return df
train = alone_family(train)

test = alone_family(test)
train.head()
def fare_cat(df):

    fare_labels = ['very cheap', 'cheap', 'moderate', 'exp', 'very exp']

    df['Fare_cat'] = pd.qcut(df['Fare'], [0, 0.1, 0.25, 0.5, 0.9, 1], precision=0, labels=fare_labels)

    del df['Fare']

    return df
train = fare_cat(train)

test = fare_cat(test)
train.head()
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)

test['Embarked'].fillna(test['Embarked'].mode()[0], inplace=True)

train.head()
def data_clean(drop_col, dummies, df):

    df = df.drop(drop_col, axis=1)

    df = pd.get_dummies(data=df, columns=dummies)

    return df
col_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']

col_dummies = ['Sex', 'Embarked', 'age_cat', 'Fare_cat', 'Pclass', 'Title']
train_clean = data_clean(col_drop, col_dummies, train)

train_clean.head()
test_clean = data_clean(col_drop, col_dummies, test)

test_clean.head()
X = train_clean.drop('Survived', axis=1)

y = train_clean['Survived']
X = np.array(X, dtype='float32')

y = np.array(y, dtype='float32')
class TitanicModel(nn.Module):

    def __init__(self, neurons=10, dropout=0.2):

        super(TitanicModel, self).__init__()

        

        self.dense0 = nn.Linear(X.shape[1], neurons)

        self.activation0 = nn.ReLU()

        self.dropout0 = nn.Dropout(dropout)

        self.dense1 = nn.Linear(neurons, neurons)

        self.activation1 = nn.ReLU()

        self.dropout1 = nn.Dropout(dropout)

        self.dense2 = nn.Linear(neurons, 1)

        self.output = nn.Sigmoid()

        

    def forward(self, x):

        x = self.dense0(x)

        x = self.activation0(x)

        x = self.dropout0(x)

        x = self.dense1(x)

        x = self.activation1(x)

        x = self.dropout1(x)

        x = self.dense2(x)

        x = self.output(x)

        return x
model = NeuralNetBinaryClassifier(module=TitanicModel,

                                          lr = 0.001, 

                                          optimizer__weight_decay = 0.001,

                                          verbose=0,

                                          train_split=False,

                                          device='cuda')
params = {'batch_size': [10],

          'max_epochs': [25],

          'optimizer': [torch.optim.Adam],

          'lr': [0.01, 0.001],

          'criterion': [nn.BCELoss],

          'module__neurons': [10, 15, 20, 25],

          'module__dropout': [0, 0.2]}

        
params
%%time

model_grid = GridSearchCV(estimator=model, param_grid=params,

                          scoring = 'accuracy', cv=3, verbose=0)



model_grid = model_grid.fit(X, y)
print(f'Accuracy: {model_grid.best_score_ * 100 :.2f}%')

print(model_grid.best_params_)
test_clean = np.array(test_clean, dtype='float32')
pred = model_grid.predict(test_clean)
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred})

submission.head()
submission.to_csv('submission.csv', index=False)