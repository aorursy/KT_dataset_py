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
data = pd.read_csv('../input/titanic/train.csv')

data.head()
print(data.shape)

data.describe()
data.groupby('Pclass')['Survived'].mean()
data.groupby('Sex')['Survived'].mean()
data['Age'].plot.hist(bins = 50)
baby_data = data.loc[data.Age <=3, :]

baby_data.head()
baby_data.describe()
data.groupby('Pclass')['Fare'].mean()
data.groupby('Pclass')['Fare'].std()
data.corr()
import seaborn as sns



sns.heatmap(data.corr())
import matplotlib.pyplot as plt



data.plot(x='Age', y='Fare', style='o')
# data.hist(by='Pclass', column='Fare', bins = 30, sharex=True)

data.hist(by='Pclass', column = 'Fare', bins=20)
data.loc[data.Fare>400, :]
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')



print(train.shape, test.shape)

print(train.columns, '\n', test.columns)

print(pd.DataFrame({'Train': train.isna().sum(), 'Test':test.isna().sum()}).dropna() )
# An overview of statistics:

print(train.describe())

print(test.describe())
# Categorical Comparison:

def consistency(train, test, col):

    train_weight = train[col].value_counts(normalize=True)

    test_weight = test[col].value_counts(normalize=True)

    compare_table = pd.DataFrame({'Train': train_weight, 'Test': test_weight}).fillna(0)

    compare_table.index.name = col

    active_weight = (compare_table.Train - compare_table.Test).abs().sum()/2    

#     print(compare_table.Train - compare_table.Test)

    return active_weight, compare_table



col = 'Sex'

col_list = ['Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked']

for col in col_list:

    active, table = consistency(train, test, col)

#     print(f'Active weight of {col}: {active:.4f}; \n {table}')

    print(f'Active weight of {col}: {active:.4f}')

# Numerical Comparisom:
# Adversarial Validation:

train.drop('Survived', axis=1, inplace=True)

target_col = 'Target'

train[target_col]=0

test[target_col]=1



combined_df = pd.concat([train, test], axis=0, sort=False)

print(combined_df.shape)

combined_df.head()
# Drop: PassengerId, Name, Ticket. (too many values)

# One-hot: Sex, Cabin, Embarked



one_hot_col = ['Sex', 'Cabin', 'Embarked']

drop_col =['PassengerId', 'Name', 'Ticket']



for col in combined_df.columns:

    print(f'Different values in {col}: {combined_df[col].nunique()}')



from sklearn.preprocessing import OneHotEncoder



encoder = OneHotEncoder()



combined_df[one_hot_col] = combined_df[one_hot_col].fillna('Unknown')

# print(combined_df.isna().sum())

temp = encoder.fit_transform(combined_df[one_hot_col])

temp

encoder.get_feature_names()

for col in one_hot_col:

    col_dummies =  pd.get_dummies(combined_df[col])

    col_dummies.columns = col_dummies.columns.map(lambda s: s+'_'+col)

    combined_df = pd.concat([combined_df.drop(col, axis=1), col_dummies], axis=1, sort=False)



combined_df.drop(drop_col, axis=1, inplace=True)

combined_df.shape
import xgboost as xgb



X = combined_df.drop(target_col, axis=1).values

y = combined_df[target_col].values

dtrain = xgb.DMatrix(X, label =y)



param = {'max_depth':2, 'eta':0.2, 'silent':1, 'objective':'binary:logistic'}

num_round = 3

model = xgb.cv(param, dtrain, num_round, nfold = 10, metrics = 'auc', verbose_eval=True)

# From the model, adversarial validation shows that the training data and testing data are very consistent. We can use usual k-fold validation as validation strategy.
combined_df.head()
def preprocessing(train_data,test_data=None, one_hot_col=None, drop_col=None, min_max_col=None, normal_col=None, inplace=False):

    df = train_data.copy()

#     if test_data:

#         test_df = test_data.copy()

#         test_df = test_df.drop(drop_col, axis=1)

        

    if drop_col:

        df = df.drop(drop_col, axis=1)



    if one_hot_col:

        df[one_hot_col] = df[one_hot_col].fillna('Unknown')

        for col in one_hot_col:

            col_dummies =  pd.get_dummies(df[col])

            col_dummies.columns = col_dummies.columns.map(lambda s: s+'_'+col)

            df = pd.concat([df.drop(col, axis=1), col_dummies], axis=1, sort=False)

    if min_max_col:

        for col in min_max_col:

            col_max = max(df[col])

            col_min = min(df[col])

            if col_max-col_min<1e8:

                df[col]=0

            else:

                df[col] = (df[col]-col_min)/(col_max-col_min)

    if normal_col:

        for col in normal_col:

            col_mean = df[col].mean()

            col_std = df[col].std()

            if col_std<1e8:

                df[col]=0

            else:

                df[col] = (df[col] - col_mean)/(col_std)

                

    return df



train.columns
from math import log

# ((train.Age-train.Age.mean())/train.Age.std()).plot.hist()

train_select = train.Fare[train.Fare<300]

train_select.plot.hist(bins = 100)
from sklearn.linear_model import LogisticRegressionCV

from sklearn.model_selection import train_test_split



# train = pd.read_csv('../input/titanic/train.csv')

# test = pd.read_csv('../input/titanic/test.csv')



# # From EDA:

# one_hot_col = ['Sex', 'Cabin', 'Embarked']

# drop_col =['PassengerId', 'Name', 'Ticket']

# min_max_col = ['Age', 'Fare']



# X, y = train.drop('Survived', axis=1), train['Survived']

# X = preprocessing(X, one_hot_col = one_hot_col, drop_col=drop_col, min_max_col = min_max_col)

# X['Age'].fillna(X.Age.mean(), inplace=True)



# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 21)



# nn_model = LogisticRegressionCV(cv = 5, verbose=1, max_iter=1e4)

# nn_model.fit(X_train, y_train)

# nn_model.score(X_test, y_test)
from sklearn.linear_model import LogisticRegressionCV

from sklearn.model_selection import train_test_split



# train = pd.read_csv('../input/titanic/train.csv')

# test = pd.read_csv('../input/titanic/test.csv')



# # From EDA:

# one_hot_col = ['Sex', 'Cabin', 'Embarked']

# drop_col =['PassengerId', 'Name', 'Ticket']

# min_max_col = ['Age', 'Fare']



# X_combined = pd.concat([train.drop('Survived', axis=1), test], axis=0)

# y_train = train['Survived']

# X_combined = preprocessing(X_combined, one_hot_col = one_hot_col, drop_col=drop_col, min_max_col = min_max_col)

# missing_col = ['Age', 'Fare']

# for col in missing_col:

#     X_combined[col].fillna(X_combined[col].mean(), inplace=True)



# X_train = X_combined.iloc[:train.shape[0], :]

# X_test =  X_combined.iloc[train.shape[0]:, :]



# nn_model = LogisticRegressionCV(cv = 5, verbose=1, max_iter=1e4)

# nn_model.fit(X_train, y_train)

# y_predict = nn_model.predict(X_test)

# submission = pd.DataFrame({

#         "PassengerId": test["PassengerId"],

#         "Survived": y_predict

#     })

# submission.to_csv('nn_submission.csv', index=False)



# print(submission.shape)

# submission.head()
import xgboost as xgb

from sklearn.model_selection import train_test_split



train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')



# From EDA:

one_hot_col = ['Sex', 'Cabin', 'Embarked']

drop_col =['PassengerId', 'Name', 'Ticket']



# Preprocessing for training:

X_combined = pd.concat([train.drop('Survived', axis=1), test], axis=0)

y_train = train['Survived']

X_combined = preprocessing(X_combined, one_hot_col = one_hot_col, drop_col=drop_col)

X_train = X_combined.iloc[:train.shape[0], :]

X_test =  X_combined.iloc[train.shape[0]:, :]



dtrain = xgb.DMatrix(X_train, label =y_train.values)

param = {'max_depth':5, 'eta':0.1, 'silent':1, 'objective':'binary:hinge'}

num_round = 100

model = xgb.train(param, dtrain, num_round)

# model = xgb.cv(param, dtrain, num_round, nfold = 5, stratified=True,

#                metrics = ['auc', 'error'], verbose_eval=True)

dtest = xgb.DMatrix(X_test)

y_predict = model.predict(dtest).astype(int)



submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": y_predict

    })

submission.to_csv('xgb_submission.csv', index=False)



print(submission.shape)

submission.head(10)
xgb.plot_importance(model, max_num_features=10)
y_predict.astype(int)