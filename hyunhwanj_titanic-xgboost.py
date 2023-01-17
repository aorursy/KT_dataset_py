import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline

%config InlineBackend.figure_format = 'retina'
def read_data(file_name):

    dat = pd.read_csv(file_name)

    prefix_dict = {

        'Mlle' : 'Ms',

        'Lady' : 'Ms',

        'the Countess' : 'Dona',

        'Miss' : 'Ms',

        'Mme' : 'Mrs',

        'Capt' : 'Military',

        'Col' : 'Military',

        'Major' : 'Military',

        'Sir' : 'Military',

        'Master' : 'Military',

        'Don' : 'Mr',

        'Jonkheer' : 'Mr'

    }

    prefix = dat.Name.str.split(', ').str[1].str.split('.').str[0]

    prefix = [prefix_dict[p] if p in prefix_dict else p for p in prefix]

    dat['prefix'] = prefix

    new_dat = pd.get_dummies(dat, 

                             columns=['prefix', 'Pclass', 'Embarked'])

    new_dat = new_dat.drop(['PassengerId', 'Cabin', 'Ticket', 'Name'], axis=1)

    new_dat['Sex'] = [1 if x == 'female' else 0 for x in new_dat['Sex']]

    

    return new_dat
train = read_data("../input/train.csv")

train.head()
test = read_data("../input/test.csv")

test.head()
train.drop(['Survived'], axis=1).columns == test.columns
train_label = train.Survived.values

train = train.drop(['Survived'], axis=1)

train_label[1:10]
train.head()
test.head()
full_data = train.append(test)
full_data.isnull().astype(int).sum()
train = train.fillna(-1)

test = test.fillna(-1)
X_train = np.array(train)

Y_train = np.array(train_label)

X_test = np.array(test)
import xgboost as xgb
n_folds = 10
best_param = {}

best_err = 1.00



for eta in [0.01, 0.1]:

    for max_depth in [2, 3]:

        for lmd in range(3):

            params = {'eta': eta, 

                      'max_depth': max_depth, 

                      'subsample': 0.2, 

                      'objective': 'binary:hinge', 

                      'seed': 123, 

                      'eval_metric':'error',

                      'lambda' : lmd,

                      'nthread':-1}



            xg_train = xgb.DMatrix(X_train, label=Y_train)



            cv = xgb.cv(params, xg_train, 1000, nfold=n_folds)

            test_error = cv.iloc[-1,2]

            if test_error < best_err:

                print("update {} => {}".format(best_err, test_error))

                print(params)

                print(cv.iloc[-1,:])

                best_param = params

                best_err = test_error
best_model = xgb.train(best_param, xg_train, 1000)

xgb_train = xgb.DMatrix(X_train)

X_pred = np.array([int(round(x)) for x in best_model.predict(xgb_train)])

print("The accuracy of the dumb test (test == train):", sum(X_pred.T == Y_train) / len(Y_train))

xg_test = xgb.DMatrix(X_test)

Y_test = [int(round(x)) for x in best_model.predict(xg_test)]
print("best parameter for xgboost:", best_param)

print("The average accuracy:", 1-best_err)
!mkdir output
import datetime

now = datetime.datetime.now().strftime("%Y%m%d_%H%M")



PassengerId = pd.read_csv("../input/test.csv").PassengerId.values

pd.DataFrame( 

    {

        'PassengerId' : PassengerId, 

        'Survived' : Y_test

    }).to_csv('output/output_{}.txt'.format(now),index=False)
datetime.datetime.now().strftime("%Y-%m-%d %H:%M")