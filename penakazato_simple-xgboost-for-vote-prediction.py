# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/basic_income_dataset_dalia.csv")

data.head()
#rename the columns to be less wordy

data.rename(columns = {'question_bbi_2016wave4_basicincome_awareness':'awareness',

            'question_bbi_2016wave4_basicincome_vote':'vote',

            'question_bbi_2016wave4_basicincome_effect':'effect',

            'question_bbi_2016wave4_basicincome_argumentsfor':'arg_for',

            'question_bbi_2016wave4_basicincome_argumentsagainst':'arg_against'},

           inplace = True)
data['vote'].value_counts().plot(kind = 'bar')
def for_against(row):

    if row == 'I would not vote': return(0)

    elif row == 'I would probably vote for it': return(1)

    elif row == 'I would vote against it': return(1)

    elif row == 'I would vote for it': return(1)

    else: return(0)



data['yes_no'] = data['vote'].apply(for_against)

data['yes_no'].value_counts().plot(kind = 'bar')
OBJ_COLS = list(data.select_dtypes(include = ['object']).columns)

OBJ_COLS.remove('uuid')

OBJ_COLS.remove('vote')

#columns have too many unique values

OBJ_COLS.remove('arg_for') 

OBJ_COLS.remove('arg_against')

OBJ_COLS.remove('weight')



data_all = pd.get_dummies(data, columns = OBJ_COLS)

data_all.columns
PREDS = list(data_all.columns)

PREDS.remove('uuid')

PREDS.remove('vote')

PREDS.remove('yes_no')

PREDS.remove('arg_for')

PREDS.remove('arg_against')

PREDS.remove('weight')



y = data_all['yes_no'].values
from sklearn.model_selection import train_test_split



X = data_all[PREDS]



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,random_state = 123)

import xgboost as xgb



params = {}

params["objective"] = "binary:logistic"

params['eval_metric'] = 'logloss'

params["eta"] = 0.02

params["subsample"] = 0.7

params["min_child_weight"] = 1

params["colsample_bytree"] = 0.7

params["max_depth"] = 4

params["silent"] = 1

params["seed"] = 1632



d_train = xgb.DMatrix(X_train, label=y_train)

d_valid = xgb.DMatrix(X_test, label=y_test)



watchlist = [(d_train, 'train'), (d_valid, 'valid')]



bst = xgb.train(params, d_train, 300, watchlist, early_stopping_rounds=50, verbose_eval=10)



xgb.plot_importance(bst)
from sklearn.metrics import accuracy_score

test_preds = bst.predict(d_valid)



test_preds[test_preds > 0.5] = 1

test_preds[test_preds <= 0.5] = 0



print('Classifier is {} % accurate!'.format(round(accuracy_score(y_test,test_preds) * 100),1))