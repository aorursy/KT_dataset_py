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


import pandas as pd

import numpy as np

import seaborn as sns



import matplotlib.pyplot as plt



from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import ElasticNetCV



import xgboost as xgb

%matplotlib inline
df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv', low_memory=False)

def amount_cut(dataset, edge=2500):

    

    data = dataset.copy()

    data.loc[data['Amount'].values > edge, 'Amount'] = edge

    data['Amount'] = np.log(data['Amount'].values + 0.1)

    

    return data



def to_log(dataset, columns):

    

    data = dataset.copy()

    if type(columns) == list:

        for column in columns:

            data[column] = np.log(dataset[column].values + 0.1)

    elif type(columns) == str:

        data[columns] = np.log(dataset[columns].values + 0.1)

    else:

        return data

    

    return data



def normalize_amount(dataset, columns):

    

    data = dataset.copy()

    if type(columns) == list:

        for column in columns:

            data[column] = np.log(dataset[column].values + 0.1)

    elif type(columns) == str:

        data[columns] = np.log(dataset[columns].values + 0.1)

    else:

        print("Columns empty")

    

    return data



def hours(dataset, night_time):

    

    data = dataset.copy()

    hours_shrink = {key: key - 24 for key in range(25, 48)}

    data['Time_hours'] = pd.cut(df['Time'], 48, labels=list(range(48))).replace(hours_shrink)

    data['Night'] = 0

    data.loc[data['Time_hours'].isin(list(range(night_time[0], night_time[1] + 1))), 'Night'] = 1

    

    return data



def normalization(dataset, columns):

    

    data = dataset.copy()

    if type(columns) == list:

        columns = [x for x in data.columns if x not in columns]

        for column in columns:

            min_val, max_val = data[column].min(), data[column].max()

            data[column] = (data[column] - min_val) / (max_val - min_val)

    elif type(columns) == str:

        column = columns

        min_val, max_val = data[column].min(), data[column].max()

        data[column] = (data[column] - min_val) / (max_val - min_val)

    else:

        return data

    

    return data

def preprocessing(dataset, normalization_columns=None, except_normalization_columns=['Class', 'Night'], night_time=(1, 6)):

    

    data = normalize_amount(dataset, normalization_columns)

    data = hours(data, night_time)

    data = to_log(data, None)

    data = amount_cut(data)

    data = normalization(data, except_normalization_columns)

    return data









data = preprocessing(df, normalization_columns='Amount')
data
data.Class.value_counts()
plt.figure(figsize = (5,5))

plt.title('Credit Card Transactions features correlation plot (Pearson)')

corr = data.corr()

sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,cmap="Reds")

plt.show()
data.hist(column='Amount')
fraud =len(data[data['Class']==1])

notfraud = len(data[data['Class']==0])



# Data to plot

labels = 'Fraud','Not Fraud'

sizes = [fraud,notfraud]



# Plot

plt.figure(figsize=(7,6))

plt.pie(sizes, explode=(0.1, 0.1), labels=labels, colors=sns.color_palette("BuPu"),

autopct='%1.1f%%', shadow=True, startangle=0)

plt.title('Pie Chart Ratio of Transactions by their Class\n', fontsize=16)

sns.set_context("paper", font_scale=1.2)
## Define predictors and target values

predictors = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',\

       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',\

       'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',\

       'Amount','Time_hours','Night']

target = 'Class'



#TRAIN/VALIDATION/TEST SPLIT

#VALIDATION

VALID_SIZE = 0.3 # simple validation using train_test_split

TEST_SIZE = 0.3 # test size using_train_test_split

#CROSS-VALIDATION

NUMBER_KFOLDS = 5 #number of KFolds for cross-validation

RANDOM_STATE = 2018
train_df, test_df = train_test_split(data, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True )

train_df, valid_df = train_test_split(data, test_size=VALID_SIZE, random_state=RANDOM_STATE, shuffle=True )

import gc

from datetime import datetime 

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn import svm

import xgboost as xgb
# Prepare the train and valid datasets

dtrain = xgb.DMatrix(train_df[predictors], train_df[target].values)

dvalid = xgb.DMatrix(valid_df[predictors], valid_df[target].values)

dtest = xgb.DMatrix(test_df[predictors], test_df[target].values)



#What to monitor (in this case, **train** and **valid**)

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]



# Set xgboost parameters

params = {}

params['objective'] = 'binary:logistic'

params['eta'] = 0.04

params['silent'] = True

params['max_depth'] = 2

params['subsample'] = 0.8

params['colsample_bytree'] = 0.9

params['eval_metric'] = 'auc'

params['random_state'] = RANDOM_STATE
model = xgb.train(params, 

                dtrain)
preds = model.predict(dtest)

## without night shift

roc_auc_score(test_df[target].values, preds)