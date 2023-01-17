# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import KFold

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!unzip /kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv.zip

!unzip /kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv.zip

!unzip /kaggle/input/walmart-recruiting-store-sales-forecasting/sampleSubmission.csv.zip

!unzip /kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip
dataset = pd.read_csv("/kaggle/working/train.csv", parse_dates=['Date'], names=['Store','Dept','Date','weeklySales','isHoliday'],sep=',', header=0)

features = pd.read_csv("/kaggle/working/features.csv", parse_dates=['Date'], sep=',', header=0,

                       names=['Store','Date','Temperature','Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4',

                              'MarkDown5','CPI','Unemployment','IsHoliday']).drop(columns=['IsHoliday'])

stores = pd.read_csv("../input/walmart-recruiting-store-sales-forecasting/stores.csv", names=['Store','Type','Size'],sep=',', header=0)

dataset = dataset.merge(stores, how='left').merge(features, how='left')

dataset
import seaborn as sns; sns.set(style="ticks", color_codes=True)

import matplotlib.pyplot as plt



def scatter(dataset, column):

    plt.figure()

    plt.scatter(dataset[column] , dataset['weeklySales'])

    plt.ylabel('weeklySales')

    plt.xlabel(column)



scatter(dataset, 'Fuel_Price')

scatter(dataset, 'Size')

scatter(dataset, 'CPI')

scatter(dataset, 'Type')

scatter(dataset, 'isHoliday')

scatter(dataset, 'Unemployment')

scatter(dataset, 'Temperature')

scatter(dataset, 'Store')

scatter(dataset, 'Dept')
fig = plt.figure(figsize=(18, 14))

corr = dataset.corr()

c = plt.pcolor(corr)

plt.yticks(np.arange(0.5, len(corr.index), 1), corr.index)

plt.xticks(np.arange(0.5, len(corr.columns), 1), corr.columns)

fig.colorbar(c)
dataset = pd.get_dummies(dataset, columns=["Type"])

dataset[['MarkDown1','MarkDown2','MarkDown3','MarkDown4', 'MarkDown5']] = dataset[['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']].fillna(0)

dataset['dow'] = pd.to_datetime(dataset['Date']).dt.dayofweek

dataset['day'] = pd.to_datetime(dataset['Date']).dt.day

dataset['month'] = pd.to_datetime(dataset['Date']).dt.month

dataset['year'] = pd.to_datetime(dataset['Date']).dt.year

dataset['week'] = pd.to_datetime(dataset['Date']).dt.week



from datetime import datetime as dt



dataset.loc[(dataset["Date"] >= dt(2010, 2, 5)) & (dataset["Date"] <= dt(2010, 2, 13)),"Special_day"] = 1

dataset.loc[(dataset["Date"] >= dt(2010, 7, 5)) & (dataset["Date"] <= dt(2010, 7, 14)),"Special_day"] = 1

dataset.loc[(dataset["Date"] >= dt(2010, 11, 9)) & (dataset["Date"] <= dt(2010, 11, 29)),"Special_day"] = 1

dataset.loc[(dataset["Date"] >= dt(2010, 12, 10)) & (dataset["Date"] <= dt(2010, 12, 31)),"Special_day"] = 1

dataset["Special_day"] = dataset["Special_day"].fillna(0)

dataset = dataset.drop(columns=["Date", "CPI", "Fuel_Price", 'Unemployment', 'MarkDown3'])

dataset
def extraTreesRegressor():

    clf = ExtraTreesRegressor(n_estimators=100,max_features='auto', verbose=1, n_jobs=1)

    return clf



def predict_(m, test_x):

    return pd.Series(m.predict(test_x))



def model_():

#     return knn()

    return extraTreesRegressor()

#     return svm()

#     return nn()

#     return randomForestRegressor()    



def train_(train_x, train_y):

    m = model_()

    m.fit(train_x, train_y)

    return m



def train_and_predict(train_x, train_y, test_x):

    m = train_(train_x, train_y)

    return predict_(m, test_x), m
def calculate_error(test_y, predicted, weights):

    return mean_absolute_error(test_y, predicted, sample_weight=weights)
kf = KFold(n_splits=5)

splited = []

# dataset2 = dataset.copy()

for name, group in dataset.groupby(["Store", "Dept"]):

    group = group.reset_index(drop=True)

    trains_x = []

    trains_y = []

    tests_x = []

    tests_y = []

    if group.shape[0] <= 5:

        f = np.array(range(5))

        np.random.shuffle(f)

        group['fold'] = f[:group.shape[0]]

        continue

    fold = 0

    for train_index, test_index in kf.split(group):

        group.loc[test_index, 'fold'] = fold

        fold += 1

    splited.append(group)



splited = pd.concat(splited).reset_index(drop=True)
best_model = None

error_cv = 0

best_error = np.iinfo(np.int32).max

for fold in range(5):

    dataset_train = splited.loc[splited['fold'] != fold]

    dataset_test = splited.loc[splited['fold'] == fold]

    train_y = dataset_train['weeklySales']

    train_x = dataset_train.drop(columns=['weeklySales', 'fold'])

    test_y = dataset_test['weeklySales']

    test_x = dataset_test.drop(columns=['weeklySales', 'fold'])

    print(dataset_train.shape, dataset_test.shape)

    predicted, model = train_and_predict(train_x, train_y, test_x)

    weights = test_x['isHoliday'].replace(True, 5).replace(False, 1)

    error = calculate_error(test_y, predicted, weights)

    error_cv += error

    print(fold, error)

    if error < best_error:

        print('Find best model')

        best_error = error

        best_model = model

error_cv /= 5
error_cv
best_error
dataset_test = pd.read_csv("/kaggle/working/test.csv",  parse_dates=['Date'], names=['Store','Dept','Date','isHoliday'],sep=',', header=0)

features = pd.read_csv("/kaggle/working/features.csv", parse_dates=['Date'], sep=',', header=0,

                       names=['Store','Date','Temperature','Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4',

                              'MarkDown5','CPI','Unemployment','IsHoliday']).drop(columns=['IsHoliday'])

stores = pd.read_csv("../input/walmart-recruiting-store-sales-forecasting/stores.csv", names=['Store','Type','Size'],sep=',', header=0)

dataset_test = dataset_test.merge(stores, how='left').merge(features, how='left')
dataset_test = pd.get_dummies(dataset_test, columns=["Type"])

dataset_test[['MarkDown1','MarkDown2','MarkDown3','MarkDown4', 'MarkDown5']] = dataset_test[['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']].fillna(0)

dataset_test = dataset_test.fillna(0)

column_date = dataset_test['Date']

dataset_test['dow'] = pd.to_datetime(dataset_test['Date']).dt.dayofweek

dataset_test['day'] = pd.to_datetime(dataset_test['Date']).dt.day

dataset_test['month'] = pd.to_datetime(dataset_test['Date']).dt.month

dataset_test['year'] = pd.to_datetime(dataset_test['Date']).dt.year

dataset_test['week'] = pd.to_datetime(dataset_test['Date']).dt.week



dataset_test.loc[(dataset_test["Date"] >= dt(2010, 2, 5)) & (dataset_test["Date"] <= dt(2010, 2, 13)),"Special_day"] = 1

dataset_test.loc[(dataset_test["Date"] >= dt(2010, 7, 5)) & (dataset_test["Date"] <= dt(2010, 7, 14)),"Special_day"] = 1

dataset_test.loc[(dataset_test["Date"] >= dt(2010, 11, 9)) & (dataset_test["Date"] <= dt(2010, 11, 29)),"Special_day"] = 1

dataset_test.loc[(dataset_test["Date"] >= dt(2010, 12, 10)) & (dataset_test["Date"] <= dt(2010, 12, 31)),"Special_day"] = 1

dataset_test["Special_day"] = dataset_test["Special_day"].fillna(0)

dataset_test = dataset_test.drop(columns=["Date", "CPI", "Fuel_Price", 'Unemployment', 'MarkDown3'])

dataset_test
predicted_test = best_model.predict(dataset_test)
dataset_test['weeklySales'] = predicted_test

dataset_test['Date'] = column_date

dataset_test['id'] = dataset_test['Store'].astype(str) + '_' +  dataset_test['Dept'].astype(str) + '_' +  dataset_test['Date'].astype(str)

dataset_test = dataset_test[['id', 'weeklySales']]

dataset_test = dataset_test.rename(columns={'id': 'Id', 'weeklySales': 'Weekly_Sales'})
dataset_test.to_csv('submission_walmart.csv', index=False)