# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns; sns.set(style="ticks", color_codes=True)

from sklearn.model_selection import train_test_split

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor 

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import ElasticNet

from sklearn.svm import SVR

from sklearn import linear_model

from sklearn.feature_selection import RFE

from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import KFold

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv("../input/train.csv", names=['Store','Dept','Date','weeklySales','isHoliday'],sep=',', header=0)

features = pd.read_csv("../input/features.csv",sep=',', header=0,

                       names=['Store','Date','Temperature','Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4',

                              'MarkDown5','CPI','Unemployment','IsHoliday']).drop(columns=['IsHoliday'])

stores = pd.read_csv("../input/stores.csv", names=['Store','Type','Size'],sep=',', header=0)

dataset = dataset.merge(stores, how='left').merge(features, how='left')



# dataset["nextWeekHoliday"] = dataset["isHoliday"].shift(-1).fillna(False)

# dataset["next2WeekHoliday"] = dataset["isHoliday"].shift(-2).fillna(False)

dataset
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
sns.pairplot(dataset, vars=['weeklySales', 'Fuel_Price', 'Size', 'CPI', 'Dept', 'Temperature', 'Unemployment'])
sns.pairplot(dataset.fillna(0), vars=['weeklySales', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'])
for name, group in dataset.groupby(["Store", "Dept"]):

    plt.title(name)

    plt.scatter(range(len(group)), group["weeklySales"])

    plt.show()

    break
dataset = pd.get_dummies(dataset, columns=["Type"])

dataset[['MarkDown1','MarkDown2','MarkDown3','MarkDown4', 'MarkDown5']] = dataset[['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']].fillna(0)

dataset['Month'] = pd.to_datetime(dataset['Date']).dt.month

dataset = dataset.drop(columns=["Date", "CPI", "Fuel_Price", 'Unemployment', 'MarkDown4'])

dataset
def knn():

    knn = KNeighborsRegressor(n_neighbors=10)

    return knn



def extraTreesRegressor():

    clf = ExtraTreesRegressor(n_estimators=100,max_features='auto', verbose=1, n_jobs=1)

    return clf



def randomForestRegressor():

    clf = RandomForestRegressor(n_estimators=100,max_features='log2', verbose=1)

    return clf



def svm():

    clf = SVR(kernel='rbf', gamma='auto')

    return clf



def nn():

    clf = MLPRegressor(hidden_layer_sizes=(10,),  activation='relu', verbose=3)

    return clf



def lm():

    clf = linear_model.LinearRegression(normalize=True)

    return clf



def gbr():

    clf = GradientBoostingRegressor(loss="huber")

    return clf



def elasticNet():

    clf = linear_model.ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True)

    return clf



def predict_(m, test_x):

    return pd.Series(m.predict(test_x))



def model_():

#     return knn()

     return extraTreesRegressor()

#     return svm()

#     return nn()

#     return randomForestRegressor()

#     return lm()

#     return gbr()

#     return elasticNet()



def train_(train_x, train_y):

    m = model_()

    m.fit(train_x, train_y)

    return m



def train_and_predict(train_x, train_y, test_x):

    m = train_(train_x, train_y)

    return predict_(m, test_x), m

#
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
splited
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
dataset_test = pd.read_csv("../input/test.csv", names=['Store','Dept','Date','isHoliday'],sep=',', header=0)

features = pd.read_csv("../input/features.csv",sep=',', header=0,

                       names=['Store','Date','Temperature','Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4',

                              'MarkDown5','CPI','Unemployment','IsHoliday']).drop(columns=['IsHoliday'])

stores = pd.read_csv("../input/stores.csv", names=['Store','Type','Size'],sep=',', header=0)

dataset_test = dataset_test.merge(stores, how='left').merge(features, how='left')
dataset_test = pd.get_dummies(dataset_test, columns=["Type"])

dataset_test[['MarkDown1','MarkDown2','MarkDown3','MarkDown4', 'MarkDown5']] = dataset_test[['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']].fillna(0)

dataset_test = dataset_test.fillna(0)

column_date = dataset_test['Date']

dataset_test['Month'] = pd.to_datetime(dataset_test['Date']).dt.month

dataset_test = dataset_test.drop(columns=["Date", "CPI", "Fuel_Price", 'Unemployment', 'MarkDown3'])

dataset_test
predicted_test = best_model.predict(dataset_test)
dataset_test['weeklySales'] = predicted_test

dataset_test['Date'] = column_date

dataset_test['id'] = dataset_test['Store'].astype(str) + '_' +  dataset_test['Dept'].astype(str) + '_' +  dataset_test['Date'].astype(str)

dataset_test = dataset_test[['id', 'weeklySales']]

dataset_test = dataset_test.rename(columns={'id': 'Id', 'weeklySales': 'Weekly_Sales'})
dataset_test.to_csv('output.csv', index=False)