!pip install catboost==0.17.5
import catboost

catboost.__version__
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from catboost import CatBoostRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from copy import deepcopy

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load Training & Test data

data = pd.read_csv('../input/tcdml1920-income-ind/tcd ml 2019-20 income prediction training (with labels).csv', index_col='Instance')

trainData = deepcopy(data.drop(['Income in EUR','Wears Glasses', 'Hair Color'], axis=1))

data1 = pd.read_csv('../input/tcdml1920-income-ind/tcd ml 2019-20 income prediction test (without labels).csv', index_col='Instance')

testData = deepcopy(data1.drop(['Income','Wears Glasses', 'Hair Color'], axis=1))
trainData.info()
testData.info()
pd.options.display.float_format = '{:.2f}'.format

data['Income in EUR'].describe()
# Row Duplication check Train & Test

duplicateRowsTrain = trainData[trainData.duplicated()]

print('train' +duplicateRowsTrain)



duplicateRowsTest = testData[testData.duplicated()]

print('test' + duplicateRowsTest)
# Null check Train

trainData.isnull().sum().sort_values()
# Replace train missing values with 'unknown', 'others' and dropping NaN

trainData1 = deepcopy(trainData)

trainData1['Gender']            = trainData1['Gender'].replace({np.nan: 'unknown'})

trainData1['University Degree'] = trainData1['University Degree'].replace({np.nan: 'unknown'})

trainData1['Profession']        = trainData1['Profession'].replace({np.nan: 'others'})

trainData1 = trainData1.dropna()
# Null check Test

testData.isnull().sum().sort_values()
# Replace test missing values with 'unknown', 'others' and replace NaN with mean

testData1 = deepcopy(testData)

testData1['Gender']            = testData1['Gender'].replace({np.nan: 'unknown'})

testData1['University Degree'] = testData1['University Degree'].replace({np.nan: 'unknown'})

testData1['Profession']        = testData1['Profession'].replace({np.nan: 'others'})

testData1['Year of Record']    = testData1['Year of Record'].replace({np.nan: testData1['Year of Record'].mean()})

testData1['Age']               = testData1['Age'].replace({np.nan: testData1['Age'].mean()})

testData1 = testData1.dropna()
# unknown and zero mapping to downsize categories



trainData1['University Degree'].where(trainData1['University Degree'] == 0, 'unknown')

trainData1['Gender'].where(trainData1['Gender'] == 0, 'unknown')



testData1['University Degree'].where(testData1['University Degree'] == 0, 'unknown')

testData1['Gender'].where(testData1['Gender'] == 0, 'unknown')
trainData1.info()
testData1.info()
normColumn = ['Year of Record', 'Age', 'Size of City', 'Body Height [cm]']

for col in normColumn:

    trainData1[col] = (trainData1[col] - trainData1[col].mean()) / trainData1[col].std()

    testData1[col] = (testData1[col] - testData1[col].mean()) / testData1[col].std()
#Creating a training set for modeling and validation set to check model performance

X = trainData1

y = data.loc[trainData1.index, 'Income in EUR']





X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.9, random_state=18313172)

categorical_features_indices = np.array([1, 3, 5, 6])



model=CatBoostRegressor(

    iterations=5000,

    depth=7,

    learning_rate=0.1,

    loss_function='RMSE',

    task_type='GPU',

    border_count=32,

    verbose=500,

    l2_leaf_reg = 150,

    random_seed=18313172

)

model.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_validation, y_validation),plot=True)
print(model.get_feature_importance(prettified=True))
submission = pd.DataFrame()

submission['Instance'] = testData1.index

submission['Income'] = model.predict(testData1)

submission.to_csv('Submission.csv', index = False)