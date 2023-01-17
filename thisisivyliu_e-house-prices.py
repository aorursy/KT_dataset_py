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
trainPath = '/kaggle/input/house-prices-advanced-regression-techniques/train.csv'

testPath = '/kaggle/input/house-prices-advanced-regression-techniques/test.csv'

samplePath = '/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv'



traindf = pd.read_csv(trainPath, index_col='Id')

testdf = pd.read_csv(testPath)

sampledf = pd.read_csv(samplePath)



print('Load Successfully')
traindf.head()
testdf.head()
sampledf.head()
traindf.shape
testdf.shape
def checkMissingValues(df):

    missing_column = (df.isnull().sum())

    print(missing_column[missing_column > 0])



checkMissingValues(traindf)

print(' ')

checkMissingValues(testdf)
colList = traindf.columns

print(sorted(colList))
colList = testdf.columns

print(sorted(colList))
# Drop the rows with NaN in number columns

# Replace NaN with None in object columns



def cleanData(df):

#     print(df.shape)

    numCols = [col for col in df.columns if df[col].dtypes!='object']

    objectCols = [col for col in df.columns if col not in numCols]

    

    df[numCols] = df[numCols].replace(np.nan, 0)

    df = df.fillna('None')

#     print(df.shape)

    return df



traindf = cleanData(traindf)

testdf = cleanData(testdf)



print('Successfully clean the data')
# If the number of types in training set is different from the number of types in test set, then drop the column



def checkTypes(df):

    colList = [col for col in df.columns if df[col].dtypes=='object']

    #   print(colList)

    typeDict = {}

    

    for col in colList:

        tple = (len(df[col].unique()), df[col].unique())

        typeDict[col] = tple

    

#     for k, v in typeDict.items():

#         print(k, end=' ')

#         print(v)

    return typeDict



trainDict = checkTypes(traindf)

# print(' ')

testDict = checkTypes(testdf)



# print(len(trainDict.keys()))



needtodrop = []

for k in trainDict.keys():

    if trainDict[k][0] != testDict[k][0]:

        needtodrop.append(k)

    if k not in needtodrop and sorted(trainDict[k][1]) != sorted(testDict[k][1]):

        needtodrop.append(k)



# print(len(needtodrop))



drop_traindf = traindf.drop(needtodrop, axis=1)

drop_testdf = testdf.drop(needtodrop, axis=1)



print('Successfully drop the columns')
# newTrainDict = checkTypes(drop_traindf)

# newTestDict = checkTypes(drop_testdf)



# for k in newTrainDict.keys():

#     print(k)

#     print(newTrainDict[k][0], end=' ')

#     print(newTestDict[k][0])

#     print(newTrainDict[k][1], end=' ')

#     print(newTestDict[k][1])
from sklearn.preprocessing import LabelEncoder



encoder = LabelEncoder()

colList = [col for col in drop_traindf.columns if drop_traindf[col].dtypes=='object']



label_traindf = drop_traindf.copy()

label_testdf = drop_testdf.copy()



for col in colList:

    label_traindf[col] = encoder.fit_transform(drop_traindf[col])

    label_testdf[col] = encoder.transform(drop_testdf[col])



# label_traindf.head()
import matplotlib.pyplot as plt

import seaborn as sns



sns.kdeplot(data=label_traindf.SalePrice, shade=True)
numCols = [col for col in drop_traindf.columns if drop_traindf[col].dtypes!='object']

print(numCols)



sns.scatterplot(x=drop_traindf['GarageYrBlt'], y=drop_traindf.SalePrice)
drop_traindf = drop_traindf.drop(['MSSubClass', 'BsmtUnfSF', 'LowQualFinSF', 'BsmtFullBath', 'HalfBath', 'BedroomAbvGr', 'WoodDeckSF','EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MoSold', 'YrSold'], axis=1)

numCols = [col for col in drop_traindf.columns if drop_traindf[col].dtypes!='object']

print(numCols)
from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle



label_traindf = shuffle(label_traindf)



x = label_traindf.drop(['SalePrice'], axis=1)

y = label_traindf['SalePrice']



xTrain, xValid, yTrain, yValid = train_test_split(x, y, test_size = 0.3, random_state = 0)

xTest = label_testdf



print('Split Successfully')
from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LinearRegression



def score_model(xTrain, yTrain, xValid, yValid):

    model = LinearRegression()

    model.fit(xTrain, yTrain)

    pred = model.predict(xValid)

    print('mse : {}'.format(mean_absolute_error(yValid, pred)))
score_model(xTrain, yTrain, xValid, yValid)
model = LinearRegression()

model.fit(xTrain, yTrain)

print(xTest.loc[:, xTest.columns != 'Id'].shape)

preds = model.predict(xTest.loc[:, xTest.columns != 'Id'])



output = pd.DataFrame({'Id': xTest.Id, 'SalePrice': preds})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")