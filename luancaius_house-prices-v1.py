



import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from time import time

import seaborn as sns

import os

print(os.listdir("../input"))



inputTrain = pd.read_csv('../input/train.csv')

inputTest = pd.read_csv('../input/test.csv')
inputTrain.head(10)
inputTest.head(5)
def replacingMissingValues(dataset):

    for i in dataset.columns:

        if dataset[i].isnull().sum() > 0:

            dataset[i].fillna(np.mean(dataset[i]), inplace=True)

    return dataset 

def dropColumns(data):

    dropColumns = ['Id','YearBuilt','LotFrontage', 'MasVnrArea', 'GarageYrBlt']

    if 'SalePrice' in data.columns:

        dropColumns.append('SalePrice')

    print(dropColumns)

    features = data.drop(dropColumns, axis = 1)   

    return features

def getCategories(features):

    categories = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 

                  'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

                 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 

                  'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 

                  'SaleCondition' ]

    for category in categories: 

        features[category] = features[category].astype("category").cat.codes

    return features

def getTarget(data):

    return data['SalePrice']

def getFeatures(data):

    features = dropColumns(data)

    features  = getCategories(features)

    replacingMissingValues(features)

    print(features.shape)

    return features

def prepareData(data):

    target = getTarget(data)

    features = getFeatures(data)

    return features,target
from sklearn.model_selection import KFold

from sklearn.metrics import f1_score

   

def train_predict(clf, X_train, y_train, X_test, y_test):

    print("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))

    clf.fit(X_train, y_train)

    y_pred = clf.predict(features)

    return clf.score(features, y_pred)     

def split_data(features):

    n_splits = 4

    folds = KFold(n_splits=n_splits, random_state=42)

    return enumerate(folds.split(features))

def training(clf, features, target):

    score = 0

    start_fold = time()

    for fold, (train_idx, test_idx) in split_data(features):

        print("\nFold ", fold)

        X_train = features.iloc[train_idx]

        y_train = target.iloc[train_idx]

        X_test = features.iloc[test_idx]

        y_test = target.iloc[test_idx]            

        score = train_predict(clf, X_train, y_train, X_test, y_test)

        print(score)        

    end_fold = time()

    print('Training folds in {:.4f}'.format(end_fold - start_fold))

    return score 
def submissionFile(clf):

    print('Creating submission file')

    sub = pd.DataFrame(inputTest['Id'], columns=['Id','SalePrice'])

    features=getFeatures(inputTest)

    pred = clf.predict(features) 

    sub['SalePrice']=pred

    print('submit_{}.csv'.format(clf.__class__.__name__))

    sub.to_csv('submit_{}.csv'.format(clf.__class__.__name__), index=False) 
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoLars, ElasticNet, ElasticNetCV



clfs = [

    LinearRegression(),

    Ridge(alpha=.5),

    Lasso(alpha=0.1),

    LassoLars(alpha=.1),

    ElasticNet(random_state=0),

    ElasticNetCV(cv=3)

]



print("Starting")

start_init = time()



features, target = prepareData(inputTrain)

for clf in clfs:

    score=training(clf, features, target)

    #clf=tuning(clf,features, target, score)    

    submissionFile(clf)

end_init = time()

print("Finished in {:.4f} seconds".format(end_init - start_init))