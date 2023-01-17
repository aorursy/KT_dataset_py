import numpy as np

import pandas as pd 

import matplotlib.pylab as plt

%matplotlib inline

import seaborn as sns

sns.set(color_codes=True)

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
#histogram and normal probability plot

from scipy.stats import norm

from scipy import stats



def norm_plot(dataset, features):

    n = len(features)

    fig = plt.figure(figsize=(10,5*n))



    for i in range(n):

        f = features[i]

        p1 = fig.add_subplot(n,2,(i+1)*2-1)



        X = dataset[f]

        X.dropna(inplace=True)



        sns.distplot(X, fit=norm);

        

        p1 = fig.add_subplot(n,2,(i+1)*2)

        res = stats.probplot(X, plot=plt)   

    

def log_norm_plot(dataset, features):

    logfeatures = []

    for feature in features:

        logfeature = "Log" + feature

        dataset[feature].replace(0,np.nan, inplace=True)

        dataset[feature].dropna(inplace=True)

        dataset[logfeature] = np.log(dataset[feature] + 1)

        logfeatures.append(logfeature)

    norm_plot(dataset, logfeatures)

log_norm_plot(train, ["SalePrice"])
continuous_features =  ['LotArea', 'BsmtFinSF1', 'BsmtFinSF2',

       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 

       'GrLivArea', 

       'GarageArea', 'WoodDeckSF', 'OpenPorchSF',

       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',

       ]
norm_plot(train, continuous_features)

log_norm_plot(train,continuous_features)
from scipy import stats

train["BoxLotArea"], _ = stats.boxcox(train["LotArea"])
norm_plot(train, ["BoxLotArea"])
norm_plot(train, ["LogLotArea"])
# Fill Na with mean



for feature in continuous_features:

    X = train["Log"+ feature]

    X = X.fillna(X.mean(), inplace=True)



cols = [

    'LogSalePrice', 

    'Log1stFlrSF',

    'LogGrLivArea',

    'LogLotArea', 

    'LogBsmtFinSF1',

    'LogGarageArea',

    'LogTotalBsmtSF',

    'Log2ndFlrSF',

#         'BsmtFinSF1', 'BsmtFinSF2',

#      'BsmtUnfSF', 

#     'TotalBsmtSF', 

#     '1stFlrSF', '2ndFlrSF', 

#     'Log2ndFlrSF',

#        'LogGarageArea', 'LogWoodDeckSF', 'LogOpenPorchSF',

#        'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',

]

sns.set()



sns.pairplot(train[cols])

plt.show();
continuous_features.append("SalePrice")
#correlation matrix

corrmat = train.loc[:,continuous_features].corr()

corrmat.sort_values(by=["SalePrice"], ascending=False, inplace=True, axis=1)

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, cmap=sns.color_palette("Blues"));
sns.boxplot(x="OverallCond", y="SalePrice", data=train)
sns.boxplot(x="YrSold", y="SalePrice", data=train)
sns.boxplot(x="OverallQual", y="SalePrice", data=train)