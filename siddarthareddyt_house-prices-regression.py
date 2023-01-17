# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import matplotlib.pyplot as plt

from scipy.stats import skew



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

holdout = pd.read_csv("../input/test.csv")



#sale price starts at 34k till 755k with a mean of 180k. 

train['SalePrice'].describe()

correlation_matrix = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(correlation_matrix, vmax=.8, square=True)
#data exploration to find SalePrice relation to some important numeric variables



numeric = [feature for feature in train.columns if train.dtypes[feature] != 'object']

numeric.remove('Id')

numreicMostCorr = ['LotFrontage', 'OverallQual', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'TotalBsmtSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'GarageArea', 'GarageCars']



for feature in numreicMostCorr:

    featureDF = pd.concat([train['SalePrice'], train[feature]], axis=1)

    featureDF.plot.scatter(x=feature, y='SalePrice', ylim=(0,800000))
#High correlation between some features themselves. So, we can choose anyone of the pair



pairs = [('GarageArea', 'GarageCars'), 

         ('YearBuilt', 'YearRemodAdd'), 

         ('TotalBsmtSF', 'TotRmsAbvGrd'),

         ('GrLivArea', 'FullBath'),

         ('TotalBsmtSF', '1stFlrSF'),

         ('GrLivArea', '2ndFlrSF')

        ]



for pair in pairs:

    featureDF = pd.concat([train[pair[0]], train[pair[1]]], axis=1)

    featureDF.plot.scatter(x=pair[0], y=pair[1])
categorical = [feature for feature in train.columns if train.dtypes[feature] == 'object']

for category in categorical:

    data = pd.concat([train[category], train['SalePrice']], axis=1)

    data[category] = data[category].astype('category')

    if data[category].isnull().any():

        data[category] = data[category].cat.add_categories(['MISSING'])

        data[category] = data[category].fillna('MISSING')

    cat_data = pd.concat([data['SalePrice'], data[category]], axis=1)

    f, ax = plt.subplots(figsize=(8, 6))

    fig = sns.boxplot(x=category, y="SalePrice", data=cat_data)

    fig.axis(ymin=0, ymax=800000)

    plt.show()
def removeFromList(sourceList, filterList):

    filteredList = list(filter(lambda x: x not in filterList, sourceList))

    return filteredList



def process_missing(df, numreicCorr, categoricalC):

    numeric_h_correlated = ['GarageCars', '1stFlrSF', '2ndFlrSF', 'YearRemodAdd', 'FullBath']

    categorical_h_correlated = ['Alley', 

                                               'LotShape', 

                                               'LandSlope', 

                                               'BldgType', 

                                               'Exterior1st',

                                               'Exterior2nd',

                                               'ExterCond',

                                               'BsmtCond',

                                               'BsmtExposure',

                                               'BsmtFinType1',

                                               'BsmtFinType2',

                                               'HeatingQC',

                                               'GarageFinish',

                                               'GarageType',

                                               'GarageCond',

                                               'Fence'

                                              ]

    #missing data

    numeric_missing = df[numreicCorr].isnull().sum().sort_values(ascending=False)

    categorical_missing = df[categoricalC].isnull().sum().sort_values(ascending=False)

    

    #delete missing data that's more than 30% percent

    numeric_to_delete = (numeric_missing[numeric_missing > 438]).index

    categorical_to_delete = (categorical_missing[categorical_missing > 438]).index



    numreicCorr = removeFromList(numreicCorr, numeric_to_delete)

    categoricalC = removeFromList(categoricalC, categorical_to_delete)



    #delete highly correlated numeric features

    numreicCorr = removeFromList(numreicCorr, numeric_h_correlated)

    categoricalC = removeFromList(categoricalC, categorical_h_correlated)

    

    return (df, numreicCorr, categoricalC)



train, numericMostCorr_train, categorical_train = process_missing(train, numreicMostCorr, categorical)

holdout, numericMostCorr_holdout, categorical_holdout = process_missing(holdout, numreicMostCorr, categorical)



all_columns = numericMostCorr_train + categorical_train + ['SalePrice']

train = train[all_columns]



holdoutids = holdout.Id

holdout = holdout[numericMostCorr_holdout + categorical_holdout]



print(train.columns.values)



def process_na(df):

    #handle numeric n/a

    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())

    #df[numreicMostCorr].isnull().sum().sort_values(ascending=False)



    #handle categorical n/a

    df['BsmtQual'] = df['BsmtQual'].fillna("Missing")

    df['GarageQual'] = df['GarageQual'].fillna("Missing")

    df['MasVnrType'] = df['MasVnrType'].fillna("Missing")

    df = df.drop(df.loc[df['Electrical'].isnull()].index)

    return df



    #train[categorical].isnull().sum().sort_values(ascending=False)



train = process_na(train)

holdout = process_na(holdout)



train.head(5)





holdout[numericMostCorr_holdout].isnull().sum().sort_values(ascending=False)

holdout['GarageArea'] = holdout['GarageArea'].fillna(holdout['GarageArea'].mean())

holdout['TotalBsmtSF'] = holdout['TotalBsmtSF'].fillna(holdout['TotalBsmtSF'].mean())
#skewed saleprice

saleprice = pd.DataFrame({"saleprice_skewed" :train['SalePrice']})

saleprice.hist()
#handle numeric skewness

from scipy.stats import skew

def process_skewness(df, numericCorr, isholdout=False):

    if isholdout:

        skewed_cols = numericCorr

    else:

        skewed_cols = numericCorr + ['SalePrice']

    skewed = df[skewed_cols].apply(lambda x: skew(x.dropna()))

    skewed = skewed[skewed > 0.75]

    skewed = skewed.index

    df[skewed] = np.log1p(df[skewed])

    return (df,skewed_cols)



train, skewed_cols_train = process_skewness(train, numericMostCorr_train)

holdout, skewed_cols_holdout = process_skewness(holdout, numericMostCorr_holdout, True)



for numer in skewed_cols_train:

    numerFeature = pd.DataFrame({"unskewed_"+numer:train[numer]})

    numerFeature.hist()
#create dummies for categorical features

def get_dummies(df):

    return pd.get_dummies(df)



all_data = pd.concat((train.loc[:,'LotFrontage':'SaleCondition'],

                      holdout.loc[:,'LotFrontage':'SaleCondition']))

all_data = get_dummies(all_data)
from sklearn.ensemble import RandomForestRegressor

from sklearn.cross_validation import train_test_split

from sklearn.linear_model import LassoLarsCV



y = train.SalePrice

X = all_data[:train.shape[0]]

holdout_X = all_data[train.shape[0]:]



regressor = RandomForestRegressor(n_estimators = 150, random_state = 0)



regressor.fit(X, y)



preds = np.expm1(regressor.predict(holdout_X))



solution = pd.DataFrame({"id":holdoutids, "SalePrice":preds})

solution.to_csv("HousePrice.csv", index = False)