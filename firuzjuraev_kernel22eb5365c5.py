import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
def side_by_side(*objs, **kwds):

    from pandas.io.formats.printing import adjoin

    space = kwds.get('space', 4)

    reprs = [repr(obj).split('\n') for obj in objs]

    print (adjoin(space, *reprs))

    print()

    return
house_file_path = '../input/house-prices/train.csv'

house_data = pd.read_csv(house_file_path) 

house_data.columns
house_data.shape
house_data.drop(["Alley", "PoolQC", "Fence", "MiscFeature", "FireplaceQu"], axis=1, inplace = True) 
side_by_side(house_data.isnull().sum(), house_data.count())
house_data.shape
corr = house_data.corr()

sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns)
def correlationHeatmap(aDF, Features, title):

    fig,ax = plt.subplots(figsize=(20,8))

    fig.suptitle("Heatmap Plot - "+title, fontsize=30)

    corrcoef = aDF[Features].corr()

    mask = np.array(corrcoef)

    mask[np.tril_indices_from(mask)] = False

# sns.heatmap(corrcoef, mask=mask, vmax=.8, square=True, annot=True, ax=ax)

    sns.heatmap(corrcoef, mask=mask, vmax=.8, annot=True, ax=ax)

    plt.show();
fig,ax = plt.subplots(figsize=(20,8))

fig.suptitle("Heatmap Plot", fontsize=30)

corrcoef = house_data[['SalePrice', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 

                         'GarageArea', 'GarageQual', 'GarageCond', 'GarageType']].corr()

mask = np.array(corrcoef)

mask[np.tril_indices_from(mask)] = False

sns.heatmap(corrcoef, mask=mask, vmax=.8, annot=True, ax=ax)

plt.show();
fig,ax = plt.subplots(figsize=(20,8))

fig.suptitle("Heatmap Plot", fontsize=30)

corrcoef = house_data[['SalePrice', 'LotFrontage', 'LotArea', 'YearBuilt', 'FullBath']].corr()

mask = np.array(corrcoef)

mask[np.tril_indices_from(mask)] = False

sns.heatmap(corrcoef, mask=mask, vmax=.8, annot=True, ax=ax)

plt.show();
fig,ax = plt.subplots(figsize=(20,8))

fig.suptitle("Heatmap Plot", fontsize=30)

corrcoef = house_data[['SalePrice', 'OverallQual', '2ndFlrSF', '1stFlrSF', 'GrLivArea']].corr()

mask = np.array(corrcoef)

mask[np.tril_indices_from(mask)] = False

sns.heatmap(corrcoef, mask=mask, vmax=.8, annot=True, ax=ax)

plt.show();
fig,ax = plt.subplots(figsize=(20,8))

fig.suptitle("Heatmap Plot", fontsize=30)

corrcoef = house_data[['SalePrice', 'Fireplaces', 'GarageYrBlt', 'HalfBath', 'BedroomAbvGr',]].corr()

mask = np.array(corrcoef)

mask[np.tril_indices_from(mask)] = False

sns.heatmap(corrcoef, mask=mask, vmax=.8, annot=True, ax=ax)

plt.show();
house_data.drop(["MSSubClass", "OverallCond", "PoolArea", "SaleCondition", 

                 "GarageCond", "GarageQual", "GarageType"], axis=1, inplace = True) 
house_data.drop(["MoSold", "YrSold", "SaleType", "ScreenPorch", 

                 "MiscVal", "3SsnPorch", "KitchenAbvGr", "Exterior2nd"], axis=1, inplace = True) 
house_data.drop(["EnclosedPorch", "BsmtHalfBath"], axis=1, inplace = True) 
house_data.drop(["GarageFinish", "BsmtQual", 'BsmtFinSF2' ], axis=1, inplace = True) 
corr = house_data.corr()

sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns)
house_data.shape
Features = ["OverallQual", "GrLivArea", "GarageCars", "GarageArea", "1stFlrSF", "FullBath", "TotRmsAbvGrd",

            "YearBuilt", "YearRemodAdd", "GarageYrBlt","MasVnrArea", "Fireplaces", "LotFrontage","WoodDeckSF",

           "OpenPorchSF", "2ndFlrSF", "HalfBath", "LotArea", "BedroomAbvGr", "SalePrice"]

house_data2 = house_data[Features]
fig,ax = plt.subplots(figsize=(20,8))

fig.suptitle("Correlation Scores", fontsize=30)

corrcoef = house_data2[Features].corr()

mask = np.array(corrcoef)

mask[np.tril_indices_from(mask)] = False

sns.heatmap(corrcoef, mask=mask, vmax=.8, annot=True, ax=ax)

plt.show();
house_data2 = house_data.dropna(axis=0)
house_data2.shape
house_data2.describe()
house_features = ["OverallQual", "GrLivArea", "GarageCars", "GarageArea", "1stFlrSF", "FullBath", "TotRmsAbvGrd",

            "YearBuilt", "YearRemodAdd", "GarageYrBlt","MasVnrArea", "Fireplaces", "LotFrontage","WoodDeckSF",

           "OpenPorchSF", "2ndFlrSF", "HalfBath", "LotArea", "BedroomAbvGr"]

X = house_data2[house_features]

Y = house_data2.SalePrice
house_model = DecisionTreeRegressor(random_state=1)

house_model.fit(X, Y) # Fit model 
house_test_data2 = pd.read_csv("../input/house-prices-test/test.csv") 

Features = ["Id", "OverallQual", "GrLivArea", "GarageCars", "GarageArea", "1stFlrSF", "FullBath", "TotRmsAbvGrd",

            "YearBuilt", "YearRemodAdd", "GarageYrBlt","MasVnrArea", "Fireplaces", "LotFrontage","WoodDeckSF",

           "OpenPorchSF", "2ndFlrSF", "HalfBath", "LotArea", "BedroomAbvGr"]

house_test_data2 = house_test_data2[Features]



house_test_data2.fillna(0, inplace=True) # this time, I fill with zero when there is NaN 

house_test_data2.shape
Features = ["OverallQual", "GrLivArea", "GarageCars", "GarageArea", "1stFlrSF", "FullBath", "TotRmsAbvGrd",

            "YearBuilt", "YearRemodAdd", "GarageYrBlt","MasVnrArea", "Fireplaces", "LotFrontage","WoodDeckSF",

           "OpenPorchSF", "2ndFlrSF", "HalfBath", "LotArea", "BedroomAbvGr"]

X = house_test_data2[Features]

predicted_prices = house_model.predict(X)

len(predicted_prices)
predicted_prices
submission_data = pd.DataFrame()

submission_data["ID"] = house_test_data2["Id"]

submission_data["SalePrice"] = 0.0
submission_data.shape
for i in range(0, len(predicted_prices)):

    submission_data["SalePrice"].iloc[i] = predicted_prices[i]
submission_data.to_csv('submission.csv')