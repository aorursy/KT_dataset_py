import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn import linear_model

from sklearn.svm import SVR



import xgboost



import matplotlib.pyplot as plt



%config InlineBackend.figure_format = 'png' #set 'png' here when working on notebook



%matplotlib inline
#Define all of the required data frames upfront

train = pd.read_csv("../input/train.csv")

output = pd.read_csv("../input/sample_submission.csv")

test = pd.read_csv("../input/test.csv")



all_data = pd.concat((train, test))
#Convert the numerical-categorical data into strings

MoSold = all_data['MoSold']

all_data['MoSold'] = all_data['MoSold'].astype(str)

all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)

all_data['MSZoning'] = all_data['MSZoning'].astype(str)
#Make Categorical Data Columns

all_data = pd.get_dummies(all_data)

all_data['MoSold'] = MoSold
all_data.shape
#Impute data



nan_count = train.isnull().sum()

nan_count.sort_values(inplace = True)

print(nan_count)
all_data["GarageYrBlt"].fillna(0)

all_data["LotFrontage"].fillna(0)

all_data = all_data.fillna(all_data.mean())
all_data["LotArea"].describe()
#Adding New Features

#Physical Features

all_data.loc[:,'TotalFullBaths'] = all_data["BsmtFullBath"] + all_data["FullBath"]

all_data.loc[:,'TotalHalfBaths'] = all_data["BsmtHalfBath"] + all_data["HalfBath"]

all_data.loc[:,'TotalBaths'] = all_data["TotalFullBaths"] + all_data["TotalHalfBaths"]



#Time based features

all_data.loc[:,'YearsSince2000AtSale'] = all_data["YrSold"] - 2000

all_data.loc[:,'InflationFactor'] = (all_data["YrSold"] - all_data["YrSold"].min()) * 1.02

all_data.loc[:,'AgeAtSale'] = all_data["YrSold"] - all_data["YearBuilt"] 

all_data.loc[:,'TimeSinceRemod'] = all_data["YrSold"] - all_data["YearRemodAdd"]



all_data.loc[:,'SoldInWinter'] = np.where(((all_data['MoSold'] >= 11) | (all_data['MoSold'] <= 2)), 1, 0)

all_data.loc[:,'SoldInSpring'] = np.where(((all_data['MoSold'] > 2) & (all_data['MoSold'] < 6)), 1, 0)

all_data.loc[:,'SoldInSummer'] = np.where(((all_data['MoSold'] >= 6) & (all_data['MoSold'] < 9)), 1, 0)

all_data.loc[:,'SoldInFall'] = np.where(((all_data['MoSold'] >= 9) & (all_data['MoSold'] < 11)), 1, 0)



#Break Rankings into subsets, idea being, that marginal utility isn't linear

all_data.loc[:,'OverallQual_poor'] = np.where(all_data['OverallQual'] <= 3, 1, 0)

all_data.loc[:,'OverallQual_great'] = np.where(all_data['OverallQual'] >= 7, 1, 0)

all_data.loc[:,'OverallQual_average'] = np.where(((all_data['OverallQual'] < 7) & (all_data['OverallQual'] > 3)), 1, 0)



all_data.loc[:,'OverallCond_poor'] = np.where(all_data['OverallCond'] <= 3, 1, 0)

all_data.loc[:,'OverallCond_great'] = np.where(all_data['OverallCond'] >= 7, 1, 0)

all_data.loc[:,'OverallCond_average'] = np.where(((all_data['OverallCond'] < 7) & (all_data['OverallCond'] > 3)), 1, 0)



min = all_data['LotArea'].quantile(q = 0)

max = all_data['LotArea'].quantile(q = .2)

all_data.loc[:,'VerySmallLot'] = np.where(((all_data['LotArea'] < max) & (all_data['LotArea'] > min)), 1, 0)



min = all_data['LotArea'].quantile(q = .2)

max = all_data['LotArea'].quantile(q = .4)

all_data.loc[:,'SmallLot'] = np.where(((all_data['LotArea'] < max) & (all_data['LotArea'] > min)), 1, 0)



min = all_data['LotArea'].quantile(q = .4)

max = all_data['LotArea'].quantile(q = .6)

all_data.loc[:,'RegularLot'] = np.where(((all_data['LotArea'] < max) & (all_data['LotArea'] > min)), 1, 0)



min = all_data['LotArea'].quantile(q = .6)

max = all_data['LotArea'].quantile(q = .8)

all_data.loc[:,'LargeLot'] = np.where(((all_data['LotArea'] < max) & (all_data['LotArea'] > min)), 1, 0)



min = all_data['LotArea'].quantile(q = .8)

max = all_data['LotArea'].quantile(q = 1)

all_data.loc[:,'VeryLargeLot'] = np.where(((all_data['LotArea'] < max) & (all_data['LotArea'] > min)), 1, 0)
#Test how new features are correlated:



all_corr = all_data[all_data.columns[1:-1]].apply(lambda x: x.corr(all_data['SalePrice']))

main_corr = all_corr.sort_values().head(15)

main_corr = main_corr.append(all_corr.sort_values().tail(15))

main_corr.plot(kind="barh")

plt.title("Strongest Correlations")

plt.rcParams['figure.figsize'] = (15.0, 10.0)
originalPrices = all_data.loc[:,'SalePrice']

#New range should be between -1 and 1

all_data = (1--1) * (all_data - all_data.min()) / (all_data.max() - all_data.min()) + -1

all_data['SalePrice'] = originalPrices

all_data.describe()
prices = pd.DataFrame({"original":train['SalePrice'], "ln(prices)":np.log1p(train['SalePrice'])})

prices.hist(bins = 50)
all_data['SalePrice'] = np.log1p(all_data['SalePrice'])
#creating matrices for Machine Learning:



train = all_data[:train.shape[0]]

test = all_data[train.shape[0]:]



trainX = train.drop("SalePrice", axis = 1)

testX = test.drop("SalePrice", axis = 1)

trainY = train.SalePrice
#Make a regression

reg = linear_model.LassoCV()

reg.fit(trainX, trainY)

#Use said regression to find extreme outliers

predictions = reg.predict(trainX)

residuals = predictions-trainY

residuals.plot.box()



original_prices = train.SalePrice

train.loc[:,'residuals'] = residuals



print(len(train))



extreme_max = residuals.mean() + 4.5 * residuals.std()

extreme_min = residuals.mean() - 4.5 * residuals.std()



train = train[residuals < extreme_max]

train = train[residuals > extreme_min]

train = train.drop('residuals', axis = 1)

print(len(train))
#Recreate Matricies

trainY = train.SalePrice

trainX = train.drop("SalePrice", 1)



len(trainX)
#Rebuild model

reg = linear_model.LassoCV()

reg.fit(trainX, trainY)
coef = pd.Series(reg.coef_, index = trainX.columns)

major_coef = pd.concat([coef.sort_values().head(10),

                     coef.sort_values().tail(10)])



major_coef.plot(kind = "barh")

plt.title("Lasso Coefficients")
#See how the engineered features matched up



#Lots

print(coef['VeryLargeLot'])

print(coef['LargeLot'])

print(coef['RegularLot'])

print(coef['SmallLot'])

print(coef['VerySmallLot'])

print(coef['InflationFactor'])



print(coef['SoldInWinter'])

print(coef['SoldInSpring'])

print(coef['SoldInSummer'])

print(coef['SoldInFall'])
#rfc = RandomForestClassifier()

#rfc.fit(trainX, trainY)



#xgb = xgboost.XGBRegressor(n_estimators=300, max_depth=2, learning_rate=0.1)

#xgb.fit(trainX, trainY)
answersLASSO = reg.predict(testX)

#answersRFC = rfc.predict(testX)

#answersXGB = xgb.predict(testX)
#comparison = pd.DataFrame({"LASSO":answersLASSO, "RFC":answersRFC, "XGB":answersXGB})

#comparison.hist(bins = 50)
answers = answersLASSO

#answers = answersLASSO * .9 + answersRFC * .01 + answersXGB * .09
output['SalePrice'] = np.expm1(answers)



#Handle negative values in the output (Probably caused by lack of data)

output.loc[output['SalePrice']<0,'SalePrice'] = 0 

print(output['SalePrice'].describe())

print(np.expm1(trainY).describe())
plt.rcParams['figure.figsize'] = (13.0, 6.0)



prices = pd.DataFrame({"TestY":output["SalePrice"], "trainY":np.expm1(trainY)})

prices.hist(bins = 50)
plt.rcParams['figure.figsize'] = (13.0, 6.0)



prices = pd.DataFrame({"predictedY":output["SalePrice"], "trainY":np.expm1(trainY)})

prices.hist(bins = 75)
output.to_csv('Predictions.csv', index=False)