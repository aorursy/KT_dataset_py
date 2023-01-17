# Handle table-like data and matrices

import numpy as np

import pandas as pd

import math



# Modelling Algorithms

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression, LassoLarsCV,Ridge

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor



# Modelling Helpers

from sklearn.preprocessing import Imputer , Normalizer , scale, OneHotEncoder

from sklearn.cross_validation import train_test_split , StratifiedKFold

from sklearn.feature_selection import RFECV

from sklearn.preprocessing import LabelEncoder,OneHotEncoder



# Visualisation

import matplotlib 

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns



# Configure visualisations

%matplotlib inline
traindf = pd.read_csv("../input/train.csv")

testdf = pd.read_csv("../input/test.csv")
# create a single dataframe of both the training and testing data

wholedf = pd.concat([traindf,testdf])
wholedf.info()
wholedf.head(5)
sns.lmplot(x="GrLivArea", y="SalePrice", data=wholedf);

plt.title("Linear Regression of Above Grade Square Feet and Sale Price")

plt.ylim(0,)

plt.show()
sns.lmplot(x="1stFlrSF", y="SalePrice", data=wholedf);

plt.title("Linear Regression of First Floor Square Feet and Sale Price")

plt.ylim(0,)

plt.show()
traincorr = traindf.corr()['SalePrice']

# convert series to dataframe so it can be sorted

traincorr = pd.DataFrame(traincorr)

# correct column label from SalePrice to correlation

traincorr.columns = ["Correlation"]

# sort correlation

traincorr2 = traincorr.sort_values(by=['Correlation'], ascending=False)

traincorr2.head(15)
corr = wholedf.corr()

sns.heatmap(corr)

plt.show()
countmissing = wholedf.isnull().sum().sort_values(ascending=False)

percentmissing = (wholedf.isnull().sum()/wholedf.isnull().count()).sort_values(ascending=False)

wholena = pd.concat([countmissing,percentmissing], axis=1)

wholena.head(36)
#wholedf[["Utilities", "Id"]].groupby(['Utilities'], as_index=False).count()

wholedf['Utilities'] = wholedf['Utilities'].fillna("AllPub")



# wholedf[["Electrical", "Id"]].groupby(['Electrical'], as_index=False).count()

wholedf['Electrical'] = wholedf['Electrical'].fillna("SBrkr")



# wholedf[["Exterior1st", "Id"]].groupby(['Exterior1st'], as_index=False).count()

wholedf['Exterior1st'] = wholedf['Exterior1st'].fillna("VinylSd")



#wholedf[["Exterior2nd", "Id"]].groupby(['Exterior2nd'], as_index=False).count()

wholedf['Exterior2nd'] = wholedf['Exterior2nd'].fillna("VinylSd")
# Missing interger values replace with the median in order to return an integer

wholedf['BsmtFullBath']= wholedf.BsmtFullBath.fillna(wholedf.BsmtFullBath.median())

wholedf['BsmtHalfBath']= wholedf.BsmtHalfBath.fillna(wholedf.BsmtHalfBath.median())

wholedf['GarageCars']= wholedf.GarageCars.fillna(wholedf.GarageCars.median())



# Missing float values were replaced with the mean for accuracy 

wholedf['BsmtUnfSF']= wholedf.BsmtUnfSF.fillna(wholedf.BsmtUnfSF.mean())

wholedf['BsmtFinSF2']= wholedf.BsmtFinSF2.fillna(wholedf.BsmtFinSF2.mean())

wholedf['BsmtFinSF1']= wholedf.BsmtFinSF1.fillna(wholedf.BsmtFinSF1.mean())

wholedf['GarageArea']= wholedf.GarageArea.fillna(wholedf.GarageArea.mean())

wholedf['MasVnrArea']= wholedf.MasVnrArea.fillna(wholedf.MasVnrArea.mean())
wholedf.GarageYrBlt.fillna(wholedf.YearBuilt, inplace=True)

wholedf.TotalBsmtSF.fillna(wholedf['1stFlrSF'], inplace=True)



sns.lmplot(x="TotalBsmtSF", y="1stFlrSF", data=wholedf)

plt.title("Linear Regression of Basement SF and 1rst Floor SQ ")

plt.xlim(0,)

plt.ylim(0,)

plt.show()
lot = wholedf[['LotArea','LotConfig','LotFrontage','LotShape']]

lot = pd.get_dummies(lot)

lot.corr()['LotFrontage']
lot["LotAreaUnSq"] = np.sqrt(lot['LotArea'])
sns.regplot(x="LotAreaUnSq", y="LotFrontage", data=lot);

plt.xlim(0,)

plt.ylim(0,)

plt.title("Lot Area to Lot Frontage")

plt.show()
# Remove all lotfrontage is missing values

lot = lot[lot['LotFrontage'].notnull()]

# See the not null values of LotFrontage

lot.describe()['LotFrontage']
wholedf['LotFrontage']= wholedf.LotFrontage.fillna(np.sqrt(wholedf.LotArea))

wholedf['LotFrontage']= wholedf['LotFrontage'].astype(int)
# Distribution of values after replacement of missing frontage

sns.kdeplot(wholedf['LotFrontage']);

sns.kdeplot(lot['LotFrontage']);

sns.kdeplot(lot['LotAreaUnSq']);

plt.title("Distribution of Lot Frontage")

plt.show()
countmissing = wholedf.isnull().sum().sort_values(ascending=False)

percentmissing = (wholedf.isnull().sum()/wholedf.isnull().count()).sort_values(ascending=False)

wholena = pd.concat([countmissing,percentmissing], axis=1)

wholena.head(3)
Livingtotalsq = wholedf['TotalBsmtSF'] + wholedf['1stFlrSF'] + wholedf['2ndFlrSF'] + wholedf['GarageArea'] + wholedf['WoodDeckSF'] + wholedf['OpenPorchSF']

wholedf['LivingTotalSF'] = Livingtotalsq



# Total Living Area divided by LotArea

wholedf['PercentSQtoLot'] = wholedf['LivingTotalSF'] / wholedf['LotArea']



# Total count of all bathrooms including full and half through the entire building

wholedf['TotalBaths'] = wholedf['BsmtFullBath'] + wholedf['BsmtHalfBath'] + wholedf['HalfBath'] + wholedf['FullBath']



# Percentage of total rooms are bedrooms

wholedf['PercentBedrmtoRooms'] = wholedf['BedroomAbvGr'] / wholedf['TotRmsAbvGrd']



# Number of years since last remodel, if there never was one it would be since it was built

wholedf['YearSinceRemodel'] = 2016 - ((wholedf['YearRemodAdd'] - wholedf['YearBuilt']) + wholedf['YearBuilt'])
sns.barplot(x="OverallCond", y="LivingTotalSF", data=wholedf)

plt.title("Total Square Footage by Overall Condition Rating")

plt.show()



sns.lmplot(x="LivingTotalSF", y="SalePrice", data=wholedf)

plt.title("Relation of Sale Price to Total Square Footage")

plt.xlim(0,)

plt.ylim(0,)

plt.show()
ax = sns.barplot(x="TotRmsAbvGrd", y="TotalBaths",data=wholedf)

plt.title("Total Rooms Versus Total Bathrooms")

plt.show()
sns.swarmplot(x="MoSold", y="SalePrice", data=wholedf)

plt.title("Sale Price by Month")

plt.show()



sns.kdeplot(wholedf['MoSold']);

plt.title("Distribution of Month Sold")

plt.xlim(1,12)

plt.show()
plt.figure(figsize = (12, 6))

sns.boxplot(x = 'Neighborhood', y = 'SalePrice',  data = wholedf)

plt.xticks(rotation=45)

plt.show()
pricing1 = wholedf[['Id','SalePrice','MiscVal']]



neigh = wholedf[['Neighborhood','MSZoning','MSSubClass','BldgType','HouseStyle']]



dates = wholedf[['YearBuilt','YearRemodAdd','GarageYrBlt','YearSinceRemodel']]



quacon = wholedf[['ExterQual','BsmtQual','PoolQC','Condition1','Condition2','SaleCondition',

                  'BsmtCond','ExterCond','GarageCond','KitchenQual','GarageQual','HeatingQC','OverallQual','OverallCond']]



features =  wholedf[['Foundation','RoofStyle','RoofMatl','Exterior1st','Exterior2nd',

                     'MiscFeature','PavedDrive','Utilities',

                     'Heating','CentralAir','Electrical','Fence']]



sqfoot = wholedf[['LivingTotalSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea',

                  'GarageArea','WoodDeckSF','OpenPorchSF','LotArea','PercentSQtoLot','LowQualFinSF']]



roomfeatcount = wholedf[['PercentBedrmtoRooms','TotalBaths','FullBath','HalfBath',

                         'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','GarageType','EnclosedPorch']]



# Splits out sale price for the training set and only has not null values

pricing = wholedf['SalePrice']

pricing = pricing[pricing.notnull()]



# Bringing it all together

wholedf = pd.concat([pricing1,neigh,dates,quacon,features,sqfoot,roomfeatcount], axis=1)
wholedf = pd.get_dummies(wholedf)
traincorr = wholedf.corr()['SalePrice']

# convert series to dataframe so it can be sorted

traincorr = pd.DataFrame(traincorr)

# correct column label from SalePrice to correlation

traincorr.columns = ["Correlation"]

# sort correlation

traincorr2 = traincorr.sort_values(by=['Correlation'], ascending=False)

traincorr2.head(15)
train_X = wholedf[wholedf['SalePrice'].notnull()]

del train_X['SalePrice']

test_X =  wholedf[wholedf['SalePrice'].isnull()]

del test_X['SalePrice']
# Create all datasets that are necessary to train, validate and test models

train_valid_X = train_X

train_valid_y = pricing

test_X = test_X

train_X , valid_X , train_y , valid_y = train_test_split( train_valid_X , train_valid_y , train_size = .7 )
# model = RandomForestRegressor()

model = Ridge()

# model = LassoLarsCV()



# Models that performed substantially worse

# model = LinearSVC()

# model = KNeighborsClassifier(n_neighbors = 3)

# model = GaussianNB()

# model = LogisticRegression()

# model = SVC()
model.fit( train_X , train_y )



# Print the Training Set Accuracy and the Test Set Accuracy in order to understand overfitting

print (model.score( train_X , train_y ) , model.score( valid_X , valid_y ))
id = test_X.Id

result = model.predict(test_X)



# output = pd.DataFrame( { 'id': id , 'SalePrice': result}, columns=['id', 'SalePrice'] )

output = pd.DataFrame( { 'id': id , 'SalePrice': result} )

output = output[['id', 'SalePrice']]



output.to_csv("solution.csv", index = False)

output.head(10)