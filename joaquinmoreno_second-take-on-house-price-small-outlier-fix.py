# Import the set of libraries we will be using in this analysis:

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from sklearn.preprocessing import LabelEncoder



# Initiate the matplotlib notebook magic

%matplotlib inline
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.describe()
print('Training set has shape {} \nTest set has shape {}'.format(train.shape,test.shape))
# Plot the distribution of SalePrice form the training set. Fit a normal distribution

plt.subplots(figsize=(10,5))

sns.distplot(train['SalePrice'], fit=stats.norm)



# Retreive the mu and sigmna from the fit

(mu, sigma) = stats.norm.fit(train['SalePrice'])



# Create a plot that shows all values and legends

plt.legend(["Normal Distribution ($/mu=$ {:.2f} and $/sigma=$ {:.2f})".format(mu,sigma)])

plt.ylabel('Frequency')

plt.xlabel('Sale Price')



# Adding a Probability plot (quantiles for a probability plot)

fig = plt.figure()

plt.subplots(figsize=(10,5))

stats.probplot(train['SalePrice'],plot=plt)

plt.show()
train['LogSalePrice'] = np.log1p(train.SalePrice) 
# Lets plot again and see a fitted Normal Distribution for the sample

plt.subplots(figsize=(10,5))

sns.distplot(train.LogSalePrice,fit=stats.norm)



# Retrive again the parameters

(mu,sigma) = stats.norm.fit(train.LogSalePrice)



# Add legend and labels to the plot

plt.ylabel('Frequency')

plt.xlabel('Log Sale Price')

plt.legend(['Nomal distribution ($/mu=$ {:.2f} and $/sigma=$ {:.2f})'.format(mu,sigma)], loc='best')



# Adding again a Probability plot (quantiles for a probability plot)

fig = plt.figure()

plt.subplots(figsize=(10,5))

stats.probplot(train['LogSalePrice'],plot=plt)

plt.show()
# What features (columns) have missing values? 

train.columns[train.isnull().any()]
features_wnull = ['LotFrontage', 'Alley', 'MasVnrType', 'MasVnrArea', 'BsmtQual','BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','Electrical', 'FireplaceQu', 'GarageType', 'GarageYrBlt','GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence','MiscFeature']

train[features_wnull].count()
train.columns[train.isnull().any()]
plt.figure(figsize=(15,7))

sns.heatmap(train.isnull())

plt.show()
isnull_values = train.isnull().sum()/len(train)*100

isnull_values = isnull_values[isnull_values>0]

isnull_values.sort_values(inplace=True, ascending=False)

print(isnull_values)
isnull_values = isnull_values.to_frame()
isnull_values.columns = ['Count']

isnull_values.index.names = ['Name']

isnull_values['Name'] = isnull_values.index
plt.figure(figsize=(10,5))

sns.set(style='darkgrid') #style='whitegrid' is good too

sns.barplot(x = "Name", y = "Count", data=isnull_values)

plt.xticks(rotation=90) # This makes the names of the x axis to be vertical

plt.show()
# We are going to take the numerical features only

train_corr = train.select_dtypes(include=[np.number])

train_corr.shape
del train_corr['Id'] # Id is a feature that do not tell us anything so we need to take it out
train_corr.shape
# Now we perform a Pearson correlation

correl = train_corr.corr()

plt.subplots(figsize=(23,20))

sns.heatmap(correl, annot=True, fmt='.0%')
#Lets find the best features (in this case, Correlation higher than ABS(0.5))

best_features = correl.index[abs(correl['SalePrice']>0.5)]
best_features
plt.subplots(figsize=(10,5))

best_corr = train[best_features].corr()

sns.heatmap(best_corr, annot=True)

plt.show()
sns.barplot(train.OverallQual, train.SalePrice)
sns.barplot(train.OverallQual, train.LogSalePrice)
train.OverallQual.unique()
plt.figure(figsize=(10,5))

sns.boxplot(x=train.OverallQual, y=train.LogSalePrice)
col = ['OverallQual', 'YearBuilt', 'TotalBsmtSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'GarageCars', 'LogSalePrice']
# this also serves as an outlier spotter (check SalePrice line)

sns.pairplot(train[col], height=3, kind='reg')
correl_full = train.corr()

correl_full.sort_values(['LogSalePrice'], ascending=False, inplace=True)
# Correlation against LogSalePrice

correl.LogSalePrice
train.columns[train.isnull().any()]
for col in ['MiscFeature','Alley','Fence','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond',\

            'GarageYrBlt','MasVnrType','PoolQC','BsmtFinType1','BsmtFinType2','BsmtExposure','BsmtCond','BsmtQual']:

    train[col] = train[col].fillna('None')
train.columns[train.isnull().any()]
train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
train['MasVnrArea'] = train['MasVnrArea'].fillna(int(0))
train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])
plt.figure(figsize=(5,3))

sns.barplot(train.isnull())
cols = train.columns

cols
cols_str = [ col  for col, x in train.dtypes.items() if x == object]
cols_str = list(cols_str)

cols_str
count_cols_str = 0

for i in cols_str:

    count_cols_str += 1
print("The set has", count_cols_str, "features with object values")
# Now we will encode all the columns that have string values with sklearn LabelEncoder

for c in cols_str:

    lbl = LabelEncoder()  #Initialize the encoder

    lbl.fit(list(train[c].values))

    train[c] = lbl.transform(list(train[c].values))
# taking out the outliers

train = train[train['TotalBsmtSF'] < 5000]

train = train[train['GrLivArea']  < 4750]

train = train[train['TotRmsAbvGrd']  < 13]
y = train.SalePrice
train.SalePrice.shape
X = train.copy()
del X['SalePrice']

del X['LogSalePrice']

#y = y.values

#X = X.values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
from sklearn import linear_model

model = linear_model.LinearRegression()
model.fit(X_train,y_train)
# Make the predictions based on the data we have on X_test

linear_model_predicts = model.predict(X_test)
# Meassure the Root Mean Squared Error to test performance

from sklearn.metrics import mean_squared_error

from math import sqrt



rmse = sqrt(mean_squared_error(y_test, linear_model_predicts))

acurracy = model.score(X_test,y_test)





print("The RMSE is: ",rmse)

print("The Accuracy of the model is:", round(acurracy*100,2),"%")
from  sklearn.ensemble  import RandomForestRegressor
model = RandomForestRegressor(n_estimators=1000)

model.fit(X_train, y_train)

random_forest_predicts = model.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, random_forest_predicts))

acurracy = model.score(X_test,y_test)





print("The RMSE is: ",rmse)

print("The Accuracy of the model is:", round(acurracy*100,2),"%")
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(n_estimators=100, max_depth=4)

model.fit(X_train, y_train)
gradient_boosting_predicts = model.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, gradient_boosting_predicts))

acurracy = model.score(X_test,y_test)





print("The RMSE is: ",rmse)

print("The Accuracy of the model is:", round(acurracy*100,2),"%")
X_test
test.columns[test.isnull().any()]
for col in ['MiscFeature','Alley','Fence','FireplaceQu','GarageType','GarageFinish','GarageQual',\

            'GarageCond','GarageYrBlt','MasVnrType','PoolQC','BsmtFinType1','BsmtFinType2',\

            'BsmtExposure','BsmtCond','BsmtQual','MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',\

            'BsmtUnfSF', 'TotalBsmtSF','BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea']:

    test[col] = test[col].fillna('None')
test['Electrical'] = test['Electrical'].fillna(test['Electrical'].mode()[0])
test['LotFrontage'] = test.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
test['MasVnrArea'] = test['MasVnrArea'].fillna(int(0))
cols_str = [ col  for col, x in test.dtypes.items() if x == object]
cols_str = list(cols_str)

cols_str
count_cols_str = 0

for i in cols_str:

    count_cols_str += 1
for c in cols_str:

    lbl = LabelEncoder()  #Initialize the encoder

    lbl.fit(list(test[c].values))

    test[c] = lbl.transform(list(test[c].values))
from xgboost import XGBRegressor

xgb_model = xgboost = XGBRegressor(learning_rate=0.01, n_estimators=3460,

                                     max_depth=3, min_child_weight=0,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,

                                     objective='reg:linear', nthread=-1,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.00006)

xgb_model.fit(X, y)
xgb_model_predicts = xgb_model.predict(test)
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
test.index
output = pd.DataFrame({'Id': test['Id'],

                       'SalePrice': xgb_model_predicts})

output.to_csv('submission.csv', index=False)
#submission = pd.read_csv('submission.csv')
#submission
test.Id