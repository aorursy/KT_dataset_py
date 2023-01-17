import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

import os

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
train.head(2)



#LotFrontage, Alley, FirePlaceQC,
sns.scatterplot('YearBuilt','GarageYrBlt',data=train)


train['YrSold'] = train['YrSold'].astype(int)

train['DeltaYears'] = train['YearBuilt'] - train['YrSold']



train["LotFrontage"] = train.groupby(['MSZoning','MSSubClass'])["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))

train['MasVnrArea'] = train.groupby(['MSZoning','MSSubClass'])["MasVnrArea"].transform(

    lambda x: x.fillna(x.median()))

train['GarageYrBlt'] = train['GarageYrBlt'].fillna(train.apply(lambda x: x.YearBuilt,axis=1))

train['GarageYrBlt'] = train['GarageYrBlt'].fillna((train['GarageYrBlt'].mean()))

train['GarageYrBlt'] = train['GarageYrBlt'].astype(int)

train["PoolQC"] = train["PoolQC"].fillna("None")

train["MiscFeature"] = train["MiscFeature"].fillna("None")

train["Alley"] = train["Alley"].fillna("None")

train["Fence"] = train["Fence"].fillna("None")

train["FireplaceQu"] = train["FireplaceQu"].fillna("None")



for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    train[col] = train[col].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    train[col] = train[col].fillna(0)

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    train[col] = train[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    train[col] = train[col].fillna('None')



train['MSSubClass'] = train['MSSubClass'].apply(str)

train['OverallCond'] = train['OverallCond'].astype(str)

#train['YrSold'] = train['YrSold'].astype(str)

train['MoSold'] = train['MoSold'].astype(str)

train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']

#train.info()<1460

#plt.figure(figsize = (10,8))

#sns.heatmap(train.isnull(), yticklabels=False,cbar=False,cmap='coolwarm')
#train['LotArea'] = train['LotArea'].apply(lambda x: x if x<50000 else 50000)

train.plot.scatter(x='TotalSF',y='SalePrice')
qualTrain = train.select_dtypes(include=['object'])

quantTrain = train.select_dtypes(exclude=['object'])

Quantscores = quantTrain.drop("SalePrice", axis=1).apply(lambda x: x.corr(quantTrain.SalePrice)).sort_values(ascending=False)

quantColstoRemove = Quantscores[Quantscores<0.05].index

#Remove variables with less than ±0.1 effect on SalePrice

Id = quantTrain['Id']

quantTrain.drop(quantColstoRemove,axis=1,inplace=True)

quantTrain = quantTrain[quantTrain['LotArea']<60000]



plt.figure(figsize = (16,10))

mask = np.zeros_like(quantTrain.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)]=True

sns.heatmap(quantTrain.corr(), cmap='coolwarm',annot=True,mask=mask,linewidths=0.1)
#I take all qualitative columns from the dataset and then eliminate "bad" ones

somequalTrain = pd.concat([qualTrain,quantTrain['SalePrice']], axis=1)

dummies = pd.get_dummies(somequalTrain)

Qualscores  = dummies.drop("SalePrice", axis=1).apply(lambda x: x.corr(dummies.SalePrice)).sort_values(ascending=True)

qualColstoKeep = Qualscores[(Qualscores<-0.05) | (Qualscores>0.05)].index

finalQual = dummies[qualColstoKeep]
Qualscores
finalTrain = pd.concat([quantTrain,finalQual],axis=1)

len(finalTrain.columns)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn.datasets import make_moons

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import Imputer
X = finalTrain.drop(['SalePrice'],axis=1).fillna(0.).values

y = finalTrain['SalePrice'].fillna(0.).values

# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y,test_size=0.30, random_state=42)
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):

    return np.sqrt(mean_squared_error(y_true, y_pred))
rf_model = RandomForestRegressor(n_estimators=600,random_state=42, max_depth=12, max_features='auto')

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for XGBRegressor: {:,.0f}".format(rf_val_mae))

print("Mean Absolute Error : " + str(rmse(rf_val_predictions, val_y)))

print("accuracy on training set: %f" % rf_model.score(train_X, train_y))

print("accuracy on test set: %f" % rf_model.score(val_X, val_y))
from xgboost import XGBRegressor

my_model = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.02, max_depth=10, 

                             min_child_weight=1.7817, n_estimators=2500,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =42, nthread = -1)

# Add silent=True to avoid printing out updates with each cycle

my_model.fit(train_X, train_y, early_stopping_rounds=5, eval_set=[(val_X, val_y)], verbose=False)

# make predictions

predictions = my_model.predict(val_X)

rf_val_mae = mean_absolute_error(predictions, val_y)



print("Validation MAE for XGBRegressor: {:,.0f}".format(rf_val_mae))

print("Mean Absolute Error : " + str(rmse(predictions, val_y)))

print("accuracy on training set: %f" % my_model.score(train_X, train_y))

print("accuracy on test set: %f" % my_model.score(val_X, val_y))
my_pipeline = make_pipeline(RandomForestRegressor())

from sklearn.model_selection import cross_val_score

scores = cross_val_score(my_pipeline, X, y, scoring='neg_mean_absolute_error')

print("accuracy on set: %f" % my_model.score(X, y))
my_pipe2 = make_pipeline(my_model)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(my_pipe2, X, y)#, scoring='neg_mean_absolute_error')

#print('Mean Absolute Error %2f' %(-1 * scores.mean()))

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print("accuracy on set: %f" % my_model.score(X, y))
my_model.fit(X, y, early_stopping_rounds=5, eval_set=[(val_X, val_y)], verbose=False)

print("accuracy on set: %f" % my_model.score(X, y))
# path to file you will use for predictions

test_data_path = '../input/test.csv'

# read test data file using pandas

test_data = pd.read_csv(test_data_path)

#test_data.drop(NAcols,axis=1,inplace=True)



test_data['YrSold'] = test_data['YrSold'].astype(int)

test_data['DeltaYears'] = test_data['YearBuilt'] - test_data['YrSold']



#.bmeta['year_built'] = bmeta['year_built'].fillna(bmeta.groupby(['primary_use','site_id'])['year_built'].transform('median').round(0))



test_data["LotFrontage"] = test_data["LotFrontage"].fillna(test_data.groupby(['MSZoning','MSSubClass'])['LotFrontage'].transform('median').round(0))

test_data['MasVnrArea'] = test_data["MasVnrArea"].fillna(test_data.groupby(['MSZoning','MSSubClass'])['MasVnrArea'].transform('median').round(0))

test_data['GarageYrBlt'] = test_data['GarageYrBlt'].fillna(train.apply(lambda x: x.YearBuilt,axis=1))

test_data['GarageYrBlt'] = test_data['GarageYrBlt'].fillna((test_data['GarageYrBlt'].mean()))

test_data['GarageYrBlt'] = test_data['GarageYrBlt'].astype(int)

test_data["PoolQC"] = test_data["PoolQC"].fillna("None")

test_data["MiscFeature"] = test_data["MiscFeature"].fillna("None")

test_data["Alley"] = test_data["Alley"].fillna("None")

test_data["Fence"] = test_data["Fence"].fillna("None")

test_data["FireplaceQu"] = test_data["FireplaceQu"].fillna("None")

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    test_data[col] = test_data[col].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    test_data[col] = test_data[col].fillna(0)

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    test_data[col] = test_data[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    test_data[col] = test_data[col].fillna('None')



test_data['MSSubClass'] = test_data['MSSubClass'].apply(str)

test_data['OverallCond'] = test_data['OverallCond'].astype(str)

#test_data['YrSold'] = test_data['YrSold'].astype(str)

test_data['MoSold'] = test_data['MoSold'].astype(str)

test_data['TotalSF'] = test_data['TotalBsmtSF'] + test_data['1stFlrSF'] + test_data['2ndFlrSF']



dummiest = pd.get_dummies(test_data.select_dtypes(include=['object']))

finalQualt = dummiest[qualColstoKeep]

finalQuant = test_data[quantTrain.drop('SalePrice',axis=1).columns]
htdata = pd.concat([finalQuant,finalQualt],axis=1)

Xt = htdata.fillna(0.).values

# make predictions which we will submit. 

test_preds = my_model.predict(Xt)#iowa_model.predict(Xt)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)