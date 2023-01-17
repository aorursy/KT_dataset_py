import numpy as np
import pandas as pd 
# import data
data = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# keep all the numeric type attributes
#predictors = list(data.select_dtypes(include=['int64','float64']).columns)
#predictors.remove('SalePrice') # remove Y column
# print all columns with null values as a list
print(data.columns[data.isna().any()].tolist())
print(len(data.columns[data.isna().any()].tolist()))
null_columns = data.columns[data.isna().any()].tolist()
data[null_columns].isnull().sum()
# Handling the null/NA values.
# LotFrontage - Linear Feet of street connected to property. Assuming that NA means 0 feet of street connected.
data.loc[:, "LotFrontage"] = data.loc[:, "LotFrontage"].fillna(0)
test.loc[:, "LotFrontage"] = test.loc[:, "LotFrontage"].fillna(0)
# No alley
data.loc[:, "Alley"] = data.loc[:, "Alley"].fillna("None")
test.loc[:, "Alley"] = test.loc[:, "Alley"].fillna("None")
# NA for veneer type probably means there's no stone veneer in the house
data.loc[:, "MasVnrType"] = data.loc[:, "MasVnrType"].fillna("None")
test.loc[:, "MasVnrType"] = test.loc[:, "MasVnrType"].fillna("None")
# Presumably, no stone veneer means 0 veneer area
data.loc[:, "MasVnrArea"] = data.loc[:, "MasVnrArea"].fillna(0)
test.loc[:, "MasVnrArea"] = test.loc[:, "MasVnrArea"].fillna(0)
# NA for basement means no basement
data.loc[:, "BsmtQual"] = data.loc[:, "BsmtQual"].fillna("None")
test.loc[:, "BsmtQual"] = test.loc[:, "BsmtQual"].fillna("None")
# Condition of basement - Doesn't really make sense to have a value for this if there's no basement
data.loc[:, "BsmtCond"] = data.loc[:, "BsmtCond"].fillna("None")
test.loc[:, "BsmtCond"] = test.loc[:, "BsmtCond"].fillna("None")
data.loc[:, "BsmtExposure"] = data.loc[:, "BsmtExposure"].fillna("None")
test.loc[:, "BsmtExposure"] = test.loc[:, "BsmtExposure"].fillna("None")
data.loc[:, "BsmtFinType1"] = data.loc[:, "BsmtFinType1"].fillna("None")
test.loc[:, "BsmtFinType1"] = test.loc[:, "BsmtFinType1"].fillna("None")
data.loc[:, "BsmtFinType2"] = data.loc[:, "BsmtFinType2"].fillna("None")
test.loc[:, "BsmtFinType2"] = test.loc[:, "BsmtFinType2"].fillna("None")
data.loc[:, "BsmtFinSF1"] = data.loc[:, "BsmtFinSF1"].fillna(0)
test.loc[:, "BsmtFinSF1"] = test.loc[:, "BsmtFinSF1"].fillna(0)
data.loc[:, "BsmtFinSF2"] = data.loc[:, "BsmtFinSF2"].fillna(0)
test.loc[:, "BsmtFinSF2"] = test.loc[:, "BsmtFinSF2"].fillna(0)
# Unfinished square feet of basement area
data.loc[:, "BsmtUnfSF"] = data.loc[:, "BsmtUnfSF"].fillna(0)
test.loc[:, "BsmtUnfSF"] = test.loc[:, "BsmtUnfSF"].fillna(0)
# Total square feet of basement area
data.loc[:, "TotalBsmtSF"] = data.loc[:, "TotalBsmtSF"].fillna(0)
test.loc[:, "TotalBsmtSF"] = test.loc[:, "TotalBsmtSF"].fillna(0)
# basement full bathrooms
data.loc[:, "BsmtFullBath"] = data.loc[:, "BsmtFullBath"].fillna(0)
test.loc[:, "BsmtFullBath"] = test.loc[:, "BsmtFullBath"].fillna(0)
# basement half bathrooms
data.loc[:, "BsmtHalfBath"] = data.loc[:, "BsmtFullBath"].fillna(0)
test.loc[:, "BsmtHalfBath"] = test.loc[:, "BsmtFullBath"].fillna(0)


# There's only 1 value in the electrical attribute with an NA value. I'm setting it to SBrkr
data.loc[:, "Electrical"] = data.loc[:, "Electrical"].fillna("SBrkr")
test.loc[:, "Electrical"] = test.loc[:, "Electrical"].fillna("SBrkr")
# No fireplace
data.loc[:, "FireplaceQu"] = data.loc[:, "FireplaceQu"].fillna("None")
test.loc[:, "FireplaceQu"] = test.loc[:, "FireplaceQu"].fillna("None")
# No garage
data.loc[:, "GarageType"] = data.loc[:, "GarageType"].fillna("None")
test.loc[:, "GarageType"] = test.loc[:, "GarageType"].fillna("None")
data.loc[:, "GarageYrBlt"] = data.loc[:, "GarageYrBlt"].fillna(0)
test.loc[:, "GarageYrBlt"] = test.loc[:, "GarageYrBlt"].fillna(0)
data.loc[:, "GarageFinish"] = data.loc[:, "GarageFinish"].fillna("None")
test.loc[:, "GarageFinish"] = test.loc[:, "GarageFinish"].fillna("None")
data.loc[:, "GarageQual"] = data.loc[:, "GarageQual"].fillna("None")
test.loc[:, "GarageQual"] = test.loc[:, "GarageQual"].fillna("None")
data.loc[:, "GarageCond"] = data.loc[:, "GarageCond"].fillna("None")
test.loc[:, "GarageCond"] = test.loc[:, "GarageCond"].fillna("None")
data.loc[:, "GarageCars"] = data.loc[:, "GarageCars"].fillna(0)
test.loc[:, "GarageCars"] = data.loc[:, "GarageCars"].fillna(0)
data.loc[:, "GarageArea"] = data.loc[:, "GarageArea"].fillna(0)
test.loc[:, "GarageArea"] = test.loc[:, "GarageArea"].fillna(0)
# No swimmming pool
data.loc[:, "PoolQC"] = data.loc[:, "PoolQC"].fillna("None")
test.loc[:, "PoolQC"] = test.loc[:, "PoolQC"].fillna("None")
# No fence
data.loc[:, "Fence"] = data.loc[:, "Fence"].fillna("None")
test.loc[:, "Fence"] = test.loc[:, "Fence"].fillna("None")
# No miscellaneous features
data.loc[:, "MiscFeature"] = data.loc[:, "MiscFeature"].fillna("None")
test.loc[:, "MiscFeature"] = test.loc[:, "MiscFeature"].fillna("None")
# Assume NA means all utilities- Because AllPub is the val for most rows.
data.loc[:, "Utilities"] = data.loc[:, "Utilities"].fillna("AllPub")
test.loc[:, "Utilities"] = test.loc[:, "Utilities"].fillna("AllPub")
# Functionality of House. Assume Typ if NA, from description.
data.loc[:, "Functional"] = data.loc[:, "Functional"].fillna("Typ")
test.loc[:, "Functional"] = test.loc[:, "Functional"].fillna("Typ")
# Zoning classification of Sale. Assuming RL for NA, since its the majority value.
data.loc[:, "MSZoning"] = data.loc[:, "MSZoning"].fillna("RL")
test.loc[:, "MSZoning"] = test.loc[:, "MSZoning"].fillna("RL")
# Assume kitchen quality is TA(typical/avg) for NA
data.loc[:, "KitchenQual"] = data.loc[:, "KitchenQual"].fillna("TA")
test.loc[:, "KitchenQual"] = test.loc[:, "KitchenQual"].fillna("TA")
# Saletype - Assume WD(warranty deed) for NA as it is the majority value.
data.loc[:, "SaleType"] = data.loc[:, "SaleType"].fillna("WD")
test.loc[:, "SaleType"] = test.loc[:, "SaleType"].fillna("WD")



Y = data.SalePrice
data = data.drop(['SalePrice'], axis=1)
train_objs_len = len(data)
# Merge the training set and test set
# Use get_dummies to encode the values and split them back into train-test.
train_plus_test = pd.concat(objs=[data, test], axis=0)
train_plus_test = pd.get_dummies(train_plus_test.drop(['Exterior1st', 'Exterior2nd'],axis=1))
# Extract training and test data from the merged dataset.
train = train_plus_test.iloc[:train_objs_len] 
test = train_plus_test.iloc[train_objs_len:]
print(len(train))
print(len(test))
# Train-test split
from sklearn.model_selection import train_test_split
X_encoded_train, X_encoded_val, Y_encoded_train, Y_encoded_val = train_test_split(train, Y, random_state=0)

# Build the model with one-hot-encoded data
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
gbr_encoded = GradientBoostingRegressor()
gbr_encoded.fit(X_encoded_train, Y_encoded_train)
predicted_Y_encoded = gbr_encoded.predict(X_encoded_val)
print(mean_absolute_error(Y_encoded_val, predicted_Y_encoded))
# Use the model to make predictions
predicted_saleprices = gbr_encoded.predict(test)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_saleprices)
# create submission files
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_saleprices})

# you could use any filename. We choose submission here
my_submission.to_csv('output1.csv', index=False)
