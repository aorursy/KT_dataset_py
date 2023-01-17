import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import sys
print(os.listdir("../input"))

data_train_file = "../input/train.csv"


df_train = pd.read_csv(data_train_file)

# keep the original training data for occasional use
df_train_original = df_train


# get the number of missing data points per column
print(df_train.shape)
print(df_test.shape)

missing_values_count_train = df_train.isnull().sum()
print(missing_values_count_train)

# how many total missing values do we have?
total_cells = np.product(df_train.shape)
total_missing = missing_values_count_train.sum()

print("percent of data that is missing")
(total_missing/total_cells) * 100

# The total number of training records is 1460, Several features have in excess of 1100 missing values.
# These columns (Alley, PoolQC, Fence, and MiscFeature) are being deleted from the training and test data
df_train.drop(["Alley", "PoolQC", "Fence","MiscFeature"], axis=1, inplace=True)

# Houses, where the garage area is 0, have all garage related attributes as NA. Assuming that garage area
# contains sufficient information for the regression, let us remove the additional garage related columns
# We can come back to these, if we suspect that they have additional influence on the outcome
df_train.drop(["GarageType", "GarageYrBlt", "GarageFinish","GarageQual","GarageCond"], axis=1, inplace=True)
# Same goes for "LotFrontage" (259 missing values)
df_train.drop(["LotFrontage"], axis=1, inplace=True)
#print(df_train.head(5))
print(df_train.info())

#Find correlation of each attribute with the sale price. Select a set with maximum correlations.
#Then find correlations between the attributes themselves, and keep only one among the correlated groups
corr=df_train.corr()["SalePrice"]
print(corr[np.argsort(corr, axis=0)[::-1]])



# Find correlation of the sale price with nominal attributes likeOverallQual
#df_train.boxplot('SalePrice', by='OverallQual')
df_train.boxplot('SalePrice', by='MSZoning')
df_train.boxplot('SalePrice', by='Street')
df_train.boxplot('SalePrice', by='LandContour')
df_train.boxplot('SalePrice', by='Utilities')
df_train.boxplot('SalePrice', by='LotConfig')
df_train.boxplot('SalePrice', by='Neighborhood')
df_train.boxplot('SalePrice', by='Condition1')
df_train.boxplot('SalePrice', by='BldgType')
df_train.boxplot('SalePrice', by='ExterQual')
df_train.boxplot('SalePrice', by='HouseStyle')
df_train.boxplot('SalePrice', by='ExterCond')

df_train.boxplot('SalePrice', by='HeatingQC')
df_train.boxplot('SalePrice', by='CentralAir')
df_train.boxplot('SalePrice', by='KitchenQual')
# As a starting point, look at the correlation indices for integer valued attributes, and use
# those attributes with correlation coefficients >= 0.5 or <= -0.5
# These attributes are as follows: 
# OverallQual      0.790982
# GrLivArea        0.708624
# GarageCars       0.640409
# GarageArea       0.623431
# TotalBsmtSF      0.613581
# 1stFlrSF         0.605852
# FullBath         0.560664
# TotRmsAbvGrd     0.533723
# YearBuilt        0.522897
# YearRemodAdd     0.507101

# First check if any of these are correlated amongst each other
import matplotlib.pyplot as plt


feature_names = ['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd']
data = df_train[feature_names]
correlations = data.corr()
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,10,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(feature_names, rotation='vertical')
ax.set_yticklabels(feature_names)
plt.show()



from sklearn import linear_model

feature_names = ['OverallQual','GrLivArea','GarageCars','1stFlrSF','FullBath','YearBuilt','YearRemodAdd']
X = df_train[feature_names]
y = df_train['SalePrice']

lm = linear_model.LinearRegression()
model = lm.fit(X,y)
#Predict
predictions = lm.predict(X)
print(predictions[0:5])

print(lm.score(X,y))
# we are getting r square value of 0.7698 

# Add some categorical variables - Neighborhood, ExterQual, KitchenQual,'MSZoning','Street', 'Utilities','CentralAir'
# - these look promising from the boxplots
df_train2 = pd.get_dummies(df_train, columns=['Neighborhood', 'ExterQual', 'KitchenQual', 'MSZoning','Street', 'Utilities','CentralAir'], drop_first=True)
df_train2[0:5]

feature_names = ['OverallQual','GrLivArea','GarageCars','1stFlrSF','FullBath','YearBuilt','YearRemodAdd',\
                'Neighborhood_Blueste','Neighborhood_BrDale','Neighborhood_BrkSide','Neighborhood_ClearCr','Neighborhood_CollgCr','Neighborhood_Crawfor',\
                 'Neighborhood_Edwards','Neighborhood_Gilbert','Neighborhood_IDOTRR','Neighborhood_MeadowV','Neighborhood_Mitchel','Neighborhood_NAmes',\
                 'Neighborhood_NPkVill','Neighborhood_NWAmes','Neighborhood_NoRidge','Neighborhood_NridgHt','Neighborhood_OldTown','Neighborhood_SWISU',\
                 'Neighborhood_Sawyer','Neighborhood_SawyerW','Neighborhood_Somerst','Neighborhood_StoneBr','Neighborhood_Timber','Neighborhood_Veenker',\
                 'ExterQual_Fa','ExterQual_Gd','ExterQual_TA','KitchenQual_Fa','KitchenQual_Gd','KitchenQual_TA',\
                 'MSZoning_FV','MSZoning_RH','MSZoning_RL','MSZoning_RM','Street_Pave','Utilities_NoSeWa','CentralAir_Y']

X = df_train2[feature_names]
y = df_train2['SalePrice']

# Create linear regression object
lm = linear_model.LinearRegression()
# Train the model using the training sets
lm.fit(X,y)
#Predict
predictions = lm.predict(X)
print(predictions[0:5])

print(lm.score(X,y))
# we are getting r square value of 0.834

import matplotlib.pyplot as plt
plt.figure(1)
plt.scatter(y, predictions - y , c='b', s=40, alpha=0.5)
plt.scatter(y, predictions, c='g', s=40, alpha=0.5)
plt.hlines(y=0, xmin=0, xmax=800000)
plt.ylabel('Residuals')

df_train_not_expensive = df_train2.loc[df_train2['SalePrice'] < 500000]
X = df_train_not_expensive[feature_names]
y = df_train_not_expensive['SalePrice']

lm = linear_model.LinearRegression()
model = lm.fit(X,y)
#Predict
predictions = lm.predict(X)
print(predictions[0:5])

print(lm.score(X,y))
# we are getting r square value of 0.8417

plt.figure(2)
plt.scatter(y, predictions - y , c='b', s=40, alpha=0.5)
plt.scatter(y, predictions, c='g', s=40, alpha=0.5)
plt.hlines(y=0, xmin=0, xmax=500000)
plt.ylabel('Residuals') 


from sklearn.ensemble import RandomForestRegressor
# Create random forest regressor object
radm_clf = RandomForestRegressor(oob_score=True,n_estimators=100 )
# Train the model using the training sets
X = df_train2[feature_names]   #Use the training data that contains the expensive houses >= 500,000
y = df_train2['SalePrice']
radm_clf.fit(X, y)
#Predict
predictions = radm_clf.predict(X)
print(predictions[0:5])

print(radm_clf.score(X,y))
# we are getting r square value of 0.978, this changes a bit from one execution to the next.

plt.figure(2)
plt.scatter(y, predictions - y , c='b', s=40, alpha=0.5)
plt.scatter(y, predictions, c='g', s=40, alpha=0.5)
plt.hlines(y=0, xmin=0, xmax=800000)
plt.ylabel('Residuals')



data_test_file = "../input/test.csv"
df_test = pd.read_csv(data_test_file)

missing_values_count_test = df_test.isnull().sum()
print(missing_values_count_test)

df_test.drop(["Alley", "PoolQC", "Fence","MiscFeature"], axis=1, inplace=True)
df_test.drop(["GarageType", "GarageYrBlt", "GarageFinish","GarageQual","GarageCond"], axis=1, inplace=True)
df_test.drop(["LotFrontage"], axis=1, inplace=True)

# Replace missing values with mode values
for column in df_test.columns:
    df_test[column].fillna(df_test[column].mode().iloc[0], inplace=True)

# Add one hot encoding for the categorical variables.
df_test2 = pd.get_dummies(df_test, columns=['Neighborhood', 'ExterQual', 'KitchenQual', 'MSZoning','Street', 'Utilities','CentralAir'], drop_first=True)
# Utilities_NoSeWa - the NoSeWa level is not included in the test data, so we have to add a column
df_test2['Utilities_NoSeWa'] = 0

feature_names = ['OverallQual','GrLivArea','GarageCars','1stFlrSF','FullBath','YearBuilt','YearRemodAdd',\
                'Neighborhood_Blueste','Neighborhood_BrDale','Neighborhood_BrkSide','Neighborhood_ClearCr','Neighborhood_CollgCr','Neighborhood_Crawfor',\
                 'Neighborhood_Edwards','Neighborhood_Gilbert','Neighborhood_IDOTRR','Neighborhood_MeadowV','Neighborhood_Mitchel','Neighborhood_NAmes',\
                 'Neighborhood_NPkVill','Neighborhood_NWAmes','Neighborhood_NoRidge','Neighborhood_NridgHt','Neighborhood_OldTown','Neighborhood_SWISU',\
                 'Neighborhood_Sawyer','Neighborhood_SawyerW','Neighborhood_Somerst','Neighborhood_StoneBr','Neighborhood_Timber','Neighborhood_Veenker',\
                 'ExterQual_Fa','ExterQual_Gd','ExterQual_TA','KitchenQual_Fa','KitchenQual_Gd','KitchenQual_TA',\
                 'MSZoning_FV','MSZoning_RH','MSZoning_RL','MSZoning_RM','Street_Pave','Utilities_NoSeWa','CentralAir_Y']


X_test = df_test2[feature_names]

#Predict
predictions = radm_clf.predict(X_test)


# Create the submission dataframe and file
submission_df = pd.DataFrame() #creates a new dataframe that's empty
submission_df['Id'] = df_test2['Id']
submission_df['SalePrice'] = predictions.tolist()
submission_df.head()
submission_df.to_csv("submission.csv", index=False)

