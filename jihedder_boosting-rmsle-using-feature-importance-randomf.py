import numpy as np 

import pandas as pd 

df=pd.read_csv('../input/train.csv')

df.head()
df.isnull().values.any()

df.isnull().sum()

#=df.drop(['PoolQC','Fence','MiscFeature','Alley','Street'], axis=1)
df=df[["Id","MSSubClass","LotFrontage","LotArea","HouseStyle","OverallQual","Foundation","RoofStyle","OverallCond",

"BldgType","Neighborhood","TotalBsmtSF","1stFlrSF","2ndFlrSF","Heating","CentralAir","YearBuilt","MasVnrArea",

"GrLivArea","BedroomAbvGr","KitchenAbvGr","TotRmsAbvGrd","Fireplaces","Electrical","GarageArea","GarageYrBlt",

"GarageCars","PoolArea","YrSold","WoodDeckSF","OpenPorchSF","3SsnPorch","MiscVal","MoSold","BsmtFullBath",

"BsmtHalfBath","SaleCondition","SalePrice"]]
import random

df=df.fillna({'LotFrontage' :50})

df=df.fillna({'GarageYrBlt' :random.randint(1879,2010)})

df=df.fillna({'Electrical' :'SBrkr'})

df=df.fillna({'MasVnrArea' :0})

df.isnull().sum()

df.isnull().values.any()
X=df.loc[:,['MSSubClass','LotArea','TotalBsmtSF','1stFlrSF','2ndFlrSF','OverallQual','OverallCond','CentralAir','Heating','RoofStyle','Foundation',

'YearBuilt','MasVnrArea','GrLivArea','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','WoodDeckSF','OpenPorchSF',

'3SsnPorch','GarageArea','MiscVal','MoSold','BsmtFullBath','BsmtHalfBath','PoolArea','YrSold']]

X = pd.get_dummies(X, columns=['CentralAir','Heating','RoofStyle','Foundation'])

X.head()

Y=df.loc[:,'SalePrice']
import numpy as np

feature_list=list(X.columns)

X=np.array(X)

y=np.array(Y)
from sklearn.ensemble import RandomForestRegressor

num_training = int(0.8 * len(X))

X_train, y_train = X[:num_training], y[:num_training]

X_test, y_test = X[num_training:], y[num_training:]
RF_regressor = RandomForestRegressor(max_depth=30, random_state=7,n_estimators=1000)

RF_regressor.fit(X_train, y_train)
# extracting feature importances

importances=list(RF_regressor.feature_importances_)

feature_importances=[(feature,round(importance,2)) for feature, importance in zip(feature_list, importances)]

# SORTED the feature_imprtances 

feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

for feature, importance in feature_importances:

    print('Variable:', feature, end='')

    print(' importance:', importance)
# extract the names of the most important features

important_feature_names=[feature[0] for feature in feature_importances[:16]] # 16: all these important features have been selected

#Select the important indices 

important_indices=[feature_list.index(feature) for feature in important_feature_names]
#Re-training using our new features

num_training = int(0.8 * len(X))

X_train_important, y_train_important = X[:num_training, important_indices], y[:num_training]

X_test_important, y_test_important = X[num_training:, important_indices], y[num_training:]
RF_regressor = RandomForestRegressor(max_depth=100, random_state=7,n_estimators=700)

RF_regressor.fit(X_train_important, y_train_important)

y_test_pred = RF_regressor.predict(X_test_important)
# Evaluate the model using RMSLE

from sklearn.metrics import mean_squared_log_error

print("Root Mean Squared Logarithmic Score:", np.sqrt(mean_squared_log_error( y_test_important, y_test_pred)))