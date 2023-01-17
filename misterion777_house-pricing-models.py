import pandas as pd

import numpy as np

import warnings



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from math import sqrt
train = pd.read_csv("../input/train.csv")

train.head()
test = pd.read_csv("../input/test.csv")

test.head()
# To operate on both dataframes simulatenously

combine=[train,test]



# Check if any data is duplicated

sum(train['Id'].duplicated()),sum(test['Id'].duplicated())

miss_cols=['MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',

           'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond']

zero_list=['LotFrontage','MasVnrArea','GarageArea','GarageCars','BsmtFinSF1','BsmtFinSF2',

           'BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']

drop_cols=['PoolQC','Fence','MiscFeature','GarageYrBlt','Alley']

for df in combine:

    # Fill missing values in cetgorical variables with None

    for col in miss_cols:

        df[col]=df[col].fillna('None')

    # Fill missing values in numerical variables with 0

    for col in zero_list:

        df[col]=df[col].fillna(0)

    # Drop columns with large number of missing values

    df.drop(columns=drop_cols,inplace=True)

    

# Fill missing value in Electrical in train dataset

index=train[train['Electrical'].isnull()].index

train.loc[index,'Electrical']=train['Electrical'].mode()[0]



# Fill missing values in categorical variables in test dataset

mode_list=['Utilities','Exterior1st','Exterior2nd','KitchenQual','Functional','SaleType','MSZoning']

for col in mode_list:

    mode=test[col].mode()

    test[col]=test[col].fillna(mode[0])

    

# Check if the above operations worked correctly

train.isnull().sum().max(),test.isnull().sum().max()
categorized_features = ['MSZoning','SaleCondition','SaleType','PavedDrive','GarageCond','GarageQual','GarageFinish',

                    'GarageType','FireplaceQu','Functional','KitchenQual','Heating','HeatingQC','CentralAir',

                     'Electrical','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure',

                     'BsmtFinType1','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Street',

                     'LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1',

                     'Condition2','BldgType','HouseStyle','BsmtFinType2']

lb = LabelEncoder()

for column in categorized_features:

    train[column] = lb.fit_transform(train[column])

    test[column] = lb.fit_transform(test[column])

train.drop(columns='Id',inplace=True)

X_test = test.drop(columns='Id')

X_train = train.drop(columns=['SalePrice'])

Y_train = train['SalePrice']
X_train, X_val, Y_train, Y_val = train_test_split(X_train,Y_train,random_state=42,test_size=0.30)
def rmse_log(y_actual, y_predicted):

    return sqrt(mean_squared_error(np.log(y_actual), np.log(y_predicted)))
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(X_train, Y_train)

Y_test = lr.predict(X_test)

rmse_log(Y_val,lr.predict(X_val))
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(random_state=0)

dtr.fit(X_train, Y_train)

Y_test = dtr.predict(X_test)

rmse_log(Y_val,dtr.predict(X_val))
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(random_state=0, n_estimators=100)

rfr.fit(X_train, Y_train)

Y_test = rfr.predict(X_test)

rmse_log(Y_val,rfr.predict(X_val))
from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

warnings.filterwarnings('ignore')
mlp = make_pipeline(StandardScaler(),MLPRegressor(random_state=42, 

                   hidden_layer_sizes=(700,),activation='logistic', solver='lbfgs',alpha=1))

mlp.fit(X_train, Y_train)

Y_test = mlp.predict(X_test)

rmse_log(Y_val,mlp.predict(X_val))
from mlens.ensemble import SuperLearner 


clf_array = [lr,dtr,rfr,mlp]

ensemble = SuperLearner(random_state = 42, folds = 3,verbose=2)

ensemble.add(clf_array)

ensemble.add_meta(rfr)

ensemble.fit(X_train, Y_train)

Y_test = mlp.predict(X_test)

rmse_log(Y_val, ensemble.predict(X_val))
# Seems like random forest is the best for now:(

Y_test = rfr.predict(X_test)



y_test_pred = pd.DataFrame(data={'Id':test['Id'],'SalePrice':Y_test})


y_test_pred.to_csv('rf.csv',index=False, header=True, encoding='utf-8')