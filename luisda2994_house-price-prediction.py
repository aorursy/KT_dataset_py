# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.describe(include='all')
#Deleting outliers

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column

train["SalePrice"] = np.log1p(train["SalePrice"])
from sklearn.impute import SimpleImputer



#my_imputer = SimpleImputer()



#numeric_columns = (train.dtypes == 'int64') | (train.dtypes == 'float64')

#train_numeric = (train[numeric_columns.index[numeric_columns]])

#train_non_numeric = train.drop(numeric_columns.index[numeric_columns],axis=1)

#imputed_X_train = pd.DataFrame(my_imputer.fit_transform(train_numeric))

#imputed_X_train.columns = train_numeric.columns

#Now there are no null values for the numeric attributes
ntrain = train.shape[0]

ntest = test.shape[0]

y = train.SalePrice.values

all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)

print("all_data size is : {}".format(all_data.shape))



all_data["PoolQC"] = all_data["PoolQC"].fillna("None")

all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

all_data["Alley"] = all_data["Alley"].fillna("None")

all_data["Fence"] = all_data["Fence"].fillna("None")

all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))



for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    all_data[col] = all_data[col].fillna('None')

    

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    all_data[col] = all_data[col].fillna(0)



for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    all_data[col] = all_data[col].fillna(0)



for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    all_data[col] = all_data[col].fillna('None')





all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)



all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])



all_data = all_data.drop(['Utilities'], axis=1)



all_data["Functional"] = all_data["Functional"].fillna("Typ")



all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])



all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])



all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])



all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])



all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")





#MSSubClass=The building class

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)





#Changing OverallCond into a categorical variable

all_data['OverallCond'] = all_data['OverallCond'].astype(str)





#Year and month sold are transformed into categorical features.

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)



from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')

# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(all_data[c].values)) 

    all_data[c] = lbl.transform(list(all_data[c].values))



# shape        

print('Shape all_data: {}'.format(all_data.shape))





# Adding total sqfootage feature 

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']



all_data = pd.get_dummies(all_data)

print(all_data.shape)



X = all_data[:ntrain]

X_test = all_data[ntrain:]





#train = pd.concat([imputed_X_train, train_non_numeric], axis=1)

#X = train.drop(['Id','SalePrice'],axis=1)



#Converting categorical features into their one-hot-encoding versions

#X = pd.get_dummies(X)

#print(X.columns.values)

#y = train['SalePrice']

#print('Shape of the X train tensor: ',X.shape)

#print('Shape of the y train tensor: ',y.shape)





#test = pd.read_csv('../input/test.csv')

#X_test = test.drop('Id',axis = 1)

#X_test = pd.get_dummies(X_test)

#print('Shape of the X test tensor: ',X_test.shape)



#We gotta eliminate those columns generated by the one-hot-encoding in the train not present in the test

#X = X[X_test.columns]

#print('New shape for the X train tensor: ',X.shape)



#print(X.columns)

#print(X_test.columns)

#(X.columns == X_test.columns).all()



from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectFromModel



#clf = ExtraTreesClassifier(n_estimators=1000, n_jobs = -1)

#clf = clf.fit(X, y)



#model = SelectFromModel(clf, prefit=True)

#X = model.transform(X)

#X.shape 
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error



X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size = 0.1)
from sklearn.ensemble import RandomForestRegressor

from math import log10, sqrt



n_estimators = 1000

regr = RandomForestRegressor(n_estimators=n_estimators, n_jobs = -1)

regr.fit(X_train, y_train)



y_pred = (regr.predict(X_val))



print('RMSE (of the logs) for the random forest with', n_estimators, ' estimator = ' ,(sqrt(mean_squared_error(np.log(y_val),np.log(y_pred)))))
from xgboost import XGBRegressor



n_estimators_XGB = 1000

XGB_reg = XGBRegressor(n_estimators=n_estimators_XGB, learning_rate=0.01, n_jobs=-1)

XGB_reg.fit(X_train, y_train, 

             early_stopping_rounds=5, 

             eval_set=[(X_val, y_val)], 

             verbose=False)



y_pred = (XGB_reg.predict(X_val))

print('RMSE for the XGBoost with', n_estimators_XGB, ' estimator = ' ,(sqrt(mean_squared_error(np.log(y_val),np.log(y_pred)))))
from keras.models import Sequential

from keras.layers import Dense



MLP = Sequential()

MLP.add(Dense(512, activation = 'relu', input_shape=(X_train.shape[1],)))

MLP.add(Dense(512, activation = 'relu'))

MLP.add(Dense(1, activation = 'linear'))



MLP.compile(optimizer = 'adam', loss = 'mean_squared_error')



#MLP.fit(X_train,y_train, validation_data = (X_val,y_val), epochs = 10, batch_size = None, 

#          steps_per_epoch = 1000, validation_steps = 10)
#y_pred_MLP = np.expm1(MLP.predict(X_val))

#print('RMSE for the MLP = ' ,(sqrt(mean_squared_error(np.log(y_val),np.log(y_pred_MLP)))))

X_test_scaled = scaler.transform(X_test)
y_pred_test = np.expm1(XGB_reg.predict(X_test_scaled))
df_csv = pd.DataFrame(data=test['Id'])

df_csv['SalePrice'] = y_pred_test

df_csv.head()
df_csv.to_csv('submission.csv',index=False)