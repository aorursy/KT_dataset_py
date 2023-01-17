import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.shape
train.head()
train.SalePrice.describe()
train.info()
train["LotFrontage"]=train["LotFrontage"].fillna(train["LotFrontage"].mean())
train.drop(["Alley"],axis=1,inplace=True)
train['BsmtCond'].mode()[0]
train['BsmtCond']=train['BsmtCond'].fillna(train['BsmtCond'].mode()[0])
train['BsmtQual']=train['BsmtQual'].fillna(train['BsmtQual'].mode()[0])
train['FireplaceQu'].mode()[0]
train['FireplaceQu']=train['FireplaceQu'].fillna(train['FireplaceQu'].mode()[0])
train['GarageType']=train['GarageType'].fillna(train['GarageType'].mode()[0])
train.drop(['GarageYrBlt'],axis=1,inplace=True)

train['GarageFinish']=train['GarageFinish'].fillna(train['GarageFinish'].mode()[0])
train['GarageQual']=train['GarageQual'].fillna(train['GarageQual'].mode()[0])
train['GarageCond']=train['GarageCond'].fillna(train['GarageCond'].mode()[0])
train.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
train.shape
train.drop(['Id'],axis=1,inplace=True)
train.isnull().sum()
train['MasVnrType']=train['MasVnrType'].fillna(train['MasVnrType'].mode()[0])
train['MasVnrArea']=train['MasVnrArea'].fillna(train['MasVnrArea'].mode()[0])
train['BsmtExposure']=train['BsmtExposure'].fillna(train['BsmtExposure'].mode()[0])
train['BsmtFinType2']=train['BsmtFinType2'].fillna(train['BsmtFinType2'].mode()[0])
train.dropna(inplace=True)
train.shape
train.head()
columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
         'Condition2','BldgType','Condition1','HouseStyle','SaleType',
        'SaleCondition','ExterCond',
         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',
         'CentralAir',
         'Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']
len(columns)
def category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:
        
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final
main_df=train.copy()
test.shape
test.head()
test.isnull().sum()
final_df=pd.concat([train,test],axis=0)
final_df
final_df['SalePrice']
final_df.shape
final_df=category_onehot_multcols(columns)
final_df.shape
final_df =final_df.loc[:,~final_df.columns.duplicated()]
final_df.shape
final_df
df_Train=final_df.iloc[:1422,:]
df_Test=final_df.iloc[1422:,:]
df_Train.head()
df_Test.head()
df_Train.shape

X_train=df_Train.drop(['SalePrice'],axis=1)
y_train=df_Train['SalePrice']
!pip install xgboost
import xgboost
classifier=xgboost.XGBRegressor()
regressor=xgboost.XGBRegressor()
booster=['gbtree','gblinear']
base_score=[0.25,0.5,0.75,1]
n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 10, 15]
booster=['gbtree','gblinear']
learning_rate=[0.05,0.1,0.15,0.20]
min_child_weight=[1,2,3,4]
hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster,
    'base_score':base_score
    }
from sklearn.model_selection import RandomizedSearchCV
random_cv = RandomizedSearchCV(estimator=regressor,
            param_distributions=hyperparameter_grid,
            cv=5, n_iter=50,
            scoring = 'neg_mean_absolute_error',n_jobs = 4,
            verbose = 5, 
            return_train_score = True,
            random_state=42)
X_train.drop(['Alley'],axis=1,inplace=True)
X_train.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
y_train=y_train.dropna(axis=0)
X_train.shape
y_train.shape
random_cv.fit(X_train,y_train)
random_cv.best_estimator_
random_cv.best_estimator_
regressor=xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=2, min_child_weight=1, missing=None, n_estimators=900,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)
regressor.fit(X_train,y_train)

import pickle
filename = 'finalized_model.pkl'
pickle.dump(classifier, open(filename, 'wb'))
df_Test=df_Test.drop(['SalePrice'],axis=1)
df_Test.shape
df_Test.head()
df_Test.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
y_pred=regressor.predict(df_Test)
y_pred

temp_df=df_Train['SalePrice'].copy()
temp_df.column=['SalePrice']
df_Train.drop(['SalePrice'],axis=1,inplace=True)
df_Train=pd.concat([df_Train,temp_df],axis=1)
df_Test.head()
df_Train=pd.concat([df_Train,df_Test],axis=0)
df_Train.shape
X_train=df_Train.drop(['SalePrice'],axis=1)
y_train=df_Train['SalePrice']
!pip install keras
!pip install tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim= 50, init = 'he_uniform',activation='relu',input_dim = 174))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 25, init = 'he_uniform',activation='relu'))

# Adding the third hidden layer
classifier.add(Dense(output_dim = 50, init = 'he_uniform',activation='relu'))
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'he_uniform'))

# Compiling the ANN
classifier.compile(loss=root_mean_squared_error, optimizer='Adamax')

# Fitting the ANN to the Training set
model_history=classifier.fit(X_train.values, y_train.values,validation_split=0.20, batch_size = 10, nb_epoch = 1000)
ann_pred=classifier.predict(df_Test.drop(['SalePrice'],axis=1).values)
pred=pd.DataFrame(ann_pred)
sub_df=pd.read_csv('sample_submission.csv')
datasets=pd.concat([sub_df['Id'],pred],axis=1)
datasets.columns=['Id','SalePrice']
datasets.to_csv('sample_submission.csv',index=False)
pred.columns=['SalePrice']
df_Test=pd.concat([df_Test,pred],axis=1)

from keras import backend as K
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
k=root_mean_squared_error(y_true, y_pred)
print(k)