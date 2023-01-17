
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualisation
import seaborn as sns 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor #Random forest libraries
from sklearn.model_selection import cross_validate #cross validation
from sklearn.impute import SimpleImputer           #Treatment of missing values
from sklearn.preprocessing import OrdinalEncoder   #Ordinal Encoder package
from sklearn.preprocessing import LabelEncoder     #For Label Encoding
from sklearn.metrics import mean_squared_log_error #Mean Squared Log Error metric from sklearn
import xgboost as xgb                              #XGboost package
from sklearn.model_selection import GridSearchCV   #Grid search for finding out the best package


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

testID = test["Id"]
train.head()
test.head()
train.info()
train.describe()
numcols=train._get_numeric_data().columns

for cols in numcols:
    plt.figure()
    sns.scatterplot(x = cols,y = 'SalePrice',data = train)

data_num = train[numcols]
plt.show()
train = train.drop(train[(train['LotFrontage']> 200)|(train['LotArea']> 100000)|(train['BsmtFinSF1']> 2000)|(train['BsmtFinSF2']> 1200)
                        | (train['1stFlrSF']> 3000)|(train['GrLivArea']> 4000)].index)
train.describe()
catcols = list(set(train.columns) - set(numcols))

#for cols in catcols:
#    plt.figure()
#    sns.violinplot(x = cols, y = 'SalePrice',data = train)


train.info()
test.info()
train['LotFrontage'].fillna(train['LotFrontage'].median(), inplace = True)
train['Alley'].fillna('None', inplace = True)
train['MasVnrType'].fillna('None',inplace = True)
train['MasVnrArea'].fillna(0,inplace = True)

train['BsmtExposure'].fillna('NA',inplace = True)

train['BsmtFinType1'].fillna('NA',inplace = True)
train['BsmtFinType2'].fillna('NA',inplace = True)
train['Electrical'].fillna('SBrkr',inplace = True)
train['GarageType'].fillna('Attchd',inplace = True)
train['GarageYrBlt'].fillna(2005,inplace = True)
train['GarageFinish'].fillna('NA',inplace = True)

train['Fence'].fillna('NA',inplace = True)
train['MiscFeature'].fillna('NA',inplace = True)


test['LotFrontage'].fillna(train['LotFrontage'].median(), inplace = True)
test['Alley'].fillna('None', inplace = True)
test['MasVnrType'].fillna('None',inplace = True)
test['MasVnrArea'].fillna(0,inplace = True)

test['BsmtExposure'].fillna('NA',inplace = True)

test['BsmtFinType1'].fillna('NA',inplace = True)
test['BsmtFinType2'].fillna('NA',inplace = True)
test['Electrical'].fillna('SBrkr',inplace = True)
test['GarageType'].fillna('Attchd',inplace = True)
test['GarageYrBlt'].fillna(2005,inplace = True)
test['GarageFinish'].fillna('NA',inplace = True)

test['Fence'].fillna('NA',inplace = True)
test['MiscFeature'].fillna('NA',inplace = True)

test['TotalBsmtSF'].fillna(0,inplace = True)
test['Functional'].fillna('Typ',inplace = True)
test['TotalBsmtSF'].fillna(0,inplace = True)
test['BsmtFullBath'].fillna(0,inplace = True)
test['BsmtHalfBath'].fillna(0,inplace = True)
test['Exterior1st'].fillna('VinylSd', inplace = True)
test['Exterior2nd'].fillna('VinylSd', inplace = True)
test['MSZoning'].fillna('RL',inplace = True)
test['SaleType'].fillna('WD',inplace = True)
test['Utilities'].fillna('AllPub',inplace = True)
test['BsmtFinSF1'].fillna(0,inplace = True)
test['BsmtFinSF2'].fillna(0,inplace = True)
test['BsmtUnfSF'].fillna(0,inplace = True)
test['GarageCars'].fillna(2,inplace = True)
test['GarageArea'].fillna(400,inplace = True)
train.isnull().sum().sort_values(ascending = False).head(10)
test.isnull().sum().sort_values(ascending = False).head(10)
mapping = {"NA":0, "Po": 1, "Fa":2, "TA": 3, "Gd": 4, "Ex": 5}

quality_cols = ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','GarageQual','GarageCond','PoolQC','FireplaceQu']



for cols in quality_cols:
    cols1 = cols + 'o'
    train[cols].fillna('NA',inplace = True)
    train[cols1] = train[cols].apply(lambda x: mapping[x])
    test[cols].fillna('NA',inplace = True)
    test[cols1] = test[cols].apply(lambda x: mapping[x])
train.info()
neighborhoods = train.groupby('Neighborhood')['SalePrice'].median().sort_values()
neighborhoods.head()
plt.figure(figsize = (20,10))
sns.barplot(neighborhoods.index,neighborhoods.values)
neighborhoods_mean = train.groupby('Neighborhood')['SalePrice'].mean().sort_values()

plt.figure(figsize = (20,10))
sns.barplot(neighborhoods_mean.index,neighborhoods_mean.values)
neighborhood = pd.DataFrame(neighborhoods)
neighborhood.loc[neighborhood.SalePrice >= 50000,'NFlag'] = 0
neighborhood.loc[neighborhood.SalePrice >= 150000,'NFlag'] = 1
neighborhood.loc[neighborhood.SalePrice >= 200000,'NFlag'] = 2
neighborhood.loc[neighborhood.SalePrice >= 250000,'NFlag'] = 3
#neighborhood['Neighborhood'] = neighborhood.index
neighborhood = neighborhood.drop(labels = ['SalePrice'], axis = 1)
neighborhood.reset_index()
train = pd.merge(train,neighborhood, left_on = 'Neighborhood', right_on = 'Neighborhood')
train.info()
test = pd.merge(test,neighborhood, left_on = 'Neighborhood', right_on = 'Neighborhood')
test.info()
def add_features(train):
    train['YrBuildSold'] = train['YrSold'] - train['YearBuilt']
    train['TotalArea'] = train['GrLivArea'] + train['TotalBsmtSF'] 
    train['AreabyRooms'] = train['GrLivArea']/train['TotRmsAbvGrd']
    train['GarageAreaCar'] = train['GarageArea']/train['GarageCars']
    train["TotalBath"] = train["FullBath"] + 0.5*train["HalfBath"] + train["BsmtFullBath"] + 0.5*train["BsmtHalfBath"]
    train['roomsbath'] = train['TotRmsAbvGrd']/train['TotalBath']
    
    train['GarageAreaCar'].fillna(0, inplace = True)
add_features(train)
add_features(test)
train['GarageAreaCar'].fillna(0, inplace = True)
test['GarageAreaCar'].fillna(0, inplace = True)
train['PricePerSF'] = train['SalePrice']/train['GrLivArea']

neighborhood_price = train.groupby('Neighborhood')['PricePerSF'].median().sort_values(ascending = False)
plt.figure(figsize = (20,10))
sns.barplot(neighborhood_price.index,neighborhood_price.values)
npriceSF = pd.DataFrame(neighborhood_price.reset_index())
npriceSF.rename(columns = {'PricePerSF': 'MedianPricePerSF'}, inplace = True)
train= pd.merge(train,npriceSF, left_on = 'Neighborhood', right_on = 'Neighborhood')
test = pd.merge(test,npriceSF, left_on = 'Neighborhood', right_on = 'Neighborhood')
train.drop(columns = ['PricePerSF'], inplace = True)

train.info()
numcols=train._get_numeric_data().columns
train_num=train.loc[:,numcols]

corr = train_num.corr(method='pearson')
df_all_corr = train.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
df_all_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
df_all_corr[df_all_corr['Feature 1'] == 'SalePrice']
k=corr.unstack().sort_values().drop_duplicates()
k[(k>0.8) | (k<-0.8)]
train_len = len(train)


train.drop(columns = quality_cols, axis = 1, inplace = True)
test.drop(columns = quality_cols, axis = 1, inplace = True)
y_train = np.log(train['SalePrice'])
train.drop(columns = ['SalePrice','Fireplaces'], axis = 1, inplace = True)
test.drop(columns = ['Fireplaces'], axis = 1, inplace = True)
train.describe()
test.info()
dataset = pd.concat([train,test],sort = True).reset_index(drop = True)
numcols = dataset._get_numeric_data().columns

catcols = list(set(train.columns) - set(numcols))

dataset = pd.get_dummies(dataset,catcols)

dataset.drop(columns = ['GarageCars','PoolArea','MSSubClass','YearBuilt'])
for col in numcols:
    dataset[col] = (dataset[col] - np.mean(dataset[col]))/np.std(dataset[col])
dataset.describe()
train_len = len(train)
x_train = dataset[:train_len]
x_competition = dataset[train_len:]
x_competition.sort_values('Id', inplace = True)
x_competition.drop(columns = ['Id'])
x_train.drop(columns = ['Id'])
X_train, X_test, Y_train, Y_test = train_test_split(x_train,y_train, test_size = 0.2, random_state = 1)

regressor = RandomForestRegressor(n_estimators=100, random_state=1,max_depth=10,max_features=10,min_samples_leaf=5)  
#feature_list=x_train.columns

# # fit the regressor with X and Y data 
model=regressor.fit(X_train,Y_train) 

#Make predictions on test dataset
pred_rf=model.predict(X_test)


#Calculate Cross Validation results
cv_results = cross_validate(regressor, X_train, Y_train, cv=5,scoring='r2')
sorted(cv_results.keys())

scoreOfModel = model.score(X_test, Y_test)
print("RSquared value for Model",scoreOfModel)

print("RMSLE for RFR Model",np.sqrt(mean_squared_log_error(np.exp(Y_test),np.exp(pred_rf) )))
param_grid = {
    'bootstrap': [True],
    'max_depth': [5, 3,7,10,15],
    'max_features': [5, 12,10,8],
  #  'max_leaf_nodes': [4, 8, 16,32],
    'min_samples_split': [5, 10, 8,3],
    'n_estimators': [100, 200,150,300]
}
rf = RandomForestRegressor()
grid_search_rf = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)
# Fit the grid search to the data
grid_search_rf.fit(X_train,Y_train)
grid_search_rf.best_params_

best_grid_rf = grid_search_rf.best_estimator_

pred_gs_rf=best_grid_rf.predict(X_test)

scoreOfModel = best_grid_rf.score(X_test, Y_test)
print("Rqusared for the Model",scoreOfModel)

pred_gs_rf=best_grid_rf.predict(X_test)

print("RMSLE for the Random Forest Model",np.sqrt(mean_squared_log_error(np.exp(Y_test),np.exp(pred_gs_rf) )))


actuals=np.exp(Y_test)
predictions=np.exp(pred_gs_rf)
residuals=predictions-actuals


plt.scatter(predictions, residuals)
plt.xlabel("Predicted SalePrice")
plt.ylabel("Residuals")
plt.show()
plt.scatter(predictions,actuals)
plt.xlabel("Predicted SalePrice")
plt.ylabel("Actual SalePrice")
plt.show()
param_grid = {
    'max_depth': [8],
    'learning_rate': [0.05, 0.1],
    'colsample_bytree': [0.7],
    'lambda':[0.75],
    'max_leaves':[4,8,16],
    'min_child_weight':[2,4,6],
    'subsample': [0.7, 0.8],
    'n_estimators': [150,250]
}
xg = xgb.XGBRegressor()
grid_search_xg = GridSearchCV(estimator = xg, param_grid = param_grid, 
                          cv = 2, n_jobs = -1, verbose = 2)
grid_search_xg.fit(X_train,Y_train)
best_grid_xg=grid_search_xg.best_params_



best_estimator_xg = grid_search_xg.best_estimator_
pred_gs_xg = best_estimator_xg.predict(X_test)
print("RMSLE for the xgboost Model",np.sqrt(mean_squared_log_error(np.exp(Y_test),np.exp(pred_gs_xg) )))
print(grid_search_xg.best_params_)
from sklearn.linear_model import Lasso, LassoCV

lasso = Lasso(max_iter = 10000)

parameters = {'alpha': [  1e-3, 1e-2, 1e-1]}

lasso_regressor=GridSearchCV(lasso,parameters,scoring='r2',cv=5)
lasso_regressor.fit(X_train,Y_train)

print(lasso_regressor.best_params_)

best_estimator_lr = lasso_regressor.best_estimator_

pred_gs_lr = best_estimator_lr.predict(X_test)
print("RMSLE for the Lasso Model",np.sqrt(mean_squared_log_error(np.exp(Y_test),np.exp(pred_gs_lr) )))
from sklearn.linear_model import Ridge, RidgeCV

ridge = Ridge(max_iter = 10000)

parameters = {'alpha': [ 1e-1, 3e-1, 1, 3, 5, 10, 30, 50, 100]}

ridge_regressor=GridSearchCV(ridge,parameters,scoring='r2',cv=5)
ridge_regressor.fit(X_train,Y_train)

print(ridge_regressor.best_params_)

best_estimator_rr = ridge_regressor.best_estimator_

pred_gs_rr = best_estimator_rr.predict(X_test)
print("RMSLE for the ridge Model",np.sqrt(mean_squared_log_error(np.exp(Y_test),np.exp(pred_gs_rr) )))

from sklearn.linear_model import ElasticNet, ElasticNetCV

elastic = ElasticNet(max_iter = 10000)

parameters = {'alpha': [1e-3,1e-2,1e-1,1,3,5,10],
              'l1_ratio': [0.1,0.3,0.5,0.7,0.9]
             }

elasticnet_regressor = GridSearchCV(elastic,parameters,scoring = 'r2',cv=5)
elasticnet_regressor.fit(X_train,Y_train)

print(elasticnet_regressor.best_params_)

best_estimator_enr = elasticnet_regressor.best_estimator_

pred_gs_enr = best_estimator_enr.predict(X_test)
print("RMSLE for the elasticnet Model",np.sqrt(mean_squared_log_error(np.exp(Y_test),np.exp(pred_gs_enr) )))
pred_agg = (pred_gs_rr + pred_gs_lr + pred_gs_xg + pred_gs_enr)/4
print("RMSLE for Mean Values",np.sqrt(mean_squared_log_error(np.exp(Y_test),np.exp(pred_agg) )))



best_estimator_xg = grid_search_xg.best_estimator_.fit(x_train,y_train)
best_estimator_lr = lasso_regressor.best_estimator_.fit(x_train,y_train)
best_estimator_rr = ridge_regressor.best_estimator_.fit(x_train,y_train)
best_estimator_enr = elasticnet_regressor.best_estimator_.fit(x_train,y_train)


import tensorflow as tf
from tensorflow.keras import layers
x_train.isnull().sum().sort_values(ascending= False)
model = tf.keras.Sequential([
    layers.Dense(128, activation = 'relu', input_shape=[len(x_train.keys())]),
    #layers.Dropout(0.15),
    #layers.Dense(64, activation='relu'),
    #layers.Dropout(0.2),
    #layers.Dense(32, activation='relu'),
    layers.Dense(1),
    
  ])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

model.compile(loss='mse',
                optimizer=optimizer,
                metrics=[tf.keras.metrics.MeanSquaredLogarithmicError()])
history = model.fit(
  x_train,y_train,
  epochs=100, validation_split = 0.1, verbose=1)
acc = np.sqrt(history.history['mean_squared_logarithmic_error'])
val_acc = np.sqrt(history.history['val_mean_squared_logarithmic_error'])
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training Mae')
plt.plot(epochs, val_acc, 'b', label='Validation Mae')
plt.title('Training and validation Mae')
plt.legend(loc=0)

plt.ylim(0,.2)

plt.show()
pred_nn = model.predict(x_competition)
#nn_output = np.exp(pred_agg)
out = pred_nn.reshape([-1])
prediction_nn = pd.Series(np.exp(out), name = 'SalePrice')

results_nn = pd.concat([testID,prediction_nn],axis=1)

results_nn.to_csv("prediction_nn.csv",index=False)
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
testID = test["Id"]


len(test)
pred_rr = best_estimator_rr.predict(x_competition)
pred_xg = best_estimator_xg.predict(x_competition)
pred_lr = best_estimator_lr.predict(x_competition)
pred_enr = best_estimator_enr.predict(x_competition)

pred_agg = np.mean([pred_rr,pred_xg,pred_lr], axis = 0)
len(pred_agg)

prediction = pd.Series(np.exp(pred_agg), name = 'SalePrice')

results = pd.concat([testID,prediction],axis=1)

results.to_csv("prediction3.csv",index=False)
results.info()
x_competition.to_csv('testing.csv', index = False)