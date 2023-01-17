import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_test =  pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
df_sample_submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

print(df_train.shape)
print(df_test.shape)
print(df_sample_submission.shape)
# df_train.head(2)
# df_test.head(2)
# df_sample_submission.head(2)
df_train.columns
df_train.dtypes.value_counts()
plt.figure(figsize=(12,8))
sns.distplot(df_train['SalePrice'])
# .corr() = compute pairwise correlation of columns, excluding NA/null values
# int64      35
# float64     3

cm = df_train.corr()

plt.subplots(figsize=(12,8))
sns.heatmap(cm)
cm.nlargest(15, 'SalePrice')['SalePrice']
f = plt.subplots(figsize=(12,6))
sns.boxplot(x='OverallQual',y='SalePrice',data=df_train)
f = plt.subplots(figsize=(12,6))
sns.scatterplot(x='GrLivArea',y='SalePrice',data=df_train)
f = plt.subplots(figsize=(12,6))
sns.boxplot(x='GarageCars',y='SalePrice',data=df_train)
f = plt.subplots(figsize=(12,6))
sns.scatterplot(x='GarageArea',y='SalePrice',data=df_train)
f = plt.subplots(figsize=(12,6))
sns.scatterplot(x='TotalBsmtSF',y='SalePrice',data=df_train)
f = plt.subplots(figsize=(12,6))
sns.scatterplot(x='1stFlrSF',y='SalePrice',data=df_train)
f = plt.subplots(figsize=(15,8))
sns.boxplot(x='YearBuilt',y='SalePrice',data=df_train)
plt.figure(figsize=(10,8))
sns.heatmap(df_train.isnull())
# .isnull() gives back the same df but with booleans (True if NAN)
# .sum() sums up all the True's per column 

df_train.isnull().sum().sort_values(ascending=False).head(21)
percentage_missing = (df_train.isnull().sum())/(1460)
percentage_missing.sort_values(ascending=False).head(10)
type((df_train['LotFrontage'].iloc[1]))
sns.scatterplot(df_train['LotFrontage'],df_train['SalePrice'])
cm = df_train.corr()
cm.nlargest(5, 'LotFrontage')['LotFrontage']
sns.regplot(df_train['1stFlrSF'],df_train['LotFrontage'])
from pylab import *
from scipy import stats
the_x = df_train['1stFlrSF'].to_numpy()
the_y = df_train['LotFrontage'].to_numpy()
mask = ~np.isnan(the_x) & ~np.isnan(the_y)

slope, intercept, r_value, p_value, std_err = stats.linregress(the_x[mask], the_y[mask])

print('slope = ' + str(slope))
print('intercept = '+ str(intercept))
def predict_LotFrontage(x):
    return slope * x + intercept

fitLine = predict_LotFrontage(the_x)

plt.scatter(the_x,the_y)
plt.plot(the_x, fitLine, c='r')
plt.xlabel('1stFlrSF')
plt.ylabel('LotFrontage')
plt.show()
def replace_nans_LotFrontage(a):
    
    first_floor_sf = a[0]
    lot_frontage = a[1]
    
    if pd.isnull(lot_frontage):
        return predict_LotFrontage(first_floor_sf)    
    else:
        return lot_frontage
df_train['LotFrontage'] = df_train[['1stFlrSF','LotFrontage']].apply(replace_nans_LotFrontage,axis=1)
df_test['LotFrontage'] = df_test[['1stFlrSF','LotFrontage']].apply(replace_nans_LotFrontage,axis=1)
df_train['LotFrontage'].isnull().sum()
df_test['LotFrontage'].isnull().sum()
sns.regplot(df_train['1stFlrSF'],df_train['LotFrontage'])
the_x = df_train['1stFlrSF'].to_numpy()
the_y = df_train['LotFrontage'].to_numpy()
mask = ~np.isnan(the_x) & ~np.isnan(the_y)

slope, intercept, r_value, p_value, std_err = stats.linregress(the_x[mask], the_y[mask])

print('slope = ' + str(slope))
print('intercept = '+ str(intercept))
percentage_missing = (df_train.isnull().sum())/(1460)
percentage_missing.sort_values(ascending=False).head(10)
cm = df_train.corr()
cm.nlargest(5, 'GarageYrBlt')['GarageYrBlt']
sns.scatterplot(df_train['YearBuilt'],df_train['GarageYrBlt'])
sns.regplot(df_train['YearBuilt'],df_train['GarageYrBlt'])
the_x = df_train['YearBuilt'].to_numpy()
the_y = df_train['GarageYrBlt'].to_numpy()
mask = ~np.isnan(the_x) & ~np.isnan(the_y)

slope, intercept, r_value, p_value, std_err = stats.linregress(the_x[mask], the_y[mask])

print('slope = ' + str(slope))
print('intercept = '+ str(intercept))
def predict_GarageYrBlt(x):
    return slope * x + intercept

fitLine = predict_LotFrontage(the_x)

plt.scatter(the_x,the_y)
plt.plot(the_x, fitLine, c='r')
plt.xlabel('YearBuilt')
plt.ylabel('GarageYrBlt')
plt.show()
def replace_nans_GarageYrBlt(a):
    
    YearBuilt = a[0]
    GarageYrBlt = a[1]
    
    if pd.isnull(GarageYrBlt):
        return predict_GarageYrBlt(YearBuilt)    
    else:
        return GarageYrBlt
df_train['GarageYrBlt'] = df_train[['YearBuilt','GarageYrBlt']].apply(replace_nans_GarageYrBlt,axis=1)
df_test['GarageYrBlt'] = df_test[['YearBuilt','GarageYrBlt']].apply(replace_nans_GarageYrBlt,axis=1)
df_train['GarageYrBlt'].isnull().sum()
df_test['GarageYrBlt'].isnull().sum()
the_x = df_train['YearBuilt'].to_numpy()
the_y = df_train['GarageYrBlt'].to_numpy()
mask = ~np.isnan(the_x) & ~np.isnan(the_y)

slope, intercept, r_value, p_value, std_err = stats.linregress(the_x[mask], the_y[mask])

print('slope = ' + str(slope))
print('intercept = '+ str(intercept))
percentage_missing = (df_train.isnull().sum())/(1460)
percentage_missing.sort_values(ascending=False).head(10)
cm = df_train.corr()
cm.nlargest(15, 'SalePrice')['SalePrice']
plt.figure(figsize=(10,8))
sns.heatmap(df_train.isnull())
still_missing = (df_train.isnull().sum())/(1460)
still_missing[still_missing>0].sort_values(ascending=False)
columns_missing_data = pd.DataFrame(still_missing[still_missing>0])
columns_missing_data.index.name = 'predictor'
print(df_train.shape)
print(df_test.shape)
df_train = df_train.drop((columns_missing_data.index),axis=1)
df_test = df_test.drop((columns_missing_data.index),axis=1)
print(df_train.shape)
print(df_test.shape)
plt.figure(figsize=(10,8))
sns.heatmap(df_train.isnull())
df_train.isnull().sum().max()
plt.figure(figsize=(10,8))
sns.heatmap(df_test.isnull())
df_test[df_test['MSZoning'].isnull()]
print(df_test['MSZoning'].value_counts())
print(df_test['MSZoning'].isnull().sum())
#Impute the values using scikit-learn SimpleImpute Class

from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer( strategy='most_frequent')

trial_array = df_test['MSZoning'].values.reshape(-1, 1)

# Fit the imputer on all the MSZoning categorical values
imp_mean.fit(trial_array)
# Impute all missing values in the MSZoning column with the most_frequent categorical feature
df_test['MSZoning'] = imp_mean.transform(trial_array)
print(df_test['MSZoning'].value_counts())
print(df_test['MSZoning'].isnull().sum())
#Do this for all the other categorical features

cat_columns = ['Utilities','Exterior1st','Exterior2nd','KitchenQual','Functional','SaleType']

for column in cat_columns:
    
    # Fit the imputer 
    imp_mean.fit(df_test[str(column)].values.reshape(-1, 1))
    # Impute all missing values with the most_frequent categorical feature in each respective column
    df_test[str(column)] = imp_mean.transform(df_test[str(column)].values.reshape(-1, 1))
plt.figure(figsize=(10,8))
sns.heatmap(df_test.isnull())
#Might as wel do the same for the numerical columns.. Since so few datapoints are missing..

num_columns = ['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageCars','GarageArea']

for column in num_columns:
    
    # Fit the imputer 
    imp_mean.fit(df_test[str(column)].values.reshape(-1, 1))
    # Impute all missing values the most_frequent value in each respective column
    df_test[str(column)] = imp_mean.transform(df_test[str(column)].values.reshape(-1, 1))


plt.figure(figsize=(10,8))
sns.heatmap(df_test.isnull())
df_test.shape
cm = df_train.corr()
cm.nlargest(6,'SalePrice')['SalePrice']
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize = (15, 10))
sns.scatterplot(df_train['OverallQual'],df_train['SalePrice'],ax=ax1)
sns.distplot(df_train['OverallQual'],ax=ax2)
sns.scatterplot(df_train['GrLivArea'],df_train['SalePrice'],ax=ax3)
sns.distplot(df_train['GrLivArea'],ax=ax4)
#decided to remove all 4 points>4000

df_train[df_train['GrLivArea']>4000]
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
df_train = df_train.drop(df_train[df_train['Id'] == 692].index)
df_train = df_train.drop(df_train[df_train['Id'] == 1183].index)
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train[df_train['GrLivArea']>4000]
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize = (15, 10))
sns.scatterplot(df_train['OverallQual'],df_train['SalePrice'],ax=ax1)
sns.distplot(df_train['OverallQual'],ax=ax2)
sns.scatterplot(df_train['GrLivArea'],df_train['SalePrice'],ax=ax3)
sns.distplot(df_train['GrLivArea'],ax=ax4)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize = (15, 10))
sns.scatterplot(df_train['GarageCars'],df_train['SalePrice'],ax=ax1)
sns.distplot(df_train['GarageCars'],ax=ax2)
sns.scatterplot(df_train['GarageArea'],df_train['SalePrice'],ax=ax3)
sns.distplot(df_train['GarageArea'],ax=ax4)

df_train[df_train['GarageArea']>1220]
df_train = df_train.drop(df_train[df_train['Id'] == 582].index)
df_train = df_train.drop(df_train[df_train['Id'] == 1062].index)
df_train = df_train.drop(df_train[df_train['Id'] == 1191].index)
df_train[df_train['GarageArea']>1220]
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize = (15, 10))
sns.scatterplot(df_train['GarageCars'],df_train['SalePrice'],ax=ax1)
sns.distplot(df_train['GarageCars'],ax=ax2)
sns.scatterplot(df_train['GarageArea'],df_train['SalePrice'],ax=ax3)
sns.distplot(df_train['GarageArea'],ax=ax4)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize = (15, 5))
sns.scatterplot(df_train['TotalBsmtSF'],df_train['SalePrice'],ax=ax1)
sns.distplot(df_train['TotalBsmtSF'],ax=ax2)
df_train[df_train['TotalBsmtSF']>2500]
df_train = df_train.drop(df_train[df_train['Id']==333].index)
df_train = df_train.drop(df_train[df_train['Id']==441].index)
df_train = df_train.drop(df_train[df_train['Id']==497].index)
df_train = df_train.drop(df_train[df_train['Id']==1045].index)
df_train = df_train.drop(df_train[df_train['Id']==1374].index)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize = (15, 5))
sns.scatterplot(df_train['TotalBsmtSF'],df_train['SalePrice'],ax=ax1)
sns.distplot(df_train['TotalBsmtSF'],ax=ax2)
df_train.dtypes.value_counts()
df_train.columns.to_series().groupby(df_train.dtypes).groups
df_test.columns.to_series().groupby(df_test.dtypes).groups
print(df_train.isnull().sum().max())
print(df_test.isnull().sum().max())
print(df_train.shape)
print(df_test.shape)
all_data = pd.concat((df_train,df_test))

for column in all_data.select_dtypes(include=[np.object]).columns:

    print(column, all_data[column].unique())
from pandas.api.types import CategoricalDtype
# The following was not working:

    # all_data = pd.concat((df_train,df_test))

    # for column in all_data.select_dtypes(include=[np.object]).columns:
    #     df_train[column] = df_train[column].astype('category', categories = all_data[column].unique())
    #     df_test[column] = df_test[column].astype('category', categories = all_data[column].unique())

# Workaround: https://stackoverflow.com/questions/37952128/pandas-astype-categories-not-working
all_data = pd.concat((df_train,df_test))

for column in all_data.select_dtypes(include=[np.object]).columns:
    df_train[column] = df_train[column].astype(CategoricalDtype(categories = all_data[column].unique()))
    df_test[column] = df_test[column].astype(CategoricalDtype(categories = all_data[column].unique()))
df_train = pd.get_dummies(df_train)
df_test = pd.get_dummies(df_test) 
print(df_train.shape)
print(df_test.shape)
X = df_train.drop('SalePrice',axis=1)
y = df_train['SalePrice']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
from sklearn.linear_model import LinearRegression
lrm = LinearRegression(normalize=False)
lrm.fit(X_train,y_train)
coeff_df = pd.DataFrame(lrm.coef_,X.columns,columns=['Coefficient'])
coeff_df.sort_values(by='Coefficient', ascending=False)

#Interpreting the coefficients:
#Holding all other features fixed, a 1 unit increase in a certain predictor is associated with an increase/decrease of 'Coefficient' dollar
predictions = lrm.predict(X_test)
plt.figure(figsize=(10,8))

#predictions
ax = sns.scatterplot(y_test,predictions)
ax.set(xlabel="y_test", ylabel = "predictions")

#perfect predictions
plt.plot(y_test,y_test,'-r')
plt.figure(figsize=(10,8))
sns.distplot((y_test-predictions))
plt.xlabel("Prediction Error [Dollar]")
_ = plt.ylabel("Count")

residuals_sklearn = (y_test-predictions)
print('1σ sklearn = '+str(np.std(residuals_sklearn)))
print('2σ sklearn = '+str(2*np.std(residuals_sklearn)))
print('mean sklearn  = '+str(np.mean(residuals_sklearn)))
print('median sklearn = '+str(np.median(residuals_sklearn)))
from sklearn import metrics
mae_sklearn =metrics.mean_absolute_error(y_test,predictions)
mse_sklearn =metrics.mean_squared_error(y_test,predictions) 
rmse_sklearn =np.sqrt(metrics.mean_squared_error(y_test,predictions))
print('MAE sklearn:',mae_sklearn)
print('MSE sklearn:',mse_sklearn)
print('RMSE sklearn:',rmse_sklearn)
predictions_scikit = lrm.predict(df_test)
predictions_scikit
print(type(predictions_scikit))
predictions_scikit.shape
Ids = df_test['Id'].values.reshape(1459,1)
Ids_scikit = df_test['Id'].values.reshape(1459,1)
submission_scikit = pd.DataFrame(data={'Id':Ids[0:,0], 'SalePrice':predictions_scikit[0:]}, index=Ids_scikit[0:,0])
submission_scikit
submission_scikit.to_csv(r'submission_sklearn_house_prices.csv')
plt.figure(figsize=(15,6))
sns.scatterplot(x=submission_scikit['Id'],y=submission_scikit['SalePrice'])
X = df_train.drop('SalePrice',axis=1).values #.values because TF may complain. TF can't work with Pandas series or DF. It passes a numeric array.  
y = df_train['SalePrice'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=101)
from sklearn.preprocessing import MinMaxScaler

# Transform features by scaling each feature to a given range.
# This estimator scales and translates each feature individually such that it is in the given range on the training set, e.g. between zero and one.
# The transformation is given by:
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train) #fit and transform in one step possible with MinMaxScaler
# fit = Compute the minimum and maximum to be used for later scaling.
# transform = Scale features of X according to feature_range (default feature_range=(0, 1))

X_test = scaler.transform(X_test) #don't fit, because we don't want to assume prior info about the testset
print(X_train.shape)
print(X_test.shape)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
model_all_data = Sequential()
model_all_data.add(Dense(217,activation='relu')) # (217 features in training data)
model_all_data.add(Dense(109,activation='relu'))
model_all_data.add(Dense(50,activation='relu'))
model_all_data.add(Dense(25,activation='relu'))
model_all_data.add(Dense(12,activation='relu'))
model_all_data.add(Dense(1)) #output neuron = predicted price
model_all_data.compile(optimizer='adam',loss='mse')
model_all_data.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),batch_size=128,epochs=300) 

#validation data:
#after each epoch we will check our loss on the testdata 
#like this we can see how well the model perform on our test data and training data
#weights and biases are not affected by the test data! (Keras is not going to update your model based on test data)
#In this way we can see if we are overfitting at some point (when validation loss starts to increase)


#batch size:
#The smaller the batch size, the longer the training is going to take, but the less likely you are going to overfit to the data.
#Instead you are passing in these small batches with all different predictor situations.

#epoch:
#an arbitrary cutoff, generally defined as "one pass over the entire dataset",
#used to separate training into distinct phases, which is useful for logging and periodic evaluation.
#Compare training vs. test performance
losses = pd.DataFrame(model_all_data.history.history)
losses.plot(figsize=(12,6))
plt.xlabel("Epoch")
_ = plt.ylabel("Loss")
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
predictions = model_all_data.predict(X_test)

plt.figure(figsize=[12,8])
# The predictions
plt.scatter(y_test,predictions)
plt.xlabel("y_test")
_ = plt.ylabel("Predictions")

# Perfect predictions
plt.plot(y_test,y_test,'r') 
mae_tf =metrics.mean_absolute_error(y_test,predictions)
mse_tf =metrics.mean_squared_error(y_test,predictions) 
rmse_tf =np.sqrt(metrics.mean_squared_error(y_test,predictions))
print('MAE tf:',mae_tf)
print('MSE tf:',mse_tf)
print('RMSE tf:',rmse_tf )
print('MAE improvement of tf wrt sklearn: '+str(mae_sklearn-mae_tf))
print('MSE improvement of tf wrt sklearn: '+str(mse_sklearn-mse_tf))
print('RMSE improvement of tf wrt sklearn: '+str(rmse_sklearn-rmse_tf))
print('TF is: ~'+str(rmse_sklearn-rmse_tf) +' dollars less wrong in predicting')
#Explained variance regression score function
#Best possible score is 1.0, lower values are worse.

explained_variance_score(y_test,predictions)
plt.figure(figsize=(10,8))
sns.distplot((y_test-predictions))
plt.xlabel("Prediction Error [dollar]")
_ = plt.ylabel("Count")

residuals = y_test-predictions
print('1σ tf = '+str(np.std(residuals)))
print('2σ tf = '+str(2*np.std(residuals)))
print('mean tf = '+str(np.mean(residuals)))
print('median tf = '+str(np.median(residuals)))
print('1σ sklearn = '+str(np.std(residuals_sklearn)))
print('2σ sklearn = '+str(2*np.std(residuals_sklearn)))
print('mean sklearn  = '+str(np.mean(residuals_sklearn)))
print('median sklearn = '+str(np.median(residuals_sklearn)))
test_df = df_train.drop('SalePrice',axis=1)
test_df.iloc[0]
single_house = test_df.iloc[0]
single_house = scaler.transform(single_house.values.reshape(-1,217))
model_all_data.predict(single_house)
df_train['SalePrice'].head(1)
print('Prediction is: ' +str((model_all_data.predict(single_house))-(df_train['SalePrice'].iloc[0]))+' dollar off from the real sale price')
scaler = MinMaxScaler()
df_test_final = scaler.fit_transform(df_test) 
predictions_all_data_df_test = model_all_data.predict(df_test_final)
submission_all_data_df_test = pd.DataFrame(data={'Id':Ids[0:,0], 'SalePrice':predictions_all_data_df_test[0:,0]}, index=Ids[0:,0])
plt.figure(figsize=(15,6))
sns.scatterplot(x=submission_all_data_df_test['Id'],y=submission_all_data_df_test['SalePrice'],color='g')

submission_all_data_df_test.to_csv(r'submission_all_data_TF_house_prices.csv')
plt.figure(figsize=(10,8))
sns.distplot(df_train['SalePrice'])
df_train_reduced = df_train[df_train['SalePrice']<350000]
df_train_reduced.shape
X = df_train_reduced.drop('SalePrice',axis=1).values #.values because TF may complain. TF can't work with Pandas series or DF. It passes a numeric array.  
y = df_train_reduced['SalePrice'].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=101)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train) 

X_test = scaler.transform(X_test) #don't fit, because we don't want to assume prior info about the testset

print(X_train.shape)
print(X_test.shape)
model = Sequential()
model.add(Dense(217,activation='relu')) # (217 features in training data)
model.add(Dense(109,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(25,activation='relu'))
model.add(Dense(12,activation='relu'))
model.add(Dense(1)) #output neuron = predicted price
model.compile(optimizer='adam',loss='mse')
model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),batch_size=128,epochs=300) 


#Compare training vs. test performance
losses = pd.DataFrame(model.history.history)
losses.plot(figsize=(12,6))
plt.xlabel("Epoch")
_ = plt.ylabel("Loss")
predictions = model.predict(X_test)
plt.figure(figsize=[12,8])
# The predictions
plt.scatter(y_test,predictions)
plt.xlabel("y_test")
_ = plt.ylabel("Predictions")

# Perfect predictions
plt.plot(y_test,y_test,'r') 
explained_variance_score(y_test,predictions)
residuals_tf_ted = y_test-predictions
print('1σ tf red = '+str(np.std(residuals_tf_ted)))
print('2σ tf red = '+str(2*np.std(residuals_tf_ted)))
print('mean tf red = '+str(np.mean(residuals_tf_ted)))
print('median tf red = '+str(np.median(residuals_tf_ted)))
plt.figure(figsize=(10,8))
sns.distplot((y_test-predictions))
plt.xlabel("Prediction Error [dollar]")
_ = plt.ylabel("Count")
mae_tf_red =metrics.mean_absolute_error(y_test,predictions)
mse_tf_red =metrics.mean_squared_error(y_test,predictions) 
rmse_tf_red =np.sqrt(metrics.mean_squared_error(y_test,predictions))
print('MAE tf red:',mae_tf_red)
print('MSE tf red:',mse_tf_red)
print('RMSE tf red:',rmse_tf_red )
# RMSE sklearn: 28351.633756075065
# RMSE tf: 23625.95715406359
# RMSE tf reduced: 20810.27072576839
df_test.shape
df_test['Id']
df_test.head()
scaler = MinMaxScaler()
df_test_final = scaler.fit_transform(df_test) 
final_predictions = model.predict(df_test_final)
print(type(final_predictions))
print(final_predictions.shape)
Ids = df_test['Id'].values.reshape(1459,1)
print(type(Ids))
print(Ids.shape)
submission = pd.DataFrame(data={'Id':Ids[0:,0], 'SalePrice':final_predictions[0:,0]}, index=Ids[0:,0])
import os
os.getcwd()
submission
submission.to_csv(r'submission_house_prices.csv')
plt.figure(figsize=(15,6))
sns.scatterplot(x=submission['Id'],y=submission['SalePrice'])
plt.figure(figsize=(15,6))
sns.scatterplot(x=submission_scikit['Id'],y=submission_scikit['SalePrice']) #sklearn
sns.scatterplot(x=submission['Id'],y=submission['SalePrice'],color='r') #TF
plt.figure(figsize=(15,6))
sns.scatterplot(x=submission_all_data_df_test['Id'],y=submission_all_data_df_test['SalePrice'],color='g')
sns.scatterplot(x=submission['Id'],y=submission['SalePrice'],color='r')
# np.sqrt(np.mean((y-y_pred)**2))