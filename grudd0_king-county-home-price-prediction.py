import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb
import time
df = pd.read_csv('../input/housesalesprediction/kc_house_data.csv')
df.head()
df.describe()
zips = df['zipcode'].unique()
zips.shape
fig = plt.figure(figsize=(15,6))
df_98023 = df[df['zipcode']==98023]
df_98198 = df[df['zipcode']==98198]
plt.boxplot([df_98023['price'],df_98198['price']])
plt.xlabel('zipcode')
plt.ylabel('Price dollars');
fig = plt.figure(figsize=(10,6))
bins = np.linspace(0,2000000,20)
plt.hist(df['price'],bins=bins,color='skyblue', edgecolor='gray',linewidth=2)
plt.xlabel('Price Dollars')
plt.ylabel('count')
ax = plt.gca()
ax.set_facecolor('lightgray')
ax.tick_params(direction='out', length=6, width=2, colors='black',grid_color='gray', grid_alpha=0.5,left=True,bottom=True)
#plt.grid(color='lightgray', linestyle='-', linewidth=2)
x_ticks = np.linspace(0,2000000,20)
x_labels = ['0','','','','','500000','','','','','1000000','','','','','1500000','','','','','2000000','']
plt.xticks(x_ticks,x_labels);
plt.xlim([0,2000000]);
# calculate correlation coefficient and plot as heatmap
cols = ['price', 'bedrooms','bathrooms','sqft_lot','sqft_living','floors']

import numpy as np
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                cbar=True,
                annot=True,
                square = True,
                fmt='.2f',
                annot_kws={'size':15},
                yticklabels=cols,
                xticklabels=cols)
plt.show()
cols = ['price','waterfront','view','condition','grade','sqft_above']

import numpy as np
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                cbar=True,
                annot=True,
                square = True,
                fmt='.2f',
                annot_kws={'size':15},
                yticklabels=cols,
                xticklabels=cols)
plt.show()
cols = ['price','sqft_basement','yr_built','yr_renovated','zipcode','lat']

import numpy as np
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                cbar=True,
                annot=True,
                square = True,
                fmt='.2f',
                annot_kws={'size':15},
                yticklabels=cols,
                xticklabels=cols)
plt.show()
cols = ['price','long','sqft_living15','sqft_lot15']

import numpy as np
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                cbar=True,
                annot=True,
                square = True,
                fmt='.2f',
                annot_kws={'size':15},
                yticklabels=cols,
                xticklabels=cols)
plt.show()
df_pred = df[['price','bathrooms', 'sqft_living','grade','sqft_above']]
df_pred.head()
fig = plt.figure(figsize=(10,6))
plt.scatter(df_pred['bathrooms'],df_pred['price']/1000,s=4)
plt.xlabel('bathrooms')
plt.ylabel('Price thousands of dollars');
fig = plt.figure(figsize=(10,6))
plt.scatter(df_pred['sqft_living'],df_pred['price']/1000,s=4)
plt.xlabel('sqft_living')
plt.ylabel('Price thousands of dollars');
fig = plt.figure(figsize=(10,6))
plt.scatter(df_pred['grade'],df_pred['price']/1000,s=4)
plt.xlabel('grade')
plt.ylabel('Price thousands of dollars');
fig = plt.figure(figsize=(10,6))
plt.scatter(df_pred['sqft_above'],df_pred['price']/1000,s=4)
plt.xlabel('sqft_above')
plt.ylabel('Price thousands of dollars');
# multiple regression with traing and test data
X = df_pred.iloc[:,1:].values
y = df_pred['price'].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
slr = LinearRegression()
slr.fit(X_train,y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)
# plot results using a residual plot
fig = plt.figure(figsize=(10,6))
plt.scatter(y_train_pred,y_train_pred - y_train,c='steelblue',
           marker='o',edgecolor='white',label='Training data')
plt.scatter(y_test_pred,y_test_pred - y_test,c='limegreen',
           marker='s',edgecolor='white',label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='lower left')
plt.hlines(y=0,xmin=-10,xmax=3e6,color='black',lw=2)
plt.xlim([-10,3e6])
plt.show()
# coefficient of determination calculation
print('R^2 train: {0:.3f}, test: {1:.3f}'.format(r2_score(y_train, y_train_pred),
r2_score(y_test, y_test_pred)))
#df_zipcode = df.groupby('zipcode')['price'].mean()
df_98023 = df[df['zipcode']==98023]
df_pred = df_98023[['price','bathrooms', 'sqft_living','grade','sqft_above']]
df_pred.head()
# multiple regression with traing and test data
X = df_pred.iloc[:,1:].values
y = df_pred['price'].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
slr = LinearRegression()
slr.fit(X_train,y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)
# plot results using a residual plot
fig = plt.figure(figsize=(10,6))
plt.scatter(y_train_pred,y_train_pred - y_train,c='steelblue',
           marker='o',edgecolor='white',label='Training data')
plt.scatter(y_test_pred,y_test_pred - y_test,c='limegreen',
           marker='s',edgecolor='white',label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='lower left')
plt.hlines(y=0,xmin=-10,xmax=3e6,color='black',lw=2)
plt.xlim([-10,1e6])
plt.show()
# coefficient of determination calculation
print('R^2 train: {0:.3f}, test: {1:.3f}'.format(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))
df_zipcode_mean = df.groupby('zipcode')['price'].mean()
df_zipcode_std = df.groupby('zipcode')['price'].std()
zipcodes = df_zipcode_mean.index
df_zipcode_mean.head()
# plot error bar plot with mean and std
x = df_zipcode_mean.index
y = df_zipcode_mean.values
yerr = df_zipcode_std.values
fig = plt.figure(figsize=(10,6))
plt.xlabel('Zipcode')
plt.ylabel('mean and std deviation range for prices')
plt.errorbar(x,y,yerr,marker='s',ecolor='b',markerfacecolor='b',ls='none');
fig = plt.figure(figsize=(10,6))
plt.xlabel('Zipcode')
plt.ylabel('r2_score')
for z in zipcodes:
    X = df[df['zipcode'] == z][['bathrooms', 'sqft_living','grade','sqft_above']]
    y = df[df['zipcode'] == z][['price']]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
    slr = LinearRegression()
    slr.fit(X_train,y_train)
    y_train_pred = slr.predict(X_train)
    y_test_pred = slr.predict(X_test)
    plt.plot(z,(r2_score(y_test, y_test_pred))**0.5,marker='s',markerfacecolor='r',ls='none')
    plt.plot(z,(r2_score(y_train, y_train_pred))**0.5,marker='o',markerfacecolor='b',ls='none');
df_prediction = df[['price','bathrooms', 'sqft_living','grade','sqft_above','zipcode']]

df_zip = pd.get_dummies(df_prediction['zipcode'],prefix='zip',drop_first=True)
df_pred = pd.concat([df_prediction,df_zip],axis=1)
df_pred = df_pred.drop('zipcode',axis=1)
df_pred.head()
# multiple regression with traing and test data
X = df_pred.iloc[:,1:].values
y = df_pred['price'].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
slr = LinearRegression()
slr.fit(X_train,y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)
# plot results using a residual plot
fig = plt.figure(figsize=(10,6))
plt.scatter(y_train_pred,y_train_pred - y_train,c='steelblue',
           marker='o',edgecolor='white',label='Training data')
plt.scatter(y_test_pred,y_test_pred - y_test,c='limegreen',
           marker='s',edgecolor='white',label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend()
plt.hlines(y=0,xmin=-10,xmax=3e6,color='black',lw=2)
plt.xlim([-10,1e6])
plt.show()
# coefficient of determination calculation
print('R^2 train: {0:.3f}, test: {1:.3f}'.format(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))
# xgboost with traing and test data
X = df_pred.iloc[:,1:].values
y = df_pred['price'].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
regressor = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0,
             importance_type='gain', learning_rate=0.1, max_delta_step=0,
             max_depth=4, min_child_weight=1, missing=None, n_estimators=150,
             n_jobs=6, nthread=None, objective='reg:squarederror',
             random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
             seed=None, silent=None, subsample=0.5, verbosity=1)
regressor.fit(X_train, y_train)
y_train_pred = regressor.predict(X_train)
y_test_pred = regressor.predict(X_test)
# plot results using a residual plot
fig = plt.figure(figsize=(10,6))
plt.scatter(y_train_pred,y_train_pred - y_train,c='steelblue',
           marker='o',edgecolor='white',label='Training data')
plt.scatter(y_test_pred,y_test_pred - y_test,c='limegreen',
           marker='s',edgecolor='white',label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend()
plt.hlines(y=0,xmin=-10,xmax=3e6,color='black',lw=2)
plt.xlim([-10,1e6])
plt.show()
# coefficient of determination calculation
print('R^2 train: {0:.3f}, test: {1:.3f}'.format(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))
n_est_vec = [50,100,150,200,250,300,350,400]
score_test = []
score_train = []
for k in n_est_vec:
    # xgboost with traing and test data
    # X = df_pred.iloc[:,1:].values
    # y = df_pred['price'].values
    # X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
    regressor = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=1,
             importance_type='gain', learning_rate=0.1, max_delta_step=0,
             max_depth=3, min_child_weight=6, missing=None, n_estimators=k,
             n_jobs=6, nthread=None, objective='reg:squarederror',
             random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
             seed=None, silent=None, subsample=0.5, verbosity=1)

    regressor.fit(X_train, y_train)
    y_train_pred = regressor.predict(X_train)
    y_test_pred = regressor.predict(X_test)
    # coefficient of determination calculation
    score_train.append(r2_score(y_train, y_train_pred))
    score_test.append(r2_score(y_test, y_test_pred))
    
fig = plt.figure(figsize=(10,6))
plt.scatter(n_est_vec,score_test,c='steelblue',label="test")
plt.scatter(n_est_vec,score_train,c='red',label="train")
plt.xlabel('n_estimators')
plt.ylabel('r2_score')
plt.legend()
plt.title("test score and training score as function of n_estimators")
plt.show()
max_d = 8
score_test = []
score_train = []
for k in range(1,max_d):
    # xgboost with traing and test data
    # X = df_pred.iloc[:,1:].values
    # y = df_pred['price'].values
    # X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
    regressor = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=1,
             importance_type='gain', learning_rate=0.1, max_delta_step=0,
             max_depth=k, min_child_weight=6, missing=None, n_estimators=250,
             n_jobs=6, nthread=None, objective='reg:squarederror',
             random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
             seed=None, silent=None, subsample=0.5, verbosity=1)

    regressor.fit(X_train, y_train)
    y_train_pred = regressor.predict(X_train)
    y_test_pred = regressor.predict(X_test)
    # coefficient of determination calculation
    score_train.append(r2_score(y_train, y_train_pred))
    score_test.append(r2_score(y_test, y_test_pred))
    
fig = plt.figure(figsize=(10,6))
plt.scatter(list(range(1,max_d)),score_test,c='steelblue',label="test")
plt.scatter(list(range(1,max_d)),score_train,c='red',label="train")
plt.xlabel('max depth')
plt.ylabel('r2_score')
plt.legend()
plt.title("test score and training score as function of max_depth")
plt.show()
# 'max_depth and min_child_weight

# xgboost with traing and test data
# X = df_pred.iloc[:,1:].values
# y = df_pred['price'].values
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

params = {'gamma':[i/10.0 for i in range(0,1)],
         'min_child_weight':range(4,6,1)}

regressor = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0,
             importance_type='gain', learning_rate=0.1, max_delta_step=0,
             max_depth=3, min_child_weight=0.5, missing=None, n_estimators=250,
             n_jobs=1, objective='reg:squarederror',
             random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
             seed=None, silent=None, subsample=0.5, verbosity=1)

grid = GridSearchCV(regressor, params)

start = time.time()
grid.fit(X_train, y_train)
elapsed = time.time() - start
print("time elapsed grid search: {0:12.3f}".format(elapsed))

# Print the r2 score
print('train score: {0:12.3f}'.format(r2_score(y_train, grid.best_estimator_.predict(X_train)))) 
print('test score: {0:12.3f}'.format(r2_score(y_test, grid.best_estimator_.predict(X_test))))
grid.best_params_
# xgboost with traing and test data
X = df_pred.iloc[:,1:].values
y = df_pred['price'].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
regressor = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0,
             importance_type='gain', learning_rate=0.1, max_delta_step=0,
             max_depth=3, min_child_weight=4, missing=None, n_estimators=250,
             n_jobs=6, nthread=None, objective='reg:squarederror',
             random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
             seed=None, silent=None, subsample=0.5, verbosity=1)

regressor.fit(X_train, y_train)
y_train_pred = regressor.predict(X_train)
y_test_pred = regressor.predict(X_test)
# coefficient of determination calculation
print('R^2 train: {0:.3f}, test: {1:.3f}'.format(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))
