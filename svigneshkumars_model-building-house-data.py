import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns





from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import LabelEncoder,RobustScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics



import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor as vif



import math
# importing statsmodels package

# !python -m pip install statsmodels
df_org = pd.read_csv("../input/dataset/house_data_eda.csv")
df_org.head()
df_org.drop("Unnamed: 0", inplace=True,axis=1)
# garageyrblt colm has inf values

# so its replaced with one of the values in dataset

df_org['garageyrblt'].replace(np.inf,2010.0,inplace = True)
df = df_org.copy()
df.head()
df.head()
df.info()
list(set(df.dtypes.tolist()))
df_num = df.select_dtypes(include = ['float64','int64'])
df_num.head()
df_num.describe()
for i in range(0, len(df_num.columns), 5):

    sns.pairplot(data=df_num,

                x_vars=df_num.columns[i:i+5],

                y_vars=['saleprice'])
df_num.columns
df_cat = df.select_dtypes("O")
df_cat.head()
df_cat_colms = df_cat.columns

df_cat_colms
le = LabelEncoder()
for i in range(len(df_cat.columns)):

    df[df_cat.columns[i]] = le.fit_transform( df[df_cat.columns[i]])
df.head()
df.info()
df_num.columns
# scaling Numerical Data

# robust Scaler

# its not affected by outliers because it takes Median values to scale.

# robust scaling formula 



# xi = xi - Q1(x)/ Q3(x) - Q1(x)



# like MinMaxScaler it takes the percentile values
df['garageyrblt'].groupby(df['garageyrblt']).count()
rs = RobustScaler()

df[list(df_num.columns)] = rs.fit_transform(df[list(df_num.columns)])

df[list(df_num.columns)].head()
df.head(10)
# train test splitting

y = df[["saleprice"]]

X = df.drop(["saleprice"],axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.20,random_state = 42)
y.head()
X.head(10)
X_train.shape
X_test.shape
X_train_sm = sm.add_constant(X_train)

lr = sm.OLS(y_train,X_train_sm)

lr_model = lr.fit()

lr_model.params

lr_model.summary()
y_train_pred = lr_model.predict(X_train_sm)

y_train_pred.head()
y_train.head()
y_train["saleprice_pred_ols"] = y_train_pred
y_train.head()
# Residual Analysis

y_train['res_ols'] = y_train['saleprice'] - y_train['saleprice_pred_ols']
y_train.head()
plt.figure(figsize = (15,8))

sns.distplot(y_train.res_ols)
# residuals are normally distributed
X_test_sm = sm.add_constant(X_test)

y_test_pred = lr_model.predict(X_test_sm)
r2_score(y_test,y_test_pred)
y_test['saleprice_pred_ols'] = y_test_pred
y_test['res_ols'] = y_test.saleprice - y_test.saleprice_pred_ols
y_test.head()
plt.figure(figsize = (15,8))

sns.distplot(y_test.res_ols)
# mean squared error

mse = mean_squared_error(y_test.saleprice,y_test.saleprice_pred_ols)

print(mse)



# mean absolute error

mae = mean_absolute_error(y_test.saleprice,y_test.saleprice_pred_ols)

print(mae)



# root mean squared error

rmse = math.sqrt(mse)

print(rmse)
lr = LinearRegression()
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)
lr.fit(X_train,y_train.saleprice)
y_train_pred = lr.predict(X_train)
print('{}\n'.format(repr(y_train_pred)))
print('Coefficients: {}\n'.format(repr(lr.coef_)))
print('Intercept: {}\n'.format(lr.intercept_))
y_train["saleprice_pred_lr"] = y_train_pred
y_train.head()
r2_score(y_train.saleprice,y_train.saleprice_pred_lr)
y_train['res_lr'] = y_train.saleprice - y_train.saleprice_pred_lr
y_train.head()
plt.figure(figsize = (15,8))

sns.distplot(y_train.res_lr)
# the residuals are normally distributed
y_test_pred = lr.predict(X_test)
y_test['saleprice_pred_lr'] = y_test_pred
y_test['res_lr'] = y_test.saleprice - y_test.saleprice_pred_lr
y_test.head()
r2_score(y_test.saleprice,y_test.saleprice_pred_lr)
plt.figure(figsize = (15,8))

sns.distplot(y_test.res_lr)
# the residuals in test data also has a normal distribution.

# so most of the data points are has residuals zero

# so the model is better
# mean squared error

mse = mean_squared_error(y_test.saleprice,y_test.saleprice_pred_lr)

print(mse)



# mean absolute error

mae = mean_absolute_error(y_test.saleprice,y_test.saleprice_pred_lr)

print(mae)



# root mean squared error

rmse = math.sqrt(mse)

print(rmse)

dcr = DecisionTreeRegressor(max_depth=5)
dcr.fit(X_train,y_train.saleprice)
y_train_pred = dcr.predict(X_train)
y_train_pred
y_train['saleprice_pred_dcr'] = y_train_pred
y_train.head()
y_train['res_dcr'] = y_train.saleprice - y_train.saleprice_pred_dcr
y_train.head()
plt.figure(figsize = (15,8))

sns.distplot(y_train.res_dcr)
# r2 value

r2_score(y_train.saleprice,y_train.saleprice_pred_dcr)
# this graph shows that there is 86 score
y_test_pred = dcr.predict(X_test)
y_test['saleprice_pred_dcr'] = y_test_pred
y_test.head()
y_test['res_dcr'] = y_test.saleprice - y_test.saleprice_pred_dcr
y_test.head()
plt.figure(figsize = (15,8))

sns.distplot(y_test.res_dcr)
# this residuals are also normally distributed

# this says that the model is performing well
# r2 value

r2_score(y_test.saleprice,y_test.saleprice_pred_dcr)
# The above models give about 80 and 79 r2 value

# the above models are pretty good than DCR
# mean squared error

mse = mean_squared_error(y_test.saleprice,y_test.saleprice_pred_dcr)

print(mse)



# mean absolute error

mae = mean_absolute_error(y_test.saleprice,y_test.saleprice_pred_dcr)

print(mae)



# root mean squared error

rmse = math.sqrt(mse)

print(rmse)
# In this DCR the error terms or values are higher than above 2 models

# so this model
rfr = RandomForestRegressor(n_estimators = 100)
rfr.fit(X_train,y_train.saleprice)
y_train_pred = rfr.predict(X_train)
y_train['saleprice_pred_rfr'] = y_train_pred
y_train.head()
y_train['res_rfr'] = y_train.saleprice - y_train.saleprice_pred_rfr
y_train.head()
r2_score(y_train.saleprice,y_train.saleprice_pred_rfr)
# the random forest regressor is giving training score of 98
plt.figure(figsize = (15,8))

sns.distplot(y_train.res_rfr)
# this graph shows its in normal distribution

# so most of the data points residuals are around 0
y_test_pred = rfr.predict(X_test)
y_test['saleprice_pred_rfr'] = y_test_pred
y_test.head()
r2_score(y_test.saleprice,y_test.saleprice_pred_rfr)
# the testing accuracy score is about 86
y_test['res_rfr'] = y_test.saleprice - y_test.saleprice_pred_rfr
y_test.head()
plt.figure(figsize = (15,8))

sns.distplot(y_test.res_rfr)
# this testing data is also normally distributed so most of the data points are nearer to 0
# mean squared error

mse = mean_squared_error(y_test.saleprice,y_test.saleprice_pred_rfr)

print(mse)



# mean absolute error

mae = mean_absolute_error(y_test.saleprice,y_test.saleprice_pred_rfr)

print(mae)



# root mean squared error

rmse = math.sqrt(mse)

print(rmse)
# to above all the models this random forest regressor gives the best score and low error values
df = df_org.copy()
df.head()
# Robust Scaling

df[list(df_num.columns)] = rs.fit_transform(df[list(df_num.columns)])

df[list(df_num.columns)].head()
# one hot encoding

dum = pd.get_dummies(df[list(df_cat.columns)],drop_first = True)
dum.head()
dum.shape
len(df_cat.columns)
78-41
df.drop(list(df_cat.columns),axis = 1,inplace = True)
df.shape
df1 = pd.concat([df,dum], axis =1)
df1.shape
df1.head(10)
df1.shape
y1 = df1[["saleprice"]]

X1 = df1.drop(["saleprice"],axis=1)

X_train1,X_test1,y_train1,y_test1=train_test_split(X1,y1,test_size = 0.20,random_state = 42)
X_train1.shape
X_test1.shape
# creating OLS model (Ordinary Least Squares)

X_train_sm1 = sm.add_constant(X_train1)

lr = sm.OLS(y_train1,X_train_sm1)

lr_model = lr.fit()

lr_model.params

lr_model.summary()
# this model gives 94% variation among Independant variables

# the Adj R2 value is nearer to r2

# so the model is pretty doing well

# the prob(F-stat)=0 says that confidence interval is high
y_train_pred1 = lr_model.predict(X_train_sm1)

y_train_pred1.head()
y_train1["saleprice_pred_ols"] = y_train_pred1
# Residual Analysis

y_train1['res_ols'] = y_train1['saleprice'] - y_train1['saleprice_pred_ols']
y_train1.head()
plt.figure(figsize = (15,8))

sns.distplot(y_train1.res_ols)

# the residuals are perfectly normally distributed
X_test_sm1 = sm.add_constant(X_test1)

y_test_pred1 = lr_model.predict(X_test_sm1)
r2_score(y_test1,y_test_pred1)
y_test1['saleprice_pred_ols'] = y_test_pred1
y_test1['res_ols'] = y_test1.saleprice - y_test1.saleprice_pred_ols
plt.figure(figsize = (15,8))

sns.distplot(y_test1.res_ols)

# the test data residuals are also normally distributed
# mean squared error

mse = mean_squared_error(y_test1.saleprice,y_test1.saleprice_pred_ols)

print(mse)

# mean absolute error

mae = mean_absolute_error(y_test1.saleprice,y_test1.saleprice_pred_ols)

print(mae)



# root mean squared error

rmse = math.sqrt(mse)

print(rmse)
lr = LinearRegression()
lr.fit(X_train1,y_train1.saleprice)
y_train1_pred = lr.predict(X_train1)
y_train1['saleprice_pred_lr'] = y_train1_pred
y_train1.head()
y_train1['res_lr'] = y_train1.saleprice - y_train1.saleprice_pred_lr
y_train1.head()
r2_score(y_train1.saleprice,y_train1.saleprice_pred_lr)
# training score is about 94
plt.figure(figsize = (15,8))

sns.distplot(y_train1.res_lr)
y_test1_pred = lr.predict(X_test1)
#lr.summary()
y_test1.head()
y_test1['saleprice_pred_lr'] = y_test1_pred
y_test1['res_lr'] = y_test1.saleprice - y_test1.saleprice_pred_lr
y_test1.head()
r2_score(y_test1.saleprice,y_test1.saleprice_pred_lr)
# the test score is all about 80
plt.figure(figsize = (15,8))

sns.distplot(y_test1.res_lr)
# Regression metrics

#explained_variance=metrics.explained_variance_score(y_test1.saleprice, y_test1.saleprice_pred_dcr)

#mean_absolute_error=metrics.mean_absolute_error(y_test1.saleprice, y_test1.saleprice_pred_dcr) 

#mse=metrics.mean_squared_error(y_test1.saleprice, y_test1.saleprice_pred_dcr) 

#mean_squared_log_error=metrics.mean_squared_log_error(y_test1.saleprice, y_test1.saleprice_pred_dcr)

#median_absolute_error=metrics.median_absolute_error(y_test1.saleprice, y_test1.saleprice_pred_dcr)

#r2=metrics.r2_score(y_test1.saleprice, y_test1.saleprice_pred_dcr)

#

#print('explained_variance: ', round(explained_variance,4))    

#print('mean_squared_log_error: ', round(mean_squared_log_error,4))

#print('r2: ', round(r2,4))

#print('MAE: ', round(mean_absolute_error,4))

#print('MSE: ', round(mse,4))

#print('RMSE: ', round(np.sqrt(mse),4))
dcr = DecisionTreeRegressor(max_depth=5)

dcr.fit(X_train1,y_train1.saleprice)
y_train1_pred = dcr.predict(X_train1)
r2_score(y_train1.saleprice,y_train1_pred)
y_train1['saleprice_pred_dcr'] = y_train1_pred
y_train1['res_dcr'] = y_train1.saleprice - y_train1.saleprice_pred_dcr
plt.figure(figsize = (15,8))

sns.distplot(y_train1.res_dcr)
# this training set score is 86
y_test1_pred = dcr.predict(X_test1)
y_test1['saleprice_pred_dcr'] = y_test1_pred
y_test1['res_dcr'] = y_test1.saleprice - y_test1.saleprice_pred_dcr
y_test1.head()
r2_score(y_test1.saleprice,y_test1.saleprice_pred_dcr)
# this test score is  low

# training score is 86

# testing  score is 0.70
plt.figure(figsize = (15,8))

sns.distplot(y_test1.res_dcr)
# mean squared error

mse = mean_squared_error(y_test1.saleprice,y_test1.saleprice_pred_dcr)

print(mse)



# mean absolute error

mae = mean_absolute_error(y_test1.saleprice,y_test1.saleprice_pred_dcr)

print(mae)



# root mean squared error

rmse = math.sqrt(mse)

print(rmse)
rfr = RandomForestRegressor(n_estimators = 500)
rfr.fit(X_train1,y_train1.saleprice)
y_train1_pred = rfr.predict(X_train1)
y_train1['saleprice_pred_rfr'] = y_train1_pred
y_train1['res_rfr'] = y_train1.saleprice - y_train1.saleprice_pred_rfr
r2_score(y_train1.saleprice,y_train1.saleprice_pred_rfr)
# its score is too good as 98
plt.figure(figsize = (15,8))

sns.distplot(y_train1.res_rfr)
# the data distribution too good for training data
y_test1_pred = rfr.predict(X_test1)
y_test1['saleprice_pred_rfr'] = y_test1_pred



y_test1['res_rfr'] = y_test1.saleprice - y_test1.saleprice_pred_rfr

r2_score(y_test1.saleprice,y_test1.saleprice_pred_rfr)
#  this score is pretty good for trainning model
plt.figure(figsize = (15,8))

sns.distplot(y_test1.res_rfr)
# this data has some values distributed around 0 in residuals
# mean squared error

mse = mean_squared_error(y_test1.saleprice,y_test1.saleprice_pred_rfr)

print(mse)



# mean absolute error

mae = mean_absolute_error(y_test1.saleprice,y_test1.saleprice_pred_rfr)

print(mae)



# root mean squared error

rmse = math.sqrt(mse)

print(rmse)
# the feature selection can be done in many ways

# forward selection

# backward elimination

# Recursive feature elimination

# some methods like Boruta, SHAP, Null importance
df.head()
df1.head()
X_train.head()
X_train1.head()
vif_df = pd.DataFrame()

vif_df["features"] = X_train.columns

vif_df["VIF"] = [vif(X_train.values,i) for i in range(X_train.shape[1])]

vif_df["VIF"] = round(vif_df["VIF"],3)

vif_df = vif_df.sort_values(by = "VIF",ascending = False)

vif_df
vif_less_ten = vif_df[vif_df.VIF < 10.0]
vif_less_ten
vif_less_ten.shape
# this has selected 69 columns form 77 columns (label encod & robust scaling)
vif_less_five = vif_df[vif_df.VIF < 5.0]
vif_less_five
vif_less_two = vif_df[vif_df.VIF < 2.0]
vif_less_two
vif_less_two.shape
# the vif <2 has selected 40 features
# OLS

# vif < 10

X_train_sm = sm.add_constant(X_train[list(vif_less_ten.features)])

lr = sm.OLS(y_train.saleprice,X_train_sm)

lr_model = lr.fit()

lr_model.params

lr_model.summary()
y_train_pred = lr_model.predict(X_train_sm)

y_train_pred.head()
y_train["saleprice_pred_vif10_ols"] = y_train_pred
# Residual Analysis

y_train['res_vif10_ols'] = y_train['saleprice'] - y_train['saleprice_pred_vif10_ols']
plt.figure(figsize = (15,8))

sns.distplot(y_train.res_vif10_ols)
# see the errors are normally distributed
X_test_sm = sm.add_constant(X_test[list(vif_less_ten.features)])

y_test_pred = lr_model.predict(X_test_sm)
r2_score(y_test.saleprice,y_test_pred)
# the model score is 85 and test score is 75

# pretty good model
# mean squared error

mse = mean_squared_error(y_test.saleprice,y_test_pred)

print(mse)



# mean absolute error

mae = mean_absolute_error(y_test.saleprice,y_test_pred)

print(mae)



# root mean squared error

rmse = math.sqrt(mse)

print(rmse)
# the values of errors are high
# vif < 5

X_train_sm = sm.add_constant(X_train[list(vif_less_five.features)])

lr = sm.OLS(y_train.saleprice,X_train_sm)

lr_model = lr.fit()

lr_model.params

lr_model.summary()
X_test_sm = sm.add_constant(X_test[list(vif_less_five.features)])
y_test_pred = lr_model.predict(X_test_sm)

y_test_pred.head()
r2_score(y_test.saleprice,y_test_pred)
# mean squared error

mse = mean_squared_error(y_test.saleprice,y_test_pred)

print(mse)



# mean absolute error

mae = mean_absolute_error(y_test.saleprice,y_test_pred)

print(mae)



# root mean squared error

rmse = math.sqrt(mse)

print(rmse)
# vif < 2

X_train_sm = sm.add_constant(X_train[list(vif_less_two.features)])

lr = sm.OLS(y_train.saleprice,X_train_sm)

lr_model = lr.fit()

lr_model.params

lr_model.summary()
X_test_sm = sm.add_constant(X_test[list(vif_less_two.features)])
y_test_pred = lr_model.predict(X_test_sm)

y_test_pred.head()
r2_score(y_test.saleprice,y_test_pred)
# mean squared error

mse = mean_squared_error(y_test.saleprice,y_test_pred)

print(mse)



# mean absolute error

mae = mean_absolute_error(y_test.saleprice,y_test_pred)

print(mae)



# root mean squared error

rmse = math.sqrt(mse)

print(rmse)
# Linear Regression
lr = LinearRegression()
lr.fit(X_train[list(vif_less_ten.features)],y_train.saleprice)
lr.score(X_train[list(vif_less_ten.features)],y_train.saleprice)
# the score of training is 85 for vif < 10
y_test_pred = lr.predict(X_test[list(vif_less_ten.features)])
r2_score(y_test.saleprice,y_test_pred)
# mean squared error

mse = mean_squared_error(y_test.saleprice,y_test_pred)

print(mse)



# mean absolute error

mae = mean_absolute_error(y_test.saleprice,y_test_pred)

print(mae)



# root mean squared error

rmse = math.sqrt(mse)

print(rmse)

lr = LinearRegression()
lr.fit(X_train[list(vif_less_five.features)],y_train.saleprice)



train_score = lr.score(X_train[list(vif_less_five.features)],y_train.saleprice)

print("train score : ",train_score)



y_test_pred = lr.predict(X_test[list(vif_less_five.features)])



test_score = r2_score(y_test.saleprice,y_test_pred)

print("test score: ",test_score)



# mean squared error

mse = mean_squared_error(y_test.saleprice,y_test_pred)

print(mse)



# mean absolute error

mae = mean_absolute_error(y_test.saleprice,y_test_pred)

print(mae)



# root mean squared error

rmse = math.sqrt(mse)

print(rmse)
# this is also the best model

# as train and test score is same
rfr = RandomForestRegressor(n_estimators = 100)
rfr.fit(X_train[list(vif_less_ten.features)],y_train.saleprice)



train_score = rfr.score(X_train[list(vif_less_ten.features)],y_train.saleprice)

print("train score : ",train_score)



y_test_pred = rfr.predict(X_test[list(vif_less_ten.features)])



test_score = r2_score(y_test.saleprice,y_test_pred)

print("test score: ",test_score)



# mean squared error

mse = mean_squared_error(y_test.saleprice,y_test_pred)

print(mse)



# mean absolute error

mae = mean_absolute_error(y_test.saleprice,y_test_pred)

print(mae)



# root mean squared error

rmse = math.sqrt(mse)

print(rmse)
# the training score is 97

# the testing score is 83

# moderate model
rfr = RandomForestRegressor(n_estimators = 500)

rfr.fit(X_train[list(vif_less_five.features)],y_train.saleprice)



train_score = rfr.score(X_train[list(vif_less_five.features)],y_train.saleprice)

print("train score : ",train_score)



y_test_pred = rfr.predict(X_test[list(vif_less_five.features)])



test_score = r2_score(y_test.saleprice,y_test_pred)

print("test score: ",test_score)



# mean squared error

mse = mean_squared_error(y_test.saleprice,y_test_pred)

print(mse)



# mean absolute error

mae = mean_absolute_error(y_test.saleprice,y_test_pred)

print(mae)



# root mean squared error

rmse = math.sqrt(mse)

print(rmse)
rfr = RandomForestRegressor(n_estimators = 100)

rfr.fit(X_train[list(vif_less_two.features)],y_train.saleprice)



train_score = rfr.score(X_train[list(vif_less_two.features)],y_train.saleprice)

print("train score : ",train_score)



y_test_pred = rfr.predict(X_test[list(vif_less_two.features)])



test_score = r2_score(y_test.saleprice,y_test_pred)

print("test score: ",test_score)



# mean squared error

mse = mean_squared_error(y_test.saleprice,y_test_pred)

print(mse)



# mean absolute error

mae = mean_absolute_error(y_test.saleprice,y_test_pred)

print(mae)



# root mean squared error

rmse = math.sqrt(mse)

print(rmse)
vif_df = pd.DataFrame()

vif_df["features"] = X_train1.columns

vif_df["VIF"] = [vif(X_train1.values,i) for i in range(X_train1.shape[1])]

vif_df["VIF"] = round(vif_df["VIF"],3)

vif_df = vif_df.sort_values(by = "VIF",ascending = False)

vif_df
vif_df[vif_df.VIF.isnull()]
vif_nan = list(vif_df[vif_df.VIF.isnull()].features)
X_train1[vif_nan].describe()
# these variables has all the values are 0

# so we can eliminate the values
vif_less_ten = vif_df[vif_df.VIF < 10.0]

vif_less_ten
# vif < 10

X_train_sm = sm.add_constant(X_train1[list(vif_less_ten.features)])

lr = sm.OLS(y_train.saleprice,X_train_sm)

lr_model = lr.fit()

#lr_model.params

print(lr_model.summary())







X_test_sm = sm.add_constant(X_test1[list(vif_less_ten.features)])

y_test_pred = lr_model.predict(X_test_sm)



test_score = r2_score(y_test1.saleprice,y_test_pred)

print("\ntest score: ",test_score)



# mean squared error

mse = mean_squared_error(y_test1.saleprice,y_test_pred)

print(mse)



# mean absolute error

mae = mean_absolute_error(y_test1.saleprice,y_test_pred)

print(mae)



# root mean squared error

rmse = math.sqrt(mse)

print(rmse)
# here the training score is 88 , 76

# its a moderate model
lr = LinearRegression()

lr.fit(X_train1[list(vif_less_ten.features)],y_train1.saleprice)



train_score = lr.score(X_train1[list(vif_less_ten.features)],y_train1.saleprice)

print("train score : ",train_score)



y_test_pred = lr.predict(X_test1[list(vif_less_ten.features)])



test_score = r2_score(y_test1.saleprice,y_test_pred)

print("test score: ",test_score)



# mean squared error

mse = mean_squared_error(y_test1.saleprice,y_test_pred)

print(mse)



# mean absolute error

mae = mean_absolute_error(y_test1.saleprice,y_test_pred)

print(mae)



# root mean squared error

rmse = math.sqrt(mse)

print(rmse)
# the score for linear Regression model is 88 and 76
rfr = RandomForestRegressor(n_estimators = 100)

rfr.fit(X_train1[list(vif_less_ten.features)],y_train1.saleprice)



train_score = rfr.score(X_train1[list(vif_less_ten.features)],y_train1.saleprice)

print("train score : ",train_score)



y_test_pred = rfr.predict(X_test1[list(vif_less_ten.features)])



test_score = r2_score(y_test1.saleprice,y_test_pred)

print("test score: ",test_score)



# mean squared error

mse = mean_squared_error(y_test1.saleprice,y_test_pred)

print(mse)



# mean absolute error

mae = mean_absolute_error(y_test1.saleprice,y_test_pred)

print(mae)



# root mean squared error

rmse = math.sqrt(mse)

print(rmse)
# this RFR has a scores of 97 and 82
vif_less_five = vif_df[vif_df.VIF < 5.0]

vif_less_five
# vif < 5

X_train_sm = sm.add_constant(X_train1[list(vif_less_five.features)])

lr = sm.OLS(y_train1.saleprice,X_train_sm)

lr_model = lr.fit()

#lr_model.params

print(lr_model.summary())





X_test_sm = sm.add_constant(X_test1[list(vif_less_five.features)])

y_test_pred = lr_model.predict(X_test_sm)



test_score = r2_score(y_test1.saleprice,y_test_pred)

print("\ntest score: ",test_score)



# mean squared error

mse = mean_squared_error(y_test1.saleprice,y_test_pred)

print(mse)



# mean absolute error

mae = mean_absolute_error(y_test1.saleprice,y_test_pred)

print(mae)



# root mean squared error

rmse = math.sqrt(mae)

print(rmse)
# the model has score of 74 and 65

# its a moderate model
rfr = RandomForestRegressor(n_estimators = 100)

rfr.fit(X_train1[list(vif_less_five.features)],y_train1.saleprice)



train_score = rfr.score(X_train1[list(vif_less_five.features)],y_train1.saleprice)

print("train score : ",train_score)



y_test_pred = rfr.predict(X_test1[list(vif_less_five.features)])



test_score = r2_score(y_test1.saleprice,y_test_pred)

print("test score: ",test_score)



# mean squared error

mse = mean_squared_error(y_test1.saleprice,y_test_pred)

print(mse)



# mean absolute error

mae = mean_absolute_error(y_test1.saleprice,y_test_pred)

print(mae)



# root mean squared error

rmse = math.sqrt(mse)

print(rmse)
# its score is 95 and 71 

# its a pretty bad model

# the gap bw the scores is high