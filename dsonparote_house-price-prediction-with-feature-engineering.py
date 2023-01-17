import pandas as pd 

import numpy as np

import seaborn as sns

%matplotlib inline

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

import statsmodels.formula.api as smf

from sklearn import linear_model 

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn import preprocessing 



house_df_main = pd.read_csv('/kaggle/input/housing/housing.csv')

house_df = house_df_main.copy()



house_df.head(10)
house_df.describe(include='all')
house_df.info()
house_df.isnull().sum()
house_df.fillna(house_df.mean(), inplace=True)
house_df.isnull().sum()
#initial Correlation

plt.figure(figsize = (16,10))

sns.heatmap(house_df.corr(),annot=True,center=0 )
dic = {}
wmhouse_df = house_df.copy()

wmhouse_df.columns
scaler = MinMaxScaler()

column_names_to_normalize = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',

       'total_bedrooms', 'population', 'households', 'median_income',

       'median_house_value']

x = wmhouse_df[column_names_to_normalize].values

x_scaled = scaler.fit_transform(x)

df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = wmhouse_df.index)

wmhouse_df[column_names_to_normalize] = df_temp
#worst model

wm_data = wmhouse_df.copy()

wmY = wm_data['median_house_value']

wm_data.drop(columns=['ocean_proximity','median_house_value'],inplace=True)

wmX_train, wmX_test, wmy_train, wmy_test = train_test_split(wm_data, wmY, test_size=0.2, random_state=1)

wm_data.columns

model = linear_model.LinearRegression()

model.fit(wmX_train, wmy_train)
wmy_pred = model.predict(wmX_test)
MSE = metrics.mean_squared_error(wmy_test, wmy_pred)

RMSE = np.sqrt(metrics.mean_squared_error(wmy_test, wmy_pred))

scores = model.score(wmX_train, wmy_train)

dic['BaseModel'] = (RMSE,scores*100)

print(MSE,RMSE,scores*100)
house_df.hist(bins=50, figsize=(15, 15))
sns.pairplot(house_df, x_vars=['housing_median_age', 'population', 'households', 'median_income'],y_vars ='median_house_value',hue = 'ocean_proximity')
house_df.columns
house_df['ocean_proximity'].value_counts()
house_df['ocean_proximity'].value_counts().plot(kind='bar')
sns.boxplot(x="ocean_proximity", y="median_house_value", data=house_df)
sns.boxplot(x="ocean_proximity", y="median_income", data=house_df)
plt.figure(figsize=(15,10))

sns.distplot(house_df['total_bedrooms'],color='red')

sns.distplot(house_df['total_rooms'],color='blue')

plt.show()
plt.figure(figsize=(10,6))

sns.distplot((house_df['total_rooms']/house_df['total_bedrooms']),color='green')

plt.show()
house_df['room_bed'] = (house_df['total_rooms']/house_df['total_bedrooms'])
house_df['room_bed']
house_df1 = house_df.copy()

house_df1 = house_df1.loc[ house_df1['room_bed']<10.0]
plt.figure(figsize=(10,6))

sns.boxplot(house_df['room_bed'],color='green',)



plt.figure(figsize=(10,6))

sns.boxplot(house_df1['room_bed'],color='red')



plt.show()
house_df1.info()


house_df1.plot.scatter(x='housing_median_age', y='population')

house_df1 = house_df1.loc[ house_df['population']<20000]
house_df1.info()
house_df1.plot.scatter(x='housing_median_age', y='population')
house_df1.columns
house_df1['house_pop'] = house_df1['households'] / house_df1['population']
house_df1.hist(bins=50, figsize=(15, 15))
sns.pairplot(house_df1, x_vars=['housing_median_age', 'population', 'households', 'median_income',

       'room_bed','house_pop'],y_vars ='median_house_value')
house_df1 = pd.get_dummies(house_df1,columns=['ocean_proximity'])

house_df1.columns

house_df1_norm = house_df1.copy()
scaler = MinMaxScaler()

column_names_to_normalize = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',

       'total_bedrooms', 'population', 'households', 'median_income',

       'median_house_value', 'room_bed', 'house_pop']

x = house_df1_norm[column_names_to_normalize].values

x_scaled = scaler.fit_transform(x)

df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = house_df1_norm.index)

house_df1_norm[column_names_to_normalize] = df_temp
house_df1_norm.head()
house_df1_norm.hist(bins=50,figsize=(15, 15))
#test model 1

m1_data = house_df1.copy()

m1Y = m1_data['median_house_value']

m2_data = house_df1.copy()

m2_data.drop(columns=['total_rooms',

       'total_bedrooms','population','households','ocean_proximity_ISLAND'],inplace=True)

m1_data.drop(columns=['median_house_value','total_rooms',

       'total_bedrooms','population','households','ocean_proximity_ISLAND'],inplace=True)

m1_data.columns

m1X_train, m1X_test, m1y_train, m1y_test = train_test_split(m1_data, m1Y, test_size=0.2,random_state=1)
model1 = linear_model.LinearRegression()

model1.fit(m1X_train, m1y_train)
m1y_pred = model1.predict(m1X_test)

# m1y_pred_train = model1.predict(m1X_train)
print(metrics.mean_squared_error(m1y_test, m1y_pred))

print(np.sqrt(metrics.mean_squared_error(m1y_test, m1y_pred)))
pd.DataFrame(zip(m1_data.columns,model1.coef_))
pd.DataFrame(zip(m1y_test,m1y_pred),columns=['True','Predicted'])
model1.score(m1X_train, m1y_train)
m2_data.rename(columns={'ocean_proximity_<1H OCEAN': 'ocean_proximity1', 'ocean_proximity_NEAR BAY': 'ocean_proximity2', 'ocean_proximity_NEAR OCEAN': 'ocean_proximity3'}, inplace=True)
lm2 = smf.ols(formula='median_house_value ~ longitude + latitude + housing_median_age  + median_income + room_bed + house_pop + ocean_proximity1 + ocean_proximity_INLAND + ocean_proximity2 + ocean_proximity3', data=m2_data).fit()
lm2.summary()
model_dt = make_pipeline(preprocessing.StandardScaler(),DecisionTreeRegressor(random_state=0))

scores = cross_val_score(model_dt, m1X_train, m1y_train, cv=10)

model_dt.fit(m1X_train, m1y_train)

pred = model_dt.predict(m1X_test)



MSE = metrics.mean_squared_error(m1y_test, pred)

RMSE = np.sqrt(metrics.mean_squared_error(m1y_test, pred))



dic['DecisionTree'] = (RMSE,scores.mean() *100)



print(MSE,RMSE)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean() *100, scores.std() * 2))
model_lr = make_pipeline(preprocessing.StandardScaler(),linear_model.LinearRegression())

scores = cross_val_score(model_lr, m1X_train, m1y_train, cv=10)



model_lr.fit(m1X_train, m1y_train)

pred = model_lr.predict(m1X_test)



MSE = metrics.mean_squared_error(m1y_test, pred)

RMSE = np.sqrt(metrics.mean_squared_error(m1y_test, pred))



dic['LinearRegression'] = (RMSE,scores.mean() *100)



print(MSE,RMSE)



print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean() *100, scores.std() * 2))
ridge_reg = linear_model.Ridge()

params_Ridge = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20] , "fit_intercept": [True, False], "solver": ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}

Ridge_GS = GridSearchCV(ridge_reg, param_grid=params_Ridge, n_jobs=-1)

Ridge_GS.fit(m1X_train, m1y_train)

print(Ridge_GS.best_params_)

model_rr = make_pipeline(preprocessing.StandardScaler(),linear_model.Ridge(random_state=0, **Ridge_GS.best_params_))

scores = cross_val_score(model_rr, m1X_train, m1y_train, cv=10)



model_rr.fit(m1X_train, m1y_train)

pred = model_rr.predict(m1X_test)



MSE = metrics.mean_squared_error(m1y_test, pred)

RMSE = np.sqrt(metrics.mean_squared_error(m1y_test, pred))





dic['RidgeRegression'] = (RMSE,scores.mean() *100)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean() *100, scores.std() * 2))
lasso_reg = linear_model.Lasso()

params_Lasso = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20] , "fit_intercept": [True, False]}

Lasso_GS = GridSearchCV(lasso_reg, param_grid=params_Lasso, n_jobs=-1)

Lasso_GS.fit(m1X_train, m1y_train)

print(Lasso_GS.best_params_)

model_lasso = make_pipeline(preprocessing.StandardScaler(),linear_model.Lasso(random_state=0, **Lasso_GS.best_params_, max_iter=1e+5))

scores = cross_val_score(model_lasso, m1X_train, m1y_train, cv=10)



model_lasso.fit(m1X_train, m1y_train)

pred = model_lasso.predict(m1X_test)



MSE = metrics.mean_squared_error(m1y_test, pred)

RMSE = np.sqrt(metrics.mean_squared_error(m1y_test, pred))





dic['LassoRegression'] = (RMSE,scores.mean() *100)

print(MSE,RMSE)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean() *100, scores.std() * 2))
model_rf = make_pipeline(preprocessing.StandardScaler(),RandomForestRegressor(max_depth=2, random_state=0))

scores = cross_val_score(model_rf, m1X_train, m1y_train, cv=10)

model_rf.fit(m1X_train, m1y_train)

pred = model_rf.predict(m1X_test)



MSE = metrics.mean_squared_error(m1y_test, pred)

RMSE = np.sqrt(metrics.mean_squared_error(m1y_test, pred))



dic['RFD_2'] = (RMSE,scores.mean() *100)

print("Depth:2 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean() *100, scores.std() * 2))



model_rf = make_pipeline(preprocessing.StandardScaler(),RandomForestRegressor(max_depth=5, random_state=0))

scores = cross_val_score(model_rf, m1X_train, m1y_train, cv=10)





model_rf.fit(m1X_train, m1y_train)

pred = model_rf.predict(m1X_test)



MSE = metrics.mean_squared_error(m1y_test, pred)

RMSE = np.sqrt(metrics.mean_squared_error(m1y_test, pred))



dic['RFD_5'] = (RMSE,scores.mean() *100)



print("Depth:5 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean() *100, scores.std() * 2))



model_rf = make_pipeline(preprocessing.StandardScaler(),RandomForestRegressor(max_depth=10, random_state=0))

scores = cross_val_score(model_rf, m1X_train, m1y_train, cv=10)



model_rf.fit(m1X_train, m1y_train)

pred = model_rf.predict(m1X_test)



MSE = metrics.mean_squared_error(m1y_test, pred)

RMSE = np.sqrt(metrics.mean_squared_error(m1y_test, pred))



dic['RFD_10'] = (RMSE,scores.mean() *100)



print("Depth:10 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean() *100, scores.std() * 2))



model_rf = make_pipeline(preprocessing.StandardScaler(),RandomForestRegressor(max_depth=20, random_state=0))

scores = cross_val_score(model_rf, m1X_train, m1y_train, cv=10)



model_rf.fit(m1X_train, m1y_train)

pred = model_rf.predict(m1X_test)



MSE = metrics.mean_squared_error(m1y_test, pred)

RMSE = np.sqrt(metrics.mean_squared_error(m1y_test, pred))



dic['RFD_20'] = (RMSE,scores.mean() *100)



print("Depth:20 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean() *100, scores.std() * 2))





model_rf = make_pipeline(preprocessing.StandardScaler(),RandomForestRegressor(max_depth=50, random_state=0))

scores = cross_val_score(model_rf, m1X_train, m1y_train, cv=10)



model_rf.fit(m1X_train, m1y_train)

pred = model_rf.predict(m1X_test)



MSE = metrics.mean_squared_error(m1y_test, pred)

RMSE = np.sqrt(metrics.mean_squared_error(m1y_test, pred))



dic['RFD_50'] = (RMSE,scores.mean() *100)



print("Depth:50 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean() *100, scores.std() * 2))

test = model_rf.fit(m1X_train, m1y_train)
pred = test.predict(m1X_test)
print(np.sqrt(metrics.mean_squared_error(m1y_test, pred)))
result = pd.DataFrame(zip(m1y_test,pred),columns=['True','Predicted'])
result
accuracy_model = pd.DataFrame(dic,index=['RMSE','Training_Accuracy'])

accuracy_model = accuracy_model.T

accuracy_model
accuracy_model.sort_values(by=['RMSE'],inplace=True)
from matplotlib import pyplot

fig, ax = pyplot.subplots(figsize=(15,8))

sns.barplot(x=accuracy_model.index,y='RMSE',data=accuracy_model,ax=ax,palette='Greens')



accuracy_model.sort_values(by=['Training_Accuracy'],inplace=True)
fig1, ax1 = pyplot.subplots(figsize=(15,8))

sns.barplot(x=accuracy_model.index,y='Training_Accuracy',data=accuracy_model,ax=ax1,palette='Blues')