import math 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn import metrics

import pickle

from joblib import dump, load



from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LassoCV

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor



from tensorflow.keras import Sequential, callbacks

from tensorflow.keras.models import load_model

from tensorflow.keras.optimizers import SGD, Adam

from tensorflow.keras.layers import Dense, Dropout, Activation



pd.set_option('display.max_rows',30)

%matplotlib inline
df = pd.read_csv('../input/craigslist-carstrucks-data/vehicles.csv')
df.describe()
df.info()
df.head()
fig, ax = plt.subplots(figsize=(10,2))

ax.set_title('Box plot of the prices')

sns.boxplot(x='price', data = df)
Q1 = df['price'].quantile(0.25)

Q3 = df['price'].quantile(0.75)

IQR = Q3 - Q1

filter = (df['price'] >= Q1 - 1.5 * IQR) & (df['price'] <= Q3 + 1.5 *IQR)

init_size = df.count()['id']

df = df.loc[filter]  

filtered_size = df.count()['id']

print(init_size-filtered_size,'(', '{:.2f}'.format(100*(init_size-filtered_size)/init_size), '%',')', 'outliers removed from dataset')
fig, ax = plt.subplots(figsize=(10,5))

ax.set_title('Distribution of the prices')

sns.distplot(df['price'], bins=30, kde=False)
df = df[df['price']>600]
fig, axs = plt.subplots(2, figsize=(20,10))

sns.distplot(df['odometer'], ax = axs[0])

axs[0].set_title('Distribution of the odometer')

axs[1].set_title('Box plot of the odometer')

sns.boxplot(x='odometer', data = df, ax=axs[1])
Q1 = df['odometer'].quantile(0.25)

Q3 = df['odometer'].quantile(0.75)

IQR = Q3 - Q1

filter = (df['odometer'] <= Q3 + 3 *IQR)

init_size = df.count()['id']

df = df.loc[filter]  

filtered_size = df.count()['id']

print(init_size-filtered_size,'(', '{:.2f}'.format(100*(init_size-filtered_size)/init_size), '%',')', 'outliers removed from dataset')
fig, ax = plt.subplots(figsize=(20,15))

ax.set_title('Geographical distribution of the cars colored by prices')

sns.scatterplot(x= 'long', y='lat', data = df, hue = 'price', ax=ax )
fig, ax = plt.subplots(figsize=(15,10))

ax.set_title('Mean car price of each state')

df.groupby(['state']).mean()['price'].plot.bar(ax=ax)
df = df.drop(columns = ['id', 'url', 'region', 'region_url', 'title_status', 'vin', 'image_url', 'description', 'county', 'state', 'long', 'lat'])
df.head()
df_man = df['manufacturer'].to_frame()
df = df.drop(columns = ['manufacturer'])
fig, ax = plt.subplots(figsize=(8,6))

ax.set_title('Distribution of the missing values (yellow records)')

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df = df.drop(columns = ['size'])
rm_rows = ['year', 'model', 'fuel', 'transmission', 'drive', 'type', 'paint_color']

for column in rm_rows:

    df = df[~df[column].isnull()]
df = df.replace(np.nan, 'null', regex=True)
df.info()

fig, ax = plt.subplots(figsize=(8,6))

ax.set_title('Distribution of the missing values (yellow records)')

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
fig, ax = plt.subplots(figsize=(20,15))

ax.set_title('Scatter plot between mileages and prices')

sns.scatterplot(x='odometer', y='price', data=df)
df = df[(df['price']+df['odometer'])>5000]
df = df[df['year']>1960]
df_man['manufacturer'].value_counts()
rm_brands = ['harley-davidson', 'alfa-romeo', 'datsun', 'tesla', 'land rover', 'porche', 'aston-martin', 'ferrari']

for brand in rm_brands:

    df_man = df_man[~(df_man['manufacturer'] == brand)]
df = df.groupby('model').filter(lambda x: len(x) > 50)
df['model'].value_counts()
df.info()
fig, axs = plt.subplots(2, figsize=(14, 10))

axs[0].set_title('Box plot of the prices')

sns.boxplot(x='price', data = df, ax = axs[0])

axs[1].set_title('Distribution of the prices')

sns.distplot(df['price'], ax=axs[1], bins=30, kde=False)
fig, axs = plt.subplots(2, figsize=(14, 10))

sns.distplot(df['odometer'], ax = axs[1], bins=30, kde=False)

axs[1].set_title('Box plot of the odometer')

sns.boxplot(x='odometer', data = df, ax=axs[0])

axs[0].set_title('Distribution of the odometer')
df['paint_color'].value_counts()
fig, ax = plt.subplots(figsize=(20,15))

ax.set_title('Box plot of the prices on each color')

sns.boxplot(x='paint_color', y='price', data = df)
print(df['type'].value_counts())

fig, ax = plt.subplots(figsize=(20,15))

ax.set_title('Box plot of the prices on each car type')

sns.boxplot(x='type', y='price', data = df)
print('Condition:')

print(df['condition'].value_counts())

print('\nCylinders:')

print(df['cylinders'].value_counts())

print('\nFuel:')

print(df['fuel'].value_counts())

print('\nTransmission:')

print(df['transmission'].value_counts())

print('\nDrive:')

print(df['drive'].value_counts())



fig=plt.figure(figsize=(25,37))

fig.add_subplot(3, 2, 1)

 

sns.boxplot(x='condition', y='price', data = df)

fig.add_subplot(3, 2, 2)

sns.boxplot(x='cylinders', y='price', data = df)

fig.add_subplot(3, 2, 3)

sns.boxplot(x='fuel', y='price', data = df)

fig.add_subplot(3, 2, 4)

sns.boxplot(x='transmission', y='price', data = df)

fig.add_subplot(3, 2, 5)

sns.boxplot(x='drive', y='price', data = df)
fig, ax = plt.subplots(figsize=(20,15))

ax.set_title('Scatter plot of the prices in each year, colored by transmission type')

#sns.scatterplot(x='year', y='price', data=df[(df['transmission'] =='manual') | (df['transmission'] =='other')], hue = 'transmission')

sns.scatterplot(x='year', y='price', data=df, hue = 'transmission')
fig, ax = plt.subplots(figsize=(20,15))

ax.set_title('Scatter plot of the prices in each year, colored by drive type')

sns.scatterplot(x='year', y='price', data=df, hue = 'drive')
fig, ax = plt.subplots(figsize=(20,15))

ax.set_title('Scatter plot of the prices in each year, colored by fuel type')

sns.scatterplot(x='year', y='price', data=df, hue = 'fuel')
fig, ax = plt.subplots(figsize=(20,15))

ax.set_title('Scatter plot of the prices in each year, colored by cylinder type')

sns.scatterplot(x='year', y='price', data=df, hue = 'cylinders')
fig, ax = plt.subplots(figsize=(20,15))

ax.set_title('Scatter plot of the odometer of the cars in each year, colored by car condition')

sns.scatterplot(x='year', y='odometer', data=df, hue = 'condition')
fig, ax = plt.subplots(figsize=(35,15))

ax.set_title('Count plot of all cars group by manufacturer')

sns.countplot(x='manufacturer', data=df_man)
fig, ax = plt.subplots(figsize=(12,10))

ax.set_title('Person correlations among prices, years and mileages')

sns.heatmap(df.corr())
cate_Columns = ['model', 'condition', 'cylinders', 'fuel', 'transmission', 'drive', 'type', 'paint_color']

for column in cate_Columns:

    column = pd.get_dummies(df[column],drop_first=True)

    df = pd.concat([df,column],axis=1)

df = df.drop(columns = cate_Columns)
df.head()
std_scaler = StandardScaler()



for column in ['year', 'odometer']:

    df[column] = std_scaler.fit_transform(df[column].values.reshape(-1,1))
df.head()
X_train, X_test, y_train, y_test = train_test_split(df.drop('price',axis=1), 

                                                    df['price'], test_size=0.30, 

                                                    random_state=141)
model_score = pd.DataFrame(columns=('r2', 'rmse'))
lrmodel = LinearRegression()

lrmodel.fit(X_train,y_train)
lr_predict = lrmodel.predict(X_test)



lr_r2 = metrics.r2_score(y_test, lr_predict)

lr_rmse = math.sqrt(metrics.mean_squared_error(y_test, lr_predict))



model_score = model_score.append(pd.DataFrame({'r2':[lr_r2], 'rmse':[lr_rmse]}, index = ['Linear Regression']))



print('For the linear regressor, the root mean square error for the testing set is:', lr_rmse)

print('The r2 score for the testing set is:', lr_r2)



fig, ax = plt.subplots(figsize=(20,15))

ax.set_title('Comparison between predicted prices and actual prices in testing set, linear regrssion')

plt.scatter(y_test, lr_predict)
lr_predict_train = lrmodel.predict(X_train)



lr_r2_train = metrics.r2_score(y_train, lr_predict_train)

lr_rmse_train = math.sqrt(metrics.mean_squared_error(y_train, lr_predict_train))



print('For the linear regressor, the root mean square error for the training set is:', lr_rmse_train)

print('The r2 score for the testing set is:', lr_r2_train)



fig, ax = plt.subplots(figsize=(20,15))

ax.set_title('Comparison between predicted prices and actual prices in training set, linear regrssion')

plt.scatter(y_train, lr_predict_train)
alphas = np.logspace(-4,4,12)

lasso = LassoCV(max_iter=10**6, alphas=alphas)

lasso.fit(X_train, y_train)
lasso_predict = lasso.predict(X_test)



lasso_r2 = metrics.r2_score(y_test, lasso_predict)

lasso_rmse = math.sqrt(metrics.mean_squared_error(y_test, lasso_predict))



model_score = model_score.append(pd.DataFrame({'r2':[lasso_r2], 'rmse':[lasso_rmse]}, index = ['Lasso Regression']))



print('For the Lasso linear regressor, the root mean square error for the testing set is:', lasso_rmse)

print('The r2 score for the testing set is:', lasso_r2)



fig, ax = plt.subplots(figsize=(20,15))

ax.set_title('Comparison between predicted prices and actual prices in testing set, Lasso regrssion')

plt.scatter(y_test, lasso_predict)
callback = callbacks.EarlyStopping(monitor='loss', patience=3)

nn_model = Sequential()

nn_model.add(Dense(input_dim = X_train.shape[1], units = 2000, activation = 'relu'))

#nn_model.add(Dropout(0.3)) There seems to be no overfitting problem, so no need for dropout.

nn_model.add(Dense(units = 2000, activation = 'relu'))

nn_model.add(Dense(units=1))

nn_model.compile(loss='mean_squared_error', optimizer = 'adam', metrics=['mae', 'mse'])
#nn_model.fit(X_train, y_train, batch_size=5000, epochs=800, callbacks=[callback], verbose=0)   Here in Kaggle we use the trained model to save time

nn_model = load_model("../input/models-needed/nn0.917") #Here in Kaggle we use the trained model to save time
nn_predict = nn_model.predict(X_test)



nn_rmse = math.sqrt(metrics.mean_squared_error(y_test, nn_predict))

nn_r2 = metrics.r2_score(y_test, nn_predict)



model_score = model_score.append(pd.DataFrame({'r2':[nn_r2], 'rmse':[nn_rmse]}, index = ['MLP']))



print('For the MLP model, the root mean square error for the testing set is:', nn_rmse)

print('The r2 score for the testing set is:', nn_r2)



fig, ax = plt.subplots(figsize=(20,15))

ax.set_title('Comparison between predicted prices and actual prices in testing set, MLP')

plt.scatter(y_test, nn_predict)
knnReg = KNeighborsRegressor()



param_grid = [

     {

         'weights':['uniform'],

         'n_neighbors':[i for i in range(1,7)]

     }]



grid_search_knn = GridSearchCV(knnReg, param_grid,n_jobs=-1,verbose=2)

grid_search_knn.fit(X_train, y_train) 
knn_best = grid_search_knn.best_estimator_

knn_best
knn_predict = knn_best.predict(X_test)



knn_r2 = metrics.r2_score(y_test, knn_predict)

knn_rmse = math.sqrt(metrics.mean_squared_error(y_test, knn_predict))



model_score = model_score.append(pd.DataFrame({'r2':[knn_r2], 'rmse':[knn_rmse]}, index = ['K - Nearest Neighbor']))



print('For the K-NN regressor, the root mean square error for the testing set is:', knn_rmse)

print('The r2 score for the testing set is:', knn_r2)

fig, ax = plt.subplots(figsize=(20,15))

plt.scatter(y_test, knn_predict)
dt_model = DecisionTreeRegressor(random_state=0)

dt_model.fit(X_train, y_train)
dt_predict = dt_model.predict(X_test)



dt_r2 = metrics.r2_score(y_test, dt_predict)

dt_rmse = math.sqrt(metrics.mean_squared_error(y_test, dt_predict))



model_score = model_score.append(pd.DataFrame({'r2':[dt_r2], 'rmse':[dt_rmse]}, index = ['Decision Tree']))



print('For the decision tree regressor, the root mean square error for the testing set is:', dt_rmse)

print('The r2 score for the testing set is:', dt_r2)

fig, ax = plt.subplots(figsize=(20,15))

plt.scatter(y_test, dt_predict)
ranF_model = RandomForestRegressor(max_depth=8, random_state=0)

ranF_model.fit(X_train, y_train)
ranF_predict = ranF_model.predict(X_test)



ranF_r2 = metrics.r2_score(y_test, ranF_predict)

ranF_rmse = math.sqrt(metrics.mean_squared_error(y_test, ranF_predict))



model_score = model_score.append(pd.DataFrame({'r2':[ranF_r2], 'rmse':[ranF_rmse]}, index = ['Random Forest']))



print('For the random forest regressor, the root mean square error for the testing set is:', ranF_rmse)

print('The r2 score for the testing set is:', ranF_r2)

fig, ax = plt.subplots(figsize=(20,15))

plt.scatter(y_test, ranF_predict)
svr_model = SVR(C = 1, epsilon = 0.2, kernel = 'rbf', max_iter=10000)

svr_model.fit(X_train, y_train)
svr_predict = svr_model.predict(X_test)



svr_r2 = metrics.r2_score(y_test, svr_predict)

svr_rmse = math.sqrt(metrics.mean_squared_error(y_test, svr_predict))



model_score = model_score.append(pd.DataFrame({'r2':[svr_r2], 'rmse':[svr_rmse]}, index = ['SVM_gaus']))



print('For the support vector regressor with gaussian kernel, the root mean square error for the testing set is:', svr_rmse)

print('The r2 score for the testing set is:', svr_r2)

fig, ax = plt.subplots(figsize=(20,15))

plt.scatter(y_test, svr_predict)
svr_model2 = SVR(C = 1, epsilon = 0.2, kernel = 'linear', max_iter=10000)

svr_model2.fit(X_train, y_train)

svr_predict2 = svr_model2.predict(X_test)



svr2_r2 = metrics.r2_score(y_test, svr_predict2)

svr2_rmse = math.sqrt(metrics.mean_squared_error(y_test, svr_predict2))



model_score = model_score.append(pd.DataFrame({'r2':[svr2_r2], 'rmse':[svr2_rmse]}, index = ['SVM_linear']))



print('For the support vector regressor with linear kernal, the root mean square error for the testing set is:', svr2_rmse)

print('The r2 score for the testing set is:', svr2_r2)

fig, ax = plt.subplots(figsize=(20,15))

plt.scatter(y_test, svr_predict2)
gb_model = GradientBoostingRegressor(random_state=0)

gb_model.fit(X_train, y_train)
gb_predict = gb_model.predict(X_test)



gb_r2 = metrics.r2_score(y_test, gb_predict)

gb_rmse = math.sqrt(metrics.mean_squared_error(y_test, gb_predict))



model_score = model_score.append(pd.DataFrame({'r2':[gb_r2], 'rmse':[gb_rmse]}, index = ['GBDT']))



print('For the gradient boosting regressor, the root mean square error for the testing set is:', gb_rmse)

print('The r2 score for the testing set is:', gb_r2)

fig, ax = plt.subplots(figsize=(20,15))

plt.scatter(y_test, gb_predict)
xgb_model = XGBRegressor()



df_noDuplicate = df.loc[:,~df.columns.duplicated()]



X_train_nd, X_test_nd, y_train_nd, y_test_nd = train_test_split(df_noDuplicate.drop('price',axis=1), 

                                                    df['price'], test_size=0.30, 

                                                    random_state=141)





xgb_model.fit(X_train_nd, y_train_nd)
xgb_predict = xgb_model.predict(X_test_nd)



xgb_r2 = metrics.r2_score(y_test_nd, xgb_predict)

xgb_rmse = math.sqrt(metrics.mean_squared_error(y_test_nd, xgb_predict))



model_score = model_score.append(pd.DataFrame({'r2':[xgb_r2], 'rmse':[xgb_rmse]}, index = ['XGBoost']))



print('For the XGboosting regressor, the root mean square error for the testing set is:', xgb_rmse)

print('The r2 score for the testing set is:', xgb_r2)

fig, ax = plt.subplots(figsize=(20,15))

plt.scatter(y_test_nd, xgb_predict)
model_score.sort_values(by=['r2'], ascending=False)