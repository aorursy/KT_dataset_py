import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import FastMarkerCluster
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


%matplotlib inline



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv', index_col='id')
listings = pd.read_csv('/kaggle/input/ab-ny-august-2019/listings.csv', index_col='id')
df.head()
listings.head()
target_columns = ["property_type", "accommodates",  "review_scores_value", "review_scores_cleanliness", "review_scores_location", "review_scores_accuracy", "review_scores_communication", "review_scores_checkin", "review_scores_rating", "maximum_nights", "host_is_superhost", "host_response_time", "host_response_rate",  'bathrooms', 'bedrooms', 'beds']
data = pd.merge(df, listings[target_columns], on='id', how='left')
data.info()
data.isnull().sum()
data.describe()
data['reviews_per_month'] = data['number_of_reviews'] / 12

reviews = ['review_scores_value', 'review_scores_cleanliness', 'review_scores_location', 'review_scores_accuracy', 'review_scores_communication', 'review_scores_checkin', 'review_scores_rating']
for i in reviews:
  data[i].fillna(data[i].mean(), inplace=True)

data['accommodates'].fillna(data['accommodates'].mean(), inplace=True)
data['maximum_nights'].fillna(data['maximum_nights'].mean(), inplace=True)

data.drop(columns=['last_review'], inplace=True)

cat_columns = ['host_response_time', 'host_response_rate', 'property_type', 'host_is_superhost']

for i in cat_columns:
  data[i].fillna(data[i].value_counts().idxmax(), inplace=True)

a = ['bathrooms', 'beds', 'bedrooms']
for i in a:
  data[i].fillna(data[i].mean(), inplace=True)


data.isnull().sum()

plt.figure(figsize=(6,6))
sns.boxplot(y=data['price'])
plt.title("Distribution of Price")
plt.show()
mean = data['price'].mean()
std = data['price'].std()
upper_limit = mean + 3 * std
data = data[data['price'] <= upper_limit]
plt.figure(figsize=(6,6))
sns.boxplot(y=data['price'])
plt.title("Distribution of Price")
plt.show()
data.head()
plt.figure(figsize=(6,6))
numbers = data['neighbourhood_group'].value_counts()
plt.pie(numbers.values, labels=numbers.index, colors=['b', 'r', 'g', 'cyan', 'gray'], autopct='%1.1f%%')
plt.title('Numbers in Each Neigbourhoods')
plt.figure(figsize=(6,6))
numbers = data['room_type'].value_counts()
plt.pie(numbers.values, labels=numbers.index, colors=['cyan', 'green', 'pink'], autopct='%1.1f%%', shadow=True,startangle=90)
plt.title('Numbers in Each Room Types')
fig = plt.figure(figsize=(15,6))

ax1 = fig.add_subplot(121)
sns.scatterplot(data['longitude'], data['latitude'], hue=data['neighbourhood_group'], ax=ax1)
ax1.set_title('Distribution in Map')

ax2 = fig.add_subplot(122)
sns.scatterplot(data['longitude'], data['latitude'], hue=data['room_type'], ax=ax2)
ax2.set_title('Distribution of Room Types in the Map')

plt.show()
plt.figure(figsize=(10,4))
sns.countplot(data['neighbourhood_group'], hue=data['room_type'])
plt.title('Distribution of the room types in each neighbourhood group')
plt.show()
### Price Distribution in Each Neighbourhood
plt.figure(figsize=(15,6))
sns.boxplot(data=data, x='neighbourhood_group', y='price', palette='GnBu_d')
plt.title('Density and distribution of prices for each neighbourhood group', fontsize=15)
plt.xlabel('Neighbourhood group')
plt.ylabel("Price")
plt.figure(figsize=(15,6))
sns.boxplot(data=data, x='room_type', y='price', palette='GnBu_d')
plt.title('Density and distribution of prices for each Room Type', fontsize=15)
plt.xlabel('Room Type')
plt.ylabel("Price")
latitudes = np.array(data['latitude'])
longitudes = np.array(data['longitude'])
la_mean = latitudes.mean()
lo_mean = longitudes.mean()
locations = list(zip(latitudes, longitudes))

m = folium.Map(location=[la_mean, lo_mean], zoom_start= 11.5)
FastMarkerCluster(data=locations).add_to(m)
m
plt.figure(figsize=(6, 6))
sns.distplot(data['accommodates'])
plt.title('Distribution of Accommodates in New York')
plt.show()
plt.figure(figsize=(6, 6))
sns.distplot(data['price'], kde=False)
plt.title('Distribution of price')
data['log_price'] = np.log10(data['price'] + 1)
plt.figure(figsize=(6, 6))
sns.distplot(data['log_price'], kde=False)
plt.title('Distribution of price in Logarithm')
a = data.groupby('neighbourhood')['price'].mean().sort_values(ascending=True).head(20)
d = data.groupby('neighbourhood')['price'].mean().sort_values(ascending=False).head(20)
fig = plt.figure(figsize=(20,10))

ax1 = fig.add_subplot(121)
sns.barplot(y=a.index, x=a.values, ax=ax1)
ax1.set_title('The cheapest 20 neighbourhood')

ax2 = fig.add_subplot(122)
sns.barplot(y=d.index, x=d.values, ax=ax2)
ax2.set_title('The most expensive 20 neighbourhood')
plt.show()
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 9))

for ax, name in zip(axes.flatten(), reviews):
  ax.hist(data[name], bins=20)
  ax.set_title(f"Distribution of {name}")

plt.show()


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))

rooms = ['bedrooms', 'bathrooms']

for ax, name in zip(axes.flatten(), rooms):
  ax.hist(data[name], bins=20)
  ax.set_title(f"Distribution of {name}")

plt.show()

plt.figure(figsize=(6,6))

plt.hist(data['availability_365'], bins=20)

plt.title("Distribution of Availability in 365 Days")

plt.show()
corr = data.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

newdf = data.select_dtypes(include=numerics)
nrows = int(len(newdf.columns) / 3) + 1
fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(24, 6*nrows))
fig.subplots_adjust(hspace=0.5)

for ax, name in zip(axes.flatten(), newdf.columns):
  
  sns.regplot(x=name, y='price', data=newdf, ax=ax)
  ax.set_title(f"Correlation between {name} and the price")

plt.show()
a = data.groupby(['neighbourhood_group', 'neighbourhood'])['price'].mean().sort_values(ascending=False).head(50)
a = a.reset_index()
a
plt.figure(figsize=(12, 6))
df_pivot = data.pivot_table(values='price', index='room_type', columns='neighbourhood_group', aggfunc='mean')
sns.heatmap(df_pivot, annot=True, fmt='.1f', cmap='Blues')
plt.suptitle('Mean Price')
plt.plot()
a = data.groupby(['neighbourhood_group', 'property_type'])['price'].mean().sort_values(ascending=False).head(20)
a = a.reset_index()
a
data = data[data.price > 0]
data.columns
data.drop(columns=['name', 'host_id', 'host_name', 'reviews_per_month'], inplace=True)
data.dtypes
from sklearn.preprocessing import LabelEncoder

categorical = data.select_dtypes(include=['object']).columns

for i in categorical:
  data[i] = LabelEncoder().fit_transform(data[i])


data.dtypes
data = data.reset_index(drop=True)

data.head()
data.drop(columns=['log_price'], inplace=True)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = data.drop(columns=['price'])
y = data['price']

columns = X.columns
scaler = StandardScaler()
X[columns] = scaler.fit_transform(X[columns])



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"There are {X_train.shape[0]} traning data")
print(f"There are {X_test.shape[0]} test data")


X.head()
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor


knn = KNeighborsRegressor(5, metric="euclidean")
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

mse = mean_squared_error(y_pred, y_test)
mae = mean_absolute_error(y_pred, y_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error: {}".format(mse))
print("Mean Absolute Error: {}".format(mae))
print("Root Mean Absolute Error: {}".format(rmse))
print("R2 score: {}".format(r2))
prediction_dictionaries = {'KNN-Default': y_pred}
prediction_list = pd.DataFrame({'Actual Values': np.array(y_test).flatten(), 'KNN-Default': y_pred.flatten()}).head(20)
prediction_list.set_index('Actual Values', inplace=True)
prediction_list
error_dict = {'KNN Default': [mse, r2]}
error_list = pd.DataFrame()
error_list['KNN Default'] = [mse, r2]
error_list.reset_index(inplace=True, drop=True)
#error_list.rename(columns={0: 'MSE KNN Default'}, inplace=True)
error_list.index =['Mean Squared Error', 'R2 Score']
error_list.T
plt.figure(figsize=(6,6))
sns.regplot(y_pred, y_test)
plt.title("The correlation line between predictions and the actual values")
plt.show()
def plot_all_r2():
  length = len(prediction_dictionaries)
  n_col = 2
  if length < 2:
    n_col = length % 2
  
  nrow = 1
  if(length > 2):
    nrow = int(length / 2) 
    if length % 2 != 0:
      nrow+=1
  
  fig, axes = plt.subplots(nrow, n_col, figsize=( 16, 3 * length))
  for ax, key in zip(axes.flatten(), prediction_dictionaries.keys()):
    sns.regplot(prediction_dictionaries[key], y_test, ax=ax)
    ax.set_title("The correlation line in {}".format(key))
  plt.show()
#Bu niye uzun sürdü acaba? Bir task a 36s (Bakacağım)
from sklearn.model_selection import GridSearchCV
param_grid = {'p': [1, 2],  
              'n_neighbors' : [ 5, 10, 15]
              } 

grid_knn = GridSearchCV(KNeighborsRegressor(n_jobs=-1), param_grid, refit = True, verbose = 10, n_jobs=-1, cv=5,scoring="neg_mean_squared_error") 

grid_knn.fit(X, y)


print(f"Best parameters are {grid_knn.best_params_}") 
print("Best score is {}".format(grid_knn.best_score_ * -1))
print("Best model is {}".format(grid_knn.best_estimator_))
#print("The score for hyperparameter tuning are {}".format(grid.cv_results_))
knr_best = KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
                    metric_params=None, n_jobs=-1, n_neighbors=15, p=1,
                    weights='uniform')
knr_best.fit(X_train, y_train)
y_pred_best = knr_best.predict(X_test)

mse_knn_best = mean_squared_error(y_pred_best, y_test)
mae_knn_best = mean_absolute_error(y_pred, y_test)
rmse_knn_best = np.sqrt(mse)
r2_knn_best = r2_score(y_test, y_pred_best)

print("Mean Squared Error: {}".format(mse_knn_best))
print("Mean Absolute Error: {}".format(mae_knn_best))
print("Root Mean Absolute Error: {}".format(rmse_knn_best))
print("R2 score: {}".format(r2_knn_best))
prediction_dictionaries['Knn-Best']   = y_pred_best
prediction_list['KNN-Best'] = y_pred_best[:20]
prediction_list
error_list['MSE KNN-Best'] = [mse_knn_best, r2_knn_best]
error_list.T
from sklearn.svm import LinearSVR, SVR
clf_svr = LinearSVR()
clf_svr.fit(X_train, y_train)

preds_svr = clf_svr.predict(X_test)

mse_svr = mean_squared_error(preds_svr, y_test)
mae_svr = mean_absolute_error(preds_svr, y_test)
rmse_svr = np.sqrt(mse_svr)
r2_svr = r2_score(y_test, preds_svr)

print("Mean Squared Error: {}".format(mse_svr))
print("Mean Absolute Error: {}".format(mae_svr))
print("Root Mean Absolute Error: {}".format(rmse_svr))
print("R2 Score: {}".format(r2_svr))
prediction_dictionaries['SVR - Default'] = preds_svr
plot_all_r2()
prediction_list['SVR-Default'] = np.array(preds_svr[:20])
prediction_list
error_list['SVR Default'] = [mse_svr, r2_svr]
error_list.T
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'], 
              'dual': [True, False],
              'tol': [0.0001, 0.00001]} 

grid = GridSearchCV(LinearSVR(), param_grid, refit = True, verbose = 10, n_jobs=-1, cv=5,scoring="neg_mean_squared_error") 

grid.fit(X, y)
print(f"Best parameters are {grid.best_params_}") 
print("Best score is {}".format(grid.best_score_ * -1))
print("Best model is {}".format(grid.best_estimator_))
print("scores {}".format(grid.cv_results_['mean_test_score']))
svr_best = LinearSVR(C=1, dual=True, epsilon=0.0, fit_intercept=True,
          intercept_scaling=1.0, loss='epsilon_insensitive', max_iter=1000,
          random_state=None, tol=0.0001, verbose=0)

svr_best.fit(X_train, y_train)

preds_svr_best = svr_best.predict(X_test)

mse_svr_best = mean_squared_error(preds_svr_best, y_test)
mae_svr_best = mean_absolute_error(preds_svr_best, y_test)
rmse_svr_best = np.sqrt(mse_svr_best)
r2_svr_best = r2_score(y_test, preds_svr_best)

print("Mean Squared Error: {}".format(mse_svr_best))
print("Mean Absolute Error: {}".format(mae_svr_best))
print("Root Mean Absolute Error: {}".format(rmse_svr_best))
print("R2 Score: {}".format(r2_svr_best))
prediction_dictionaries['SVR - Best'] = preds_svr_best
plot_all_r2()
prediction_list['SVR-Best'] = np.array(preds_svr_best[:20])
prediction_list
error_list['SVR Best'] = [mse_svr_best, r2_svr_best]
error_list.T
!pip install -q git+https://github.com/tensorflow/docs
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

print(tf.__version__)
def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=([X_train.shape[1]])),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model
model = build_model()

history = model.fit(
  X_train, y_train,
  epochs=200, validation_split = 0.2, verbose=0, callbacks=[tfdocs.modeling.EpochDots()])
model.summary()
hist = pd.DataFrame(history.history)
plt.figure(figsize=(10, 6))
plt.plot(hist.mse)
plt.title("MSE Graph in Neural Network")
plt.show()
preds_nn = model.predict(X_test)

mse_nn = mean_squared_error(y_test, preds_nn)
mae_nn = mean_absolute_error(y_test, preds_nn)
rmse_nn = np.sqrt(mse_nn)
r2_nn = r2_score(y_test, preds_nn)

print("Mean Squared Error: {}".format(mse_nn))
print("Mean Absolute Error: {}".format(mae_nn))
print("Root Mean Absolute Error: {}".format(rmse_nn))
print("R2 score: {}".format(r2_nn))

prediction_dictionaries['Neural Network']  = preds_nn
plot_all_r2()
prediction_list['Neural Network'] = np.array(preds_nn[:20])
prediction_list
error_list['Neural Network'] = [mse_nn, r2_nn]
error_list.T
from sklearn import tree
from sklearn import metrics

tree_model = tree.DecisionTreeRegressor()
tree_model.fit(X_train, y_train) # x -> features, y->target (price)
tree_model_prediction = tree_model.predict(X_test)

see_result = pd.DataFrame({
    'Actual': y_test, 
    'Predicted': tree_model_prediction
    })

tree_mse = metrics.mean_squared_error(y_test, tree_model_prediction)
tree_mae = metrics.mean_absolute_error(y_test, tree_model_prediction)
tree_rmse = np.sqrt(tree_mse)
tree_r2 = metrics.r2_score(y_test, tree_model_prediction)

print("Mean Squared Error: {}".format(tree_mse))
print("Mean Absolute Error: {}".format(tree_mae))
print("Root Mean Absolute Error: {}".format(tree_rmse))
print("R2 score: {}".format(tree_r2))
prediction_dictionaries['Decision Tree - Default'] = tree_model_prediction
plot_all_r2()
prediction_list['Decision Tree - Default'] = np.array(preds_svr[:20])
prediction_list
error_list['Decision Tree - Default'] = [tree_mse, tree_r2]
error_list.T
parameters = {
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8],
    'min_samples_leaf': [1, 2, 3, 4, 5],
    'min_samples_split': [2, 3, 4, 5],
}
tree_grid = GridSearchCV(tree_model, parameters, refit = True, verbose = 1, n_jobs=-1, cv=5, scoring="neg_mean_squared_error") 
tree_grid.fit(X, y)
print(f"Best parameters are {tree_grid.best_params_}") 
print("Best MSE is {}".format(tree_grid.best_score_ * -1))
tree_model_best = tree.DecisionTreeRegressor(max_depth = 7, min_samples_leaf = 4, min_samples_split = 4)
tree_model_best.fit(X_train, y_train) 
tree_model_prediction_best = tree_model_best.predict(X_test)

tree_mse_best = metrics.mean_squared_error(y_test, tree_model_prediction_best)
tree_mae_best = metrics.mean_absolute_error(y_test, tree_model_prediction_best)
tree_rmse_best = np.sqrt(tree_mse_best)
tree_r2_best = metrics.r2_score(y_test, tree_model_prediction_best)

print("Mean Squared Error: {}".format(tree_mse_best))
print("Mean Absolute Error: {}".format(tree_mae_best))
print("Root Mean Absolute Error: {}".format(tree_rmse_best))
print("R2 score: {}".format(tree_r2_best))
prediction_dictionaries['Decision Tree - Best'] = tree_model_prediction_best
plot_all_r2()
prediction_list['Decision Tree - Best'] = np.array(preds_svr_best[:20])
prediction_list
error_list['Decision Tree - Best'] = [tree_mse_best, tree_r2_best]
error_list.T
from sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor(random_state=42) #n_estimators is 100 by default
forest_model.fit(X_train, y_train)
forest_model_prediction = forest_model.predict(X_test)

forest_mse = metrics.mean_squared_error(y_test, forest_model_prediction)
forest_mae = metrics.mean_absolute_error(y_test, forest_model_prediction)
forest_rmse = np.sqrt(forest_mse)
forest_r2 = metrics.r2_score(y_test, forest_model_prediction)

print("Mean Squared Error: {}".format(forest_mse))
print("Mean Absolute Error: {}".format(forest_mae))
print("Root Mean Absolute Error: {}".format(forest_rmse))
print("R2 score: {}".format(forest_r2))
prediction_dictionaries['Random Forest - Default'] = forest_model_prediction
plot_all_r2()
prediction_list['Random Forest - Default'] = np.array(preds_svr[:20])
prediction_list
error_list['Random Forest - Default'] = [forest_mse, forest_r2]
error_list.T
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


linear_model = LinearRegression().fit(X_train, y_train)
linear_model_prediction = linear_model.predict(X_test)

linear_mse = metrics.mean_squared_error(y_test, linear_model_prediction)
linear_mae = metrics.mean_absolute_error(y_test, linear_model_prediction)
linear_rmse = np.sqrt(linear_mse)
linear_r2 = metrics.r2_score(y_test, linear_model_prediction)

print("Mean Squared Error: {}".format(linear_mse))
print("Mean Absolute Error: {}".format(linear_mae))
print("Root Mean Absolute Error: {}".format(linear_rmse))
print("R2 score: {}".format(linear_r2))
prediction_dictionaries['Linear Regression - Default'] = linear_model_prediction
plot_all_r2()
prediction_list['Linear Regression - Default'] = np.array(preds_svr[:20])
prediction_list
error_list['Linear Regression - Default'] = [linear_mse, linear_r2]
error_list.T
from sklearn.linear_model import Ridge

alpha = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 250, 500, 750, 1000, 1500, 2500, 5000, 10000, 100000, 1000000]
param_grid = {
    'alpha': alpha
}

ridge = Ridge(alpha=1).fit(X_train, y_train)
scores = cross_val_score(ridge, X_train, y_train, cv=5, scoring='r2')
scores_mse = cross_val_score(ridge, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

print("CV Mean for Ridge (r2): ", np.mean(scores))
print("CV Mean for Ridge (mse): ", np.mean(scores_mse) * -1)
grid_mse = GridSearchCV(estimator=ridge, param_grid=param_grid, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
grid_result_mse = grid_mse.fit(X_train, y_train)

grid_r2 = GridSearchCV(estimator=ridge, param_grid=param_grid, scoring='r2', verbose=1, n_jobs=-1)
grid_result_r2 = grid_r2.fit(X_train, y_train)
print('Best Score for mse: ', grid_mse.best_score_ * -1)
print('Best Params for mse: ', grid_mse.best_params_)
print()
print('Best Score for r2: ', grid_r2.best_score_)
print('Best Params for r2: ', grid_r2.best_params_)
ridge_best = Ridge(alpha=500).fit(X_train, y_train)
ridge_best.fit(X_train, y_train)
ridge_pred = ridge_best.predict(X_test)

ridge_mse_best = metrics.mean_squared_error(y_test, ridge_pred)
ridge_mae_best = metrics.mean_absolute_error(y_test, ridge_pred)
ridge_rmse_best = np.sqrt(ridge_mse_best)
ridge_r2_best = metrics.r2_score(y_test, ridge_pred)

print("Mean Squared Error: {}".format(ridge_mse_best))
print("Mean Absolute Error: {}".format(ridge_mae_best))
print("Root Mean Absolute Error: {}".format(ridge_rmse_best))
print("R2 score: {}".format(ridge_r2_best))
dict_val = {
    'Linear Model': [linear_r2, linear_mse],
    'Ridge': [ridge_r2_best, ridge_mse_best]
}
res_df_linear_ridge = pd.DataFrame(dict_val, index=['R2', 'MSE'])
res_df_linear_ridge
prediction_dictionaries['Ridge'] = ridge_pred
plot_all_r2()
prediction_list['Ridge'] = np.array(preds_svr[:20])
prediction_list
error_list['Ridge'] = [ridge_mse_best, ridge_r2_best]
error_list.T