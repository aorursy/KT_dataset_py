import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn
import sklearn.model_selection as model_selection
import time
%matplotlib inline
df = pd.read_csv("/kaggle/input/california-housing-prices/housing.csv")
df.head()
df.describe()
df.isnull().sum()
df.shape
df.dtypes
df["ocean_proximity"].unique()
dummies = pd.get_dummies(df["ocean_proximity"],drop_first=True)
df = pd.concat([df,dummies], axis = 1)
df = df.drop(["ocean_proximity"],axis = 1)
df.head(10)
sb.scatterplot(x = df.latitude, y = df.longitude)
plt.title("California")
from sklearn.cluster import KMeans
X = df.loc[:,['latitude','longitude']]
sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(X)
    df["clusters"] = kmeans.labels_
    #print(data["clusters"])
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()
df = df.drop(["clusters"], axis = 1)
# creates 5 clusters using k-means clustering algorithm.
id_n=8
kmeans = KMeans(n_clusters=id_n, random_state=0).fit(X)
df['cluster']=kmeans.labels_
ptsymb = np.array(['b.','r.','m.','g.','c.','k.','b*','r*','m*','r^']);
plt.figure(figsize=(12,12))
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
for i in range(id_n):
    cluster = np.where(df['cluster']==i)[0]
    plt.plot(X.latitude[cluster].values,X.longitude[cluster].values,ptsymb[i])
plt.show()
df
df.cluster.unique()
df.dtypes
df["cluster"] = df["cluster"].astype("category")
dummies = pd.get_dummies(df["cluster"],drop_first=True)
df = pd.concat([df,dummies], axis = 1)
df = df.drop(["cluster","latitude","longitude"], axis = 1)
df.shape
train, test = model_selection.train_test_split(df, test_size=0.2)
print(train.shape, test.shape)
print(train.isnull().sum(),test.isnull().sum())
train_notnull = train.dropna()
test_notnull = test.dropna()
train_notnull["n_rooms_per_bedroom"] = np.divide(train_notnull["total_rooms"],train_notnull['total_bedrooms'])
test_notnull["n_rooms_per_bedroom"] = np.divide(test_notnull["total_rooms"],test_notnull['total_bedrooms'])
print(train_notnull["n_rooms_per_bedroom"].describe())
print(test_notnull["n_rooms_per_bedroom"].describe())
train = train.fillna(0)
test = test.fillna(0)
def impute_bedrooms(df):
    if df["total_bedrooms"] == 0:
        return np.round(df["total_rooms"]/5)
    else:
        return df["total_bedrooms"]
train["total_bedrooms"] = train.apply(impute_bedrooms, axis = 1)
test["total_bedrooms"] = test.apply(impute_bedrooms, axis = 1)
print(train.isnull().sum())
print(test.isnull().sum())
corr = train.corr()
sb.heatmap(corr,
          vmin=-1, vmax=1, center=0,
          cmap=sb.diverging_palette(20, 220, n=200),
          square=True)
corr
sb.scatterplot(x = train["median_income"], y = train["median_house_value"])
index1 = train[ (train['median_income'] >= 12) & (train['median_house_value'] < 410000)].index
train.drop(index1, inplace = True)
index2 = train[ (train['median_income'] >= 10) & (train['median_house_value'] < 300000)].index
train.drop(index2, inplace = True)
idx1 = train[ (train['median_income'] >= 7) & (train['median_house_value'] < 100000)].index
train.drop(idx1, inplace = True)
idx2 = train[ (train['median_income'] >= 9) & (train['median_house_value'] < 200000)].index
train.drop(idx2, inplace = True)
print(train.shape)
sb.scatterplot(x = train["median_income"], y = train["median_house_value"])
train
test
y_train = train["median_house_value"]
X_train = train.drop(["median_house_value"],axis = 1)
y_test = test["median_house_value"]
X_test = test.drop(["median_house_value"],axis = 1)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
##### Random Forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=1000, n_jobs=-1 ,random_state=42)
rf.fit(X_train,y_train)
predictions = rf.predict(X_test)
mse = sklearn.metrics.mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
print('Accuracy for Random Forest',100*max(0,rmse))
from sklearn.model_selection import GridSearchCV
param_grid = {
    'bootstrap': [True],
    'max_depth': [90, 100, 110],
    'max_features': ["sqrt"],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [2, 3],
    'n_estimators': [600, 800]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
rf_grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
rf_grid_search.fit(X_train,y_train)
rf_grid_search.best_params_
def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    errors = abs(predictions - y_test)
    mape = 100 * np.mean(errors / y_test)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} $.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy
base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(X_train, y_train)
base_accuracy = evaluate(base_model, X_test, y_test)
best_random = rf_grid_search.best_estimator_
random_accuracy = evaluate(rf_grid_search, X_test, y_test)
print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))
best_random = rf_grid_search.best_estimator_
random_accuracy = evaluate(rf_grid_search, X_train, y_train)
importances = list(best_random.feature_importances_)
# List of tuples with variable and importance
feature_list = list(X_train.columns)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
X_train_new = X_train["median_income"]
X_test_new = X_test["median_income"]
from rgf.sklearn import RGFRegressor
from sklearn.model_selection import GridSearchCV
!pip install rgf-python
rgf = RGFRegressor(max_leaf=500,l2=0.1, reg_depth=1)
rgf.fit(X_train,y_train)
train_accuracy = evaluate(rgf, X_train, y_train)
test_accuracy = evaluate(rgf, X_test, y_test)