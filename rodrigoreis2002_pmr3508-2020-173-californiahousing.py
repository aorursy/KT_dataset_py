import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

import seaborn as sns 

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import make_scorer

from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

from geopy.distance import distance, geodesic
train = pd.read_csv("../input/atividade-regressao-PMR3508/train.csv")

test = pd.read_csv("../input/atividade-regressao-PMR3508/test.csv")
train.head()
train.shape
test.head()
test.shape
train = train.drop("Id", axis = 1)

testId = test.Id

test = test.drop("Id", axis = 1)
train.head()
test.head()
train.describe()
train.info()
train.isnull().sum()
test.info()
test.isnull().sum()
train["median_house_value"].hist()
plt.figure(figsize=(10,10))

plt.title("Matriz de correlação")

sns.heatmap(train.corr(), annot=True, linewidths=0.2)
coord_LA = (34.0522, -118.2437)

coord_SF = (37.7749, -122.4194)

coord_SD = (32.7157, -117.1611)



def dist_cidades(df):

    coord_casa = (df["latitude"], df["longitude"])

    df["dist_to_city"] = min(geodesic(coord_casa, coord_SF).km, geodesic(coord_casa, coord_LA).km, geodesic(coord_casa, coord_SD).km)

    

    return df
train = train.apply(dist_cidades, axis = 1)

train = train.drop(["latitude", "longitude"], axis = 1)
train["avg_rooms"] = train.total_rooms/train.households

train["avg_bedrooms"] = train.total_bedrooms/train.households

train["avg_population"] = train.population/train.households

train["bedrooms/rooms"] = train.total_bedrooms/train.total_rooms

train["people/room"] = train.population/train.total_rooms

train["people/bedroom"] = train.population/train.total_bedrooms
plt.figure(figsize=(15,15))

plt.title("Matriz de correlação")

sns.heatmap(train.corr(), annot=True, linewidths=0.2)
train = train.drop(["people/room", "avg_population", "avg_rooms", "total_bedrooms", "total_rooms"], axis = 1)

plt.figure(figsize=(10,10))

plt.title("Matriz de correlação")

sns.heatmap(train.corr(), annot=True, linewidths=0.2)
test = test.apply(dist_cidades, axis = 1)

test["avg_bedrooms"] = test.total_bedrooms/test.households

test["bedrooms/rooms"] = test.total_bedrooms/test.total_rooms

test["people/bedroom"] = test.population/test.total_bedrooms



test = test.drop(["latitude", "longitude", "total_bedrooms", "total_rooms"], axis = 1)
train.head()
test.head()
scaler = MinMaxScaler()

sel_col = ['median_age', 'population', 'households', 'median_income', 'dist_to_city', "avg_bedrooms", "bedrooms/rooms", "people/bedroom"]

trainScaled = scaler.fit_transform(train[sel_col])

X_train, X_test, Y_train, Y_test = train_test_split(trainScaled, train['median_house_value'], test_size=0.20)
testScaled = scaler.fit_transform(test[sel_col])
def rmsle(y_test,y_pred):

    return np.sqrt(np.mean((np.log(np.abs(y_pred)+1) - np.log(np.abs(y_test)+1))**2))
scorer = make_scorer(rmsle, greater_is_better=False)
LR = LinearRegression()

LR.get_params()
grid_params_LR = {"fit_intercept":[True, False],"normalize":[True,False]}

grid_LR = GridSearchCV(LR,grid_params_LR,cv=10,scoring=scorer)
grid_LR.fit(X_train, Y_train)

print(grid_LR.best_estimator_)
LR = grid_LR.best_estimator_

LR.fit(X_train, Y_train)

Y_pred = LR.predict(X_test)

LR_RMSLE = rmsle(Y_pred, Y_test)

print("RMSLE:", LR_RMSLE)
LR_score = np.mean(cross_val_score(LR, X_train, Y_train, cv = 10))

LR_score
knn = KNeighborsRegressor()

knn.get_params()
grid_params_knn = {"n_neighbors":[i for i in range(1,31)],"weights":["uniform","distance"],"p":[1,2]}

grid_knn = GridSearchCV(knn,grid_params_knn,cv=10)
grid_knn.fit(X_train,Y_train)

print(grid_knn.best_estimator_)
knn = grid_knn.best_estimator_

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

knn_RMSLE = rmsle(Y_pred, Y_test)

print("RMSLE:", knn_RMSLE)
knn_score = np.mean(cross_val_score(knn, X_train, Y_train, cv = 10))

knn_score
ridge = Ridge()

ridge.get_params()
grid_params_ridge = {"alpha":np.linspace(0.5,10.5,101).tolist()}

grid_ridge = GridSearchCV(ridge,grid_params_ridge,cv=10,scoring=scorer)
grid_ridge.fit(X_train, Y_train)

print(grid_ridge.best_estimator_)
ridge = grid_ridge.best_estimator_

ridge.fit(X_train, Y_train)

Y_pred = ridge.predict(X_test)

ridge_RMSLE = rmsle(Y_pred, Y_test)

print("RMSLE:", ridge_RMSLE)
ridge_score = np.mean(cross_val_score(ridge, X_train, Y_train, cv = 10))

ridge_score
lasso = Lasso()

lasso.get_params()
grid_params_lasso = {"alpha":np.linspace(0.5,5.5,51).tolist(),"normalize":[True,False]}

grid_lasso = GridSearchCV(lasso,grid_params_lasso,cv=10,scoring=scorer)
grid_lasso.fit(X_train, Y_train)

print(grid_lasso.best_estimator_)
lasso = grid_lasso.best_estimator_

lasso.fit(X_train, Y_train)

Y_pred = lasso.predict(X_test)

lasso_RMSLE = rmsle(Y_pred, Y_test)

print("RMSLE:", lasso_RMSLE)
lasso_score = np.mean(cross_val_score(lasso, X_train, Y_train, cv = 10))

lasso_score
resultados = {"Regressor": ["LR", "kNN", "Ridge", "LASSO"],"RMSLE":[LR_RMSLE, knn_RMSLE, ridge_RMSLE, lasso_RMSLE], "Score":[LR_score, knn_score, ridge_score, lasso_score]}

table = pd.DataFrame(resultados, columns =  ["Regressor", "RMSLE", "Score"])

print(table)
knn.fit(X_train, Y_train)
YtestPred = knn.predict(test)

predict = pd.DataFrame(testId)

predict["median_house_value"] = YtestPred

predict.to_csv("pred_knn.csv", index = False, index_label = 'Id')