import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from geopy.distance import distance, geodesic

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import make_scorer

from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
train = pd.read_csv("../input/atividade-regressao-PMR3508/train.csv")

test = pd.read_csv("../input/atividade-regressao-PMR3508/test.csv")



train.shape
train.head()
train.isnull().sum()

train = train.drop("Id", axis = 1)

testId = test.Id

test = test.drop("Id", axis = 1)



train.head()
plt.figure(figsize=(14, 7))

train['median_house_value'].hist(rwidth=0.9)

plt.ylabel('Number of houses')

plt.xlabel('Value')

plt.title('Median house values distribution')
plt.figure(figsize=(14, 7))

train['median_age'].hist(rwidth=0.9)

plt.ylabel('Number of houses')

plt.xlabel('Median house age')

plt.title('Median house values distribution')
corr_mat = train.corr()

sns.set()

plt.figure(figsize=(15,15))

sns.heatmap(corr_mat, annot=True)
#Bedrooms

train["avgBeds"] = train.total_bedrooms/train.households

train["beds/rooms"] = train.total_bedrooms/train.total_rooms

#People per room and bedroom

train["pop/room"] = train.population/train.total_rooms

train["pop/beds"] = train.population/train.total_bedrooms

#% of Bedrooms

train["beds/rooms"] = train.total_bedrooms/train.total_rooms
#Longitude and Latitude from San Diego, San Francisco and Los Angeles taken from https://www.latlong.net

longLat_SF = (37.7749, -122.4194)

longLat_SD = (32.7157, -117.1611)

longLat_LA = (34.0522, -118.2437)



#funtion to calculate distance from location to closest big city

def distCity(df):

    longLatHouse = (df["latitude"], df["longitude"])

    df["closestCity"] = min(geodesic(longLatHouse, longLat_SF).km, geodesic(longLatHouse, longLat_SD).km, geodesic(longLatHouse,longLat_LA).km)   

    return df



train = train.apply(distCity,axis=1)
corr_mat = train.corr()

sns.set()

plt.figure(figsize=(15,15))

sns.heatmap(corr_mat, annot=True)
train = train.drop(["longitude","latitude","pop/room", "total_bedrooms", "total_rooms"], axis = 1)

plt.figure(figsize=(10,10))

sns.heatmap(train.corr(), annot=True, linewidths=0.2)
test["avgBeds"] = test.total_bedrooms/test.households

test["beds/rooms"] = test.total_bedrooms/test.total_rooms

test["pop/beds"] = test.population/test.total_bedrooms



test = test.apply(distCity, axis = 1)

test = test.drop(["latitude", "longitude", "total_bedrooms", "total_rooms"], axis = 1)
train.head()
test.head()
scaler = MinMaxScaler()

cols = ['median_age', 'population', 'households', 'median_income', 'closestCity', "avgBeds", "beds/rooms", "pop/beds"]

trainScaled = scaler.fit_transform(train[cols])

testScaled = scaler.fit_transform(test[cols])
X_train, X_test, Y_train, Y_test = train_test_split(trainScaled, train['median_house_value'], test_size=0.20)
def calcRMSLE(y_test,y_pred):

    return np.sqrt(np.mean((np.log(1+np.abs(y_pred)) - np.log(1+np.abs(y_test)))**2))



scorer = make_scorer(calcRMSLE, greater_is_better=False)
LR = LinearRegression()

grid_LR = GridSearchCV(LR,{"fit_intercept":[True, False],"normalize":[True,False]},cv=10,scoring=scorer)

grid_LR.fit(X_train, Y_train)
bestLR = grid_LR.best_estimator_

bestLR.fit(X_train, Y_train)

Y_pred = bestLR.predict(X_test)

bestLR_RMSLE = calcRMSLE(Y_pred, Y_test)

print("RMSLE:", bestLR_RMSLE)
bestLR_score = np.mean(cross_val_score(bestLR, X_train, Y_train, cv = 10))

print("score:",bestLR_score)
Ridge().get_params()
ridge = Ridge()

grid_ridge = GridSearchCV(ridge,{"alpha":np.linspace(0.5,10.5,101).tolist()},cv=10,scoring=scorer)

grid_ridge.fit(X_train, Y_train)

bestRidge = grid_ridge.best_estimator_

bestRidge.fit(X_train, Y_train)

Y_pred = bestRidge.predict(X_test)

bestRidge_RMSLE = calcRMSLE(Y_pred, Y_test)

print("RMSLE:", bestRidge_RMSLE)
bestRidge_score = np.mean(cross_val_score(bestRidge, X_train, Y_train, cv = 10))

print("Score:",bestRidge_score)
Lasso().get_params()
lasso = Lasso()

grid_lasso = GridSearchCV(lasso,{"alpha":np.linspace(0.5,5.5,51).tolist(),"normalize":[True,False]},cv=10,scoring=scorer)

grid_lasso.fit(X_train, Y_train)
bestLasso = grid_lasso.best_estimator_

bestLasso.fit(X_train, Y_train)

Y_pred = bestLasso.predict(X_test)

bestLasso_RMSLE = calcRMSLE(Y_pred, Y_test)

print("RMSLE:", bestLasso_RMSLE)
bestLasso_score = np.mean(cross_val_score(bestLasso, X_train, Y_train, cv = 10))

print("Score:",bestLasso_score)
KNeighborsRegressor().get_params()
knn = KNeighborsRegressor()

gridKNN = GridSearchCV(knn,{"n_neighbors":[i for i in range(1,31)],"weights":["uniform","distance"],"p":[1,2]},cv=10)

gridKNN.fit(X_train,Y_train)
gridKNN.best_estimator_
bestKnn = gridKNN.best_estimator_

bestKnn.fit(X_train, Y_train)

Y_pred = bestKnn.predict(X_test)

bestKnn_RMSLE = calcRMSLE(Y_pred, Y_test)

print("RMSLE:", bestKnn_RMSLE)
bestKnn_score = np.mean(cross_val_score(bestKnn, X_train, Y_train, cv = 10))

print("Score:",bestKnn_score)
final = {"Regressor": ["LR", "Ridge", "LASSO", "kNN"],"RMSLE":[bestLR_RMSLE,bestRidge_RMSLE,bestLasso_RMSLE,bestKnn_RMSLE], "Score":[bestLR_score, bestRidge_score,bestLasso_score,bestKnn_score]}

compilation = pd.DataFrame(final, columns =  ["Regressor", "RMSLE", "Score"])

print(compilation)
#Fit on train set

bestKnn.fit(X_train, Y_train)

#Predict the test set

YtestPred = bestKnn.predict(test)

#Save in the output format

predictions = pd.DataFrame(testId)

predictions["median_house_value"] = YtestPred

predictions.to_csv("pred_knn.csv", index = False, index_label = 'Id')