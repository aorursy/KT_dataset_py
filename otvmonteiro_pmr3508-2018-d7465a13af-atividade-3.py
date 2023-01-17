import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model # Regressores lineares: ordinário, Lasso e Ridge
from sklearn.neighbors import KNeighborsRegressor # Regressor por KNN

from sklearn.model_selection import cross_val_score
path='../input/train.csv'
data=pd.read_csv(path)
data.head(8)
yTrain=data['median_house_value']
from sklearn import preprocessing

latLong=data.drop(columns=['Id','median_age','total_rooms','total_bedrooms','population','households','median_income','median_house_value'])
localizacao=KNeighborsRegressor(n_neighbors=15).fit(latLong,yTrain)
# scoreLocalizacao=cross_val_score(localizacao, latLong, yTrain, cv=10)
# print('Score: ',scoreLocalizacao.mean())
xTrain=data.drop(columns=['Id','median_house_value','longitude','latitude'])
xTrain['localizacao']=preprocessing.scale(localizacao.predict(latLong))
xTrain['bedrooms_per_house']=xTrain['total_bedrooms']/xTrain['households']
xTrain['rooms']=preprocessing.scale(
    (xTrain['total_rooms']-xTrain['total_bedrooms'])/xTrain['households']
                            )

xTrain
#para avaliacao dos modelos
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import make_scorer

msle=make_scorer(mean_squared_log_error)
lmo=linear_model.LinearRegression().fit(xTrain, yTrain)
lmRidge=linear_model.Ridge(alpha=1.2).fit(xTrain, yTrain)
lmLasso=linear_model.Lasso(alpha=10).fit(xTrain, yTrain)
KNN=KNeighborsRegressor(n_neighbors=10).fit(xTrain, yTrain)

scoreLinear=cross_val_score(lmo, xTrain, yTrain, cv=10, scoring=msle)
scoreRidge=cross_val_score(lmRidge, xTrain, yTrain, cv=10, scoring=msle)
scoreLasso=cross_val_score(lmLasso, xTrain, yTrain, cv=10, scoring=msle)
scoreKNN=cross_val_score(KNN, xTrain, yTrain, cv=10, scoring=msle)

print("Linear Regression Score    :  ", scoreLinear.mean())
print("Ridge Regression Score     :  ", scoreRidge.mean())
print("Lasso Regression Score     :  ", scoreLasso.mean())
print("KNeighbors Regression Score:  ", scoreKNN.mean())
testData = pd.read_csv("../input/test.csv")
#adaptando dados as tranformacoes feitas na base de treino
latLongTest=testData.drop(columns=['Id','median_age','total_rooms','total_bedrooms','population','households','median_income'])
xTest=testData.drop(columns=['Id','longitude','latitude'])
xTest['localizacao']=preprocessing.scale(localizacao.predict(latLongTest))
xTest['bedrooms_per_house']=xTest['total_bedrooms']/xTest['households']
xTest['rooms']=preprocessing.scale(
    (xTest['total_rooms']-xTest['total_bedrooms'])/xTest['households']
                            )
xTest
yTest = lmLasso.predict(xTest)
submission = pd.DataFrame(columns=['Id','median_house_value'])
submission.Id = testData.Id
submission.median_house_value = yTest
submission
submission.to_csv("submission.csv", index=False)
