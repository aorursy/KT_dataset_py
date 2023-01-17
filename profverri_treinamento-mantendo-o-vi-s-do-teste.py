import pandas as pd
import numpy as np
training = pd.read_csv(".data/training.csv.gz")
test = pd.read_csv(".data/test.csv.gz")
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.preprocessing import StandardScaler

Xtrain = training.drop(columns = ["FLIGHT", "MAINTENANCE", "AC"])
ytrain = training.MAINTENANCE
selector = SelectKBest(mutual_info_regression, k = 200)
selector.fit(Xtrain, ytrain)
Xtrain = selector.transform(Xtrain)

scaler = StandardScaler()
scaler.fit(Xtrain)

Xtrain = scaler.transform(Xtrain)
from sklearn.neural_network import MLPRegressor
regressor = MLPRegressor((100,100,))
regressor.fit(Xtrain, ytrain)
yexpected = test.MAINTENANCE
Xtest = scaler.transform(selector.transform(test.drop(columns = ["FLIGHT", "MAINTENANCE", "AC"])))
ypredicted = regressor.predict(Xtest)
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(yexpected, ypredicted))
test['PREDICTED'] = ypredicted
test[['FLIGHT', 'PREDICTED']].to_csv(".data/test,result.csv.gz", index = False)
