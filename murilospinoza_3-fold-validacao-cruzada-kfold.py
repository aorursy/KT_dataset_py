import pandas as pd
import numpy as np
import statistics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import KFold

dados = pd.read_csv("../input/hbatcsv/HBAT.csv")

# Variável independente
X = pd.DataFrame(dados[["x9","x11","x12"]])

# Variável dependente
Y = pd.DataFrame(dados["x19"])

linearRegressor = LinearRegression()

kf = KFold(n_splits=3)
kf.get_n_splits(X)

mae = []
mse = []
rmse = []
model = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

    #Regressão Linear
    model.append(linearRegressor.fit(X_train, y_train))
    y_pred = linearRegressor.predict(X_test)

    mae.append(metrics.mean_absolute_error(y_pred, y_test))
    mse.append(metrics.mean_squared_error(y_pred, y_test))
    rmse.append(np.sqrt(metrics.mean_squared_error(y_pred, y_test)))
    
    print('\nIntercepto ou Coeficiente Linear\n')
    print(linearRegressor.intercept_)
    
    print('\nCoeficiente Angular (slope)\n')
    print(linearRegressor.coef_)
    
    print('\nPrevisão\n')
    print(y_pred)

print('\n')
print("MAE médio: ", np.mean(mae), " MAE Desvio padrão: ", np.sqrt(np.var(mae)))
print("MSE médio: ", np.mean(mse), " MSE Desvio padrão: ", np.sqrt(np.var(mse)))
print("RMSE médio: ", np.mean(rmse), " RMSE Desvio padrão: ", np.sqrt(np.var(rmse)))

