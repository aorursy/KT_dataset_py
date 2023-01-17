import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
data_train = pd.read_csv("../input/train.csv")
data_train.head()
data_train = data_train.drop('Id', axis =1)
data_train.head()
data_train['average_rooms'] = data_train['total_rooms']/data_train['population']
data_train['average_bedrooms'] = data_train['total_bedrooms']/data_train['population']
data_train['average_households'] = data_train['households']/data_train['population']
data_train['average_median_income'] = data_train['median_income']/data_train['population']
data_train.head()
correlation=(data_train.corr()["median_house_value"])
correlation
data_train = data_train[['latitude','total_rooms','median_income','average_rooms','average_households','average_median_income','median_house_value']]
data_train.head()
Y_train = data_train["median_house_value"]
Y_train = Y_train[:-48]
Y_test = data_train["median_house_value"]
Y_test = Y_test[-48:]
data_train = data_train.drop('median_house_value', axis =1)
X_train = data_train[:-48] # deixa os primeiros 14400 dados de treino
X_test = data_train[-48:] # deixa os ultimos 48 dados de teste
X_train.head()
Y_train.head()
X_test.head()
Y_test.head()
from sklearn.model_selection import cross_val_score as cvs
from sklearn.neighbors import KNeighborsRegressor as knr
features = ['latitude','total_rooms','median_income','average_rooms','average_households','average_median_income']
Xtrain = X_train[features]
Ytrain = Y_train#["median_house_value"]
knn_scores = []
for i in range(1,100,5):
    knn = knr(n_neighbors = i)
    scores = cvs(knn, Xtrain, Ytrain, scoring='neg_mean_squared_error',
                cv = 5)
    knn_scores.append([i, -scores.mean()])
knn_scores = np.array(knn_scores)
knn_scores
plt.plot(knn_scores[:,0], knn_scores[:,1])
knn_scores[np.where(knn_scores[:,1] == np.amin(knn_scores[:,1]))[0]]
from sklearn.linear_model import Ridge
features = ['latitude','total_rooms','median_income','average_rooms','average_households','average_median_income']
Xtrain = X_train[features]
Ytrain = Y_train#["median_house_value"]
r = Ridge(alpha=1.0)
scores = cvs(r, Xtrain, Ytrain,
             scoring='neg_mean_squared_error',cv = 5)
print(-scores.mean())
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
features = ['latitude','total_rooms','median_income','average_rooms','average_households','average_median_income']
Xtrain = X_train[features]
Ytrain = Y_train #median_house_value"]
l = Lasso(alpha=1.0)
scores = cvs(l, Xtrain, Ytrain,
             scoring='neg_mean_squared_error',cv = 5)
print(-scores.mean())
x = np.arange(3)
scores = [6.93392478e+09, 5875799711.338691, 5875824057.38965]

fig, ax = plt.subplots()
plt.bar(x, scores)
plt.xticks(x, ('Knn', 'Ridge', 'Lasso'))
plt.title("RMSE")
plt.show()
r = Ridge(alpha=1.0)
features = ['latitude','total_rooms','median_income','average_rooms','average_households','average_median_income']
Xtrain = X_train[features]
Ytrain = Y_train #["median_house_value"]
Xtest = X_test[features]
Ytest = Y_test#["median_house_value"]
r.fit(Xtrain, Ytrain)
Predict = r.predict(Xtest)
from sklearn.metrics import mean_squared_log_error
Predict = pd.Series(Predict)
Predict.describe()

for i in range(Predict.shape[0]-1):
    if(Predict[i] < 0):
        Predict[i] = 0
Predict.describe()
from math import log, sqrt
Sum = 0
for i in range(Predict.shape[0]-1):
    Sum +=  (log(1+Predict[i]) - log(1+np.array(Ytest)[i]))*(log(1+Predict[i]) - log(1+np.array(Ytest)[i]))
score = sqrt(Sum/Predict.shape[0])
print(score)
data_test = pd.read_csv("../input/test.csv")
data_test.head()
data_test['average_rooms'] = data_test['total_rooms']/data_test['population']
data_test['average_bedrooms'] = data_test['total_bedrooms']/data_test['population']
data_test['average_households'] = data_test['households']/data_test['population']
data_test['average_median_income'] = data_test['median_income']/data_test['population']
data_test.head()
features = ['latitude','total_rooms','median_income','average_rooms','average_households','average_median_income']

X_final = data_test[features]
Y_final = r.predict(X_final)

for i in range(Y_final.shape[0]-1):
    if(Y_final[i] < 0):
        Y_final[i] = 0
Export = pd.DataFrame(columns = ["Id", "median_house_value"])
Export["Id"] = data_test["Id"]
Export["median_house_value"] = Y_final
Export.to_csv("Export.csv")
Export
