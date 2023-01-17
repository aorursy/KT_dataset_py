import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import fbeta_score, make_scorer, mean_squared_log_error
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import linear_model
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import geopy
from sklearn.neighbors import RadiusNeighborsRegressor
from keras.models import Sequential 
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_regression
df = pd.read_csv("../input/train.csv",
    sep=r'\s*,\s*',
    engine='python',
    na_values="NaN")
test = pd.read_csv("../input/test.csv",
    sep=r'\s*,\s*',
    engine='python',
    na_values="NaN")
score = make_scorer(mean_squared_log_error)
X = df.drop('median_house_value', axis = 1)
X = X.drop('Id', axis = 1)
X = StandardScaler().fit_transform(X)
Y = df['median_house_value']
reg = LinearRegression().fit(X, Y)
scores = cross_val_score(reg, X, Y, cv=100)
scores.mean()
X = pd.DataFrame()
RP = df['total_rooms']/df['population']
testRP = test['total_rooms']/test['population']
RH = df['total_rooms']/df['households']
testRH = test['total_rooms']/test['households']
for i in range(13):
    x = str(i) + 'latitude'
    X[x] = df['latitude']
    x = str(i) + 'longitude'
    X[x] = df['longitude']
for j in range(1):
    x = str(j) + 'RP'
    X[x] = RP
for k in range(1):
    x = str(k) + 'RH'
    X[x] = RH
for l in range(1):
    x = str(l) + 'median_income'
    X[x] = df['median_income']
X = StandardScaler().fit_transform(X)
Y = df['median_house_value']
knn = KNeighborsRegressor(n_neighbors=12, p = 1, weights = 'distance')
scores = cross_val_score(knn, X, Y, cv=100, scoring = score)
scores = np.sqrt(scores)
scores.mean()
Xtest = pd.DataFrame()
for i in range(13):
    x = str(i) + 'latitude'
    Xtest[x] = test['latitude']
    x = str(i) + 'longitude'
    Xtest[x] = test['longitude']
for j in range(1):
    x = str(j) + 'RP'
    Xtest[x] = testRP
for k in range(1):
    x = str(k) + 'RH'
    Xtest[x] = testRH
for l in range(1):
    x = str(l) + 'median_income'
    Xtest[x] = test['median_income']
Xtest = StandardScaler().fit_transform(Xtest)
knn.fit(X,Y)
testPred = knn.predict(Xtest)
train, test1 = train_test_split(df, test_size=0.2)
X = train.drop('median_house_value', axis = 1)
X = X.drop('Id', axis = 1)
Y = train['median_house_value'].values
model = Sequential()
model.add(Dense(4, input_dim=8, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')
model.fit(X, Y, epochs=150, verbose=2)
Xtest1 = test1.drop('median_house_value', axis = 1)
Xtest1 = Xtest1.drop('Id', axis = 1)
Ytest1 = test1['median_house_value'].values
Ytest = model.predict(Xtest1)
mean_squared_log_error(Ytest1, Ytest)**(1/2)  
testPred = knn.predict(Xtest)
sid = test.Id.values
w = pd.DataFrame(sid)
w.columns = ["Id"]

w['median_house_value'] = testPred
w.to_csv("resposta.csv", index = False)
