# Importação das bibliotecas 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import warnings 
import math
import seaborn as sns

#Lendo a base de treino
traindata = pd.read_csv("../input/california-houses/train.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
#Modificando a visalização da base de treino
traindata.iloc[0:20,:]
traindata = traindata.drop(columns = ['Id'])
traindata.info()
traindata[traindata['total_bedrooms'].isnull()]
traindata.loc[290]
# Gráficos de barra das variáveis 
traindata.hist(bins=50, figsize=(20,20))
plt.show()
traindata.corr()
plt.figure(figsize=(6,6))
plt.title('Matriz de correlação')
sns.heatmap(traindata.corr(), annot=True, linewidths=0.1)
rooms_n_bedrooms = traindata['total_rooms']/traindata['total_bedrooms']
prices = traindata['median_house_value']
plt.scatter(rooms_n_bedrooms,prices)

median_income = traindata['median_income']
plt.scatter(median_income,prices)
median_age = traindata['median_age']
plt.scatter(median_age,prices)
plt.scatter(traindata['latitude'],traindata['longitude'])
def per_capita(row):
    row['per_capita'] = row['median_income'] / row['population'] 
    return row
traindata = traindata.apply(per_capita, axis=1)
size_train = traindata.shape
trainY = traindata['median_house_value']
knn_trainX = traindata[["longitude","total_rooms","total_bedrooms","population","households","median_income"]]

from sklearn.metrics import mean_squared_error as mse
def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5
from sklearn.neighbors import KNeighborsRegressor
neighbor = KNeighborsRegressor(n_neighbors=2)
neighbor.fit(knn_trainX,trainY) 
knn_predict = neighbor.predict(knn_trainX)
df_knn = pd.DataFrame({'Y_real':trainY[:],'Y_pred':knn_predict[:]})
print(rmsle(df_knn.Y_real,df_knn.Y_pred))
lasso_trainX = traindata[["longitude","median_age","total_rooms","total_bedrooms","population","households","median_income"]]
warnings.filterwarnings("ignore")
from sklearn import linear_model
clf = linear_model.Lasso(alpha=0.5)
clf.fit(lasso_trainX,trainY)
lasso_predict = clf.predict(lasso_trainX)
df_l = pd.DataFrame({'Y_real':trainY[:],'Y_pred':lasso_predict[:]})
for i in range (len(df_l.Y_real)):
    if df_l.Y_pred[i] < 0:
        aux = df_l.Y_pred[i]*(-1)
        df_l.ix[i,'Y_pred'] = aux
print(rmsle(df_l.Y_real,df_l.Y_pred))
ridge_trainX = traindata[["longitude","median_age","total_rooms","total_bedrooms","population","households","median_income"]]
from sklearn.linear_model import Ridge
clf = Ridge(alpha=1.0)
clf.fit(ridge_trainX,trainY) 
ridge_predict = clf.predict(ridge_trainX)
df_r = pd.DataFrame({'Y_real':trainY[:],'Y_pred':ridge_predict[:]})
for i in range (len(df_r.Y_real)):
    if df_r.Y_pred[i] < 0:
        aux = df_r.Y_pred[i]*(-1)
        df_r.ix[i,'Y_pred'] = aux
print(rmsle(df_r.Y_real,df_r.Y_pred))
def rooms_pop(row):
    row['rooms_pop'] = row['total_rooms'] / row['population'] 
    return row
traindata = traindata.apply(rooms_pop, axis=1)
traindata = traindata.drop(['population'], axis=1)
plt.figure(figsize=(6,6))
plt.title('Matriz de correlação')
sns.heatmap(traindata.corr(), annot=True, linewidths=0.1)
def age_rooms(row):
    row['age_rooms'] = row['median_age'] / row['total_rooms'] 
    return row
traindata = traindata.apply(age_rooms, axis=1)
traindata = traindata.drop(['median_age'], axis=1)
plt.figure(figsize=(6,6))
plt.title('Matriz de correlação')
sns.heatmap(traindata.corr(), annot=True, linewidths=0.1)
traindata = traindata.drop(columns = ['latitude','longitude'])
newknn_trainX = traindata[["per_capita","total_bedrooms","rooms_pop","households","age_rooms"]]
neighbor = KNeighborsRegressor(n_neighbors=2)
neighbor.fit(newknn_trainX,trainY) 
knn_predict = neighbor.predict(newknn_trainX)
df_knn = pd.DataFrame({'Y_real':trainY[:],'Y_pred':knn_predict[:]})
print(rmsle(df_knn.Y_real,df_knn.Y_pred))
newlasso_trainX = traindata[["per_capita","total_bedrooms","rooms_pop","households","age_rooms"]]
warnings.filterwarnings("ignore")
from sklearn import linear_model
clf = linear_model.Lasso(alpha=0.5)
clf.fit(newlasso_trainX,trainY)
lasso_predict = clf.predict(newlasso_trainX)
df_l = pd.DataFrame({'Y_real':trainY[:],'Y_pred':lasso_predict[:]})
for i in range (len(df_l.Y_real)):
    if df_l.Y_pred[i] < 0:
        aux = df_l.Y_pred[i]*(-1)
        df_l.ix[i,'Y_pred'] = aux
print(rmsle(df_l.Y_real,df_l.Y_pred))
newridge_trainX = traindata[["per_capita","total_bedrooms","rooms_pop","households","age_rooms"]]
from sklearn.linear_model import Ridge
clf = Ridge(alpha=1.0)
clf.fit(newridge_trainX,trainY) 
ridge_predict = clf.predict(newridge_trainX)
df_r = pd.DataFrame({'Y_real':trainY[:],'Y_pred':ridge_predict[:]})
for i in range (len(df_r.Y_real)):
    if df_r.Y_pred[i] < 0:
        aux = df_r.Y_pred[i]*(-1)
        df_r.ix[i,'Y_pred'] = aux
print(rmsle(df_r.Y_real,df_r.Y_pred))
# Submissão do arquivo teste
testdata = pd.read_csv("../input/california-houses/test.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")

testdata = testdata.apply(per_capita, axis =1)
testdata = testdata.apply(rooms_pop, axis = 1)
testdata = testdata.apply(age_rooms, axis = 1)
testX = testdata[["per_capita","total_bedrooms","rooms_pop","households","age_rooms"]]
testX.shape
testdata.head()
x_val_test = testX
y_val_test = neighbor.predict(x_val_test)

dfSave = pd.DataFrame(data={"Id" : testdata["Id"], "median_house_value" : y_val_test})
dfSave['Id'] = dfSave['Id'].astype(int)
pd.DataFrame(dfSave[["Id", "median_house_value"]], columns = ["Id", "median_house_value"]).to_csv("Output.csv", index=False)
dfSave.head()
