
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#Importation des librairies
import seaborn as sns; sns.set(style="ticks", color_codes=True)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
#from sklearn.utils import check_arrays
from math import sqrt
# Lire les données
df_train = pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip")
df_test = pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv.zip")
df_store = pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv")
df_features = pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv.zip")
#df_train = df_train.merge(df_store, on='Store', how='left')
# Visiulaser les données
df_features.head()
df_features.describe()
df_train.describe()
# Fonction pour la création des shémas
def scatter(column):
    plt.figure()
    plt.scatter(df_fusion[column] , df_train['Weekly_Sales'])
    plt.ylabel('weeklySales')
    plt.xlabel(column)
# La création de l'erreur WMAE
def WMAE(dataset, real, predicted):
    weights = dataset.IsHoliday.apply(lambda x: 5 if x else 1)
    return np.round(np.sum(weights*abs(real-predicted))/(np.sum(weights)), 2)
# La créastion de l'erreur RMSE
def rmse(val_y, y_pred) :
    return sqrt(mean_squared_error(val_y, y_pred))
# Fusionner les données
df = pd.merge(df_features,df_store,on='Store',how='inner')
df_fusion = pd.merge(df,df_train,on=['Store','Date','IsHoliday']) 
# L'utilisation des 3 critères 'Store','Date','IsHoliday' en même temps aide à optimiser l'utilisation 
#de la RAM sinon Kaggle donne l'erreur 'Your notebook tried to allocate more memory than is available. It has restarted.'
 
df_fusion.head()
df_fusion.info()
sns.jointplot(x="Store", y = "Weekly_Sales", data = df_fusion)
df_fusion.describe()
# Elimination des valeurs des Weekly_Sales qui sont inf à 0
#df_fusion.drop(df_fusion[df_fusion['Weekly_Sales'] < 0].index, inplace = True)
df_fusion.describe()
# Préparation des données pour la méthodes LSTM où nous allons traiter les magazins individuelement 
L = []
for i in range(45) :
    L += [df_fusion[df_fusion['Store'] == i+1 ]]
L[0].describe()
#for i in range(45) :
#    print('the store ',i,'has :', L[i].shape[0],'lines')

df_fusion['Date'] = pd.to_datetime(df_fusion['Date'])
df_fusion['Year'] = pd.to_datetime(df_fusion['Date']).dt.year
df_fusion['Month'] = pd.to_datetime(df_fusion['Date']).dt.month
df_fusion['Week'] = pd.to_datetime(df_fusion['Date']).dt.week
df_fusion['Day'] = pd.to_datetime(df_fusion['Date']).dt.day
df_fusion.replace({'A': 1, 'B': 2,'C':3},inplace=True)

# La valeur à prédir
y = df_fusion.Weekly_Sales
# Identifier les critères que l'alghoritme doit prendre en compte
features = ['Dept', 'IsHoliday', 'Size','Temperature','Fuel_Price','CPI','Unemployment','Type']
# Au début je n'ai pas pu utliser ces critères 'Temperature','Fuel_Price','CPI','Unemployment'. En les ajoutant 
#j'ai pu optimiser le 'mean absolute error' par 630 dans la méthode 'RandomForestRegressor'
#et 100 dans la méthode 'DecisionTreeRegressor' et 650 dans la méthode 'XGBRegressor'

X = df_fusion[features]
            
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


# Les shémats
scatter('Fuel_Price')
scatter('Size')
scatter('CPI')
scatter('Type')
scatter('IsHoliday')
scatter('Unemployment')
scatter('Temperature')
scatter('Store')
scatter('Dept')
weekly_sales_2010 = df_fusion[df_fusion.Year==2010]['Weekly_Sales'].groupby(df_fusion['Week']).mean()
weekly_sales_2011 = df_fusion[df_fusion.Year==2011]['Weekly_Sales'].groupby(df_fusion['Week']).mean()
weekly_sales_2012 = df_fusion[df_fusion.Year==2012]['Weekly_Sales'].groupby(df_fusion['Week']).mean()
plt.figure(figsize=(20,8))
sns.lineplot(weekly_sales_2010.index, weekly_sales_2010.values)
sns.lineplot(weekly_sales_2011.index, weekly_sales_2011.values)
sns.lineplot(weekly_sales_2012.index, weekly_sales_2012.values)
plt.grid()
plt.xticks(np.arange(1, 53, step=1))
plt.legend(['2010', '2011', '2012'], loc='best', fontsize=16)
plt.title('Les ventes hebdomadaires pour chaque année', fontsize=18)
plt.ylabel('Ventes', fontsize=16)
plt.xlabel('Semaine', fontsize=16)
plt.show()
# Méthode Machine Learning
# Méthode RandomForestRegressor
iowa_model = RandomForestRegressor(random_state=1)
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
val_mse = mean_squared_error(val_y, val_predictions)
val_rmse = rmse(val_y, val_predictions)
val_r2 = r2_score(val_y, val_predictions)
val_wmae = WMAE(val_X, val_y, val_predictions)
print('Erreur MAE RandomForestRegressor: ', val_mae)
print('Erreur MSE RandomForestRegressor: ', val_mse)
print('Erreur RMSE RandomForestRegressor: ', val_rmse)
print('Erreur R2 RandomForestRegressor: ', val_r2)
print('Erreur WMAE RandomForestRegressor: ', val_wmae)


# Méthode DecisionTreeRegressor

iowa_model_tree = DecisionTreeRegressor(random_state=1)
iowa_model_tree.fit(train_X, train_y)
val_predictions_tree = iowa_model_tree.predict(val_X)
val_mae_tree = mean_absolute_error(val_predictions_tree, val_y)
val_mse_tree = mean_squared_error(val_y, val_predictions_tree)
val_rmse_tree = rmse(val_y, val_predictions_tree)
val_r2_tree = r2_score(val_y, val_predictions_tree)
val_wmae_tree = WMAE(val_X, val_y, val_predictions_tree)
print('Erreur MAE DecisionTreeRegressor: ', val_mae_tree)
print('Erreur MSE DecisionTreeRegressor: ', val_mse_tree)
print('Erreur RMSE DecisionTreeRegressor: ', val_rmse_tree)
print('Erreur R2 DecisionTreeRegressor: ', val_r2_tree)
print('Erreur WMAE DecisionTreeRegressor: ', val_wmae_tree)


#Méthode XGBRegressor
from xgboost import XGBRegressor
my_model = XGBRegressor(n_estimators=500)
my_model.fit(train_X, train_y, 
             early_stopping_rounds=5, 
             eval_set=[(val_X, val_y)], 
             verbose=False)
val_predictions_XG = my_model.predict(val_X)
val_mae_XG = mean_absolute_error(val_predictions_XG, val_y)
val_mse_XG = mean_squared_error(val_y, val_predictions_XG)
val_rmse_XG = rmse(val_y, val_predictions_XG)
val_r2_XG = r2_score(val_y, val_predictions_XG)
val_wmae_XG = WMAE(val_X, val_y, val_predictions_XG)
print('Erreur MAE XGBRegressor: ', val_mae_XG)
print('Erreur MSE XGBRegressor: ', val_mse_XG)
print('Erreur RMSE XGBRegressor: ', val_rmse_XG)
print('Erreur R2 XGBRegressor: ', val_r2_XG)
print('Erreur WMAE XGBRegressor: ', val_wmae_XG)

# Mise à l'échelle de la dataset
#sc = MinMaxScaler(feature_range=(0,1))
#training_set_scaled = sc.fit_transform(train_X)
#val_X = sc.fit_transform(val_X)
#print(val_X)
#print(training_set_scaled)
# LSTM en utilisant tous les données
#X_train, y_train = np.array(training_set_scaled), np.array(train_y)
#val_X, val_y = np.array(val_X), np.array(val_y)

#val_X = np.reshape(val_X, (val_X.shape[0],val_X.shape[1],1))
#X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
#print(X_train)

#LSTM en utilisant que les weekly_sales
W = L[1].Weekly_Sales
X1_train, X1_test = train_test_split(W,test_size = 0.2)


X1_train = X1_train.values.reshape(-1,1)
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(X1_train)


X_train = []
y_train = []
# Nous avons pris une longueur sup à 52 pour que le programme soit capable d'identifier plus précisement les variations
# des ventes durant toute l'année
for i in range(60,7000):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
regressor = Sequential()
# Première couche LSTM 
regressor.add(LSTM(units=200, return_sequences=True, input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))
# 2éme couche
regressor.add(LSTM(units=200, return_sequences=True))
regressor.add(Dropout(0.2))
# 3éme couche
regressor.add(LSTM(units=200, return_sequences=True))
regressor.add(Dropout(0.2))
# 4éme couche
regressor.add(LSTM(units=200))
regressor.add(Dropout(0.2))
# Couche de sortie
regressor.add(Dense(units=1))

# Compilation du RNN
regressor.compile(optimizer='rmsprop',loss='mean_squared_error')
# Adaptation
regressor.fit(X_train,y_train)
#val_X = sc.fit_transform(val_X)
X1_test = X1_test.values.reshape(-1,1)
X1_test = sc.fit_transform(X1_test)

X_test = []
y_test = []
for i in range(60,400):
    X_test.append(X1_test[i-60:i,0])
    y_test.append(X1_test[i,0])
X_test = np.array(X_test)
y_test = np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
val_prediction_LSTM = regressor.predict(X_test)
val_y = np.array(y_test)

val_mae_LSTM = mean_absolute_error(val_prediction_LSTM, y_test)
val_mse_LSTM = mean_squared_error(val_y, val_prediction_LSTM)
val_rmse_LSTM = rmse(val_y, val_prediction_LSTM)
val_r2_LSTM = r2_score(val_y, val_prediction_LSTM)
print('Erreur MAE LSTM: ', val_mae_LSTM)
print('Erreur MSE LSTM: ', val_mse_LSTM)
print('Erreur RMSE LSTM: ', val_rmse_LSTM)
print('Erreur R2 LSTM: ', val_r2_LSTM)
