import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error

import seaborn as sns

from datetime import datetime

from datetime import date

from sklearn.metrics import mean_squared_log_error



from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import GradientBoostingRegressor

import xgboost as xgb





%matplotlib inline
df = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
df.head()
print(df.shape)
df.describe()
df.info()
df_ = df.drop(['id','vendor_id','store_and_fwd_flag'],axis=1)
id_test = test['id']

test = test.drop(['id','vendor_id','store_and_fwd_flag'],axis=1)
df_.head()
df_['Hora'] = ([y[0:2] for x,y in df_['pickup_datetime'].str.split('\s').values])

test['Hora'] = ([y[0:2] for x,y in test['pickup_datetime'].str.split('\s').values])
df_['Data'] = ([datetime.strptime(x, '%Y-%m-%d').date() for x,y in df_['pickup_datetime'].str.split('\s').values])

test['Data'] = ([datetime.strptime(x, '%Y-%m-%d').date() for x,y in test['pickup_datetime'].str.split('\s').values])
df_['Dia'] = ([x.day for x in df_['Data']])

test['Dia'] = ([x.day for x in test['Data']])
df_['Mês'] = ([x.month for x in df_['Data']])

test['Mês'] = ([x.month for x in test['Data']])
df_['Mês'].unique()
dias = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-Feira', 'Sexta-feira', 'Sábado', 'Domingo']

#df_['Dia_Semana'] = ([dias[x.weekday()] for x in df_['Data']])

df_['Dia_Semana'] = ([x.weekday() for x in df_['Data']])

test['Dia_Semana'] = ([x.weekday() for x in test['Data']])
#df_['store_and_fwd_flag'] = np.where(df_['store_and_fwd_flag'] == 'N',0 , 1)

#test['store_and_fwd_flag'] = np.where(test['store_and_fwd_flag'] == 'N',0 , 1)
def turno(x):

    Turno = ['Manhã','Almoço','Tarde-Noite','Noite','Madrugada']

    if(7 <= x <= 11):

        return 0

    elif(12 <= x <= 14):

        return 1

    elif(15 <= x <= 19):

        return 2

    elif(20 <= x <= 23):

        return 3

    elif(0 <= x <= 6):

        return 4
df_['Turno'] = ([turno(int(x)) for x in df_['Hora']])

test['Turno'] = ([turno(int(x)) for x in test['Hora']])
df_.head()
df_.shape, test.shape
df_['passenger_count'].unique()
time = df_.set_index('trip_duration')



cols = ['passenger_count','Turno','Dia','Mês','Dia_Semana']



for c in cols:

    plt.figure()

    plt.title(c)

    time[c].plot(kind='hist')

    plt.show()
df_.drop(['dropoff_datetime', 'pickup_datetime','Data'],axis=1, inplace=True)

test.drop(['pickup_datetime','Data'],axis=1, inplace=True)
df_.head()
def distancia(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):

    

    #Definindo o raio da Terra (km)

    R_terra = 6371

    #Convertendo graus para radianos

    inicio_lat, inicio_lon, fim_lat, fim_lon = map(np.radians,[pickup_lat, pickup_lon, dropoff_lat, dropoff_lon])

    #Calculando as distancias de lat e long 

    dlat = fim_lat - inicio_lat

    dlon = fim_lon - inicio_lon

    

    #Calculando distancia haversine

    d = np.sin(dlat/2.0)**2 + np.cos(inicio_lat) * np.cos(fim_lat) * np.sin(dlon/2.0)**2

    

    return 2 * R_terra * np.arcsin(np.sqrt(d))
df_['Distancia'] = distancia(df_['pickup_latitude'], df_['pickup_longitude'], 

                                   df_['dropoff_latitude'] , df_['dropoff_longitude'])

test['Distancia'] = distancia(test['pickup_latitude'], test['pickup_longitude'], 

                                   test['dropoff_latitude'] , test['dropoff_longitude'])
df_['Duração'] = df_['trip_duration']

df_.drop(['trip_duration'],axis=1, inplace=True)
df_.head()
test.head()
X = df_.values[:,:-1]

y = np.log(df_['Duração'].values + 1)
y.min()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
y[0:5]
corr = df_.corr()



sns.heatmap(corr)

plt.show()
corr
corr_matrix = corr

corr_matrix["Duração"].sort_values(ascending=False)
std = StandardScaler()

X_train_str = std.fit_transform(X_train)

X_test_str = std.transform(X_test)
models = []



models.append(('Linear', LinearRegression()))

models.append(('GBooting',GradientBoostingRegressor()))

models.append(('RFR', RandomForestRegressor(n_estimators=10)))

#models.append(('SVR', SVR()))

models.append(('DTR', DecisionTreeRegressor()))

#models.append(('KNN',KNeighborsRegressor))
rmse_calc = []

rmsle = []



for nome,model in models:

    print(nome)

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    rmse_calc.append((nome, np.sqrt(mean_squared_error(y_test,y_pred))))

    rmsle.append((nome,np.sqrt(mean_squared_log_error(y_test,y_pred))))
rmse_str = []

rmsle_str = []



for nome,model in models:

    print(nome)

    model.fit(X_train_str,y_train)

    y_pred = model.predict(X_test_str)

    rmse_str.append((nome, np.sqrt(mean_squared_error(y_test,y_pred))))

    rmsle_str.append((nome,np.sqrt(mean_squared_log_error(y_test,y_pred))))
rmsle_str
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
watchlist = [(dtrain, 'train'), (dtest,'valid')]

xgb_pars = {'min_child_weight': 10, 'eta': 0.03, 'colsample_bytree': 0.3, 'max_depth': 10,

            'subsample': 0.8, 'lambda': 0.5, 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,

            'eval_metric': 'rmse', 'objective': 'reg:linear'}
model = xgb.train(xgb_pars, dtrain, 1000, watchlist,early_stopping_rounds=90,maximize=False, verbose_eval=100)
y_pred = model.predict(dtest)
y_pred
rmse_xgb = np.sqrt(mean_squared_error(y_test,y_pred))
rmsle_xgb = np.sqrt(mean_squared_log_error(y_test,y_pred))
rmse_calc
rmsle
nome_modelos = list(zip(*rmse_calc))[0]

index = np.arange(len(nome_modelos)) 



results_rmse = list(zip(*rmse_calc))[1]



bar_width = 0.55

opacity = 0.4

plt.figure(figsize=(12, 6))

plt.bar(index, results_rmse, bar_width, alpha=opacity, color='b', label='RMSE')

plt.xticks(index, nome_modelos) 

plt.xlabel('Modelos')

plt.ylabel('RSME')

plt.title('Comparação dos Modelos')

plt.show()
nome_modelos = list(zip(*rmsle))[0]

index = np.arange(len(nome_modelos)) 



results_rmse = list(zip(*rmsle))[1]



bar_width = 0.55

opacity = 0.4

plt.figure(figsize=(12, 6))

plt.bar(index, results_rmse, bar_width, alpha=opacity, color='b', label='RMSLE')

plt.xticks(index, nome_modelos) 

plt.xlabel('Modelos')

plt.ylabel('RSMLE')

plt.title('Comparação dos Modelos')

plt.show()
rmse_xgb
model.best_score
rmsle_xgb
df_.columns
xgb.plot_importance(model, height=0.7);
dtest_k = xgb.DMatrix(test.values)
pred = model.predict(dtest_k)
pred.min()
submission = pd.concat([id_test, pd.DataFrame(pred)], axis=1)

submission.columns = ['id','trip_duration']

submission.to_csv("submission.csv", index=False)