import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

import datetime

from pandas.tseries.holiday import USFederalHolidayCalendar

import time

import lightgbm as lgb

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import KFold



start_time = time.time()

X = pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv', nrows = 3_000_000)

X_test = pd.read_csv('../input/new-york-city-taxi-fare-prediction/test.csv')

X.head()
X.info()
X.describe()
# Correlação das variáveis

correlation = X.corr()



plt.figure(figsize=(8,8))

sns.heatmap(correlation, annot = True)

plt.title('Correlação das Variáveis')



plt.show()
missing_val_count_by_column = (X.isnull().sum())

print('Features com dados faltantes no DataSet:')

print(missing_val_count_by_column[missing_val_count_by_column > 0])





qtd_inicial, _ = X.shape

print('Retirando linhas que possuem valores faltantes')

X.dropna(inplace=True)

qtd_final, _ = X.shape

print('Porcentagem de dados retirados do conjunto: {:.02f}%'.format((1 - qtd_final/qtd_inicial)*100))



fig, ((ax1,ax2)) = plt.subplots(figsize = [14,6],nrows = 1, ncols = 2)

plt.subplots_adjust(left=0, bottom=None, right=1, top=0.5, wspace = 0.8, hspace=None)

sns.set(style="darkgrid")



ax1.boxplot(X['fare_amount'])

ax1.set_title('Boxplot - Preço das corridas')





sns.distplot(a=X['fare_amount'], kde=False)

ax2.set_title('Distribuição do preço das corridas')

ax2.set_xlabel('Preço')

ax2.set_ylabel('Quantidade')





plt.tight_layout()

plt.show()
def clean_fare_amount(df):

    return df[(df.fare_amount > 0) & (df.fare_amount < 200)]

qtd_inicial, _ = X.shape

X = clean_fare_amount(X)



print('Retirando linhas que possuem preços inválidos')

qtd_final, _ = X.shape

print('Porcentagem de dados retirados do conjunto: {:.02f}%'.format((1 - qtd_final/qtd_inicial)*100))

# Verificando os outliers nas longitudes e latitudes

fig, (ax1,ax2) = plt.subplots(figsize = [14, 6],nrows = 1, ncols = 2)



ax1.boxplot(X['pickup_latitude'])

ax2.boxplot(X['pickup_longitude'])



ax1.set_title('Latitude de partida')

ax2.set_title('Longitude de partida')



ax1.set_ylabel('Latitude')

ax2.set_ylabel('Longitude')





fig.show()
def clean_locations(df):

    return df[

            (df.pickup_longitude > -80) & (df.pickup_longitude < -70) &

            (df.pickup_latitude > 35) & (df.pickup_latitude < 45) &

            (df.dropoff_longitude > -80) & (df.dropoff_longitude < -70) &

            (df.dropoff_latitude > 35) & (df.dropoff_latitude < 45) 

            ]

qtd_inicial, _ = X.shape

print('Retirando linhas que possuem localizações inválidas')

X = clean_locations(X)

qtd_final, _ = X.shape

print('Porcentagem de dados retirados do conjunto: {:.02f}%'.format((1 - qtd_final/qtd_inicial)*100))

from math import radians, cos, sin, asin, sqrt

def generate_distances(df):

    """

    Calcula e adiciona a distância em linha reta e a distância de haversine ao dataframe.

    As distâncias são dadas em km

    """

    # Pegando as coordenadas (aplicando transformação para radiano)

    pickup_latitude = df['pickup_latitude']*57.2958

    pickup_longitude = df['pickup_longitude']*57.2958

    dropoff_latitude = df['dropoff_latitude']*57.2958

    dropoff_longitude = df['dropoff_longitude']*57.2958

    

    # Calculando a distância em linha reta

    

    straight_distance = (((pickup_latitude - dropoff_latitude)*1.852)**2 + ((pickup_longitude - dropoff_longitude)*1.852)**2)**0.5

    

    # Calculando a distância de haversine

    R = 6371

    phi1 = np.radians(df['pickup_latitude'])

    phi2 = np.radians(df['dropoff_latitude'])

    phi_chg = np.radians(df['pickup_latitude'] - df['dropoff_latitude'])

    delta_chg = np.radians(df['pickup_longitude'] - df['dropoff_longitude'])

    a = np.sin(phi_chg / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_chg / 2)**2

    haversine_distance = 2*R*np.arcsin(a**0.5)

   

    # Adicionando a distância em linha reata e a distância de harversina ao dataframe

    df['straight_distance'] = straight_distance

    df['haversine_distance'] = haversine_distance

    
# Implementando variáveis de distância - conjunto de treino

generate_distances(X)



# Implementando variáveis de distância - conjunto de teste

generate_distances(X_test)



X.head()
X[['straight_distance', 'haversine_distance']].describe()
fig, ((ax1,ax2)) = plt.subplots(figsize = [14,6],nrows = 1, ncols = 2)

plt.subplots_adjust(left=0, bottom=None, right=1, top=0.5, wspace = 0.8, hspace=None)

sns.set(style="darkgrid")



ax1.boxplot(X['straight_distance'])

ax1.set_title('Boxplot - Distância das corridas')





sns.distplot(a=X['straight_distance'], kde=False)

ax2.set_title('Distribuição da distância das corridas')

ax2.set_xlabel('Distância percorrida')

ax2.set_ylabel('Quantidade')





plt.tight_layout()

plt.show()
def clean_distance(df):

    return df[(df.straight_distance > 0.2) & (df.straight_distance < 40)]
qtd_inicial, _ = X.shape

print('Retirando linhas que possuem distâncias inválidas')

X = clean_distance(X)

qtd_final, _ = X.shape

print('Porcentagem de dados retirados do conjunto: {:.02f}%'.format((1 - qtd_final/qtd_inicial)*100))
# Verificando os outliers nas longitudes e latitudes

fig, (ax1,ax2) = plt.subplots(figsize = [14, 6],nrows = 1, ncols = 2)



ax1.scatter(X['pickup_longitude'],X['pickup_latitude'],alpha=0.8, c = 'b')

ax2.scatter(X['dropoff_longitude'],X['dropoff_latitude'],alpha=0.8, c = 'r')



ax1.set_title('Coordenadas de Partida')

ax2.set_title('Coordenadas de Destino')



ax1.set_ylabel('Latitude')

ax1.set_xlabel('Longitude')



ax2.set_ylabel('Latitude')

ax2.set_xlabel('Longitude')





fig.show()

def train_cluster_features(df):

    """

    Usa o dataframe de treino para treinar o algoritmo de clusterização

    """

    # Clusterização das coordenadas de partida

    pickup = df[['pickup_longitude', 'pickup_latitude']]

    pickup_model = KMeans(n_clusters = 4)



    pickup_model.fit(pickup)



    # Clusterização das variáveis de destino

    dropoff = df[['dropoff_longitude', 'dropoff_latitude']]

    dropoff_model = KMeans(n_clusters = 4)



    dropoff_model.fit(dropoff)

    

    return pickup_model, dropoff_model



def add_cluster_features(df, pickup_model, dropoff_model):

    """

    Adiciona as features de clusterização ao dataframe (df)

    """

    # Adicionando as novas features ao dataset

    pickup = df[['pickup_longitude', 'pickup_latitude']]

    pickup_labels = pickup_model.predict(pickup)

    

    df['cluster'] = pickup_labels

    pickup_clusters = pd.get_dummies(df['cluster'], prefix = 'pickup_cluster', drop_first = False).iloc[:,1:]

    df = pd.concat([df, pickup_clusters], axis =1).drop('cluster', axis = 1)



    dropoff = df[['dropoff_longitude', 'dropoff_latitude']]

    dropoff_labels = dropoff_model.predict(dropoff)

    df['cluster'] = dropoff_labels

    dropoff_clusters = pd.get_dummies(df['cluster'], prefix = 'dropoff_cluster', drop_first = False).iloc[:,1:]

    df = pd.concat([df, dropoff_clusters], axis =1).drop('cluster', axis = 1)

    

    return df



# Recebe os modelos de treinamento dos clusters

pickup_model, dropoff_model = train_cluster_features(X)



# Adiciona as features de clusterização ao conjunto de treino

X = add_cluster_features(X, pickup_model, dropoff_model)



# Adiciona as features de clusterização ao conjunto de teste

X_test = add_cluster_features(X_test, pickup_model, dropoff_model)
def dist(pickup_lat, pickup_long, dropoff_lat, dropoff_long):  

    """

    Calcula a distância em linha reta entre dois pontos - auxilia a função airport_feats

    """

    pickup_lat = pickup_lat*57.2958

    pickup_long = pickup_long*57.2958

    dropoff_lat = dropoff_lat*57.2958

    dropoff_long = dropoff_long*57.2958

    # Calculando a distância em linha reta

    

    distance = (((pickup_lat - dropoff_lat)*1.852)**2 + ((pickup_long - dropoff_long)*1.852)**2)**0.5

    

    return distance



def airport_feats(train):

    """

    Calcula se uma viagem é de algum aeroporto ou se é para algum aeroporto e adiciona as features ao dataset

    """

    for data in [train]:

        nyc = (-74.0063889, 40.7141667)

        jfk = (-73.7822222222, 40.6441666667)

        ewr = (-74.175, 40.69)

        lgr = (-73.87, 40.77)

        data['picking_at_center'] = (dist(nyc[1], nyc[0],

                                          data['pickup_latitude'], data['pickup_longitude']) < 2).astype(int)

        data['dropping_at_center'] = (dist(nyc[1], nyc[0],

                                          data['dropoff_latitude'], data['dropoff_longitude']) < 2).astype(int)

        data['picking_at_jfk'] = (dist(jfk[1], jfk[0],

                                             data['pickup_latitude'], data['pickup_longitude']) < 2).astype(int)

        data['dropping_at_jfk'] = (dist(jfk[1], jfk[0],

                                               data['dropoff_latitude'], data['dropoff_longitude']) < 2).astype(int)

        data['picking_at_ewr'] = (dist(ewr[1], ewr[0], 

                                              data['pickup_latitude'], data['pickup_longitude']) < 2).astype(int)

        data['dropping_at_ewr'] = (dist(ewr[1], ewr[0],

                                               data['dropoff_latitude'], data['dropoff_longitude']) < 2).astype(int)

        data['picking_at_lgr'] = (dist(lgr[1], lgr[0],

                                              data['pickup_latitude'], data['pickup_longitude']) < 2).astype(int)

        data['dropping_at_lgr'] = (dist(lgr[1], lgr[0],

                                               data['dropoff_latitude'], data['dropoff_longitude']) < 2).astype(int)

    return train
# Implementando variáveis de tempo - conjunto de treino



X = airport_feats(X)



# Implementando variáveis de tempo - conjunto de teste



X_test = airport_feats(X_test)

X.head()
def fare_over_distance(df):

    

    df['fare_over_distance'] = df['fare_amount']/(df['straight_distance']+ 0.0001)



def fare_over_distance_over_npass(df):

    df['fare_over_distance_over_npass'] = df['fare_amount']/(df['straight_distance']*df['passenger_count'] + 0.0001)

    

fare_over_distance(X)

fare_over_distance_over_npass(X)

X.head()
X.fare_over_distance.describe()
X.fare_over_distance_over_npass.describe()
fig, ((ax1,ax2)) = plt.subplots(figsize = [14,6],nrows = 1, ncols = 2)

plt.subplots_adjust(left=0, bottom=None, right=1, top=0.5, wspace = 0.8, hspace=None)

sns.set(style="darkgrid")



ax1.boxplot(X['fare_over_distance'])

ax1.set_title('Boxplot - Distância das corridas')





sns.distplot(a=X['fare_over_distance'], kde=False)

ax2.set_title('Distribuição da distância das corridas')

ax2.set_xlabel('Distância percorrida')

ax2.set_ylabel('Quantidade')





plt.tight_layout()

plt.show()
def clean_fare_over_distance(df):

    return df[(df.fare_over_distance < 20) & (df.fare_over_distance > 1.5)]
qtd_inicial, _ = X.shape

print('Retirando linhas que possuem preço por km muito alto ou muito baixo')

X = clean_fare_over_distance(X)

qtd_final, _ = X.shape

print('Porcentagem de dados retirados do conjunto: {:.02f}%'.format((1 - qtd_final/qtd_inicial)*100))
def time_features(df):



    df['data'] = df['pickup_datetime'].str.replace(" UTC", "")



    df['data'] = pd.to_datetime(df['data'], format = '%Y-%m-%d %H:%M:%S')

    df['hour_of_day'] = df.data.dt.hour

    df['week'] = df.data.dt.week

    df['month'] = df.data.dt.month

    df["year"] = df.data.dt.year

    df['day_of_year'] = df.data.dt.dayofyear

    df['week_of_year'] = df.data.dt.weekofyear

    df["weekday"] = df.data.dt.weekday

    df["quarter"] = df.data.dt.quarter

    df["day_of_month"] = df.data.dt.day

    df.drop('data',inplace= True, axis =1)

    

    df['pickup_datetime'] = df.pickup_datetime.apply(

    lambda x: datetime.datetime.strptime(x[:10], '%Y-%m-%d'))





    cal = USFederalHolidayCalendar()

    holidays = cal.holidays(start='2009-01-01', end='2015-12-31').to_pydatetime()



    df['is_holiday'] = df.pickup_datetime.apply(lambda x: 1 if x in holidays else 0)

    

    return df
# Implementando variáveis de tempo - conjunto de treino

X = time_features(X)

# Implementando variáveis de tempo - conjunto de teste

X_test = time_features(X_test)
X.head()
index = X.passenger_count.value_counts().index

unique_values = X.passenger_count.value_counts()

plt.title('Quantidade de viagens por número de passageiros')

plt.xlabel('Quantidade de passageiros')

plt.ylabel('Quantidade de viagens')

plt.bar(index,unique_values)



plt.show()

def clean_passenger_count(df):

    return df[(df['passenger_count'] > 0) & (df['passenger_count'] <10)]





qtd_inicial, _ = X.shape

print('Retirando linhas que possuem quantidade inválida de passageiros')

X = clean_passenger_count(X)

qtd_final, _ = X.shape

print('Porcentagem de dados retirados do conjunto: {:.02f}%'.format((1 - qtd_final/qtd_inicial)*100))

total_time = time.time() - start_time



print(total_time/60)
# Correlação das variáveis

corr = X.corr()

fare_amount_corr = corr.loc['fare_amount']

good_features = abs(fare_amount_corr).sort_values(ascending = False)

good_features_index = good_features.index





# Gŕafico de barras

plt.figure(figsize=(10,8))



plt.title("Módulo da Correlação entre Features e o Preço da Viagem de Taxi")

plt.ylabel("Features")

plt.xlabel("Correlação")



sns.barplot(y=good_features_index[1:], x=good_features[1:])



plt.show()
# Agrupamento em relação às features de data (mês, ano, dia do mês, dia da semana, hora do dia)

month = X.groupby('month').agg({'fare_amount':['mean']})

year = X.groupby('year').agg({'fare_amount':['mean']})

day_of_month = X.groupby('day_of_month').agg({'fare_amount':['mean']})

day_of_week = X.groupby('weekday').agg({'fare_amount':['mean']})

hour_of_day = X.groupby('hour_of_day').agg({'fare_amount':['mean']})



# Gerando a figura dos resultados

fig,((ax1, ax2, ax3, ax4, ax5)) = plt.subplots(figsize = [18, 14],nrows = 5, ncols = 1)





ax1.plot(year, 'b')

ax2.plot(month,'g')

ax3.plot(day_of_month, 'c')

ax4.plot(day_of_week, 'r')

ax5.plot(hour_of_day)







ax1.set_title('Fare by Year', fontsize = 18)

ax2.set_title('Fare by Month of Year', fontsize = 18)

ax3.set_title('Fare by Day of Month', fontsize = 18)

ax4.set_title('Fare by Day of Week', fontsize = 18)

ax5.set_title('Fare by Hour of Day', fontsize = 18)



ax1.set_xlabel('Year',fontsize = 18)

ax2.set_xlabel('Month of Year',fontsize = 18)

ax3.set_xlabel('Day of Month',fontsize = 18)

ax4.set_xlabel('Day of Week',fontsize = 18)

ax5.set_xlabel('Hour of Day',fontsize = 18)



ax1.set_ylabel('Fare',fontsize = 18)

ax2.set_ylabel('Fare',fontsize = 18)

ax3.set_ylabel('Fare',fontsize = 18)

ax4.set_ylabel('Fare',fontsize = 18)

ax5.set_ylabel('Fare',fontsize = 18)





plt.style.use('seaborn')

plt.tight_layout()



plt.show()
sns.kdeplot(data=X['straight_distance'], shade=True)

plt.show()
# Agrupamento em relação às features de data (mês, ano, dia do mês, dia da semana, hora do dia)

month = X.groupby('month').agg({'straight_distance':['mean']})

year = X.groupby('year').agg({'straight_distance':['mean']})

day_of_month = X.groupby('day_of_month').agg({'straight_distance':['mean']})

day_of_week = X.groupby('weekday').agg({'straight_distance':['mean']})

hour_of_day = X.groupby('hour_of_day').agg({'straight_distance':['mean']})



# Gerando a figura dos resultados

fig,((ax1, ax2, ax3, ax4, ax5)) = plt.subplots(figsize = [18, 14],nrows = 5, ncols = 1)





ax1.plot(year, 'b')

ax2.plot(month,'g')

ax3.plot(day_of_month, 'c')

ax4.plot(day_of_week, 'r')

ax5.plot(hour_of_day)



ax1.set_title('Distance by Year', fontsize = 18)

ax2.set_title('Distance by Month of Year', fontsize = 18)

ax3.set_title('Distance by Day of Month', fontsize = 18)

ax4.set_title('Distance by Day of Week', fontsize = 18)

ax5.set_title('Distance by Hour of Day', fontsize = 18)



ax1.set_xlabel('Year',fontsize = 18)

ax2.set_xlabel('Month of Year',fontsize = 18)

ax3.set_xlabel('Day of Month',fontsize = 18)

ax4.set_xlabel('Day of Week',fontsize = 18)

ax5.set_xlabel('Hour of Day',fontsize = 18)



ax1.set_ylabel('Distance',fontsize = 18)

ax2.set_ylabel('Distance',fontsize = 18)

ax3.set_ylabel('Distance',fontsize = 18)

ax4.set_ylabel('Distance',fontsize = 18)

ax5.set_ylabel('Distance',fontsize = 18)





plt.style.use('seaborn')

plt.tight_layout()



plt.show()
idx = X[X['straight_distance'] < 10].index[0:1000]





plt.scatter(X.straight_distance[idx], X.fare_over_distance[idx], color = 'r')

plt.xlabel('Distancia em km')

plt.ylabel('Tarifa por km')

plt.title('Valor da viagem por km percorrido')



plt.show()
npass = X.groupby('passenger_count').agg({'fare_amount':['mean','min','max'], 'straight_distance': ['mean', 'min', 'max']})



npass
y = X.fare_amount.copy()

key = X_test.key.copy()

X_test.drop(['pickup_datetime','key'] ,axis = 1, inplace=True)



X_train = X[X_test.columns]
# Iniciando um objeto de dataset de lgb

dtrain = lgb.Dataset(X_train, label=y, free_raw_data=False)



# Iniciando a quantiadade de folds para a validação cruzada 

folds = KFold(n_splits=5, shuffle=True, random_state=1)



# A variável predictions possuirá o valor final das predições para o conjunto de testes

predictions = np.zeros(X_test.shape[0])

params =  {'task': 'train', 'boosting_type': 'gbdt','objective': 'regression','metric': 'rmse'}



# Variável que vai guardar o valor do rmse de validação a cada época de treinamento

evals_result = {}



# Realização do treinamento e predição

for train_index, validation_index in folds.split(X_train):

    clf = lgb.train(

        params=params,

        train_set=dtrain.subset(train_index),

        valid_sets=dtrain.subset(validation_index),

        num_boost_round=1000, 

        early_stopping_rounds=125,

        evals_result=evals_result,

        verbose_eval=250

    )

    predictions = predictions + clf.predict(X_test) / folds.n_splits



# Dataframe de submissão

submission = pd.DataFrame(predictions,columns=["fare_amount"],index=X_test.index)



# Colocando de volta a coluna key

keys = pd.DataFrame(key, columns = ['key'], index=  X_test.index)

submission = pd.concat([keys, submission], axis = 1)



# Valores iniciais das predições

submission.head()

submission.to_csv('submit.csv', index = False)
lgb.plot_metric(evals_result)

plt.show()
lgb.plot_importance(clf)

plt.show()