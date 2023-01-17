import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

# para poder visualizar os gráficos no notebook
reg_test_data = pd.read_csv("../input/epistemicselecao/regression_features_test.csv")

reg_test_label = pd.read_csv("../input/epistemicselecao/regression_targets_test.csv")

reg_train_data = pd.read_csv("../input/epistemicselecao/regression_features_train.csv")

reg_train_label = pd.read_csv("../input/epistemicselecao/regression_targets_train.csv")

# Estou utilizando a ground truth como label

# Começando a analise dos dados atraves do reg_train_data

reg_train_data
tempo_data = pd.to_datetime(reg_train_data[['year','month','day','hour']])

train_data = reg_train_data.set_index(tempo_data).drop(['year', 'month', 'day', 'hour'], axis=1)



reg_train_label['day'] = 1

tempo_data = pd.to_datetime(reg_train_label[['year','month', 'day']])

train_label = reg_train_label.set_index(tempo_data).drop(['year', 'month'], axis=1)



tempo_data = pd.to_datetime(reg_test_data[['year','month','day','hour']])

test_data = reg_test_data.set_index(tempo_data).drop(['year', 'month', 'day', 'hour'], axis=1)



reg_test_label['day'] = 1

tempo_data = pd.to_datetime(reg_test_label[['year','month', 'day']])

test_label = reg_test_label.set_index(tempo_data).drop(['year', 'month'], axis=1)
#Vou iniciar removendo quais NaN presentes

test_data = test_data.dropna()

test_label = test_label.dropna()

train_data = train_data.dropna()

train_label = train_label.dropna()
train_data.describe()
train_data.loc[(train_data['DEWP'] == -9999.0)|(train_data['HUMI'] == -9999.0)]
train_data.loc[(train_data['city'] == 2)]['2013-12-14']
train_data.loc[train_data.DEWP == -9999.0, ['DEWP', 'HUMI']] = 13.2, 100

train_data.loc[(train_data['city'] == 2)]['2013-12-14']
test_data.describe()
train_data = pd.get_dummies(train_data, columns=['cbwd'])



test_data = pd.get_dummies(test_data, columns=['cbwd'])



train_data
temp_dic = {}

for i in list(train_data):

    temp_dic[i] = np.mean

temp_dic['precipitation'] = np.sum

#como pretendemos achar volume total de precipitação, queremos manter seu valor total, e não a média que nem os outros dados
diaria = pd.DataFrame()

diaria = train_data.loc[(train_data['city'] == 0)].resample('D').agg(temp_dic)

for city in range(1, 5):

    diaria = diaria.append(train_data.loc[(train_data['city'] == city)].resample('D').agg(temp_dic))

#resample dos dados para o tamanho desejado seguindo o dicionario das funções que criamos anteriormente

diaria.dropna(inplace=True)

diaria
mensal = pd.DataFrame()

mensal = train_data.loc[(train_data['city'] == 0)].resample('M').agg(temp_dic)

for city in range(1, 5):

    mensal = mensal.append(train_data.loc[(train_data['city'] == city)].resample('M').agg(temp_dic))

#analogo ao anterior, porém com tamanho diferente

mensal.dropna(inplace=True)

mensal
corr_matrix = pd.DataFrame()

corr_matrix['/hora'] = train_data.corr(method='pearson').precipitation

corr_matrix['/dia'] = diaria.corr(method='pearson').precipitation

corr_matrix['/mes'] = mensal.corr(method='pearson').precipitation



corr_matrix
temp_dic = {}

for i in list(test_data):

    temp_dic[i] = np.mean

#Dessa vez nao temos os dados 'precipitation'
teste_mensal = pd.DataFrame()

teste_mensal = test_data.loc[(test_data['city'] == 0)].resample('M').agg(temp_dic)

for city in range(1, 5):

    teste_mensal = teste_mensal.append(test_data.loc[(test_data['city'] == city)].resample('M').agg(temp_dic))

#analogo ao anterior, porém com tamanho diferente

teste_mensal.dropna(inplace=True)
treino_mes = pd.concat([mensal[['season', 'DEWP', 'HUMI', 'PRES', 'TEMP', 'cbwd_NW', 'cbwd_SE'

                                ]], mensal[['DEWP','HUMI', 'PRES', 'TEMP']].rename(columns={'DEWP':'DEWP2',

                                                                                           'HUMI':'HUMI2',

                                                                                           'PRES':'PRES2',

                                                                                           'TEMP':'TEMP2'})], axis=1)

#Ao fazer a duplicação das colunas, renomeio elas para evitar possiveis confusões

target_mes = mensal['precipitation']

treino_mes
#antes de podermos aplicar o random forest devemos normalizar os dados

treino_mes = (treino_mes - treino_mes.mean(axis=0))/treino_mes.std(axis=0)
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=50, max_depth = 40)

rf.fit(treino_mes, target_mes)
teste_mes = pd.concat([teste_mensal[['season', 'DEWP', 'HUMI', 'PRES', 'TEMP', 'cbwd_NW', 'cbwd_SE'

                                ]], teste_mensal[['DEWP','HUMI', 'PRES', 'TEMP']].rename(columns={'DEWP':'DEWP2',

                                                                                           'HUMI':'HUMI2',

                                                                                           'PRES':'PRES2',

                                                                                           'TEMP':'TEMP2'})], axis=1)

#Ao fazer a duplicação das colunas, renomeio elas para evitar possiveis confusões

teste_mes
reg = rf.predict(teste_mes)
teste_mensal['monthly_precipitation'] = reg

teste_mensal
test_label = test_label.rename_axis('MyIdx').sort_values(by = ['city', 'MyIdx'], ascending = [True, True])

test_label
from sklearn.metrics import mean_squared_error

verdade = test_label['monthly_precipitation']

pred = teste_mensal['monthly_precipitation']

mean_squared_error(verdade, pred)
temp_dic = {}

for i in list(test_data):

    temp_dic[i] = np.mean

#Dessa vez nao temos os dados 'precipitation'
teste_diaria = pd.DataFrame()

teste_diaria = test_data.loc[(test_data['city'] == 0)].resample('D').agg(temp_dic)

for city in range(1, 5):

    teste_diaria = teste_diaria.append(test_data.loc[(test_data['city'] == city)].resample('D').agg(temp_dic))

#resample dos dados para o tamanho desejado seguindo o dicionario das funções que criamos anteriormente

teste_diaria.dropna(inplace=True)

teste_diaria
treino_dia = pd.concat([diaria[['season', 'DEWP', 'HUMI', 'PRES', 'TEMP'

                                ]], diaria[['DEWP','HUMI', 'PRES']].rename(columns={'DEWP':'DEWP2',

                                                                                    'HUMI':'HUMI2',

                                                                                    'PRES':'PRES2',})], axis=1)

#Ao fazer a duplicação das colunas, renomeio elas para evitar possiveis confusões, note que dessa vez mudamos os classificadores escolhidos tal como dito em

# 1.2, removemos os cbwd e não duplicamos TEMP

target_dia = diaria['precipitation']

treino_dia
#antes de podermos aplicar o random forest devemos normalizar os dados

treino_dia = (treino_dia - treino_dia.mean(axis=0))/treino_dia.std(axis=0)
rf = RandomForestRegressor(n_estimators=50, max_depth = 40)

rf.fit(treino_dia, target_dia)
teste_dia = pd.concat([teste_diaria[['season', 'DEWP', 'HUMI', 'PRES', 'TEMP'

                                ]], teste_diaria[['DEWP','HUMI', 'PRES']].rename(columns={'DEWP':'DEWP2',

                                                                                    'HUMI':'HUMI2',

                                                                                    'PRES':'PRES2',})], axis=1)
reg = rf.predict(teste_dia)
teste_dia['daily_precipitation'] = reg

teste_dia['city'] = teste_diaria['city']

teste_dia
temp_dic = {}

for i in list(teste_dia):

    temp_dic[i] = np.mean

temp_dic['daily_precipitation'] = np.sum

#como pretendemos achar volume total de precipitação, queremos manter seu valor total, e não a média que nem os outros dados
dia_mes = pd.DataFrame()

dia_mes = teste_dia.loc[(teste_dia['city'] == 0)].resample('M').agg(temp_dic)

for city in range(1, 5):

    dia_mes = dia_mes.append(teste_dia.loc[(teste_dia['city'] == city)].resample('M').agg(temp_dic))

#analogo ao anterior, porém com tamanho diferente

dia_mes.dropna(inplace=True)

dia_mes
test_label = test_label.rename_axis('MyIdx').sort_values(by = ['city', 'MyIdx'], ascending = [True, True])

test_label
verdade = test_label['monthly_precipitation']

pred = dia_mes['daily_precipitation']

mean_squared_error(verdade, pred)
test_label.describe()
test_label['monthly_precipitation'].hist(bins=30, figsize=(6,6))

plt.show()

dia_mes['daily_precipitation'].hist(bins=30, figsize=(6,6))

plt.show()

teste_mensal['monthly_precipitation'].hist(bins=30, figsize=(6,6))

plt.show()
train_class_data = pd.read_csv("../input/epistemicselecao/classification_features_train.csv")

train_class_label = pd.read_csv("../input/epistemicselecao/classification_targets_train.csv")

test_class_data = pd.read_csv("../input/epistemicselecao/classification_features_test.csv")

test_class_label = pd.read_csv("../input/epistemicselecao/classification_targets_test.csv")
train_class_data
tempo_data = pd.to_datetime(train_class_data[['year','month','day','hour']])

train_data = train_class_data.set_index(tempo_data).drop(['year', 'month', 'day', 'hour'], axis=1)



#diferente do anterior, dessa vez o label tem os dados Dias

tempo_data = pd.to_datetime(train_class_label[['year','month', 'day']])

train_label = train_class_label.set_index(tempo_data).drop(['year', 'month', 'day'], axis=1)



tempo_data = pd.to_datetime(test_class_data[['year','month','day','hour']])

test_data = test_class_data.set_index(tempo_data).drop(['year', 'month', 'day', 'hour'], axis=1)



#idem

tempo_data = pd.to_datetime(test_class_label[['year','month', 'day']])

test_label = test_class_label.set_index(tempo_data).drop(['year', 'month', 'day'], axis=1)
#Vou iniciar removendo quais NaN presentes

test_data = test_data.dropna()

test_label = test_label.dropna()

train_data = train_data.dropna()

train_label = train_label.dropna()
train_data.describe()
train_data.loc[train_data.DEWP == -9999.0, ['DEWP', 'HUMI']] = 13.2, 100
train_data = pd.get_dummies(train_data, columns=['cbwd'])



test_data = pd.get_dummies(test_data, columns=['cbwd'])

temp_dic = {}

for i in list(train_data):

    temp_dic[i] = np.mean

temp_dic['precipitation'] = np.sum

#Dessa vez o np.sum na precipitação é para evitar zeros por causa de numeros pequenos.
diaria = pd.DataFrame()

diaria = train_data.loc[(train_data['city'] == 0)].resample('D').agg(temp_dic)

for city in range(1, 5):

    diaria = diaria.append(train_data.loc[(train_data['city'] == city)].resample('D').agg(temp_dic))

#resample dos dados para o tamanho desejado seguindo o dicionario das funções que criamos anteriormente

diaria.dropna(inplace=True)

diaria
train_data.loc[train_data['precipitation'] != 0,'precipitation'] = True

train_data.loc[train_data['precipitation'] == 0, 'precipitation'] = False

diaria.loc[diaria['precipitation'] != 0,'precipitation'] = True

diaria.loc[diaria['precipitation'] == 0, 'precipitation'] = False
correlation = pd.DataFrame(train_data.corr(method='pearson').precipitation)

corr_D = pd.DataFrame(diaria.corr(method='pearson').precipitation)

correlation.join(corr_D, rsuffix = '_Dia').sort_values(by='precipitation', ascending=True)
diaria_teste = pd.DataFrame()

diaria_teste = test_data.loc[(test_data['city'] == 0)].resample('D').mean() # não necessita do dicionario pois não tem precipitation

for city in range(1, 5):

    diaria_teste = diaria_teste.append(test_data.loc[(test_data['city'] == city)].resample('D').mean())

#resample dos dados para o tamanho desejado seguindo o dicionario das funções que criamos anteriormente

diaria_teste.dropna(inplace=True)

diaria_teste
treino_class = train_data[['PRES', 'DEWP', 'HUMI']]

target_class = train_data['precipitation']

treino_dia = pd.concat([diaria[['PRES', 'DEWP', 'HUMI', 'season', 'TEMP'

                               ]], diaria[['HUMI']].rename(columns={'HUMI':'HUMI2'})], axis=1)

target_dia = diaria['precipitation']
test_class = test_data[['PRES', 'DEWP', 'HUMI']]

test_dia = pd.concat([diaria_teste[['PRES', 'DEWP', 'HUMI', 'season', 'TEMP'

                               ]], diaria_teste[['HUMI']].rename(columns={'HUMI':'HUMI2'})], axis=1)
from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()

bnb.fit(treino_dia, target_dia)
choveu = bnb.predict(test_dia)
test_dia_temp = pd.DataFrame()

test_dia_temp = test_dia

test_dia_temp['precipitation'] = choveu

test_dia_temp['city'] = diaria_teste['city'] #para podemos verificar se estamos com o local certo ao comparar com o label

test_dia_temp.rename_axis('MyIdx', inplace=True)

test_dia_temp = test_dia_temp.reset_index(drop=False)

test_dia_temp
test_label = test_label.rename_axis('MyIdx').sort_values(by = ['city', 'MyIdx'], ascending = [True, True])

test_label.dropna(inplace = True)

test_label_temp = test_label.reset_index(drop=False)

test_label_temp
result = pd.merge(test_dia_temp, test_label_temp, on = ['MyIdx', 'city'], sort=False)

result
from sklearn.metrics import roc_auc_score

y_true = result['rain']

y_pred = result['precipitation']

roc_auc_score(y_true, y_pred)
bnb = BernoulliNB()

bnb.fit(treino_class, target_class)
choveu = bnb.predict(test_class)
test_class_temp = pd.DataFrame()

test_class_temp = test_class

test_class_temp.loc[:, 'precipitation'] = choveu

test_class_temp.loc[:, 'city'] = test_data['city'] #para podemos verificar se estamos com o local certo ao comparar com o label

test_class_temp
hora_dia = pd.DataFrame()

hora_dia = test_class_temp.loc[(test_class_temp['city'] == 0)].resample('D').mean() # não necessita do dicionario pois não tem precipitation

for city in range(1, 5):

    hora_dia = hora_dia.append(test_class_temp.loc[(test_class['city'] == city)].resample('D').mean())

#resample dos dados para o tamanho desejado seguindo o dicionario das funções que criamos anteriormente

hora_dia.dropna(inplace=True)

hora_dia.loc[hora_dia['precipitation'] != 0,'precipitation'] = True

hora_dia.loc[hora_dia['precipitation'] == 0, 'precipitation'] = False

hora_dia
hora_dia.rename_axis('MyIdx', inplace=True)

hora_dia_temp = hora_dia.reset_index(drop=False)

result = pd.merge(hora_dia, test_label_temp, on = ['MyIdx', 'city'], sort=False)

result
y_true = result['rain']

y_pred = result['precipitation']

roc_auc_score(y_true, y_pred)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=20)

knn.fit(treino_dia, target_dia)
test_dia.drop(['precipitation', 'city'], axis = 1, inplace = True)
predictions = knn.predict(test_dia)
y_true = result['rain']

y_pred = predictions

roc_auc_score(y_true, y_pred)
cluster_data = pd.read_csv("../input/epistemicselecao/clustering_dataset.csv")

cluster_data
cluster_data.describe()
cluster_data_outlier = cluster_data.loc[10, :]

cluster_data = cluster_data.drop([10], axis=0)

#removendo a linha outlier para podermos estudar os dados melhor
cluster_data.describe()
labels = list(cluster_data)

labels.remove('city')

labels.remove('season')

#lista com os labels que precisam do minmaxscale
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()



cluster_data_mm = pd.DataFrame()

cluster_data_mm = cluster_data.copy()

scaler.fit(cluster_data_mm[labels])



cluster_data_mm[labels] = scaler.transform(cluster_data_mm[labels])

cluster_data_mm.reset_index(drop=True, inplace = True)

cluster_data_mm
from sklearn.cluster import KMeans
cluster_seasons = cluster_data_mm.drop('city', axis=1)

#Perceba que como as estações serão as clusters, podemos chamar elas de "target"

cluster_seasons_target = cluster_seasons.loc[:, 'season']

cluster_seasons.drop('season', axis=1, inplace=True)
km = KMeans(n_clusters=4)

y_predicted = km.fit_predict(cluster_seasons)

y_predicted.size == cluster_seasons.shape[0] #apenas verificando se o tamanho está de acordo
cluster_seasons['cluster'] = y_predicted

cluster_seasons
labels = list(cluster_seasons)

labels.remove('average_total_seasonal_precipitation')

labels.remove('cluster')

#Vou plotar a partir da media total de chuva
target_plot = pd.DataFrame()

target_plot = cluster_seasons.copy()

target_plot['season'] = cluster_seasons_target -1 #season começa em 1, queremos que ela comece no 0 tal como a cluster
cores = ['green', 'red', 'black', 'blue']

x=1

f = plt.figure(figsize=(10,50))

for y in labels:

    for i in range(0, 4):

        ax = f.add_subplot(12,2,x)

        df_temp = cluster_seasons[cluster_seasons.cluster==i]

        ax.scatter(df_temp[y], df_temp['average_total_seasonal_precipitation'], color= cores[i])

        plt.title(y)

        

        ax2 = f.add_subplot(12,2,x+1)

        df_temp = target_plot[target_plot.season==i]

        ax2.scatter(df_temp[y], df_temp['average_total_seasonal_precipitation'], color= cores[i])

        plt.title(y + 'real')

        

    x+=2

plt.show()

    
from sklearn.metrics import silhouette_score
labels1 = list(cluster_seasons)

labels1.remove('cluster')

silhouette_score(cluster_seasons[labels1], cluster_seasons['cluster'])