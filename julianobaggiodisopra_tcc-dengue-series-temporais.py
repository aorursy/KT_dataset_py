!pip install adtk;

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from pylab import rcParams
import math
import scipy.stats as st
import statistics
from statsmodels.tsa.stattools import adfuller
import statsmodels as sm
from adtk.data import validate_series

def sumzip(*items):
    return [sum(values) for values in zip(*items)]

%matplotlib inline

meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']

def sumzip(*items):
    return [sum(values) for values in zip(*items)]

def draw_mean (X, Y) :
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)
    num = 0
    den = 0
    for i in range (len(X)) :
        num += (X[i] - X_mean) * (Y[i] - Y_mean)
        den += (X[i] - X_mean) ** 2
    m = num/den
    c = Y_mean - m * X_mean
    Y_pred = m * X + c
    plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color = 'red')


def inclui_nan(db, index):
    
    print("Iniciando tratamento de " + index)
    
    for idx, row in db.iterrows():
        if math.isnan(row[index]):
            database.loc[idx,index] = database[database["mes"] == row.mes].eval(index).mean();

def tratar_outlier(db, index):
    
    print("Iniciando tratamento de " + index)
    
    lim = 3 #outliers extremos
    #lim = 1.5 #outliers moderados
    
    p_q1 = db.eval(index).quantile(q=0.25)
    p_q3 = db.eval(index).quantile(q=0.75)
    p_amplitude = p_q3 - p_q1 #amplitude interquartílica
    p_lim_max = p_q3 + lim * p_amplitude #limites internos
    p_lim_min = p_q1 - lim * p_amplitude #limites internos

    print("Variáveis para " + index)
    print(p_q1, p_q3, p_amplitude, p_lim_max, p_lim_min)
    print("Outliers de" + index)
    print(db[db[index] > p_lim_max])
    print(db[db[index] < p_lim_min])

    for idx, row in db.iterrows():
        if row[index] > p_lim_max :
            db.loc[idx,index] = db[(db["mes"] == row.mes) & (db["ano"] != row.ano)].eval(index).mean();
            
        if row[index] < p_lim_min :
            db.loc[idx,index] = db[(db["mes"] == row.mes) & (db["ano"] != row.ano)].eval(index).mean();     
   
    print("Outliers de" + index)
    print(db[db[index] > p_lim_max])
    print(db[db[index] < p_lim_min]) 

    print("###################")
    

def adiciona_timelag(db, mesesRetroceder,indexes) :
    #timelag
    for indiceAtual in indexes:
        for mes in range(1, mesesRetroceder + 1) :
            index_name = indiceAtual + "_m" + str(mes);
            db[index_name] = db[indiceAtual].shift(mes);
   
    #coloca média nos nulos anteriores criados
    for indiceAtual in indexes:     
        for mes in range(1, mesesRetroceder + 1) :
            index_name = indiceAtual + "_m" + str(mes);
            for idx, row in db.iterrows():
                if math.isnan(row[index_name]):
                    db.loc[idx,index_name] = db[db["mes"] == row.mes].eval(index_name).mean();
                    
   
#série estacionária = ADF < -3.49, p_value <= 0.05
#série estacionária (constante ao longo do tempo), premissa para time serie
def teste_estacionaridade(variavel, nome) :
    print("Testando estacionaridade de variável: " + nome);
    result = adfuller(variavel, autolag='AIC', regression='ct') #constant and trend
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    if result[1]<=0.05:
        print("Strong evidence against Null Hypothesis")
        print("Reject Null Hypothesis - Data is Stationary")
    else:
        print("Strong evidence for  Null Hypothesis")
        print("Accept Null Hypothesis - Data is not Stationary")
        
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')

database = pd.read_csv(
    "/kaggle/input/dados-dengue-2010-2019/analise_com_falhas_em_dados_2010.csv",
    sep = ';',
    header = 0,
    error_bad_lines = False,
    parse_dates = ['periodo'],
    index_col = ['periodo'],
    date_parser = dateparse,
    #usecols = ["periodo", "focos"],
    usecols = ["periodo", "focos", "precipitacao", "temp_minima", "temp_media", "temp_maxima", "umidade"],
)

database['ano'] = database.index.year
database['mes'] = database.index.month

database.resample('MS')

database.info(verbose=True)
    
database.dtypes

database.tail()
database["focos"].sum()
database.isnull().sum()
inclui_nan(database, "precipitacao")
inclui_nan(database, "temp_minima")
inclui_nan(database, "temp_media")
inclui_nan(database, "temp_maxima")
inclui_nan(database, "umidade")
database.isnull().sum()
database["focos"].plot(figsize=(15, 4))
plt.show()
figure = plt.figure(figsize=(9,4))
data = database.groupby("ano")["focos"].sum()
data.plot.bar(title = 'Focos por ano', edgecolor = 'black', width = 0.7, linewidth = 0.5)
plt.xlabel('Ano')
plt.ylabel('Qtde. Focos')
figure = plt.figure(figsize=(9,4))
data = database.groupby("mes")["focos"].sum()
data.plot.bar(title = 'Focos por mês', edgecolor = 'black', width = 0.9)
plt.xlabel('Mês')
plt.ylabel('Qtde. Focos')
d10 = database[database["ano"] == 2010].groupby("mes")["focos"].sum().values
d11 = database[database["ano"] == 2011].groupby("mes")["focos"].sum().values
d12 = database[database["ano"] == 2012].groupby("mes")["focos"].sum().values
d13 = database[database["ano"] == 2013].groupby("mes")["focos"].sum().values
d14 = database[database["ano"] == 2014].groupby("mes")["focos"].sum().values
d15 = database[database["ano"] == 2015].groupby("mes")["focos"].sum().values
d16 = database[database["ano"] == 2016].groupby("mes")["focos"].sum().values
d17 = database[database["ano"] == 2017].groupby("mes")["focos"].sum().values
d18 = database[database["ano"] == 2018].groupby("mes")["focos"].sum().values
d19 = database[database["ano"] == 2019].groupby("mes")["focos"].sum().values

dados = np.vstack([d10, d11, d12, d13, d14, d15, d16, d17, d18, d19])
labels = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']

fig, ax = plt.subplots(figsize=(15,7))

ax.bar(meses, d10, label = '2010')
ax.bar(meses, d11, label = '2011', bottom = sumzip(d10))
ax.bar(meses, d12, label = '2012', bottom = sumzip(d10, d11))
ax.bar(meses, d13, label = '2013', bottom = sumzip(d10, d11, d12))
ax.bar(meses, d14, label = '2014', bottom = sumzip(d10, d11, d12, d13))
ax.bar(meses, d15, label = '2015', bottom = sumzip(d10, d11, d12, d13, d14))
ax.bar(meses, d16, label = '2016', bottom = sumzip(d10, d11, d12, d13, d14, d15))
ax.bar(meses, d17, label = '2017', bottom = sumzip(d10, d11, d12, d13, d14, d15, d16))
ax.bar(meses, d18, label = '2018', bottom = sumzip(d10, d11, d12, d13, d14, d15, d16, d17))
ax.bar(meses, d19, label = '2019', bottom = sumzip(d10, d11, d12, d13, d14, d15, d16, d17, d18))

ax.set_xticks(np.arange(13))
ax.set_ylabel('Focos')
#ax.set_title('Focos por mês')
ax.legend(loc='upper right')
plt.show()
fig, ax = plt.subplots(figsize=(15,7))

minima = database.groupby("mes")["temp_minima"].mean()
media = database.groupby("mes")["temp_media"].mean()
maxima = database.groupby("mes")["temp_maxima"].mean()

ax.plot(meses, minima, 'o--', label = 'Mínima')
ax.plot(meses, maxima, 'o--', label = 'Máxima')
ax.plot(meses, media, color = 'green', marker = '.', label = 'Temp Média', linewidth = 2.5, markersize = 10)
ax.set_xticks(np.arange(13))
ax.fill_between(meses, maxima, media, color = 'red', alpha=0.2)
ax.fill_between(meses, minima, media, color = 'blue', alpha=0.2)

#plt.xlabel('Mês')
plt.ylabel('Temperatura (ºC)')
plt.ylim(8, 30);
plt.legend();
database["temp_minima"].plot(figsize=(15, 4))
plt.show()
figure = plt.figure(figsize=(15,7))

data = database.groupby("mes")["precipitacao"].mean()

plt.plot(meses, data, 'o--')
plt.fill_between(meses, data, color = 'blue', alpha=0.2, linewidth= 1)

plt.xlabel('Mês')

plt.ylabel('Precipitação média mm')

plt.ylim(100, 250);
database["precipitacao"].plot(figsize=(15, 4))
plt.show()
figure = plt.figure(figsize=(15,7))

data = database.groupby("mes")["umidade"].mean()

plt.plot(meses, data, 'o--')
plt.fill_between(meses, data, color = 'blue', alpha=0.2, linewidth= 1)

plt.xlabel('Mês')

plt.ylabel('Umidade Média %')

plt.ylim(65, 85);
database["umidade"].plot(figsize=(15, 4))
plt.show()
plt.figure(figsize=(24,6))
plt.subplot(1,6,1)
fig = sb.boxplot(x='focos', data=database, orient='v', color='#ffffb2')
fig.set_title('Boxplot Focos');
fig.set_ylabel('Focos');

plt.subplot(1,6,2)
fig = sb.boxplot(x='precipitacao', data=database, orient='v', color='#ffffb2')
fig.set_title('Boxplot Precipitação');
fig.set_ylabel('Temp. Precipitação')

plt.subplot(1,6,3)
fig = sb.boxplot(x='umidade', data=database, orient='v', color='#ffffb2')
fig.set_title('Boxplot Umidade');
fig.set_ylabel('Umidade');

plt.subplot(1,6,4)
fig = sb.boxplot(x='temp_minima', data=database, orient='v', color='#ffffb2')
fig.set_title('Boxplot Temp. Mínima');
fig.set_ylabel('Temp. Mínima');

plt.subplot(1,6,5)
fig = sb.boxplot(x='temp_media', data=database, orient='v', color='#ffffb2')
fig.set_title('Boxplot Temp. Média');
fig.set_ylabel('Temp. Média')

plt.subplot(1,6,6)
fig = sb.boxplot(x='temp_maxima', data=database, orient='v', color='#ffffb2')
fig.set_title('Boxplot Temp. Máxima');
fig.set_ylabel('Temp. Máxima');
plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
fig = database.focos.hist(bins=25)
fig.set_title('Distribuição Focos');
fig.set_ylabel('Focos');


plt.subplot(1,2,2)
fig = database.precipitacao.hist(bins=25)
fig.set_title('Distribuição Precipitação');
fig.set_ylabel('Precipitação');
db_train = validate_series(database[['focos','precipitacao', 'temp_minima', 'umidade']])
from adtk.visualization import plot
plot(db_train)
from adtk.detector import InterQuartileRangeAD

dataframe_anomalias = db_train[["focos", "precipitacao"]]

iqr_ad = InterQuartileRangeAD(c=3) #outliers moderados

anomalies = iqr_ad.fit_detect(dataframe_anomalias)
plot(dataframe_anomalias, anomaly=anomalies, anomaly_color="red", anomaly_tag="marker")
from adtk.detector import PersistAD
 
dataframe_anomalias = db_train[["focos", "precipitacao"]]

#positive = procura uma anomalia à frente
persist_ad = PersistAD(c=3.0, side='positive')
persist_ad.window = 24
anomalies = persist_ad.fit_detect(dataframe_anomalias)
plot(dataframe_anomalias, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_color='red')
from adtk.detector import AutoregressionAD

dataframe_anomalias = db_train[["focos", "precipitacao"]]

autoregression_ad = AutoregressionAD(n_steps=4, step_size=12, c=3.0)
anomalies = autoregression_ad.fit_detect(dataframe_anomalias)
plot(dataframe_anomalias, anomaly=anomalies, ts_markersize=1, anomaly_color='red', anomaly_tag="marker", anomaly_markersize=5);
from adtk.detector import MinClusterDetector
from sklearn.cluster import KMeans

dataframe_anomalias = db_train[["focos", "precipitacao", "temp_minima", "umidade"]]

min_cluster_detector = MinClusterDetector(KMeans(n_clusters=3))
anomalies = min_cluster_detector.fit_detect(dataframe_anomalias)

plot(dataframe_anomalias, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_color='red', anomaly_alpha=0.3, curve_group='all');
from adtk.detector import OutlierDetector
from sklearn.neighbors import LocalOutlierFactor

dataframe_anomalias = db_train[["focos", "precipitacao", "temp_minima", "umidade"]]

outlier_detector = OutlierDetector(LocalOutlierFactor(contamination=0.05))

anomalies = outlier_detector.fit_detect(dataframe_anomalias)

plot(dataframe_anomalias, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_color='red', anomaly_alpha=0.3, curve_group='all');
from statsmodels.tsa.seasonal import seasonal_decompose

rcParams['figure.figsize'] = 18, 6
decomposition = seasonal_decompose(database["focos"], model='additive')
fig = decomposition.plot()
plt.show()
rcParams['figure.figsize'] = 18, 6
decomposition = seasonal_decompose(database["temp_minima"], model='additive')
fig = decomposition.plot()
plt.show()
rcParams['figure.figsize'] = 18, 6
decomposition = seasonal_decompose(database["precipitacao"], model='additive')
fig = decomposition.plot()
plt.show()
rcParams['figure.figsize'] = 18, 6
decomposition = seasonal_decompose(database["umidade"], model='additive')
fig = decomposition.plot()
plt.show()
from adtk.transformer import Retrospect

dataframe_anomalias = np.log(db_train[["focos"]])
dataframe_tratar = db_train[["temp_minima"]]

df = Retrospect(n_steps=3, step_size=1, till=1).transform(dataframe_tratar)
plot(pd.concat([dataframe_anomalias, df], axis=1), curve_group="all");
database_tratado = database.copy();

tratar_outlier(database_tratado, "precipitacao")
tratar_outlier(database_tratado, "focos")
from adtk.detector import InterQuartileRangeAD

db_train = validate_series(database_tratado[['focos','precipitacao', 'temp_minima', 'umidade']])

dataframe_anomalias = db_train[["focos", "precipitacao"]]

iqr_ad = InterQuartileRangeAD(c=3) #outliers moderados

anomalies = iqr_ad.fit_detect(dataframe_anomalias)
plot(dataframe_anomalias, anomaly=anomalies, anomaly_color="red", anomaly_tag="marker")
from adtk.detector import MinClusterDetector
from sklearn.cluster import KMeans

dataframe_anomalias = db_train[["focos", "precipitacao", "temp_minima", "umidade"]]

min_cluster_detector = MinClusterDetector(KMeans(n_clusters=3))
anomalies = min_cluster_detector.fit_detect(dataframe_anomalias)

plot(dataframe_anomalias, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_color='red', anomaly_alpha=0.3, curve_group='all');
plt.figure(figsize=(24,6))

plt.subplot(1,3,1)
plt.scatter(database_tratado.temp_minima, database_tratado.focos, s=20)
draw_mean(database_tratado.temp_minima, database_tratado.focos);
plt.ylabel('Qtde. Focos')
plt.xlabel('Temperatura Mínima')
plt.title('Correlação Focos x Temp. Mínima');

plt.subplot(1,3,2)
plt.scatter(database_tratado.temp_media, database_tratado.focos, s=20)
draw_mean(database_tratado.temp_media, database_tratado.focos);
plt.ylabel('Qtde. Focos')
plt.xlabel('Temperatura Média')
plt.title('Correlação Focos x Temp. Media');

plt.subplot(1,3,3)
plt.scatter(database_tratado.temp_maxima, database_tratado.focos, s=20)
draw_mean(database_tratado.temp_maxima, database_tratado.focos);
plt.ylabel('Qtde. Focos')
plt.xlabel('Temperatura Máxima')
plt.title('Correlação Focos x Temp. Máxima');
plt.figure(figsize=(24,6))

plt.subplot(1,3,1)
plt.scatter(database_tratado.precipitacao, database_tratado.focos, s=20)
draw_mean(database_tratado.precipitacao, database_tratado.focos);
plt.ylabel('Qtde. Focos')
plt.xlabel('Precipitação')
plt.title('Correlação Focos x Precipitação');

plt.subplot(1,3,2)
plt.scatter(database_tratado.umidade, database_tratado.focos, s=20)
draw_mean(database_tratado.umidade, database_tratado.focos);
plt.ylabel('Qtde. Focos')
plt.xlabel('Umidade Média')
plt.title('Correlação Focos x Umidade');

plt.subplot(1,3,3)
plt.scatter(database_tratado.precipitacao, database_tratado.umidade, s=20)
draw_mean(database_tratado.precipitacao, database_tratado.umidade);
plt.xlabel('Precipitação')
plt.ylabel('Umidade')
plt.title('Correlação Umidade x Precipitação');
rcParams['figure.figsize'] = 18, 6
decomposition = seasonal_decompose(database_tratado["temp_minima"], model='additive')
fig = decomposition.plot()
plt.show()
print(database.focos.sum())
print(database_tratado.focos.sum())

print(database.precipitacao.sum())
print(database_tratado.precipitacao.sum())
print(database.corr('spearman')["focos"])
print("#########");
print(database_tratado.corr('spearman')["focos"])
corr = database_tratado.corr('spearman')
sb.heatmap(corr, 
            xticklabels=corr.columns,
            yticklabels=corr.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5)
print(st.spearmanr(database_tratado.focos,database_tratado.temp_minima))
print(st.spearmanr(database_tratado.focos,database_tratado.temp_media))
print(st.spearmanr(database_tratado.focos,database_tratado.temp_maxima))
print(st.spearmanr(database_tratado.focos,database_tratado.umidade))
print(st.spearmanr(database_tratado.focos,database_tratado.precipitacao))
#segundo estudos, variáveis de precipitação tem um uso aplicado de até 70 dias
#segundo estudo, variáveis de temperatura tem um time lag aplicado de até 20 dias
#traremos time lag de variáveis de um, dois e três meses

adiciona_timelag(database, 6, ["precipitacao", "temp_minima", "temp_media", "temp_maxima", "umidade"])
adiciona_timelag(database_tratado, 6, ["precipitacao", "temp_minima", "temp_media", "temp_maxima", "umidade"])

print(database.head())
print(database_tratado.head())
database_tratado.corr('spearman')
print(st.spearmanr(database_tratado.focos,database_tratado.temp_minima_m1))
print(st.spearmanr(database_tratado.focos,database_tratado.temp_media_m1))
print(st.spearmanr(database_tratado.focos,database_tratado.temp_maxima_m1))
#desvio padrão
print(statistics.pstdev(database_tratado.focos), statistics.pstdev(database_tratado.temp_minima_m1), statistics.pstdev(database_tratado.umidade), statistics.pstdev(database_tratado.precipitacao))
focos_total = database_tratado.focos.resample('MS').mean()
teste_estacionaridade(focos_total, 'Focos');
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(focos_total) #, lags=12
plt.show()
import statsmodels.api as sm
import itertools

#melhores parâmetros para série
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(focos_total,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit(maxiter=60)
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
import statsmodels.api as sm

#Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors (SARIMAX)
mod = sm.tsa.statespace.SARIMAX(focos_total,
                                order=(1, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
results.summary()
results.plot_diagnostics(figsize=(16, 8))
plt.show()
pred = results.get_prediction(start=pd.to_datetime('2018-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = focos_total.plot(label='Observado')
pred.predicted_mean.plot(ax=ax, label='Um passo à frente', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Focos')
plt.legend()
plt.show()
y_forecasted = pred.predicted_mean
y_truth = focos_total
mse = ((y_forecasted - y_truth) ** 2).mean()
#erro quadrático médio. quanto menor, melhor
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

#O erro quadrático médio da raiz (RMSE) nos diz que nosso modelo foi capaz de prever as vendas médias diárias de móveis no conjunto de teste
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
mean_absolute_percentage_error(y_forecasted, y_truth['2018-01-01':])
pred_uc = results.get_forecast(steps=60)
pred_ci = pred_uc.conf_int()

ax = focos_total['2019-01-01':].plot(label='Observado', figsize=(14, 7))
pred_uc.predicted_mean['2019-01-01':].plot(ax=ax, label='Focos')

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Focos')
plt.legend()
plt.show()
endogena_ce = database_tratado.focos.resample('MS').mean()
teste_estacionaridade(endogena_ce, 'Focos');

exogenas_ce = database_tratado[['temp_minima', 'precipitacao', 'umidade']].resample('MS').mean()
#exogenas.temp_minima = np.log(exogenas['temp_minima']);
teste_estacionaridade(exogenas_ce.temp_minima, 'Temperatura Minima');
teste_estacionaridade(exogenas_ce.precipitacao, 'Precipitacao');
teste_estacionaridade(exogenas_ce.umidade, 'Umidade');

import statsmodels.api as sm
import itertools

#melhores parâmetros para série
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(endogena_ce,
                                            exogenas_ce,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit(maxiter=120)
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
import statsmodels.api as sm

#Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors (SARIMAX)
mod_ce = sm.tsa.statespace.SARIMAX(endogena_ce,
                                exogenas_ce,
                                order=(1, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results_ce = mod_ce.fit(maxiter=200)
results_ce.summary()
results_ce.plot_diagnostics(figsize=(16, 8))
plt.show()
pred_ce = results_ce.get_prediction(start=pd.to_datetime('2018-01-01'),dynamic=False)
pred_ci_ce = pred_ce.conf_int()
ax_ce = endogena_ce.plot(label='Observado')
pred_ce.predicted_mean.plot(ax=ax_ce, label='Um passo à frente', alpha=.7, figsize=(14, 7))
ax_ce.fill_between(pred_ci_ce.index,
                pred_ci_ce.iloc[:, 0],
                pred_ci_ce.iloc[:, 1], color='k', alpha=.2)
ax_ce.set_xlabel('Date')
ax_ce.set_ylabel('Focos')
plt.legend()
plt.show()
y_forecasted_ce = pred.predicted_mean
y_truth_ce = endogena_ce
mse_ce = ((y_forecasted_ce - y_truth_ce) ** 2).mean()
#erro quadrático médio. quanto menor, melhor
print('The Mean Squared Error of our forecasts is {}'.format(round(mse_ce, 2)))

#O erro quadrático médio da raiz (RMSE) nos diz que nosso modelo foi capaz de prever as vendas médias diárias de móveis no conjunto de teste
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse_ce), 2)))
mean_absolute_percentage_error(y_forecasted_ce, y_truth_ce['2018-01-01':])
predicted_exogenas = [
    database['2019-01-01':'2019-12-01'].temp_minima,
    database['2019-01-01':'2019-12-01'].precipitacao,
    database['2019-01-01':'2019-12-01'].umidade,
]
pred_uc_ce = results_ce.get_forecast(exog = predicted_exogenas, steps=12)
pred_ci_ce = pred_uc_ce.conf_int()

ax = endogena_ce['2018-01-01':].plot(label='Observado', figsize=(14, 7))
pred_uc_ce.predicted_mean['2018-01-01':].plot(ax=ax, label='Projetados')
ax.fill_between(pred_ci_ce.index,
                pred_ci_ce.iloc[:, 0],
                pred_ci_ce.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Focos')
plt.legend()
plt.show()
predicted_exogenas = [
    database['2015-01-01':'2019-12-01'].temp_minima,
    database['2015-01-01':'2019-12-01'].precipitacao,
    database['2015-01-01':'2019-12-01'].umidade,
]
pred_uc_ce = results_ce.get_forecast(exog = predicted_exogenas, steps=60)
pred_ci_ce = pred_uc_ce.conf_int()

ax = endogena_ce['2018-01-01':].plot(label='Observado', figsize=(14, 7))
pred_uc_ce.predicted_mean['2018-01-01':].plot(ax=ax, label='Projetados')
ax.fill_between(pred_ci_ce.index,
                pred_ci_ce.iloc[:, 0],
                pred_ci_ce.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Focos')
plt.legend()
plt.show()