import pandas as pd


#Dados internacionais
c=pd.read_csv('../input/cpia-local-de-datahubs-covid19-dataset/time-series-19-covid-combined.csv')
df_sumario_w = c.groupby(['Date','Country/Region'],as_index = False)['Confirmed','Deaths'].sum().pivot('Date','Country/Region').fillna(0)
df_conf_w = df_sumario_w["Confirmed"]
df_deaths_w = df_sumario_w["Deaths"]

#Dados População Internacionais
popw = pd.read_excel('../input/cpia-local-de-covid19-ecdc-dataset/COVID-19-geographic-disbtribution-world.xlsx')
popw["countriesAndTerritories"] = popw["countriesAndTerritories"].str.replace("United_States_of_America", "US", case = False)
popw['countriesAndTerritories'] = popw['countriesAndTerritories'].str.replace("_"," ")
pop = popw.groupby('countriesAndTerritories', as_index=False).max()[['countriesAndTerritories','popData2018']]
pop.rename(columns={'countriesAndTerritories':'local', 'popData2018':'pop'}, inplace=True);

# Dados Nacionais
df = pd.read_csv('../input/corona-virus-brazil/brazil_covid19.csv')
dfe = df.drop(['region'], axis=1) #
dfe_sumario = dfe.groupby(['date','state'],as_index = False)['cases','deaths'].sum().pivot('date','state').fillna(0)
dfe_conf = dfe_sumario["cases"]
dfe_deaths = dfe_sumario["deaths"]
# Por dia
def acum2pd(df_conf):
    
    df_conf_pd = df_conf.copy()

    for col in df_conf.columns:
        for i,x in enumerate(df_conf_pd[col]):
            if i != 0:
                pdia = df_conf[col][i]-df_conf[col][i-1]
                if pdia < 0:
                    df_conf_pd[col][i] = 0
                    df_conf[col][i] = df_conf[col][i-1]
                else:
                    df_conf_pd[col][i] = pdia
    return df_conf_pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def sigm_predict(y, future_days, boundary):
    def avg_err(pcov):
        return np.round(np.sqrt(np.diag(pcov)).mean(), 2)
    # function to be minimized
    def f_sigmoid(x, a, b, c):
        # a = sigmoid midpoint
        # b = curve steepness (logistic growth)
        # c = max value
        return (c / (1 + np.exp(-b*(x-a))))
  
    x = np.arange(len(y))
    
    # fitting the data on the logistic function
    #boundary = ([-np.inf, -np.inf, -np.inf],[np.inf, np.inf, np.inf])
    #boundary = ([0., 0.001, y.max()],[90., 2.5, 100*y.max()])
    popt_sig, pcov_sig = curve_fit(f_sigmoid, x[:-1], y[:-1], method='trf',bounds=boundary)#,sigma = np.linspace(0.5, 0.05, len(y)),absolute_sigma=True)#dogbox, sigma = np.linspace(0.5, 0.05, len(y)),absolute_sigma=False
    peakday = len(y) + int(popt_sig[0])
     
    x_m = np.arange(len(y)+future_days)
    y_m = f_sigmoid(x_m, *popt_sig)    
    
    return x_m, y_m, avg_err(pcov_sig), popt_sig
df_conf_w.tail()
df_deaths_w.tail()
plt.figure(figsize=(16,8))
ax = plt.subplot(1,2,1)
ax.set_title('Casos Acumulados', fontsize=18, loc='left')
plt.plot(df_conf_w['Germany'], label='Germany')
plt.plot(df_conf_w['Italy'], label='Italy')
plt.plot(df_conf_w['Brazil'], label='Brazil')
plt.xlabel("Dia")
plt.ylabel("Casos")
plt.legend();
ax = plt.subplot(1,2,2)
ax.set_title('Óbitos Acumulados', fontsize=18, loc='left')
plt.plot(df_deaths_w['Germany'], label='Germany')
plt.plot(df_deaths_w['Italy'], label='Italy')
plt.plot(df_deaths_w['Brazil'], label='Brazil')
plt.xlabel("Dia")
plt.ylabel("Óbitos")
plt.legend();
df_conf_pd = acum2pd(df_conf_w)
df_deaths_pd = acum2pd(df_deaths_w)
plt.figure(figsize=(16,8))
ax = plt.subplot(1,2,1)
ax.set_title('Casos por Dia', fontsize=18, loc='left')
plt.plot(df_conf_pd['Germany'], label='Germany')
plt.plot(df_conf_pd['Italy'], label='Italy')
plt.plot(df_conf_pd['Brazil'], label='Brazil')
plt.xlabel("Dia")
plt.ylabel("Casos")
plt.legend();
ax = plt.subplot(1,2,2)
ax.set_title('Óbitos por Dia', fontsize=18, loc='left')
plt.plot(df_deaths_pd['Germany'], label='Germany')
plt.plot(df_deaths_pd['Italy'], label='Italy')
plt.plot(df_deaths_pd['Brazil'], label='Brazil')
plt.xlabel("Dia")
plt.ylabel("Óbitos")
plt.legend();
dfe_conf_pd = acum2pd(dfe_conf)
dfe_deaths_pd = acum2pd(dfe_deaths)
## Graficos Estados do Brasil
plt.figure(figsize=(16,8))
ax = plt.subplot(1,2,1)
ax.set_title('Casos por Dia', fontsize=18, loc='left')
plt.plot(dfe_conf_pd['São Paulo'], label='SP')
plt.plot(dfe_conf_pd['Amazonas'], label='AM')
plt.plot(dfe_conf_pd['Santa Catarina'], label='SC')
plt.xlabel("Dia")
plt.ylabel("Casos")
plt.legend();
ax = plt.subplot(1,2,2)
ax.set_title('Óbitos por Dia', fontsize=18, loc='left')
plt.plot(dfe_deaths_pd['São Paulo'], label='SP')
plt.plot(dfe_deaths_pd['Amazonas'], label='AM')
plt.plot(dfe_deaths_pd['Santa Catarina'], label='SC')
plt.xlabel("Dia")
plt.ylabel("Óbitos")
plt.legend();
br = dfe_conf['Amazonas'][:75]
br1 = br.copy()
br1[-1] = br1[-2] + br1[-2] - br1[-3]
plt.figure(figsize=(16,8))
ax = plt.subplot(1,2,1)
ax.set_title('Casos Acumulados', fontsize=18, loc='left')
plt.plot(br, 'b', label='AM')
plt.plot(br1, 'r', label='Prev')
plt.xlabel("Dia")
plt.ylabel("Casos")
plt.legend();
ax = plt.subplot(1,2,2)
ax.set_title('Casos por Dia', fontsize=18, loc='left')
plt.plot(np.diff(br), 'b', label='AM')
plt.plot(np.diff(br1), 'r', label='Prev')
plt.xlabel("Dia")
plt.ylabel("Casos")
plt.legend();
#Lib scikit-learn
from sklearn.linear_model import LinearRegression
import numpy as np

#A partir do 100o caso
casos_br = df_conf_pd['Brazil'][df_conf_pd[df_conf_pd['Brazil'] > 100].index[0]:].values
#Dividindo os dados do passado e os a serem previstos
casos_br_treino = casos_br[:-5]
casos_br_teste = casos_br[-5:]
#Instanciando o modelo
model = LinearRegression()
#Botando no formato do fit
X = np.expand_dims(np.linspace(0,len(casos_br_treino), len(casos_br_treino)), axis=1)
y = np.expand_dims(casos_br_treino, axis=1)
#Fazendo o treino
model.fit(X,y)
#Prevendo com dias à frente
casos_br_prev = model.predict(np.expand_dims(np.linspace(0, len(casos_br_treino)+5,len(casos_br)), axis=1))
plt.figure(figsize=(12,8))
plt.plot(np.linspace(0,len(casos_br_treino), len(casos_br_treino)), casos_br_treino, 'b', label='Passado')
plt.plot(np.linspace(len(casos_br_treino), len(casos_br_treino)+5, 5), casos_br_teste, 'g', label = 'Futuro')
plt.plot(np.linspace(0, len(casos_br_prev), len(casos_br_prev)), casos_br_prev, 'r', label = 'Previsão')
plt.xlabel("Dia")
plt.ylabel("Casos")
plt.legend();
from sklearn.preprocessing import PolynomialFeatures 

# Instanciando Modelo  
poly = PolynomialFeatures(degree = 4) 
X_poly = poly.fit_transform(X) 
# Treino  
poly.fit(X_poly, y) 
lin2 = LinearRegression() 
lin2.fit(X_poly, y)
# Predição
casos_br_prev_poli = lin2.predict(poly.fit_transform(np.expand_dims(np.linspace(0, len(casos_br_treino)+5,len(casos_br)), axis=1)))
plt.figure(figsize=(12,8))
plt.plot(np.linspace(0,len(casos_br_treino), len(casos_br_treino)), casos_br_treino, 'b', label='Passado')
plt.plot(np.linspace(len(casos_br_treino), len(casos_br_treino)+5, 5), casos_br_teste, 'g', label = 'Futuro')
plt.plot(np.linspace(0, len(casos_br_prev), len(casos_br_prev)), casos_br_prev_poli, 'r', label = 'Previsão')
plt.xlabel("Dia")
plt.ylabel("Casos")
plt.legend();
from pandas.plotting import autocorrelation_plot

ax = plt.figure(figsize=(12,6))
ax.suptitle("Autocorrelação")
autocorrelation_plot(casos_br_treino)
ax=ax
from statsmodels.tsa.seasonal import seasonal_decompose
from pylab import rcParams

rcParams['figure.figsize'] = 11, 9

dec = seasonal_decompose(casos_br_treino, freq=7)
dec.plot();

!pip install pmdarima
from pmdarima import auto_arima

# Instancia modelo já com os melhores parâmetros buscados automaticamente
m_sarima = auto_arima(casos_br_treino,
                     start_p=0, start_q=0, max_p=10, max_q=10,
                     d=1, m=7, seasonal = True,
                     D=1, start_Q=1, start_P=1, max_P=4, max_Q=4,
                     trace=True,
                     error_action='ignore', suppress_warnings=True,
                     stepwise=False)
#Treino
m_sarima.fit(casos_br_treino)
#Predição
fitted = m_sarima.predict_in_sample(start=1, end=len(casos_br_treino))
casos_br_prev_sarima = m_sarima.predict(n_periods=len(casos_br_teste))
plt.figure(figsize=(12,8))
plt.plot(np.linspace(0,len(casos_br_treino), len(casos_br_treino)), casos_br_treino, 'b', label='Passado')
plt.plot(np.linspace(0,len(casos_br_treino), len(casos_br_treino)), fitted, 'r')
plt.plot(np.linspace(len(casos_br_treino), len(casos_br_treino)+5, 5), casos_br_teste, 'g', label='Futuro')
plt.plot(np.linspace(len(casos_br_treino), len(casos_br_treino)+5, 5), casos_br_prev_sarima, 'r', label='Previsão')
plt.xlabel("Dia")
plt.ylabel("Casos")
plt.legend()
#A partir do 10o caso
casos_al = df_conf_pd['Germany'][df_conf_pd[df_conf_pd['Germany'] > 10].index[0]:].values
plt.plot(casos_al);
#Dividindo os dados do passado e os a serem previstos
casos_al_treino = casos_al[:25]
casos_al_teste = casos_al[25:]

num_proj = 10
# Regressão Linear
model = LinearRegression()
X = np.expand_dims(np.linspace(0,len(casos_al_treino), len(casos_al_treino)), axis=1)
y = np.expand_dims(casos_al_treino, axis=1)
model.fit(X,y)
casos_al_prev = model.predict(np.expand_dims(np.linspace(0,len(casos_al_treino)+num_proj,len(casos_al_treino)+num_proj), axis=1))

#Regressão Polinimial
poly = PolynomialFeatures(degree = 3) 
X_poly = poly.fit_transform(X) 
poly.fit(X_poly, y) 
lin2 = LinearRegression() 
lin2.fit(X_poly, y)
casos_al_prev_poli = lin2.predict(poly.fit_transform(np.expand_dims(np.linspace(0, len(casos_al_treino)+num_proj,len(casos_al_treino)+num_proj), axis=1)))

#SARIMA
m_sarima = auto_arima(casos_al_treino,
                     start_p=0, start_q=0, max_p=10, max_q=10,
                     d=1, m=7, seasonal = True,
                     D=1, start_Q=1, start_P=1, max_P=4, max_Q=4,
                     trace=True,
                     error_action='ignore', suppress_warnings=True,
                     stepwise=False)
safit = m_sarima.fit(casos_al_treino)
fitted = m_sarima.predict_in_sample(start=1, end=len(casos_al_treino))
casos_al_prev_sarima = m_sarima.predict(n_periods=num_proj)
plt.figure(figsize=(10,4))
#plt.plot(np.linspace(0,len(casos_al), len(casos_al)), casos_al, 'b', label='Real')
plt.plot(np.linspace(0,len(casos_al_treino), len(casos_al_treino)), casos_al_treino, 'b', label='Passado')
plt.plot(np.linspace(len(casos_al_treino), len(casos_al), len(casos_al)-len(casos_al_treino)), casos_al[len(casos_al_treino):], 'g', label='Futuro')
plt.plot(np.linspace(0,len(casos_al_treino)+num_proj, len(casos_al_treino)+num_proj), casos_al_prev, 'r', label='Linear')
plt.plot(np.linspace(0,len(casos_al_treino)+num_proj, len(casos_al_treino)+num_proj), casos_al_prev_poli, 'y', label='Poli')
plt.plot(np.linspace(len(casos_al_treino), len(casos_al_treino)+num_proj, num_proj), casos_al_prev_sarima[:], 'c', label='SARIMA')
plt.plot(np.linspace(0,len(casos_al_treino), len(casos_al_treino)), fitted, 'c')
plt.xlabel("Dia")
plt.ylabel("Casos")
plt.legend();
#Dividindo os dados do passado e os a serem previstos
casos_al_treino = casos_al[:30]
casos_al_teste = casos_al[30:]

num_proj = 10
# Regressão Linear
model = LinearRegression()
X = np.expand_dims(np.linspace(0,len(casos_al_treino), len(casos_al_treino)), axis=1)
y = np.expand_dims(casos_al_treino, axis=1)
model.fit(X,y)
casos_al_prev = model.predict(np.expand_dims(np.linspace(0,len(casos_al_treino)+num_proj,len(casos_al_treino)+num_proj), axis=1))

#Regressão Polinimial
poly = PolynomialFeatures(degree = 3) 
X_poly = poly.fit_transform(X) 
poly.fit(X_poly, y) 
lin2 = LinearRegression() 
lin2.fit(X_poly, y)
casos_al_prev_poli = lin2.predict(poly.fit_transform(np.expand_dims(np.linspace(0, len(casos_al_treino)+num_proj,len(casos_al_treino)+num_proj), axis=1)))

#SARIMA
m_sarima = auto_arima(casos_al_treino,
                     start_p=0, start_q=0, max_p=10, max_q=10,
                     d=1, m=7, seasonal = True,
                     D=1, start_Q=1, start_P=1, max_P=4, max_Q=4,
                     trace=True,
                     error_action='ignore', suppress_warnings=True,
                     stepwise=False)
safit = m_sarima.fit(casos_al_treino)
fitted = m_sarima.predict_in_sample(start=1, end=len(casos_al_treino))
casos_al_prev_sarima = m_sarima.predict(n_periods=num_proj)
plt.figure(figsize=(10,4))
#plt.plot(np.linspace(0,len(casos_al), len(casos_al)), casos_al, 'b', label='Real')
plt.plot(np.linspace(0,len(casos_al_treino), len(casos_al_treino)), casos_al_treino, 'b', label='Passado')
plt.plot(np.linspace(len(casos_al_treino), len(casos_al), len(casos_al)-len(casos_al_treino)), casos_al[len(casos_al_treino):], 'g', label='Futuro')
plt.plot(np.linspace(0,len(casos_al_treino)+num_proj, len(casos_al_treino)+num_proj), casos_al_prev, 'r', label='Linear')
plt.plot(np.linspace(0,len(casos_al_treino)+num_proj, len(casos_al_treino)+num_proj), casos_al_prev_poli, 'y', label='Poli')
plt.plot(np.linspace(len(casos_al_treino), len(casos_al_treino)+num_proj, num_proj), casos_al_prev_sarima[:], 'c', label='SARIMA')
plt.plot(np.linspace(0,len(casos_al_treino), len(casos_al_treino)), fitted, 'c')
plt.xlabel("Dia")
plt.ylabel("Casos")
plt.legend()
from IPython.display import Image
from IPython.core.display import HTML 
Image(url= "https://statsland.files.wordpress.com/2012/08/extrapolating.jpg")
plt.figure(figsize=(16,8))
ax = plt.subplot(1,2,1)
ax.set_title('Casos Acumulados', fontsize=18, loc='left')
plt.plot(df_conf_w['Germany'], label='Germany')
plt.plot(df_conf_w['Italy'], label='Italy')
plt.plot(df_conf_w['China'], label='China')
plt.plot(df_conf_w['Brazil'], label='Brazil')
plt.xlabel("Dia")
plt.ylabel("Casos")
plt.legend();
ax = plt.subplot(1,2,2)
ax.set_title('Casos por Dia', fontsize=18, loc='left')
plt.plot(df_conf_pd['Germany'], label='Germany')
plt.plot(df_conf_pd['Italy'], label='Italy')
plt.plot(df_conf_pd['China'], label='China')
plt.plot(df_conf_pd['Brazil'], label='Brazil')
plt.xlabel("Dia")
plt.ylabel("Casos")
plt.legend();
acum_al = df_conf_w['Germany'][df_conf_pd[df_conf_pd['Germany'] > 10].index[0]:].values
acum_al_treino = acum_al[:30]
acum_al_teste = acum_al[30:]
# Aqui coloquei que o espaço de busca é pelo menos o q tivemos vezes, dois, pois vemos que não estamos no pico
boundaries = ([0., 0.05, acum_al_treino.max()*2],[90., 1.0, 100*acum_al_treino.max()])
xal, yal, erral, popt_sigal = sigm_predict(acum_al_treino, 30, boundaries)
#plt.plot(np.linspace(0,len(acum_al),len(acum_al)), acum_al, 'y', label='Real')
plt.plot(np.linspace(0,len(acum_al_treino),len(acum_al_treino)), acum_al_treino, 'b', label='Passado')
plt.plot(np.linspace(len(acum_al_treino),len(acum_al_treino)+30, 30), acum_al_teste[:30], 'g', label='Futuro')
plt.plot(np.linspace(0, len(acum_al_treino)+30,len(acum_al_treino)+30), yal, 'r', label = 'Pred. Al')
plt.xlabel("Dia")
plt.ylabel("Casos")
plt.legend();
pd_al_real = df_conf_pd['Germany'][df_conf_pd[df_conf_pd['Germany'] > 10].index[0]:].values
pd_al_prev = np.diff(yal)
plt.plot(pd_al_real, label='Real')
plt.plot(pd_al_prev, label='Prev Sigmoid')
plt.xlabel("Dia")
plt.ylabel("Casos")
plt.legend();
eixo_dias = np.linspace(0,df_conf_w.shape[0], len(df_conf_w))
plt.plot(eixo_dias, df_conf_w['Italy']*1000000/pop[pop['local'] == 'Italy']['pop'].values, label = 'Italy')
plt.plot(eixo_dias, df_conf_w['France']*1000000/pop[pop['local'] == 'France']['pop'].values, label = 'France')
plt.plot(eixo_dias, df_conf_w['Switzerland']*1000000/pop[pop['local'] == 'Switzerland']['pop'].values, label = 'Spain')
plt.plot(eixo_dias, df_conf_w['Norway']*1000000/pop[pop['local'] == 'Norway']['pop'].values, label = 'Norway')
plt.xlabel("Dia")
plt.ylabel("Casos Acumulados")
plt.legend();
#Dados por 1M, a partir de >10 casos
acum_al_1m = df_conf_w['Germany'][df_conf_pd[df_conf_pd['Germany'] > 10].index[0]:]*1000000/pop[pop['local'] == 'Germany']['pop'].values
acum_al_treino_1m = acum_al_1m[:30]
acum_al_teste_1m = acum_al_1m[30:]
# Aqui o limite máximo será algum valor entre próximo de 1500 (Norway)
boundaries = ([0., 0.05, 1500],[90., 1.0, 2000])
xal, yal_1m_min, erral, popt_sigal = sigm_predict(acum_al_treino_1m, 30, boundaries)
print(popt_sigal[0])
# Um valor próximo do maior da Itália 3500
boundaries = ([0., 0.05, 3500],[90., 1.0, 4000])
xal, yal_1m_max, erral, popt_sigal = sigm_predict(acum_al_treino_1m, 30, boundaries)
print(popt_sigal[0])
#plt.plot(np.linspace(0,len(acum_al_1m),len(acum_al_1m)), acum_al_1m, 'y', label='Real')
plt.plot(np.linspace(0,len(acum_al_treino_1m),len(acum_al_treino_1m)), acum_al_treino_1m, 'b', label='Passado')
plt.plot(np.linspace(len(acum_al_treino_1m),len(acum_al_treino_1m)+30, 30), acum_al_teste_1m[:30], 'g', label='Futuro')
plt.plot(np.linspace(0, len(acum_al_treino_1m)+30,len(acum_al_treino_1m)+30), yal_1m_min, 'r', label = 'Pred. Min')
plt.plot(np.linspace(0, len(acum_al_treino_1m)+30,len(acum_al_treino_1m)+30), yal_1m_max, 'r', label = 'Pred. Max')
plt.xlabel("Dia")
plt.ylabel("Casos")
plt.legend();
fator = 1000000/pop[pop['local'] == 'Germany']['pop'].values[0]
# Valores calculados por 1M, agora vontando aos absolutos
pd_al_prev_min = np.diff(yal_1m_min/fator)
pd_al_prev_max = np.diff(yal_1m_max/fator)
plt.plot(pd_al_real, label='Real')
plt.plot(pd_al_prev_min, 'r', label='Prev Min')
plt.plot(pd_al_prev_max, 'r', label='Prev Max')
plt.xlabel("Dia")
plt.ylabel("Casos por dia")
plt.legend();
# Precisa estar num dataframe com data
df_al = df_conf_w['Germany'][df_conf_pd[df_conf_pd['Germany'] > 10].index[0]:].copy()
df_al = df_al.reset_index()
df_al.rename(columns={'Date':'ds', 'Germany':'y'}, inplace=True)
df_al_treino = df_al[:30]
df_al_teste = df_al[30:]
from fbprophet import Prophet

prophet = Prophet(growth='logistic',
                  changepoint_range=0.95,
                  yearly_seasonality=False,
                  weekly_seasonality=True,
                  daily_seasonality=False,
                  #seasonality_prior_scale=10,
                  changepoint_prior_scale=.5)#.01
#Vamos estabelecer os limites como anteriormente
cap = 2000/fator#10*df_al_treino['y'].max()
floor = 0#2*df_al_treino['y'].max()
df_al_treino['cap'] = cap
df_al_treino['floor'] = floor
proffit = prophet.fit(df_al_treino)

#fitted = m_sarima.predict_in_sample(start=1, end=len(casos_al_treino))
futuro = prophet.make_future_dataframe(periods=30, freq='D')
futuro['cap'] = cap
futuro['floor'] = floor
casos_al_prev_prof = prophet.predict(futuro)
fig = prophet.plot(casos_al_prev_prof)
#a = add_changepoints_to_plot(fig.gca(), prophet, forecast)
plt.plot(futuro['ds'][30:], df_al_teste['y'][:30], 'g', label = 'Futuro')
plt.show()
#fig2 = prophet.plot_components(forecast)
#plt.show()
#Por dia
plt.figure(figsize=(10,4))
#plt.plot(np.linspace(0,len(casos_al), len(casos_al)), casos_al, 'b', label='Real')
plt.plot(np.linspace(0,len(casos_al_treino), len(casos_al_treino)), casos_al_treino, 'b', label='Passado')
plt.plot(np.linspace(len(casos_al_treino), len(casos_al), len(casos_al)-len(casos_al_treino)), casos_al[len(casos_al_treino):], 'g', label='Futuro')
plt.plot(np.linspace(0,len(casos_al_treino)+30, len(casos_al_treino)+30), casos_al_prev_prof['yhat'].diff().values, 'r', label='Prophet')
#plt.plot(np.linspace(0,len(casos_al_treino), len(casos_al_treino)), fitted, 'c')
plt.xlabel("Dia")
plt.ylabel("Casos")
plt.legend()
