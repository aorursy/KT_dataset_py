import pandas as pd
import glob
import math
import numpy as np
import os as os
import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
plt.style.use('fivethirtyeight')
rcParams['figure.figsize'] = 20, 8

#path = "X:/Historico_Cotacoes/teste/alpha_data/"
nomeArq = '../input/alphacart12018.csv'

#pwd = os.getcwd() # guarda o path corrente
#os.chdir(os.path.dirname(path)) # muda para o path dos arquivos

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
#df = pd.read_csv(os.path.basename(nomeArq), parse_dates=["datetime"], index_col="datetime",date_parser=dateparse)
df = pd.read_csv(nomeArq, parse_dates=["datetime"], index_col="datetime",date_parser=dateparse)
df['2018-08-28' : '2018-08-28'].head(15)
## Definindo a data de inicio das observacoes, quandotodos os ativos possuem dados
start_date = "2013-01-01"
end_date = "2018-12-31"
s = df.loc[start_date:end_date] 
df = s.copy()
combined = df.sort_index(ascending=True)
d_decimals = 2    
combined['adj_close'] = combined['adj_close'].apply(lambda x: round(x, d_decimals))
comb = combined.copy()
combined = comb.copy()
combined.head()
combined.info()
comb = combined.copy()
comb.reset_index(inplace = True)
comb.head()
# Checando e eliminando os feriados nacionais brasileiros
feriados = ['2013-01-01', '2013-01-25', '2013-02-11', '2013-02-12', '2013-02-13', '2013-02-18', '2013-03-29', '2013-04-21', '2013-05-01', '2013-05-30', '2013-07-09', '2013-09-07', '2013-11-15', '2013-11-20', '2013-12-25', '2014-01-01', '2014-01-25', '2014-03-03', '2014-03-04', '2014-03-05', '2014-04-18', '2014-04-21', '2014-05-01', '2014-06-19', '2014-07-09', '2014-09-07', '2014-11-20', '2014-12-25', '2015-01-01', '2015-01-25', '2015-02-25', '2015-02-16', '2015-02-17', '2015-02-18', '2015-04-03', '2015-04-21', '2015-05-01', '2015-06-04', '2015-07-09', '2015-09-07', '2015-10-12', '2015-11-02', '2015-11-20', '2015-12-25', '2016-01-01', '2016-01-25', '2016-02-08', '2016-02-09', '2016-02-10', '2016-03-25', '2016-04-21', '2016-05-01', '2016-05-26', '2016-07-09', '2016-09-07', '2016-10-12', '2016-11-02', '2016-11-15', '2016-12-25', '2017-01-01', '2017-01-25', '2017-02-27', '2017-02-28', '2017-03-01', '2017-04-14', '2017-04-21', '2017-05-01', '2017-06-15', '2017-07-09', '2017-09-07', '2017-10-12', '2017-11-02', '2017-11-15', '2017-11-20', '2017-12-25', '2018-01-01', '2018-01-25', '2018-02-25', '2018-02-12', '2018-02-13', '2018-02-14', '2018-03-30', '2018-04-21', '2018-05-01', '2018-05-31', '2018-07-09', '2018-09-07', '2018-10-12', '2018-11-02', '2018-11-15', '2018-11-20', '2018-12-25']
df = comb[~comb['datetime'].isin(feriados)]
combined = df.sort_index(ascending=True)
combined.head()
df_cum_ativos =  combined.groupby('datetime')['adj_close'].sum().reset_index()
df_cum_ativos.head()
# reconstruindo o indice da serie temporal de dados acumulados por data
keys = 'datetime'
df_cum_ativos.set_index(keys, drop=True, append=False, inplace=True)
# outliers:
out_drop = ['2013-04-26', '2013-07-26', '2013-12-24', '2013-12-31', '2014-06-12', '2014-12-24', '2014-12-31', \
            '2015-12-24', '2015-12-31', '2016-12-30']
df = df_cum_ativos.copy()
df.drop(pd.to_datetime(out_drop), inplace=True)
df_cum_ativos = df.copy()
# inspecionando o índice datetime reconstruído
df_cum_ativos.head()
# resumo dos 5 últimos registros da série
df_cum_ativos.tail()
# Vamos isolar agora apenas os dados que entrarão no modelo (colunas 'adj_close' e 'datetime' que é o índice)
cols = ['adj_close']
dfcot = df_cum_ativos[cols].copy() # faz uma cópia 
port = dfcot.copy() 
## Substitui preços Zero pela média
df = port.copy()
mask = (df['adj_close'] != 0)
nomask = (df['adj_close'] == 0)
Numeric_columns = ['adj_close']
means = df.loc[mask, Numeric_columns].mean()
df.loc[nomask, Numeric_columns] = means
port = df.copy()
# Tracando uma primeira visualizacao do portfolio
np.warnings.filterwarnings("ignore")

# Fig prepare
fig = plt.figure(figsize=(20, 8))
plt.ylim(200, 550)

plt.rc('axes', axisbelow=True)
ax = plt.subplot(111)

#ax.yaxis.grid(color='lightgray', linestyle='dashed')
ax.yaxis.grid(color='lightgray')
plt.title("Portfolio de Preços Diários Ajustados")
#plt.axis(['2013-01-01','2018-12-31', 0.0, 1.0], bin=100)

plt.plot(port, 'k-', alpha=0.5);
# Tracando a MM de 21 períodos (30 dias corridos)
fig = plt.figure(figsize=(20, 8))
r = port.rolling(window = 21)
plt.ylim(200, 550)
plt.plot(port, '-k', alpha = 0.50)
plt.title("Portfolio e sua Media Movel de 21 dias - MM21 - em vermelho")
plt.plot(r.mean(), '-r');
plt.show();
# Ajustando a estacionariedade da série com log
port_ret = np.log10(port).diff().dropna()
# Primera plotagem após ajuste por log
# plt.plot(port_ret, '-k', alpha=0.5);

# Fig prepare
fig = plt.figure(figsize=(20, 6))
axis = fig.add_subplot(111)
plt.ylim(-0.1, 0.1)

plt.rc('axes', axisbelow=True)
ax = plt.subplot(111)
ylabel = 'Valores Log'
xlabel = 'Últimos 5 anos (2013 a 2018)'
plt.xlabel(xlabel, fontdict=None, labelpad=None)
plt.ylabel(ylabel, fontdict=None, labelpad=None)

#ax.yaxis.grid(color='lightgray', linestyle='dashed')
ax.yaxis.grid(color='lightgray')
plt.title("Retornos do Portfolio após amortecimento por Log")
#plt.axis(['2013-01-01','2018-12-31', 0.0, 1.0], bin=100)

plt.plot(port_ret, '-k', alpha=0.50);
port1 = np.cumprod(np.r_[port.values[0], np.power(10, port_ret.values.reshape(len(port_ret)))]) # produto acumulado e exp "element wize"

# Fig prepare
fig = plt.figure(figsize=(20, 8))
axis = fig.add_subplot(111)
plt.plot(port1, alpha=0.75);
# Fig prepare
rcParams['figure.figsize'] = 20, 8
g = port_ret.hist(bins=21, alpha=0.75);
rcParams['figure.figsize'] = 20, 8
r = port_ret.rolling(window = 21)
plt.plot(port_ret, '-k', alpha = 0.50)
plt.plot(r.std(), '-r');
import statsmodels.tsa as tsa
import statsmodels.api as sm
from pandas import tseries as ts

np.warnings.filterwarnings("ignore")

model=sm.tsa.ARMA(port_ret.as_matrix(), (1,0)).fit(trend = 'c')

#model = sm.tsa.ARMA(port_ret.as_matrix(), (1,0)).fit(trend = 'c')
print(model.summary());
m = model.params
print(m);
port_ar_proc = tsa.arima_process.ArmaProcess(np.r_[1, -model.params], [1])
port_ar_sim = port_ar_proc.generate_sample(len(port_ret))*model.resid.std()
# Fig prepare
fig = plt.figure(figsize=(20, 8))
plt.rc('axes', axisbelow=True)
ax = plt.subplot(211)

#ax.yaxis.grid(color='lightgray', linestyle='dashed')
ax.yaxis.grid(color='lightgray')
plt.title("Retorno Real")
plt.ylim(-0.1, 0.1)
# plt.axis([1, 1000, -0.1, 0.1])
plt.plot(port_ret, 'k', alpha = 0.6)

ax = plt.subplot(212)
plt.title("Retorno Simulado")
plt.ylim(-0.1, 0.1)
# plt.axis([1, 1000, -0.1, 0.1])

#ax.yaxis.grid(color='lightgray', linestyle='dashed')
ax.yaxis.grid(color='lightgray')
plt.plot(port_ar_sim)

plt.show();
port1 = np.cumprod(np.r_[port.values[1], np.power(10, port_ar_sim)]) # produto acumulado e 10^p "element wize"
# Fig prepare
fig = plt.figure(figsize=(20, 8))
plt.rc('axes', axisbelow=True)
ax = plt.subplot(211)

#ax.yaxis.grid(color='lightgray', linestyle='dashed')
ax.yaxis.grid(color='lightgray')
plt.title("Série Original")
plt.ylim(100, 1000)
plt.plot(port, 'k', alpha = 0.75)

ax = plt.subplot(212)
plt.ylim(100, 1000)
#plt.xlim(150, 1250)
#plt.axis([150, 1200, 100, 700])
plt.title("Série Simulada a partir do Modelo")
#ax.yaxis.grid(color='lightgray', linestyle='dashed')
ax.yaxis.grid(color='lightgray')
plt.plot(port1)

plt.show();
# Fig prepare
fig = plt.figure(figsize=(20, 8))
plt.rc('axes', axisbelow=True)
ax = plt.subplot(211)

#ax.yaxis.grid(color='lightgray', linestyle='dashed')
ax.yaxis.grid(color='lightgray')
plt.title("Retornos (ou resíduos) Reais")
plt.ylim(-0.1, 0.1)
# plt.axis([1, 1000, -0.1, 0.1])
plt.plot(port_ret, 'k', alpha = 0.6)

ax = plt.subplot(212)
plt.title("Residuos simulados a partir do modelo ARMA")
plt.ylim(-0.1, 0.1)
# plt.axis([1, 1000, -0.1, 0.1])

#ax.yaxis.grid(color='lightgray', linestyle='dashed')
ax.yaxis.grid(color='lightgray')
plt.plot(model.resid)

plt.show();
from arch import arch_model
# O modelo envolve elevar ao quadrado valores muito pequenos. Multiplicaremos os valores da serie de retornos que será
# usada por 100, para diminuirmos a chance de divergência do modelo.

am = arch_model(port_ret*100, p=1, q=1)
res = am.fit(disp='off')
print(res.summary())
# Fig prepare
fig = plt.figure(figsize=(20, 8))
plt.rc('axes', axisbelow=True)
ax = plt.subplot(211)

#ax.yaxis.grid(color='lightgray', linestyle='dashed')
ax.yaxis.grid(color='lightgray')
plt.title("Resíduos Reais")
plt.ylim(-0.1, 0.1)
# plt.axis([0, 1000, -0.1, 0.1])
plt.plot(port_ret, 'k', alpha = 0.75)

ax = plt.subplot(212)
plt.title("Resíduos Simulados via Modelo")
plt.ylim(-0.1, 0.1)
# plt.axis([0, 1000, -0.1, 0.1])
#ax.yaxis.grid(color='lightgray', linestyle='dashed')
ax.yaxis.grid(color='lightgray')
plt.plot((res.resid/(res.conditional_volatility*100)))

plt.show();
# Função para simulação 
def simulate_garch(params, nobs, var=0):
    e = np.random.randn(nobs)
    var = np.zeros(len(e)) + var
    x = np.zeros(len(e))
    corr = 0.10200
    for t in range(len(e)):
        var[t] = params[1] + params[2] * x[t-1] ** 2 + params[3] * var[t-1]
        x[t] = params[0] + e[t] * np.sqrt(var[t])
        
    return x+corr
# Visualizando os parâmetros gerados pelo modelo
res.params
port_am_sim = simulate_garch(res.params, len(port_ret))
# Fig prepare
fig = plt.figure(figsize=(15, 8))
plt.rc('axes', axisbelow=True)
ax = plt.subplot(211)

#ax.yaxis.grid(color='lightgray', linestyle='dashed')
ax.yaxis.grid(color='lightgray')
plt.ylim(-0.1, 0.1)
plt.title("Retornos do Portfolio")
# plt.axis([0, 1000, -0.1, 0.1])
plt.plot(port_ret, 'k', alpha = 0.5)

ax = plt.subplot(212)
# plt.axis([0, 1000, -0.1, 0.1])
plt.ylim(-0.1, 0.1)
plt.title("Retornos Simulados do Portfolio - Modelo Garch")
#ax.yaxis.grid(color='lightgray', linestyle='dashed')
ax.yaxis.grid(color='lightgray')
plt.plot(port_am_sim/100)

plt.show();
# Valores foram restaurados ao original, redividindo-os por 100, antes de simulá-los
port1 = np.cumprod(np.r_[port.values[1], np.exp(port_am_sim/100)]) # produto acumulado e exp "element wize"
# Fig prepare
fig = plt.figure(figsize=(20, 8))
plt.rc('axes', axisbelow=True)
ax = plt.subplot(211)

#ax.yaxis.grid(color='lightgray', linestyle='dashed')
ax.yaxis.grid(color='lightgray')
plt.title("Valores Originais do Portfolio")
plt.ylim(200, 750)
# plt.axis([0, 1000, 2, 5])
plt.plot(port, 'k', alpha = 0.5)

ax = plt.subplot(212)
plt.title("Valores Simulados do Portfolio")
plt.ylim(200, 750)
ax.yaxis.grid(color='lightgray')
plt.plot(port1)

plt.show();
# Fig prepare
fig = plt.figure(figsize=(15, 8))
plt.rc('axes', axisbelow=True)
ax = plt.subplot(211)

#ax.yaxis.grid(color='lightgray', linestyle='dashed')
ax.yaxis.grid(color='lightgray')
plt.title("Resíduos Reais do Modelo")
plt.ylim(-0.1, 0.1)
# plt.axis([0, 1000, -0.1, 0.1])
plt.plot(port_ret, 'k', alpha = 0.5)

ax = plt.subplot(212)
plt.title("Resíduos Simulados via Modelo")
plt.ylim(-0.1, 0.1)
# plt.axis([0, 1000, -0.1, 0.1])
#ax.yaxis.grid(color='lightgray', linestyle='dashed')
ax.yaxis.grid(color='lightgray')
plt.plot((res.resid/res.conditional_volatility)/100)

plt.show();
# preparando a serie para tomarmos seus últimos 252 dias (um ano útil)
# e preparando para traçar o gráfico das predições
port2 = port.copy()
port2 = port2.reset_index()
port2.head()
# eliminando a data do índice e deixando apenas a série de valores ajustados do portfolio
port2.drop('datetime', axis=1, inplace=True)
port2.head()
len(port2)
# Preparacao da figura
fig = plt.figure(figsize=(20, 8))
plt.rc('axes', axisbelow=True)
ax = plt.subplot(111)
ax.yaxis.grid(color='lightgray')

plt.title("Estimativas para o valor do Portfolio")
plt.xlabel('Ultimos 252 dias úteis em azul e a previsão dos próximos 21 dias em verde')

plt.ylim(350, 550)
plt.axis([1160, 1435, 350, 550])

plt.plot(port2.tail(252))

paths = np.zeros((100,21))
lim = len(port2)

for i in range(100):
    port_am_sim = simulate_garch(res.params, 20)/100              ## restaurando os valores simulados dividindo por 100 
    path = np.cumprod(np.r_[port.values[-1], np.exp(port_am_sim)])
    paths[i,:] = path
    plt.plot(range(lim, lim+21), path, 'g', alpha = 0.10)

plt.plot(range(lim, lim+21), np.mean(paths, 0), '-r');

'''

## Buscando ativos no Alpha_vantage Service e gravando localmente numa pasta
#
# Parametros Default da Interface
#
# api_key: str     = None
# output_size: str = 'compact'
# datatype: str    = 'json'
# export: bool     = False
# export_path: str = '~/av_data'
# output: str      = 'csv'
# clean: bool      = False
# proxy: dict      = {}

import pandas as pd
import time
from alphaVantageAPI.alphavantage import AlphaVantage

#
# A classe AlphaVantage possui os seguintes requisitos:
#
# (1) Cadastro previo no site http://www.alphavantage.co/support/ # A api-key, aqui referida, deve se obtida
#
# (2) ATENCAO: para instalar a biblioteca alphaVantage aqui utilizada, NÃO SIGA O PROCEDIMENTO pip INDICADO NO SITE.
#     SIGA O PROCEDIMENTO ABAIXO:
#          
#     pip install alphaVantage-api
# 
# (3) O objetivo deste notebook é o de, apenas, compartilhar conhecimentos a respeito das habilidades de análise
#     das Séries Temporais, tema extremamente carente de exemplos práticos de implementação, pelo menos, em 
#     língua Portuguesa. A análise de séries temporais é reconhecidamente uma habilidade pouco desenvolvida e 
#     carente de profissionais de Data Science que as saibam modelar e interpretar. 
#
#
# (4) Valores default

api_key = 'SUA_API_KEY'
av = AlphaVantage(
        api_key=api_key,
        output_size='full',
        datatype='pandas',
        export=True,
        export_path='alpha_data\\',
        output='csv',
        clean=True,
        proxy={}
    )

# Aqui serão usados preços ajustados, obtidos com periodicidade diaria (Daily Adjusted). Outras periodicidades estão
# disponíveis via a mesma biblioteca. 
# Não há garantia, entretanto, quanto à qualidade dos dados fornecidos.

#
# portfolio =  ['ABEV3.SA', 'EZTC3.SA', 'GRND3.SA', 'HGTX3.SA', '1PETR3.SA', 'EGIE3.SA',  \
#              'VALE3.SA', 'CIEL3.SA', '1ITUB3.SA', 'ITSA3.SA', 'RADL3.SA', 'PSSA3.SA',  \
#              'WEGE3.SA', 'LREN3.SA', 'MDIA3.SA']
#

#
# O portfolio aqui estudade consiste numa carteira de ações ON de empresas brasileiras, com bons fundamentos.
# Mas poderia ser qualquer outro portfolio de ativos, que possuísse, pelo meno 50 pontos observáveis no tempo.
#
# Ao grupo de ações selecionadas foram incluídas VALE e PETROBRAS, pela magnitude do volume burstátil 
# historicamente movimentado por elas.
#
# Não há, neste estudo, qualquer sugestão ou recomendação de aquisição ou alienação de qualquer dos ativos 
# mencionados.
#
# Para a AlphaVantage PETR3 = PBR e ITUB3 = ITUB
#

carteira =  ['ABEV3.SA', 'EZTC3.SA', 'GRND3.SA', 'HGTX3.SA', 'PBR', 'EGIE3.SA',  \
             'VALE3.SA', 'CIEL3.SA', 'ITUB', 'ITSA3.SA', 'RADL3.SA', 'PSSA3.SA',  \
             'WEGE3.SA', 'LREN3.SA', 'MDIA3.SA']

for nome in carteira:
    pd = av.data(symbol=nome, function='DA') 
    print("Proc.: ", nome)
    time.sleep(7) 
print()
print('OK!')

''';