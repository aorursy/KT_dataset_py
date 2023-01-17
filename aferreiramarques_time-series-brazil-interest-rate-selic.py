import quandl

import warnings

import itertools

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



sns.set_style('whitegrid')

# definição da estética da grade

sns.set_context('talk')

# definindo-se o tamanho da fonte



from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.graphics.tsaplots import plot_pacf

from statsmodels.graphics.tsaplots import plot_acf

# acf = autocorrelation function   pacf = partial autocorrelation function
import matplotlib.pyplot as plt

params = {'legend.fontsize': 'x-large',

          'figure.figsize': (15, 5),

          'axes.labelsize': 'x-large',

          'axes.titlesize':'x-large',

          'xtick.labelsize':'x-large',

          'ytick.labelsize':'x-large'}



%matplotlib inline

plt.rcParams.update(params)



# specify to ignore warning messages

warnings.filterwarnings("ignore")
import pandas as pd

data = pd.DataFrame(pd.read_csv(r'../input/selic1.csv', sep=';', encoding = 'latin', low_memory=False,

                               parse_dates=True,infer_datetime_format=True,index_col=[0], decimal=','))

# BCB https://www.bcb.gov.br/controleinflacao/taxaselic
data.head()
data.info()

# somente objetos, não há dados numéricos
data.tail()
data.index
ts = data['y'] 

ts.head(10)

# provocou a multiplicação por 100
data.info()
data = data.reindex(pd.date_range(data.index.min(), 

                                  data.index.max(), 

                                  freq='M')).fillna(method='ffill')

# a Selic possui valor fixo para um mês inteiro, não há sentido em se ter valor diário.

data.head()
# inclui linhas no dataset para criar automaticamente as datas que serão utilizadas na projeção. Não é utilizado fillna() propositalmente, pois para os gráficos comparativos, a série com dados reais

# deverá ser mantida a mesma e estar vazia nos períodos futuros

data = data.reindex(pd.date_range(data.index.min(), 

                                  '2020-06-30', 

                                  freq='M'))

data.tail()
data.plot(figsize=(20, 6))

plt.show()

# grandes variações nos últimos anos. 

# depois de 2018, um patamar muito inferior & constante em relação aos demais

# avaliar profile para seleção dos períodos.

# há pontos tocando o eixo horizontal, com valor próximo de 0, que não faz sentido.
data.count(0)

# há 120 valores de selic = 0, o que não faz sentido.
from statsmodels.tsa.seasonal import seasonal_decompose
# A função 'seasonal_decompose', se não for passado um parametro de frequênci sazonal, ele assume que existe uma sazonalidade (com base na frequencia dos dados) e "força" os dados para isso

#decompose = seasonal_decompose(data[:"2019-11-30"], model = 'additive')

#decompose.plot();

fig, ax = plt.subplots(nrows=3,ncols=3, figsize=(30,15))

decompose_1 = seasonal_decompose(data["2012-01-01":"2019-11-30"], model = 'additive')

decompose_2 = seasonal_decompose(data["2012-01-01":"2019-11-30"], model = 'additive', freq=18)

decompose_3 = seasonal_decompose(data["2012-01-01":"2019-11-30"], model = 'additive', freq=1)



decompose_1.trend.plot(ax=ax[0,0], legend=False)

decompose_2.trend.plot(ax=ax[1,0], legend=False)

decompose_3.trend.plot(ax=ax[2,0], legend=False)



decompose_1.seasonal.plot(ax=ax[0,1], legend=False)

decompose_2.seasonal.plot(ax=ax[1,1], legend=False)

decompose_3.seasonal.plot(ax=ax[2,1], legend=False)



decompose_1.resid.plot(ax=ax[0,2], legend=False)

decompose_2.resid.plot(ax=ax[1,2], legend=False)

decompose_3.resid.plot(ax=ax[2,2], legend=False)



#decompose_1.observed.plot(ax=ax[0,3], legend=False)

#decompose_2.observed.plot(ax=ax[1,3], legend=False)

#decompose_3.observed.plot(ax=ax[2,3], legend=False)



ax[0,0].set_title('Trend')

ax[0,1].set_title('Seasonal')

ax[0,2].set_title('Residual')



ax[0,0].set_ylabel('Original')

ax[1,0].set_ylabel('Freq @ 18')

ax[2,0].set_ylabel('Freq @ 1')

plt.show()
decompose = seasonal_decompose(data[:"2019-11-30"], model = 'multiplicative')

decompose.plot();
# Original Series

from statsmodels.tsa import stattools

from statsmodels.tsa import seasonal

adf_result = stattools.adfuller(data.y.dropna(),autolag='AIC')

print('p-valor do teste Dickey-Fuller',adf_result[1])

# alfa = 0,05

# alfa menor que p-value, então aceitar H0: série não estacionária.
#Aplicando a primeira diferença 



first_order_diff = data['y'].dropna().diff(1)
first_order_diff
# gráfico original x a primeira diferença 

fig, ax = plt.subplots(2, sharex=True)

data['y'].plot(ax=ax[0],color='b')

ax[0].set_title('original_selic')

first_order_diff.plot(ax=ax[1], color ='g')

ax[1].set_title("primeira diferença_selic ")
from statsmodels.graphics.tsaplots import plot_pacf

from statsmodels.graphics.tsaplots import plot_acf
plot_pacf(data.dropna(), lags = 30);
plot_acf(data.dropna(), lags = 30);
plot_acf(data.dropna(), lags = 90);
# Selecionado o período pós governo Dilma, acredita-se que é um período de comportamento econômico mais similar ao cenário atual.

train_selic_12 = data.loc["2012-01-01":"2019-07-01"]

train_selic_17 = data.loc["2017-01-01":"2019-07-01"]

test_selic = data.loc["2019-07-02":]

X = data.loc["2012-01-01":]

X_17 = data.loc["2017-01-01":]

# necessário revisar este conjunto de datas
!pip install pmdarima
from pmdarima.arima import auto_arima
from pmdarima.arima import auto_arima



# Obs: a função auto-arima já faz o fit do modelo. Ela simultamente calcula os melhores hiperparametros ARIMA com critérios de minização de erro para o fit.

# Ou seja, já é possível, dar um predict após a chamada do auto_arima. Portanto, é exatamente nesse ponto que devemos escolher qual série de dados de referência utilizar



# 1o modelo: não sazonal e não estacionário, com série completa

modelo_nao_saz_nao_estac = auto_arima(data.dropna(),

                                      stationary = False, # Se for True, os parâmetros de 'd' e 'D' devem ser zerados

                                      start_p=1, # AR_init: número inicial de lags do modelo auto-regressivo

                                      max_p=5, # AR_max: número máximo de lags do modelo auto-regressivo

                                      d=1, # I_init: ordem inicial da diferencial da porção auto-regressiva

                                      max_d=5, # I_max: ordem máxima da diferencial da porção auto-regressiva

                                      start_q=1, # MA_init: número inicial de médias móveis

                                      max_q=5, # MA_max: número inicial de médias móveis

                                      transparams = True, # se deve ou não transformar os parâmetros para garantir estacionariedade



                                      seasonal = False, # Se for False, os parâmetros de 'P', 'D' e 'Q' devem ser zerados

                                      #m=18, # período para a diferencial da sazaonalidade

                                      #start_P=1, # número inicial de ordem da porção auto-regressiva da sazonalidade

                                      #max_P=5, # número máximo de ordem da porção auto-regressiva da sazonalidade

                                      #D=1, # S: ordem inicial da 1a diferencial da sazonalidade

                                      #max_D=5, # S: ordem máxima da 1a diferencial da sazonalidade

                                      #start_Q=1, # número inicial de ordem da porção média-móvel da sazonalidade

                                      #max_Q=5, # número máximo de ordem da porção média-móvel da sazonalidade

                                      

                                      max_order = 20, # ordem máxima do modelo, agregando 'p' e 'q'

                                      #error_action='ignore', # se não for possível fitar devido a problemas de estacionariedade, avisar (warn) ou ignorar

                                      #trace=True, 

                                      out_of_sample_size = 12, # número de períodos no fim da série que não será utilizado para fitar, porém mantido para que as projeções iniciem a partir dele

                                      suppress_warning=True, stepwise=True)

NAs = len(data[data.selic.isna()])

# Nomenclatura das colunas - ns: não sazonal, ne: não estacionário.

predict = pd.DataFrame(modelo_nao_saz_nao_estac.predict(NAs), index = data.index[-NAs:], columns=['ns_ne_full'])

predict.loc[data.index[-NAs-1]] = float(data.iloc[data.shape[0]-NAs-1])

predict = predict.sort_index()

predict['round'] = round(predict.ns_ne_full*4)/4

predict
import pandas as pd

selic1 = pd.read_csv("../input/selic1.csv")