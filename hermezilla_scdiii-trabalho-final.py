#Imports

import numpy as np
import pandas as pd
from dateutil.parser import parse 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
# Subindo o banco
base = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv', parse_dates=['Last Update'])
# Filtrando somente por Brasil
df = base[base['Country/Region'] == 'Brazil']
#Criando o campo contaminados
df['Contamined'] = df['Confirmed'] - df['Deaths'] - df['Recovered']
# Observando os dados
df
#Consertando a coluna
df[df['Last Update'] == '2020-03-08 05:31:00']
#Consertando as linhas
df.loc[7637,'Last Update'] = datetime.datetime(2020, 3, 22)
df.loc[9443,'Last Update'] = datetime.datetime(2020, 3, 28)
df.loc[9754,'Last Update'] = datetime.datetime(2020, 3, 29)
df.loc[10066,'Last Update'] = datetime.datetime(2020, 3, 30)
#Reindexando o dataframe
df.reset_index(drop=True, inplace=True)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.arima_model import ARIMA

# Criando o modelo
model = ARIMA(df.Contamined, order=(1,1,2))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# Plotando os erros
plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':120})

residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Resíduos", ax=ax[0])
residuals.plot(kind='kde', title='Densidade', ax=ax[1])
plt.show()
# Realidade vs Modelo
model_fit.plot_predict(dynamic=False)
plt.show()
!pip install pmdarima
#Achando a melhor configuração do modelo com stepwise
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm

model = pm.auto_arima(df.Contamined, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model.summary())
#Prevendo os próximos dias

# Até o fim do mês
n_periods = 11
fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = np.arange(len(df.Contamined), len(df.Contamined)+n_periods)

# Series para gerar o gráfico
fc_series = pd.Series(fc, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Gráfico
plt.plot(df.Contamined)
plt.plot(fc_series, color='darkgreen')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)

plt.title("Previsão de contaminados para o resto de Abril/2020")
plt.show()