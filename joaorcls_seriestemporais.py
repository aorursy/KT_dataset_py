import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import time

from datetime import datetime, date, timedelta

from statsmodels.graphics.tsaplots import plot_acf



pd.options.display.max_rows = 999

pd.options.display.max_columns = 999



pd.plotting.register_matplotlib_converters()



%matplotlib notebook



figsize=(10,6)
serie = pd.DataFrame(columns=['t','tendencia','sazonalidade','aleatorio','completa'])

serie['t'] = np.linspace(0, 8*np.pi, 8*180)



serie.head()
serie['tendencia'] = serie.t * 2

serie.head()
serie.plot(figsize=figsize,

                        grid=True, 

    linewidth=0.5,

    use_index=False,

          x='t',

          y='tendencia')

plt.title('Tendência')

plt.show()
serie['sazonalidade'] = 10*np.sin(serie.t )

serie.head()
serie.plot(figsize=figsize,

                        grid=True, 

    linewidth=0.5,

    use_index=False,

          x='t',

          y='sazonalidade')

plt.title('Sazonalidade')

plt.show()
serie['aleatorio'] = 5*np.random.normal(0, 1, len(serie))

serie.head()
serie.plot(figsize=figsize,

                        grid=True, 

    linewidth=0.5,

    use_index=False,

          x='t',

          y='aleatorio')

plt.title('Aleatório')

plt.show()
serie['completa'] = serie.tendencia + serie.sazonalidade + serie.aleatorio

serie.head()
serie.plot(figsize=figsize,

                        grid=True, 

    linewidth=0.5,

    use_index=False,

          x='t',

          y='completa')

plt.title('Completa')

plt.show()
serie['Diff'] = serie.completa.diff()



serie.loc[:,['completa','Diff']].plot(figsize=figsize,

                        grid=True, 

    linewidth=0.5,

    use_index=False)

plt.title('Diff')

plt.show()
serie['MediaLonga'] = serie.completa.rolling(400, 

                                                                 min_periods=1,

                                                                center =True).mean()

serie['Completa_MediaLonga'] = serie.completa - serie.MediaLonga

serie.loc[:,['completa','MediaLonga','Completa_MediaLonga']].plot(figsize=figsize,

                        grid=True, 

    linewidth=0.5,

    use_index=False)

plt.title('Média móvel')

plt.show()
serie['MediaCurta'] = serie.completa.rolling(50, 

                                                                 min_periods=1,

                                                                center =True).mean()

serie['Completa_MediaCurta'] = serie.completa - serie.MediaCurta

serie.loc[:,['completa','MediaCurta','Completa_MediaCurta']].plot(figsize=figsize,

                        grid=True, 

    linewidth=0.5,

    use_index=False)

plt.title('Média móvel')

plt.show()
fig, ax = plt.subplots(1,1,sharex=False, sharey=False,figsize=figsize)



serie.Diff.plot(kind='hist',

                figsize=figsize,

                        grid=True, 

    linewidth=0.5,

               ax=ax,

               bins=100)

plt.title('Aleatório')

plt.show()
fig, ax = plt.subplots(1,1,sharex=False, sharey=False,figsize=figsize)



serie.completa.plot(kind='hist',

                figsize=figsize,

                        grid=True, 

    linewidth=0.5,

               ax=ax,

               bins=100)

plt.title('Série completa')

plt.show()
serie['Media'] = serie.MediaCurta-serie.MediaLonga
serie.loc[:,['Media']].plot(figsize=figsize,

                        grid=True, 

    linewidth=0.5,

    use_index=False)

plt.title('Média')

plt.show()




fig, ax = plt.subplots(1,1,sharex=False, sharey=False,figsize=figsize)

plot_acf(serie.Media, 

         lags=450, ax=ax)
serie.head()
serie['Variancia_Nao_Ajustada'] = serie.completa.rolling(50, 

                                                                 min_periods=1,

                                                                center =True).var()

serie.loc[:,['completa','Variancia_Nao_Ajustada']].plot(figsize=figsize,

                        grid=True, 

    linewidth=0.5,

    use_index=False,

                                                       secondary_y='Variancia_Nao_Ajustada')

plt.title('Variancia_Nao_Ajustada')

plt.show()
tamanho_previsao = len(serie)+int(0.2*len(serie))
serie.head()
serie.plot(figsize=figsize,

                        grid=True, 

    linewidth=0.5,

    use_index=False,

          x='t',

          y='completa')

plt.title('Completa')

plt.show()
previsao = serie.loc[:,['completa']].copy()

base = datetime.today()

date_list = [base - timedelta(days=x) for x in range(len(previsao),0,-1)]

previsao['ds'] = date_list

previsao.columns=['y','ds']

previsao.head()
previsao.tail()
from fbprophet import Prophet
m = Prophet()

m.fit(previsao)
future = m.make_future_dataframe(periods=365)

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)
from fbprophet.plot import plot_plotly

import plotly.offline as py

py.init_notebook_mode()



fig = plot_plotly(m, forecast)  # This returns a plotly Figure

py.iplot(fig)
import os

print(os.listdir('../input'))
Gastos2015 = pd.read_excel('../input/Gastos.xlsx', sheet_name='2015')

#Gastos2015.head()
Gastos2016 = pd.read_excel('../input/Gastos.xlsx', sheet_name='2016')

#Gastos2016.head()
Gastos2017 = pd.read_excel('../input/Gastos.xlsx', sheet_name='2017')

#Gastos2017.head()
Gastos2018 = pd.read_excel('../input/Gastos.xlsx', sheet_name='2018')

#Gastos2018.head()
Gastos2019 = pd.read_excel('../input/Gastos.xlsx', sheet_name='2019')

#Gastos2019.head()




Gastos = pd.concat([Gastos2015,Gastos2016,Gastos2017,Gastos2018,Gastos2019])
Gastos = Gastos.loc[(Gastos.DATA>='2015-01-01')&(Gastos.DATA<='2019-08-31'),:]

Gastos.head()
Gastos['DiaSemana']=0

Gastos['DiaSemana']=Gastos['DATA'].dt.strftime( '%w').astype(copy=False, dtype ='int')

Gastos['DiaMes']=0

Gastos['DiaMes']=Gastos['DATA'].dt.strftime( '%d').astype(copy=False, dtype ='int')

#Gastos.head()
TiposGastos = Gastos.loc[:,['TIPO_DESPESA','VALOR_REEMBOLSADO']].groupby(['TIPO_DESPESA']).sum().index.values

#TiposGastos
Gastos.loc[:,['TIPO_DESPESA','VALOR_REEMBOLSADO']].groupby(['TIPO_DESPESA']).sum()
Gastos.loc[:,['DiaSemana','TIPO_DESPESA','VALOR_REEMBOLSADO']].groupby(['DiaSemana','TIPO_DESPESA']).mean()
Gastos.loc[:,['DiaSemana','VALOR_REEMBOLSADO']].groupby(['DiaSemana']).mean().plot.bar(figsize=figsize,

                        grid=True, title='Gasto médio total por dia da semana')
for tipoGasto in TiposGastos:

    Gastos.loc[Gastos.TIPO_DESPESA==tipoGasto,['DiaSemana','VALOR_REEMBOLSADO']].groupby(['DiaSemana']).mean().plot.bar(figsize=figsize,

                        grid=True, title=tipoGasto)
Gastos.loc[:,['DiaMes','VALOR_REEMBOLSADO']].groupby(['DiaMes']).mean().plot.bar(figsize=figsize,

                        grid=True)

for tipoGasto in TiposGastos:

    Gastos.loc[Gastos.TIPO_DESPESA==tipoGasto,['DiaMes','VALOR_REEMBOLSADO']].groupby(['DiaMes']).mean().plot.bar(figsize=figsize,

                        grid=True, title=tipoGasto)
#inicio = datetime(year = 2017, month = 1, day = 1, hour = 0, minute = 0, second = 0)

pd.plotting.register_matplotlib_converters()



GastosConsolidados = Gastos.loc[:,['DATA','VALOR_REEMBOLSADO']].groupby(['DATA']).sum()

GastosConsolidados['MEDIA'] = GastosConsolidados['VALOR_REEMBOLSADO'].rolling(30, 

                                                                 min_periods=1,

                                                                center =True).mean()

GastosConsolidados['DESVIO'] = GastosConsolidados['VALOR_REEMBOLSADO'].rolling(30, 

                                                                 min_periods=1,

                                                                center =True).std()
GastosConsolidados.plot(figsize=figsize,

                        grid=True, 

    linewidth=0.5)

plt.show()


fig, ax = plt.subplots(1,1,sharex=False, sharey=False,figsize=figsize)

plot_acf(GastosConsolidados.VALOR_REEMBOLSADO, 

         lags=370, ax=ax)

GastosConsolidadoTipoDespesa = Gastos.loc[Gastos.TIPO_DESPESA=='Passagens aéreas, aquáticas e terrestres nacionais',

                                          ['DATA','VALOR_REEMBOLSADO']].groupby(['DATA']).sum()



GastosConsolidadoTipoDespesa['MEDIA'] = GastosConsolidadoTipoDespesa['VALOR_REEMBOLSADO'].rolling(30, 

                                                                 min_periods=1,

                                                                center =True).mean()
GastosConsolidadoTipoDespesa.plot(figsize=figsize,

                        grid=True, 

    linewidth=0.5,

    use_index=True,

                                 title='Passagens aéreas, aquáticas e terrestres nacionais')

plt.show()