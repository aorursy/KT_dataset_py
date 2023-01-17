# Este ambiente Python 3 vem com muitas bibliotecas de análise úteis instaladas
# É definido pela imagem do kaggle / python Docker: https://github.com/kaggle/docker-python
# Por exemplo, aqui estão vários pacotes úteis para carregar

import numpy as np # álgebra Linear
import pandas as pd # processamento de dados, E / S de arquivo CSV (por exemplo, pd.read_csv)

# Arquivos de dados de entrada estão disponíveis no diretório "../input/" somente leitura
# Por exemplo, executar isso (clicando em executar ou pressionando Shift + Enter) listará todos os arquivos no diretório de entrada

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Você pode gravar até 5 GB no diretório atual (/ kaggle / working /) que é preservado como saída ao criar uma versão usando "Salvar e executar tudo"
# Você também pode gravar arquivos temporários em / kaggle / temp /, mas eles não serão salvos fora da sessão atual

import numpy as np # importa biblioteca 
import pandas as pd # importa biblioteca 
import matplotlib.pyplot as plt # importa biblitecas
import seaborn as sns # importa biblitecas
from fbprophet import Prophet # importa biblitecas para previsão do futuro
from datetime import datetime # importa bibliotecas de calendários

plt.rcParams.update({'font.size': 12}) # define o padrão de fontesC

# Carrega dados
covid = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv', parse_dates = ['ObservationDate','Last Update'])


print (covid.shape)
print ('Last update: ' + str(data.ObservationDate.max()))
covid.tail(7) # imprime 7 linhas
covid.rename(columns={'ObservationDate':'Date','Country/Region':'Country'}, inplace=True)
dias = (900)
mortes = covid.groupby('Date').sum()['Deaths'].reset_index()
mortes.tail(dias)
mortes.columns = ['ds','y']
mortes.tail()
m = Prophet(interval_width=0.95)
m.fit(mortes)
futuro = m.make_future_dataframe(periods=dias)
futuro.tail(dias)
previsao = m.predict(futuro)
previsao.tail(dias)
previsao[['ds','yhat_lower','yhat','yhat_upper']].tail(dias)
confirmed_forecast_plot = m.plot(previsao)