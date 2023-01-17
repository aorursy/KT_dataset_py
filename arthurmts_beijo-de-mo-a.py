import pandas as pd

import matplotlib.pyplot as plot

%matplotlib inline

import requests

path = "/kaggle/input/serenata_data/"
# Capturando os dados da API das tabelas do serenata

deputados_Nordeste_2019 = requests.get("https://dadosabertos.camara.leg.br/api/v2/deputados?siglaUf=MA&siglaUf=CE&siglaUf=PI&siglaUf=BA&siglaUf=SE&siglaUf=AL&siglaUf=PE&siglaUf=PB&siglaUf=RN&siglaUf=&ordem=ASC&ordenarPor=nome", headers={'Accept':'application/json'}).json()

deputados_Nordeste_2018 = requests.get('https://dadosabertos.camara.leg.br/api/v2/deputados?siglaUf=MA&siglaUf=CE&siglaUf=PI&siglaUf=BA&siglaUf=SE&siglaUf=AL&siglaUf=PE&siglaUf=PB&siglaUf=RN&siglaUf=&dataInicio=2018-01-01&dataFim=2018-12-31', headers= {'Accept':'application/json'}).json()

df_deputados_serenata_2018 = pd.read_csv(path+'Ano-2018.csv', sep=';')

df_deputados_serenata_2019 = pd.read_csv(path+'Ano-2019.csv', sep=';')
nomes_deputados_nordeste_2018 = [dep['nome'] for dep in deputados_Nordeste_2018['dados']]

nomes_deputados_nordeste_2019 = [dep['nome'] for dep in deputados_Nordeste_2019['dados']]
df_deputados_nordeste_2018 = df_deputados_serenata_2018[df_deputados_serenata_2018

                                                   ['txNomeParlamentar'].isin(nomes_deputados_nordeste_2018)]



df_deputados_nordeste_2019 = df_deputados_serenata_2019[df_deputados_serenata_2019

                                                   ['txNomeParlamentar'].isin(nomes_deputados_nordeste_2019)]
df_deputados_nordeste_2018['vlrLiquido'].sum()
df_deputados_nordeste_2019['vlrLiquido'].sum()
ceap_deputado_2018 = df_deputados_nordeste_2018.groupby(df_deputados_nordeste_2018['txNomeParlamentar']).sum()['vlrLiquido'].sort_values(ascending=False)
ceap_deputado_2018[:15].plot.bar()
ceap_deputado_2018[-15:].plot.bar()
ceap_deputado_2019 = df_deputados_nordeste_2019.groupby(df_deputados_nordeste_2019['txNomeParlamentar']).sum()['vlrLiquido'].sort_values(ascending=False)
ceap_deputado_2019[:15].plot.bar()
gastos_mensais_2018 = df_deputados_nordeste_2018.groupby(['numMes']).sum()['vlrLiquido']

gastos_mensais_2018
gastos_mensais_2018.plot()
gasto_partido_2018 = df_deputados_nordeste_2019.groupby('sgPartido').sum()['vlrLiquido'].bb(ascending=False)

gasto_partido_2018
nomes_deputados_nordeste_2018 = [dep['siglaPartido'] for dep in deputados_Nordeste_2018['dados']]

qnt_deputados_partido_2018 = pd.Series(nomes_deputados_nordeste_2018).value_counts()

qnt_deputados_partido_2018
pd.DataFrame([qnt_deputados_partido_2018, gasto_partido_2018])
df_deputados_nordeste_2019[['sgPartido','txNomeParlamentar']]
df_deputados_nordeste_2019.columns
import pandas as pd

Ano_2018 = pd.read_csv("../input/Ano-2018.csv")

Ano_2019 = pd.read_csv("../input/Ano-2019.csv")

federal_senate_2018 = pd.read_csv("../input/federal-senate-2018.csv")

federal_senate_2019 = pd.read_csv("../input/federal-senate-2019.csv")

reimbursements_2018 = pd.read_csv("../input/reimbursements-2018.csv")

reimbursements_2019 = pd.read_csv("../input/reimbursements-2019.csv")