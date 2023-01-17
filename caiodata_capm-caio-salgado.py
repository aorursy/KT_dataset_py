# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# importando base de dados
dates =['DATPRE']
df = pd.read_csv('../input/COTAHIST_A2009_to_A2018P.csv')
# ajustando dados e selecionando acoes para analise
df['DATPRE'] = pd.to_datetime(df['DATPRE'], errors='coerce')
Datas = df[df.CODNEG == 'PETR4']
Datas = df['DATPRE']
df.set_index('DATPRE', inplace=True)
Petrobras4 = df[df.CODNEG == 'PETR4']
ItauUnibanco = df[df.CODNEG == 'ITUB4']
Vale = df[df.CODNEG == 'VALE3']
Bradesco = df[df.CODNEG == 'BBDC4']
# Ambev = df[df.CODNEG == 'ABEV3']
BancodoBrasil = df[df.CODNEG == 'BBAS3']
BMFBovespa = df[df.CODNEG == 'BVMF3']
Cielo = df[df.CODNEG == 'CIEL3']
Itausa = df[df.CODNEG == 'ITSA4']
# Kroton = df[df.CODNEG == 'KROT3']
BRF = df[df.CODNEG == 'BRFS3']
Petrobras3 = df[df.CODNEG == 'PETR3']
Gerdau = df[df.CODNEG == 'GGBR4']
# BBSeguridade = df[df.CODNEG == 'BBSE3']
Usiminas = df[df.CODNEG == 'USIM5']
JBS = df[df.CODNEG == 'JBSS3']
CCR = df[df.CODNEG == 'CCRO3']
LojasRenner = df[df.CODNEG == 'LREN3']
LojasAmericanas = df[df.CODNEG == 'LAME4']
# Magaluiza = df[df.CODNEG == 'MGLU3']
Gol = df[df.CODNEG == 'GOLL4']
# CVC = df[df.CODNEG == 'CVCB3']
Localiza = df[df.CODNEG == 'RENT3']
Estacio = df[df.CODNEG == 'ESTC3']
# RAIADROGASILON = df[df.CODNEG == 'RADL3']
# QUALICORP = df[df.CODNEG == 'QUAL3']
MULTIPLAN = df[df.CODNEG == 'MULT3']
EMBRAER = df[df.CODNEG == 'EMBR3']
SABESP = df[df.CODNEG == 'SBSP3']
BRASKEM = df[df.CODNEG == 'BRKM5']
MRV = df[df.CODNEG == 'MRVE3']
MARFRIG = df[df.CODNEG == 'MRFG3']
NATURA = df[df.CODNEG == 'NATU3']

Petrobras4 = Petrobras4['PREULT']
ItauUnibanco = ItauUnibanco['PREULT']
Vale = Vale['PREULT']
Bradesco = Bradesco['PREULT']
# Ambev = Ambev['PREULT']
BancodoBrasil = BancodoBrasil['PREULT']
BMFBovespa = BMFBovespa['PREULT']
Cielo = Cielo['PREULT']
Itausa = Itausa['PREULT']
# Kroton = Kroton['PREULT']
BRF = BRF['PREULT']
Petrobras3 = Petrobras3['PREULT']
Gerdau = Gerdau['PREULT']
# BBSeguridade = BBSeguridade['PREULT']
Usiminas = Usiminas['PREULT']
JBS = JBS['PREULT']
CCR = CCR['PREULT'] 
LojasRenner = LojasRenner['PREULT']
LojasAmericanas = LojasAmericanas['PREULT']
# Magaluiza = Magaluiza['PREULT']
Gol = Gol['PREULT']
# CVC = CVC['PREULT']
Localiza = Localiza['PREULT']
Estacio = Estacio['PREULT']
# RAIADROGASILON = RAIADROGASILON['PREULT']
# QUALICORP = QUALICORP['PREULT']
MULTIPLAN = MULTIPLAN['PREULT']
EMBRAER = EMBRAER['PREULT']
SABESP = SABESP['PREULT']
BRASKEM = BRASKEM['PREULT']
MRV = MRV['PREULT']
MARFRIG = MARFRIG['PREULT']
NATURA = NATURA['PREULT']

df_brasil = pd.DataFrame(Petrobras4, index=Petrobras4.index)
df_brasil.columns = ['PETR4']
df_brasil['ITUB4'] = ItauUnibanco
df_brasil['VALE3'] = Vale
df_brasil['BBDC4'] = Bradesco
# df_brasil['ABEV3'] = Ambev
df_brasil['BBAS3'] = BancodoBrasil
df_brasil['BVMF3'] = BMFBovespa
df_brasil['CIEL3'] = Cielo
df_brasil['ITSA4'] = Itausa
# df_brasil['KROT3'] = Kroton
df_brasil['BRFS3'] = BRF
df_brasil['PETR3'] = Petrobras3
df_brasil['GGBR4'] = Gerdau
# df_brasil['BBSE3'] = BBSeguridade
df_brasil['USIM5'] = Usiminas
df_brasil['JBSS3'] = JBS
df_brasil['CCRO3'] = CCR
df_brasil['LREN3'] = LojasRenner
df_brasil['LAME4'] = LojasAmericanas
# df_brasil['MGLU3'] = Magaluiza
df_brasil['GOLL4'] = Gol
# df_brasil['CVCB3'] = CVC
df_brasil['RENT3'] = Localiza
df_brasil['ESTC3'] = Estacio
# df_brasil['RADL3'] = RAIADROGASILON
# df_brasil['QUAL3'] = QUALICORP
df_brasil['MULT3'] = MULTIPLAN
df_brasil['EMBR3'] = EMBRAER
df_brasil['SBSP3'] = SABESP
df_brasil['BRKM5'] = BRASKEM
df_brasil['MRVE3'] = MRV
df_brasil['MRFG3'] = MARFRIG
df_brasil['NATU3'] = NATURA

df_brasil.tail()
df_brasil['PETR4'].head()
# site para taxas SELIC
# https://www.bcb.gov.br/pt-br/#!/c/COPOMJUROS/
# organizar dados afim de organizar por data
df_brasil = df_brasil.sort_values(by=['DATPRE'])
df_brasil.head()
selic = 0.064
# calcular media de dias uteis por mes durante o periodo de jan/2009 ate dez/2017
# fonte https://www.dias-uteis.com/#a22
media_dias_uteis_mes = 2261/9/12
round(media_dias_uteis_mes)

# iremos usar o retorno mensal para que haja uma melhor distribuicao normal dos resultados favorecendo o modelo
# pegar o ultimo valor da acao no mes
df_brasil['PETR4'].resample('M').last().head()
# pegar o ultimo valor do 'mercado' no mes
df_brasil['BVMF3'].resample('M').last().head()
data = pd.DataFrame({'Petro4' : df_brasil['PETR4'].resample('M').last(), 'Ibovespa' : df_brasil['BVMF3'].resample('M').last()})
data
    
    
data[['ret Petro', 'ret Mercado']] = np.log(data[['Petro4','Ibovespa']]/data[['Petro4','Ibovespa']].shift(1))
data
# retirando dados faltantes
data = data.dropna()
matrz_covariancia = np.cov(data['ret Petro'], data['ret Mercado'])
matrz_covariancia
beta = matrz_covariancia[0,1]/matrz_covariancia[1,1]
beta
# usando regressao linear temos
beta,alpha = np.polyfit(data['ret Mercado'], data['ret Petro'], deg=1)
beta
# plotando o grafico
#fig, axis = plt.subplot(1,figsize=(20,10))
plt.scatter(data['ret Mercado'], data['ret Petro'], label='Data Points')
plt.plot(data['ret Mercado'], beta*data['ret Mercado']  + alpha, color= 'red', label="CAPM line")
plt.title('CAPM')
plt.xlabel('Mkt Return')
plt.ylabel('Stock Return')
plt.legend()
plt.grid(True)
retorno_esperado = selic + beta*(data['ret Mercado'].mean()*12-selic)
retorno_esperado
