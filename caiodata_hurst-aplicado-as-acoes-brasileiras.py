# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from datetime import datetime
import scipy.optimize as optimization
import mpld3
#from pandas_datareader import data as pdr
#import fix_yahoo_finance as yf
import plotly.plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go
import xlsxwriter
#yf.pdr_override()
%matplotlib inline
%load_ext autoreload
%autoreload 2
#%matplotlib notebook
style.use('classic')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

dates =['DATPRE']
df = pd.read_csv('../input/COTAHIST_A2009_to_A2018P.csv')
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
def hurst(x):
    for i in range(33):
        serie = x
        serie = serie[serie.columns[i]].fillna(0).values
        if not serie.any():
            print(0.5)
            continue
        tamanhoDosDados = serie.size
        yAcumulado = 0
        logTamanho = tamanhoDosDados - 1
        max_ = np.zeros(logTamanho)
        min_ = np.zeros(logTamanho)
        media_ = np.zeros(logTamanho)
        range_ = np.zeros(logTamanho)
        sdesvio_ = np.zeros(logTamanho)
        eixo_x = np.zeros(logTamanho)
        eixo_y = np.zeros(logTamanho)
        Y_calculo = np.zeros((tamanhoDosDados,tamanhoDosDados))
        # preenchendo médias
        # calcular total ALGO 0
        for i in np.arange(np.size(media_)):
            ch1 = 2+i
            media_[i] = serie[:ch1].sum() / ch1
        # preenchendo desvios
        for i in np.arange(np.size(sdesvio_)):
            ch1 = i + 2
            sdesvio_[i] = serie[:ch1].std()
        # preenchendo o range
        for i in np.arange(np.size(max_)):
            ch1 = i + 2
            for j in range(ch1):
                Y_calculo[i][j] = serie[j] - serie[:ch1].mean() + yAcumulado
                yAcumulado = Y_calculo[i][j]
            max_[i] = Y_calculo[i].max()
            min_[i] = Y_calculo[i].min()
            range_[i] = max_[i] - min_[i]
        # calculando o eixo x
        for i in np.arange(np.size(max_)):
            eixo_x[i] = math.log(i+2.0,2.0)
        # calculando o eixo y
        for i in np.arange(np.size(range_)):
            eixo_y[i] = math.log((range_[i] / sdesvio_[i]),2)
        # calculando MMQ
        z = eixo_x * eixo_y
        xQuad_ = eixo_x ** 2
        a = (eixo_x.size * z.sum() - eixo_x.sum() * eixo_y.sum() ) / (eixo_x.size * xQuad_.sum() - eixo_x.sum()**2)
        print(a)
hurst(df_brasil)
