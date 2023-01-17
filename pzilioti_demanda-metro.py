# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dados = pd.read_csv("/kaggle/input/demanda-metro-sp/serie.csv", sep=";", parse_dates=True, date_parser ='MM/YYYY', thousands='.')



print(dados.shape)

dados.head()
dados.dtypes
#Serie temporal da soma das demandas das linhas

fig, axs = plt.subplots(1, 1, figsize=(20, 5))

axs.plot(dados.mes, dados.passageiro_transportados)

plt.show()
#Serie temporal das demandas, por linha

fig, axs = plt.subplots(1, 1, figsize=(20, 5))

axs.plot(dados.mes, dados.linha_azul,'b')

axs.plot(dados.mes, dados.linha_verde,'g')

axs.plot(dados.mes, dados.linha_vermelha,'r')

axs.plot(dados.mes, dados.linha_prata,'k')

plt.show()
#Serie temporal das médias dos dias uteis, por linha

fig, axs = plt.subplots(1, 1, figsize=(20, 5))

axs.plot(dados.mes, dados.azul_media_dias_uteis,'b')

axs.plot(dados.mes, dados.verde_media_dias_uteis,'g')

axs.plot(dados.mes, dados.vermelha_media_dias_uteis,'r')

axs.plot(dados.mes, dados.prata_media_dias_uteis,'k')

plt.show()
#Serie temporal das médias dos sabados, por linha

fig, axs = plt.subplots(1, 1, figsize=(20, 5))

axs.plot(dados.mes, dados.azul_media_sabado,'b')

axs.plot(dados.mes, dados.verde_media_sabado,'g')

axs.plot(dados.mes, dados.vermelha_media_sabado,'r')

axs.plot(dados.mes, dados.prata_media_sabado,'k')

plt.show()
#Serie temporal das médias dos domingos, por linha

fig, axs = plt.subplots(1, 1, figsize=(20, 5))

axs.plot(dados.mes, dados.azul_media_domingo,'b')

axs.plot(dados.mes, dados.verde_media_domingo,'g')

axs.plot(dados.mes, dados.vermelha_media_domingo,'r')

axs.plot(dados.mes, dados.prata_media_domingo,'k')

plt.show()