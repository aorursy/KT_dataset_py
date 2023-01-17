# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

covid_19_clean_complete = pd.read_csv("../input/corona-virus-report/covid_19_clean_complete.csv")
df1 = covid_19_clean_complete



df1.head(20)
l = pd.to_datetime(df1.Date)

df1["Date"] = l

df1["Date"].min

df1[df1["Date"]=="1/30/20"]

primerasem = df1.loc[:2321]

cdrPrimerasem = primerasem.groupby("Country/Region")["Confirmed","Deaths","Recovered"].max()

cdrPrimerasem.loc[["China","Korea, South","Italy","Spain","Argentina"],]

cdrPrimerasem.sort_values(by="Confirmed",ascending=False).head(19)
#buscar el índice de la última entrada luego de un mes

df1[df1["Date"]=="02/22/20"].tail(1)
primermes = df1.loc[:8255,:].groupby("Country/Region")["Confirmed","Deaths","Recovered"].max()

primermes

primermes.loc[["China","Korea, South","Italy","Spain","Argentina"],:]
primermes.sort_values(by="Confirmed",ascending=False).head(20)
actual = df1.groupby("Country/Region")["Confirmed","Deaths","Recovered"].max()
ckisa_actual = actual.loc[["China","Korea, South","Italy","Spain","Argentina"],:]
global_actual = actual.sort_values(by="Confirmed",ascending=False)

global_actual
tasa_mortalidad_global = "tasa de mortalidad global: " + str((global_actual.Deaths.sum()/global_actual.Confirmed.sum())*100)+"%"

tasa_mortalidad_global

tasa_recuperacion_global = "tasa de recuperación global: " + str((global_actual.Recovered.sum()/global_actual.Confirmed.sum())*100) + "%"

tasa_recuperacion_global
dicc_tasas = {"Tasa de mortalidad":global_actual.Deaths*100/global_actual.Confirmed.values,

            "Tasa de recuperación":global_actual.Recovered.values*100/global_actual.Confirmed.values}

tasas_por_pais = pd.DataFrame(dicc_tasas, index=global_actual.index)

tasas_por_pais

tasas_por_pais.loc[["China","Korea, South","Italy","Spain","Argentina"],:]