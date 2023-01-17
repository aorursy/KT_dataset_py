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
pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

file = "../input/covid19-in-italy/covid19_italy_region.csv"

covid_data= pd.read_csv(file, index_col=0)

#prendo solo i dati dell'ultimo giorno

data = covid_data.iloc[-21:,:]

print("Setup Complete")

covid_data.loc[(covid_data.RegionName == 'Lombardia')].tail()
sns.set_style("whitegrid")

plt.figure(figsize=(25,6))

plt.title("percentuali nuovi casi per regione")

plt.xlabel("Regioni")

plt.ylabel("Percentuale")

sns.barplot(x=data.RegionName, y=100*data['NewPositiveCases']/(data['TotalPositiveCases']-data['NewPositiveCases']))
#contando anche i morti nei contagiati

print("Nuovi casi in Italia: ",data.NewPositiveCases.sum())

print("Toale casi in Italia: ",data.TotalPositiveCases.sum())

print("percentuale nuovi casi: "+ str(100*data.NewPositiveCases.sum()/(data.TotalPositiveCases.sum() -data.NewPositiveCases.sum()))+"%")
sns.set_style("whitegrid")

plt.figure(figsize=(25,6))

plt.title("totale casi in lombardia")

plt.xlabel("giorni")

plt.ylabel("Totale casi")

data1 =covid_data.loc[(covid_data.RegionName == 'Lombardia')]

sns.barplot(y =data1.TotalPositiveCases, x =data1.Date, )
sns.set_style("whitegrid")

plt.figure(figsize=(25,6))

plt.title("percentuale di morte nel tempo in lombardia")

plt.xlabel("giorni")

plt.ylabel("percentuale morte")

data1 =covid_data.loc[(covid_data.RegionName == 'Lombardia')]

sns.barplot(y =100*data1.Deaths/(data1.TotalPositiveCases+data1.Deaths), x =data1.Date, )





sns.set_style("whitegrid")

plt.figure(figsize=(25,6))

plt.title("percentuale di guariti nel tempo in lombardia")

plt.xlabel("giorni")

plt.ylabel("percentuale guariti")

data1 =covid_data.loc[(covid_data.RegionName == 'Lombardia')]

sns.barplot(y =100*data1.Recovered/data1.TotalPositiveCases, x =data1.Date, )