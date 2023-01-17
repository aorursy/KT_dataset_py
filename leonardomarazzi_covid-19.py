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
file_path="/kaggle/input/coronavirus-italian-data/dati-regioni/dpc-covid19-ita-regioni.csv"
covid = pd.read_csv(file_path,index_col=0)
print("setup")
data_lomb=covid.loc[(covid.denominazione_regione=="Lombardia")]
plt.figure(figsize=(20,6))
plt.title("percentuali nuovi casi per regione")
plt.xlabel("data")
plt.ylabel("Casi confermati")
sns.lineplot(data=data_lomb['totale_casi'], label="totale_casi")
sns.lineplot(data=data_lomb['totale_positivi'], label="totale_positivi")

data = covid.iloc[-21:,:]
sns.set_style("whitegrid")
plt.figure(figsize=(25,6))
plt.title("percentuali nuovi casi per regione")
plt.xlabel("Regioni")
plt.ylabel("Percentuale")
sns.barplot(x=data.denominazione_regione, y=100*data['nuovi_positivi']/(data['totale_casi']-data['nuovi_positivi']))
plt.figure(figsize=(20,6))

data_lomb = data_lomb.iloc[10:,:]
plt.title("Percentuale di nuovi contagiati in lombardia nel tempo")
plt.xlabel("data")
plt.ylabel("Percentuale")
sns.lineplot(data=100*data_lomb['nuovi_positivi']/(data_lomb['totale_casi']-data_lomb['nuovi_positivi']),label="percentuale_nuovi_casi")