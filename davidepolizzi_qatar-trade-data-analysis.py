# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#import zipfile



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/qatartrade1518/EVERPROD.PAN.csv")

dfcodes = pd.read_csv("/kaggle/input/qatartrade15-18/HS2code.csv",delimiter='\t')



dfcodes.set_index('code', inplace=True)



df = df.join(dfcodes, on='hs2')

#df1 = df

df = df.drop(['Unnamed: 0', 'hs8','hs6','hs4'], axis=1)



df = df.astype({"hs2": object, "year": int })
df['time'] = (((df['year']-2015) * 12) + df['month'])



df.loc[df['time'] < 30, 'emb'] = False

df.loc[df['time'] >= 30, 'emb'] = True



df = df.drop(['year','month'], axis = 1)
df.sample(5)
df.shape
df.head(10)
df.info()
df.describe().round()
df.isnull().sum()
ccol = ['iso3c','port','continent','region','hs2', 'emb','description','category']

ncol = ['time','weight','import_value']
for c in ccol:

    print(df[c].value_counts())

    print("-")
country_list = df.iso3c.unique()

time_list = df.time.unique()
country_list_sh=['USA','CHN','DEU','JPN','ARE','GBR','IND','ITA','FRA','SAU','CHE','TUR','KOR','ESP','THA','NLD','OMN','AUS','MYS','VNM','EGY']
df1 = pd.DataFrame(df[df.iso3c == 'TLS'].groupby(['time'])['import_value'].sum())

df1.rename(columns={'import_value': 'TLS'}, inplace = True)
df1.drop("TLS", axis='columns',inplace = True)
for  c in country_list_sh:

    df1 = df1.join(pd.DataFrame(df[df.iso3c == c].groupby(['time'])['import_value'].sum()))

    df1.rename(columns={'import_value': c}, inplace = True)



corr = df1.corr().round(2)

corr.style.background_gradient(cmap='PiYG')

#YlGn RdYlGn

# 'RdBu_r' & 'BrBG' are other good diverging colormaps
import seaborn as sns; sns.set()

import matplotlib.pyplot as plt

from pandas.plotting import autocorrelation_plot

from statsmodels.tsa.arima_model import ARIMA
fig, axs = plt.subplots(21,figsize=(40,250), dpi = 50, sharex=True)

i=0

x = df1.index

plt.style.use('fivethirtyeight')



for c in country_list_sh:

    y = df1[c].values

    axs[i].plot(x, y, label = c)

    axs[i].tick_params(axis='both', which='major', labelsize=30)

    axs[i].title.set_text(c)

    #axs[i].title.set_size(40)

    axs[i].axvline(x=29)

    i=i+1
plt.style.use('fivethirtyeight')



for c in country_list_sh:

    autocorrelation_plot(df1[c][:29].values)

    plt.title(c)

    plt.show()

    

#     axs[i].plot(x, y, label = c)

#     axs[i].tick_params(axis='both', which='major', labelsize=30)

#     axs[i].title.set_text(c)

    #axs[i].title.set_size(40)

#     axs[i].axvline(x=29)

#     i=i+1
model = ARIMA(df1['DEU'][:29].values, order=(0,1,1))

model_fit = model.fit(disp=0)

print(model_fit.summary())

model = ARIMA(df1['ITA'][:29].values, order=(0,1,1))

model_fit = model.fit(disp=0)

print(model_fit.summary())
residuals = pd.DataFrame(model_fit.resid)

residuals.plot()

plt.show()

residuals.plot(kind='kde')

plt.show()

print(residuals.describe())
itaforcast = model_fit.forecast(steps=18)
forca = itaforcast[0]
forcastgraph = np.append( [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],forca)
forcastgraph
(itaforcast[1])
plt.plot(df1[30:].index, df1['DEU'][30:].values)

plt.plot(df1[30:].index, forcastgraph[30:])
