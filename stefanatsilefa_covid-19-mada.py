import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from mpl_toolkits.basemap import Basemap

from itertools import chain

import seaborn as sns

sns.set_style("darkgrid")

from scipy import integrate, optimize

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
df = pd.read_csv('../input/covid19-madagacar/Mada-COVID.csv')

df.head()
mada_df = pd.DataFrame()

mada_df['Date'] = df.Date.unique()

mada_df['Cas_journalier'] = df.groupby('Date').agg({'Nouveau_cas':['sum']}).values

mada_df['Guerrison_journalier'] = df.groupby('Date').agg({'cas_Guerri':['sum']}).values

mada_df['Mort_journalier'] = df.groupby('Date').agg({'cas_mor':['sum']}).values

mada_df['Cas_cumuler'] = df.groupby('Date').agg({'Confirmé':['mean']}).values

mada_df['Guerrison_cumuler'] = df.groupby('Date').agg({'Guerrison':['mean']}).values

mada_df['Mort_cumuler'] = df.groupby('Date').agg({'Mortalite':['mean']}).values

mada_df = mada_df.set_index('Date')
cas = df.groupby('Date').agg({'Nouveau_cas':['sum']})

guerri = df.groupby('Date').agg({'cas_Guerri':['sum']})

cas_mor = df.groupby('Date').agg({'cas_mor':['sum']})

conf = df.groupby('Date').agg({'Confirmé':['mean']})

rec = df.groupby('Date').agg({'Guerrison':['mean']})

mor = df.groupby('Date').agg({'Mortalite':['mean']})
df_tana = df[df['Region']=='Analamanga']

df_tamaga = df[df['Region']=='Atsinanana']



cas_tana = df_tana.groupby('Date').agg({'Nouveau_cas':['sum']})

conf_tana = df_tana.groupby('Date').agg({'Nouveau_cas':['sum']}).cumsum()



cas_tamatave = df_tamaga.groupby('Date').agg({'Nouveau_cas':['sum']})

conf_tamatave = df_tamaga.groupby('Date').agg({'Nouveau_cas':['sum']}).cumsum()



mada_df['day_count'] = list(range(1,len(mada_df)+1))

y_mada = cas.values

xdata_mada = mada_df.day_count

x_mada = np.array(xdata_mada, dtype=float)

x_m = x_mada[:,np.newaxis]



df_tamaga['day_count'] = list(range(1,len(df_tamaga)+1))

y_tamaga = cas_tamatave.values

xdata_tamaga = df_tamaga.day_count

x_tamaga = np.array(xdata_tamaga, dtype=float)

x_t = x_tamaga[:,np.newaxis]



df_tana['day_count'] = list(range(1,len(df_tana)+1))

y_tana = df_tana['Nouveau_cas'].values

xdata_tana = df_tana.day_count

x_tana = np.array(xdata_tana, dtype=float)

x_a = x_tana[:,np.newaxis]
poly_reg = PolynomialFeatures(degree=6)

y_reg_tana = poly_reg.fit_transform(x_a)

model = LinearRegression()

model.fit(y_reg_tana, y_tana)



y_reg_tamaga = poly_reg.fit_transform(x_t)

model1 = LinearRegression()

model1.fit(y_reg_tamaga, y_tamaga)



y_reg_mada = poly_reg.fit_transform(x_m)

model2 = LinearRegression()

model2.fit(y_reg_mada, y_mada)
fig ,(ax1,ax2)= plt.subplots(1,2, figsize=(20,5))

plt.plot(model2.predict(y_reg_mada),color='red',label='Regression Polynômial')

cas.plot(ax=ax2,color='blue',label='Confirmés journalier')

guerri.plot(ax=ax1,color='green',label='Guerrison journalier')

cas_mor.plot(ax=ax1,color='black',label='Mortalite journalier')

plt.legend()



fig ,(ax1,ax2)= plt.subplots(1,2, figsize=(20,5))

rec.plot(ax=ax2,color='green',label='Guérisons')

mor.plot(ax=ax1,color='black',label='Décès')

conf.plot(ax=ax2,color='red',label='Confirmés')

plt.legend()
fig ,(ax1,ax2)= plt.subplots(1,2, figsize=(20,5))

plt.plot(model.predict(y_reg_tana),color='red',label='Regression Polynômial')

cas_tana.plot(ax=ax2,color='blue',label='Confirmés journalier')

conf_tana.plot(ax=ax1,color='black',label='Confirmés cumulée')

plt.legend()
fig ,(ax1,ax2)= plt.subplots(1,2, figsize=(20,5))

plt.plot(model1.predict(y_reg_tamaga),color='red',label='Regression Polynômial')

cas_tamatave.plot(ax=ax2,color='blue',label='Confirmés journalier')

conf_tamatave.plot(ax=ax1,color='black',label='Confirmés cumulée')

plt.legend()