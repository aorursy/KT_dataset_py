import matplotlib.pyplot as plt

plt.style.use('ggplot') 

import pandas as pd

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from scipy import stats

from xgboost import XGBRegressor

import warnings

import json

from plotnine import *

warnings.filterwarnings('ignore')

%matplotlib inline
wineDataBase = pd.read_csv('../input/wine-reviews/winemag-data-130k-v2.csv')

#verificando o tamanho da Base

wineDataBase.shape
wineDataBase.describe()
wineDataBase.head(3)
def resumetable(df):

    print(f"Dataset Shape: {df.shape}")

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.loc[0].values

    summary['Second Value'] = df.loc[1].values

    summary['Third Value'] = df.loc[2].values



    for name in summary['Name'].value_counts().index:

        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 



    return summary



def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df



resumetable(wineDataBase)


#Verificando valores nulos

total = wineDataBase.isnull().sum().sort_values(ascending=False)

percent = (wineDataBase.isnull().sum()/wineDataBase.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
import squarify 



df_raw = wineDataBase



df = df_raw.groupby('points').size().reset_index(name='counts')

labels = df.apply(lambda x: str(x[0]) + "\n (" + str(x[1]) + ")", axis=1)

sizes = df['counts'].values.tolist()

colors = [plt.cm.Spectral(i/float(len(labels))) for i in range(len(labels))]



plt.figure(figsize=(12,8), dpi= 80)

squarify.plot(sizes=sizes, label=labels, color=colors, alpha=.8)



plt.title('Treemap of Vechile Class')

plt.axis('off')

plt.show()


df = wineDataBase.head(1000).dropna()

(ggplot(df)

     + aes('points', 'price')

     + aes(color='points')

     + geom_point()

     + stat_smooth()

     + facet_wrap('~variety')

)
(ggplot(df, aes('points', 'price'))

 + geom_point()

)
plt.figure(figsize = (40, 20))

sns.boxplot(x = 'country', y = 'points',  data = wineDataBase)
b = wineDataBase.groupby('country')[['points','price']].mean()

b = b.sort_values(by=['points'],ascending=False)

a4_dims = (11.7, 8.27)

fig, ax = plt.subplots(figsize=a4_dims)

sns.scatterplot(x='points',y='price',hue= b.index, data = b)
b.head(10)
def estatistica(pais):

    media = wineDataBase[wineDataBase['country'] == pais

                       ].points.mean()

    moda = wineDataBase[wineDataBase['country'] == pais

                      ].points.mode()[0]

    mediana = wineDataBase[wineDataBase['country'] == pais

                     ].points.median()

    desvio = wineDataBase[wineDataBase['country'] == pais

                        ].points.std()

    estado = {'moda': float(moda), 'mediana': float(mediana), 'media': float(media), 'desvio_padrao': float(desvio)}

    return json.dumps(estado)
medidasEstatisticasPais = pd.DataFrame(wineDataBase.groupby('country')['points'].describe())
medidasEstatisticasPais = medidasEstatisticasPais.sort_values(by=['mean','std'],ascending =False)

medidasEstatisticasPais.head()
medidasEstatisticasProvincia = pd.DataFrame(wineDataBase.groupby('province')['points'].describe())
medidasEstatisticasProvincia = medidasEstatisticasProvincia.sort_values(by=['mean','std'],ascending =False)

medidasEstatisticasProvincia.head()