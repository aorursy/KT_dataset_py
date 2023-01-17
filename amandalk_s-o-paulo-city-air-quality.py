import datetime

import warnings



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from statsmodels.tsa.seasonal import seasonal_decompose



%matplotlib inline
warnings.simplefilter(action='ignore')



sns.set(rc={'figure.figsize':(15, 6)})



sns.set_style('white', {

    'axes.spines.left': True,

    'axes.spines.bottom': True,

    'axes.spines.right': False,

    'axes.spines.top': False

})
df = pd.read_csv('/kaggle/input/sp-air-quality/sp_air_quality.csv', parse_dates=['Datetime'])
df.shape
df.tail()
pd.unique(df['Station'])
pollutants = ['Benzene', 'CO', 'PM10', 'PM2.5', 'NO2', 'O3', 'SO2', 'Toluene', 'TRS']
# Thresholds of good quality air [https://cetesb.sp.gov.br/ar/padroes-de-qualidade-do-ar/]

thresholds = {

    'PM10': 50,

    'PM2.5': 25,

    'O3': 100,

    'CO': 9,

    'NO2': 200,

    'SO2': 20

}
# São Paulo city stations

saopaulo_stations = ['Cerqueira César', 'Cid.Universitária-USP-Ipen', 'Ibirapuera', 'Interlagos',

        'Itaim Paulista', 'Itaquera', 'Marg.Tietê-Pte Remédios', 'Mooca', 'N.Senhora do Ó', 

        'Parque D.Pedro II', 'Pinheiros', 'Santana', 'Santo Amaro', 'Pico do Jaraguá', 'Perus']
for pollutant in pollutants:

    filtered = df[df['Station'].isin(saopaulo_stations)][['Datetime', pollutant]].dropna()

    grouped = filtered.groupby('Datetime').mean().reset_index()

    grouped = grouped.resample('d', on='Datetime').mean().reset_index()



    ax = sns.lineplot(x='Datetime', y=pollutant, data=grouped)

    

    plt.title('Mean of ' + pollutant + ' in a day in São Paulo')



    plt.xlabel('')

    

    max_lim = np.max(grouped[pollutant] + 10)

    

    plt.xlim(datetime.date(2013, 1, 1), datetime.date(2020, 12, 31))

    plt.ylim(0, max_lim)

    

    if pollutant in thresholds and thresholds[pollutant] < max_lim:

        ax.axhline(thresholds[pollutant], ls='--', label='Good quality threshold', c=sns.color_palette('Greys')[1])



        plt.legend(frameon=False)



    plt.show()
for pollutant in pollutants:

    filtered = df[df['Station'].isin(saopaulo_stations)][['Datetime', pollutant]].dropna()

    grouped = filtered.groupby('Datetime').mean().reset_index()

    grouped = grouped.resample('m', on='Datetime').mean().reset_index()



    series = pd.Series(grouped[pollutant])

    series.index = grouped['Datetime']



    result = seasonal_decompose(series, model='linear')

    result.plot()

    plt.show()

    