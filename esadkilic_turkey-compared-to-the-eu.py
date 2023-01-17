%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import re

plt.style.use('ggplot')
df = pd.read_csv('/kaggle/input/world-development-indicators/Indicators.csv')



Indicator_array =  df[['IndicatorName','IndicatorCode']].drop_duplicates().values
import pandas as pd

Country = pd.read_csv("../input/world-development-indicators/Country.csv")

CountryNotes = pd.read_csv("../input/world-development-indicators/CountryNotes.csv")

Footnotes = pd.read_csv("../input/world-development-indicators/Footnotes.csv")

Indicators = pd.read_csv("../input/world-development-indicators/Indicators.csv")

Series = pd.read_csv("../input/world-development-indicators/Series.csv")

SeriesNotes = pd.read_csv("../input/world-development-indicators/SeriesNotes.csv")

chosen_country1 = 'Turkey'

chosen_indicators = ['NE.TRD.GNFS.ZS', 'SE.SEC.NENR','SP.DYN.LE00.IN', \

                     'NY.GDP.PCAP.PP.KD','SP.URB.TOTL.IN.ZS']



df_subset = df[df['IndicatorCode'].isin(chosen_indicators)]





df_EU = df_subset[df['CountryName']=="European Union"]

df_Turkey = df_subset[df['CountryName']=="Turkey"]
def plot_indicator(indicator,delta=10):

    ds_EU = df_EU[['IndicatorName','Year','Value']][df_EU['IndicatorCode']==indicator]

    try:

        title = ds_EU['IndicatorName'].iloc[0]

    except:

        title = "None"



    xeu = ds_EU['Year'].values

    yeu = ds_EU['Value'].values

    ds_Turkey = df_Turkey[['IndicatorName','Year','Value']][df_Turkey['IndicatorCode']==indicator]

    xturkey = ds_Turkey['Year'].values

    yturkey = ds_Turkey['Value'].values

    

    plt.figure(figsize=(14,4))

    

    plt.subplot(121)

    plt.plot(xeu,yeu,label='European Union')

    plt.plot(xturkey,yturkey,label='Turkey')

    plt.title(title)

    plt.legend(loc=2)
plot_indicator(chosen_indicators[0],delta=10)
plot_indicator(chosen_indicators[4],delta=10)
plot_indicator(chosen_indicators[1],delta=10)
plot_indicator(chosen_indicators[2],delta=10)

plot_indicator(chosen_indicators[3],delta=10)
arr = np.array([5.7,9.9,14.5,21.6,48.3])



def gini(arr):

    count = arr.size

    coefficient = 2 / count

    indexes = np.arange(1, count + 1)

    weighted_sum = (indexes * arr).sum()

    total = arr.sum()

    constant = (count + 1) / count

    return coefficient * weighted_sum / total - constant



def lorenz(arr):

    scaled_prefix_sum = arr.cumsum() / arr.sum()

    return np.insert(scaled_prefix_sum, 0, 0)



print(gini(arr))



lorenz_curve = lorenz(arr)



plt.plot(np.linspace(0.0, 1.0, lorenz_curve.size), lorenz_curve)



plt.plot([0,1], [0,1])

plt.show()