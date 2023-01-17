import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')



import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
Cor = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")
def country_plotter(land):    

    temp_df = Cor[Cor['Country/Region'] == land]



    growth_rate = [0]



    for i in range(1,len(temp_df['Confirmed'])): 

        growth_rate.append(temp_df.iloc[i,5]/temp_df.iloc[i-1,5])



    temp_df["Growth Rate"] = growth_rate

    temp_df['New Cases'] = temp_df['Confirmed'].diff()



    amount_of_days = range(1, len(temp_df['ObservationDate']) + 1)



    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

    fig.set_figwidth(25)



    ax1.scatter(amount_of_days, temp_df['Confirmed'], label = 'Confirmed')

    ax1.bar(amount_of_days, temp_df['New Cases'], label = 'Growth')

    ax1.legend(loc=2)

    ax1.set_title(f'Confirmed Cases and Growth in {land}')

    ax1.set_xlabel('Number of Days')



    ax2.plot(amount_of_days, temp_df['Growth Rate'])

    ax2.set_title(f'Growth Rate in {land}')

    ax2.set_xlabel('Number of Days')

    

    ax3.plot(amount_of_days, temp_df['Deaths'])

    ax3.set_title(f'Deaths in {land}')

    ax3.set_xlabel('Number of Days')    

    

    ax4.hist(temp_df['Growth Rate'])

    ax4.set_title(f'Growth Rate Histogram of {land}')
Europe = ['Germany','Netherlands','Sweden','Belgium','Spain','Finland','Norway']

for i in Cor['Country/Region'].unique():

    if i in Europe:

        plt.figure()

        country_plotter(i)

        None