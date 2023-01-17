

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # seaborn package for visualising

import plotly.express as px # plotly visualisation

import time

import matplotlib.pyplot as plt

from fastai.tabular import * 

# visualization

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import folium



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
path = ('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv')

path
#whole_data_path='/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv'



whole_data_path='/kaggle/input/corona-virus-report/covid_19_clean_complete.csv'





covid_19_data = pd.read_csv("/kaggle/input/corona-virus-report/covid_19_clean_complete.csv")

#time_series_covid_19_Confirmed = pd.read_csv("/kaggle/input/covid19-coronavirus/time_series_19-covid-Confirmed.xlsx")

#time_series_covid_19_Deaths = pd.read_csv("/kaggle/input/covid19-coronavirus/time_series_19-covid-Deaths.xlsx")

#time_series_covid_19_Recovered = pd.read_csv("/kaggle/input/covid19-coronavirus/time_series_19-covid-Recovered.xlsx")



corona_data=pd.read_csv(whole_data_path)
corona_data.shape

#recovered_data.shape

#deaths_data.shape

#confirmed_data.shape

#Total no. of rows|
corona_data.info()
if 'Last Update' in corona_data.columns :

    corona_data=corona_data.drop('Last Update',axis=1)

elif  'Sno' in corona_data.columns:

    corona_data=corona_data.drop('Sno',axis=1)

    

corona_data["Date"] = corona_data['Date'].astype('datetime64')

#corona_data["date"] = corona_data['date'].astype('datetime64')



corona_data["Confirmed"] = corona_data['Confirmed'].astype('float64')

corona_data["Deaths"] = corona_data['Deaths'].astype('float64')

corona_data["Recovered"] = corona_data['Recovered'].astype('float64')



#print('Minimum date collected - ',min(corona_data["Date"]))

#print('Maximum date collected - ',max(corona_data["Date"]))



#corona_data=corona_data.rename(columns={"date": "ObservationDate","Province/State":"Country"})
print('Total no. of confirmed cases over these days',sum(corona_data['Confirmed']))

print('Total no. of deaths over these days',sum(corona_data['Deaths']))

print('Total no. of recovered cases over these days',sum(corona_data['Recovered']))





print('Mortality Rate', sum(corona_data['Deaths'])/sum(corona_data['Confirmed']))
corona_data_date=pd.DataFrame(corona_data.groupby(by='Date').sum())

if 'Sno' in corona_data_date.columns:

    corona_data_date=corona_data_date.drop('Sno',axis=1)

corona_data_date['Date']=corona_data_date.index

corona_data_date.Date=corona_data_date.Date.apply(lambda x:x.date())
melted_data=pd.melt(corona_data_date,id_vars=['Date'])
def bar_plot(column_name):

    plt.figure(figsize=(10,15))

    plt.xticks(rotation=90)

    plt.xlabel('Date', fontsize=18)

    plot_1=sns.barplot(x='Date',y=column_name,data=corona_data_date)

    plot_1
bar_plot('Recovered')
bar_plot('Deaths')
bar_plot('Confirmed')
hm=sns.catplot(x='Date', y='value', hue='variable', data=melted_data, kind='bar',height=10,aspect =1.6,legend=True)

hm.set_xticklabels( rotation=90)
import plotly.express as px

fig = px.line(melted_data, x="Date",y='value', color='variable')

fig.show()
corona_data_country=pd.DataFrame(corona_data.groupby(by='Country/Region').sum())

if 'Sno' in corona_data_country:

    corona_data_country=corona_data_country.drop('Sno',axis=1)

corona_data_country['country']=corona_data_country.index
corona_data_country.sort_values(['Confirmed','Deaths','Recovered'],ascending=[False,False,False])