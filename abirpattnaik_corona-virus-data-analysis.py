# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # seaborn package for visualising

import plotly.express as px # plotly visualisation

import time

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
'''

List of files present

/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv

/kaggle/input/novel-corona-virus-2019-dataset/time_series_2019_ncov_recovered.csv

/kaggle/input/novel-corona-virus-2019-dataset/time_series_2019_ncov_deaths.csv

/kaggle/input/novel-corona-virus-2019-dataset/time_series_2019_ncov_confirmed.csv

'''

#whole_data_path='../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv'

whole_data_path='../input/novel-corona-virus-2019-dataset/covid_19_data.csv'



#recovered_data_path='../input/novel-corona-virus-2019-dataset/time_series_2019_ncov_recovered.csv'

#deaths_data_path='../input/novel-corona-virus-2019-dataset/time_series_2019_ncov_deaths.csv'

#confirmed_data_path='../input/novel-corona-virus-2019-dataset/time_series_2019_ncov_confirmed.csv'



corona_data= pd.read_csv(whole_data_path)

#recovered_data=pd.read_csv(recovered_data_path)

#deaths_data=pd.read_csv(recovered_data_path)

#confirmed_data=pd.read_csv(recovered_data_path)



corona_data.shape

#Total no. of rows
corona_data.info()
if 'Last Update' in corona_data.columns :

    corona_data=corona_data.drop('Last Update',axis=1)

elif  'SNo' in corona_data.columns:

    corona_data=corona_data.drop('SNo',axis=1)

    

#corona_data["Date"] = corona_data['ObservationDate'].astype('datetime64')

corona_data["ObservationDate"] = corona_data['ObservationDate'].astype('datetime64')



corona_data["Confirmed"] = corona_data['Confirmed'].astype('int64')

corona_data["Deaths"] = corona_data['Deaths'].astype('int64')

corona_data["Recovered"] = corona_data['Recovered'].astype('int64')



print('Minimum date collected - ',min(corona_data["ObservationDate"]))

print('Maximum date collected(Latest data can be retrieved now) - ',max(corona_data["ObservationDate"]))



corona_data=corona_data.rename(columns={"ObservationDate": "Date","Country/Region":"Country"})
corona_data_date=pd.DataFrame(corona_data.groupby(by='Date').sum())

if 'SNo' in corona_data_date.columns:

    corona_data_date=corona_data_date.drop('SNo',axis=1)

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
#Since this is a cumulative dataset we can take the maximum of the dataset and then perform our analysis

latest_data_corona=corona_data[(corona_data.Date==max(corona_data["Date"]))]
corona_data_country=pd.DataFrame(latest_data_corona.groupby(by='Country').sum())

if 'SNo' in corona_data_country:

    corona_data_country=corona_data_country.drop('SNo',axis=1)

corona_data_country['country']=corona_data_country.index
print('Total no. of confirmed cases over these days',sum(latest_data_corona['Confirmed']))

print('Total no. of deaths over these days',sum(latest_data_corona['Deaths']))

print('Total no. of recovered cases over these days',sum(latest_data_corona['Recovered']))
corona_data_country.sort_values(['Confirmed','Deaths','Recovered'],ascending=[False,False,False])
corona_data_country["country"].replace({"Ivory Coast": "Cote d'Ivoire", 

                                        "Mainland China": "China",

                                        "Hong Kong":"Hong Kong, China",

                                       "South Korea":"Korea, Rep.",

                                        "UK":"United Kingdom",

                                        "US":"United States",

                                        "Macau" :"China"

                                       }, inplace=True)

df = px.data.gapminder().query("year == 2007")

corona_data_country_geo_cd=pd.merge(corona_data_country,df,how='left',on='country')
#Removing NaN based rows 

corona_data_country_geo_cd=corona_data_country_geo_cd.dropna(how='any')

corona_data_country_geo_cd=corona_data_country_geo_cd.drop(['year','lifeExp','pop','gdpPercap'],axis=1)
fig = px.scatter_geo(corona_data_country_geo_cd[corona_data_country_geo_cd['country']!='China'], locations="iso_alpha",

                     size="Confirmed", # size of markers, "pop" is one of the columns of gapminder)

                    )

fig.show()
fig = px.choropleth(corona_data_country_geo_cd[corona_data_country_geo_cd['country']!='China'], locations="iso_alpha",

                    color="Confirmed", # lifeExp is a column of gapminder

                    hover_name="country", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.Electric)

fig.show()
fig = px.choropleth(corona_data_country_geo_cd[corona_data_country_geo_cd['country']!='China'], locations="iso_alpha",

                    color="Deaths", # lifeExp is a column of gapminder

                    hover_name="country", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.Oranges_r)

fig.show()
fig = px.choropleth(corona_data_country_geo_cd[corona_data_country_geo_cd['country']!='China'], locations="iso_alpha",

                    color="Recovered", # lifeExp is a column of gapminder

                    hover_name="country", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.Oranges_r)

fig.show()
#Analysis of China Region

#corona_data.head()

corona_data["Country"].replace({"Ivory Coast": "Cote d'Ivoire", 

                                        "Mainland China": "China",

                                        "Hong Kong":"Hong Kong, China",

                                       "South Korea":"Korea, Rep.",

                                        "UK":"United Kingdom",

                                        "US":"United States",

                                        "Macau" :"China"

                                       }, inplace=True)

corona_data.Date=corona_data.Date.apply(lambda x:x.date())
#Selecting data for only China regions including their province

China_data=corona_data[corona_data['Country']=='China']

print(China_data.columns)
sns.catplot(y="Province/State", x="Confirmed", data=China_data,kind='boxen',height=9,aspect=1.4);

sns.catplot(y="Province/State", x="Deaths", data=China_data,kind='boxen',height=9,aspect=1.4);

sns.catplot(y="Province/State", x="Recovered", data=China_data,kind='boxen',height=9,aspect=1.4);
