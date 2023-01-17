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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns; sns.set(color_codes=True)  # visualization tool





from sklearn.linear_model import LinearRegression



import warnings

warnings.filterwarnings("ignore")
from IPython.display import IFrame

IFrame('https://www.myheatmap.com/maps/PPk1_rfT1jQ%3D', width=800, height=600)
data= pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")
data.head()
data.info()
data.shape
data.describe()
data.isnull().sum()
#this shows we have empty values in province/state field
#data[['Confirmed', 'Deaths', 'Recovered']].sum()
# Let's get rid of the Sno column as it's redundant

data.drop(['Sno'], axis=1, inplace=True)
data['Date'] = data['Date'].apply(pd.to_datetime)
data['Last Update'] =data['Last Update'].apply(pd.to_datetime)
data.tail()
data['Province/State'].value_counts()
# Countries affected



countries = data['Country'].unique().tolist()

print(countries)



print("\nTotal countries affected by virus: ",len(countries))
#Combining China and Mainland China cases



data['Country'].replace({'Mainland China':'China'},inplace=True)

countries = data['Country'].unique().tolist()

print(countries)

print("\nTotal countries affected by virus: ",len(countries))
data
from datetime import date

d = data['Date'][-1:].astype('str')

year = int(d.values[0].split('-')[0])

month = int(d.values[0].split('-')[1])

day = int(d.values[0].split('-')[2].split()[0])
latest_nCoV_data = data[data['Date'] > pd.Timestamp(date(year,month,day))]

# Data Glimpse

latest_nCoV_data.tail()
# Getting the latest numbers



#formatted_text('***Latest Numbers Globaly***')

print('Confirmed Cases around the globe : ',latest_nCoV_data['Confirmed'].sum())

print('Deaths Confirmed around the globe: ',latest_nCoV_data['Deaths'].sum())

print('Recovered Cases around the globe : ',latest_nCoV_data['Recovered'].sum())
tempState = data['Province/State'].mode()

print(tempState)

#df['Province/State'].fillna(tempState, inplace=True)
from datetime import datetime

# 1/22/2020 12:00

# 1/26/2020 23:00

# 1/23/20 12:00 PM

# 2020-01-02 23:33:00

# 

def try_parsing_date_time(text):

    for fmt in ('%Y-%m-%d', '%d.%m.%Y', '%m/%d/%Y %h:%m', '%m/%d/%Y %H:%M','%m/%d/%Y %H:%M:%S','%m/%d/%y %I:%M %p', '%m/%d/%Y %I:%M %p', '%Y-%d-%m %H:%M:%S'):

        try:

            return datetime.strptime(text, fmt)

        except ValueError:

            pass

    raise ValueError('no valid date time format found', text)





def try_parsing_date(text):

    for fmt in ('%m/%d/%Y', '%m/%d/%y', '%Y-%d-%m', '%d.%m.%Y'):

        try:

            return datetime.strptime(text, fmt)

        except ValueError:

            pass

    raise ValueError('no valid date format found', text)
data['Date']
data['Date']
#No. Of Countries Currently affected by it.

allCountries = latest_nCoV_data['Country'].unique().tolist()

print(allCountries)



print("\nTotal countries affected by virus: ",len(allCountries))
CountryWiseData = pd.DataFrame(latest_nCoV_data.groupby('Country')['Confirmed', 'Deaths', 'Recovered'].sum())

CountryWiseData['Country'] = CountryWiseData.index

CountryWiseData.index = np.arange(1, len(allCountries)+1)



CountryWiseData = CountryWiseData[['Country','Confirmed', 'Deaths', 'Recovered']]



#formatted_text('***Country wise Analysis of ''Confirmed'', ''Deaths'', ''Recovered'' Cases***')

CountryWiseData
#formatted_text('***Country wise Analysis of ''Confirmed'', ''Deaths'', ''Recovered'' Cases***')

CountryWiseData.plot('Country',['Confirmed', 'Deaths', 'Recovered'],kind='bar',figsize=(30,20), fontsize=18)
date_wise_data = data[["Date","Confirmed","Deaths","Recovered"]]

date_wise_data.head()
date_wise_data = date_wise_data.groupby(["Date"]).sum().reset_index()



# strip off the time part from date for day-wise distribution

date_wise_data.Date = date_wise_data.Date.apply(lambda x:x.date())



#formatted_text('***Day wise distribution (WorldWide) for Confirmed, Deaths and Recovered Cases***')

date_wise_data
#*Day wise distribution (WorldWide) for Confirmed, Deaths and Recovered Cases

date_wise_data.plot('Date',['Confirmed', 'Deaths', 'Recovered'],kind='bar',figsize=(20,15), fontsize=15, rot=30)
latest_nCoV_data
china_latest_data = latest_nCoV_data[latest_nCoV_data['Country']=='China'][['Province/State','Confirmed','Deaths','Recovered']]

#data[data.Country == 'China'][['Province/State', 'Confirmed']].groupby('Province/State').max()

# Reset Index

china_latest_data.reset_index(drop=True, inplace=True)

china_latest_data.index = pd.RangeIndex(start=1, stop=len(china_latest_data['Province/State']) + 1, step=1)



#formatted_text('***Numbers in China for Confirmed, Deaths and Recovered Cases***')



# Data Glimpse

china_latest_data
china_latest_data.plot('Province/State',['Confirmed', 'Deaths', 'Recovered'],kind='bar',figsize=(20,15), fontsize=15)
rest_of_China = china_latest_data[china_latest_data['Province/State'] !='Hubei'][["Province/State", "Confirmed","Deaths","Recovered"]]



# Reset Index to start from 1

rest_of_China.reset_index(drop=True, inplace=True)

rest_of_China.index = pd.RangeIndex(start=1, stop=len(china_latest_data['Province/State']), step=1)



#formatted_text('***Numbers in rest of China for Confirmed, Deaths and Recovered Cases***')



# Data Glimpse

rest_of_China
rest_of_China.plot('Province/State',['Confirmed', 'Deaths', 'Recovered'],kind='bar',figsize=(20,15), fontsize=15)
#formatted_text('***Most number of Confirmed Cases Outside of Hubei***')

print(rest_of_China[rest_of_China['Confirmed'] > 500])
#Rest Of World 
rest_of_world = CountryWiseData[CountryWiseData['Country'] !='China'][["Country", "Confirmed","Deaths","Recovered"]]



# Reset Index

rest_of_world.reset_index(drop=True, inplace=True)

rest_of_world.index = pd.RangeIndex(start=1, stop=len(CountryWiseData['Country']), step=1)



#formatted_text('***Numbers in rest of world for Confirmed, Deaths and Recovered Cases***')



# Data Glimpse

rest_of_world
#formatted_text('***Most number of Confirmed Cases Outside of China***')

print(rest_of_world[rest_of_world['Confirmed'] > 20])
rest_of_world.plot('Country',['Confirmed', 'Deaths', 'Recovered'],kind='bar',figsize=(20,15), fontsize=15)
data.Country.nunique()
#Plots for the field after data cleaning 

data.plot(subplots=True,figsize=(18,18))

plt.show()


from datetime import date

data_2_feb = latest_nCoV_data[latest_nCoV_data['Date'] > pd.Timestamp(date(2020,2,2))]

data_2_feb.head()
import plotly.express as px

pxdf = px.data.gapminder()



country_isoAlpha = pxdf[['country', 'iso_alpha']].drop_duplicates()

country_isoAlpha.rename(columns = {'country':'Country'}, inplace=True)

country_isoAlpha.set_index('Country', inplace=True)

country_map = country_isoAlpha.to_dict('index')
def getCountryIsoAlpha(country):

    try:

        return country_map[country]['iso_alpha']

    except:

        return country

    
latest_nCoV_data['iso_alpha'] = latest_nCoV_data['Country'].apply(getCountryIsoAlpha)

latest_nCoV_data
data_plot = latest_nCoV_data.groupby('iso_alpha').sum().reset_index()

fig = px.choropleth(data_plot, locations="iso_alpha",

                    color="Confirmed", 

                    hover_name="iso_alpha", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.Plasma)

fig.show()
latest_nCoV_data.groupby('Country')['Confirmed'].sum()
latest_nCoV_data.groupby('Country')['Confirmed'].sum().sort_values(ascending=False)[0:10]

latest_nCoV_data.groupby('Country')['Deaths'].sum().sort_values(ascending=False)
#Initial Case time

data.sort_values(by='Date')['Date'][0]
#latest Case 

latest_nCoV_data['Date'].max()
latest_nCoV_data[latest_nCoV_data.Country == 'China'][['Province/State', 'Confirmed']].groupby('Province/State').max()
latest_nCoV_data[['Confirmed', 'Deaths', 'Recovered']].max().plot(kind='bar')
plt.figure(figsize=(12,7))

chart = sns.countplot(data=latest_nCoV_data, x='Country', palette='Set1')

chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right', fontweight='light');