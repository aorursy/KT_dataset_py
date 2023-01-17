# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Plotting libraries

import matplotlib.pyplot as plt

import plotly.express as px



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv',parse_dates=['Date'])

#print(len(data))

#print(data.dtypes)

data.head()
data.rename(columns={'Id': 'id',

                     'Province/State':'state',

                     'Country/Region':'country',

                     'Lat':'lat',

                     'Long': 'long',

                     'Date': 'date', 

                     'ConfirmedCases': 'confirmed',

                     'Fatalities':'deaths',

                    }, inplace=True)

data.head()
data = data.sort_values(by=['country','date'],ascending=[True,True])

#data.head()
 # Sorting by country and date

data = data.groupby(['country','date'])['country','date','confirmed','deaths'].sum().reset_index()

#print(len(data))

data.head()
# Worldwide

worldwide = data.groupby(['date'])[['date','confirmed','deaths']].sum().reset_index()

worldwide.tail()
fig = px.line(worldwide,'date','confirmed',title='Worldwide confirmed cases')

fig.show()



fig = px.line(worldwide,'date','confirmed',title='Worldwide confirmed cases (Log scale)',log_y = True)

fig.show()
# China

grouped_china = data[data.country=='China']



fig = px.line(grouped_china,'date','confirmed',title='China confirmed cases')

fig.show()



fig = px.line(grouped_china,'date','confirmed',title='China confirmed cases (Log scale)',log_y = True)

fig.show()
# USA

grouped_usa = data[data.country=='US']



fig = px.line(grouped_usa,'date','confirmed',title='USA confirmed cases')

fig.show()



fig = px.line(grouped_usa,'date','confirmed',title='USA confirmed cases (Log scale)',log_y = True)

fig.show()
# India

grouped_india = data[data.country=='India']



fig = px.line(grouped_india,'date','confirmed',title='India confirmed cases')

fig.show()



fig = px.line(grouped_india,'date','confirmed',title='India confirmed cases (Log scale)',log_y = True)

fig.show()
# Cases per population

pop = pd.read_csv('../input/countryinfo/covid19countryinfo.csv')

pop.head()
# Treat values

pop['pop'] = pop['pop'].str.replace(',','').fillna(0).astype(int)
# Treat missing values

# Check missing values per column

pop.isnull().sum()
# Fill missing with mean

pop['smokers'] = pop['smokers'].fillna(pop['smokers'].mean())

# And since this data is on 100 scale so making it %

pop['smokers'] = pop['smokers']/100

pop['urbanpop'] = pop['urbanpop']/100
# Clean population data

pop_data = pop[['country','pop','density','medianage','urbanpop','hospibed','smokers']]

pop_data.describe()
fig = px.bar(pop.sort_values(by="pop", ascending=False)[:10],'country','pop',title = "Population country wise")

fig.update_layout(xaxis_title = 'Country',yaxis_title = 'Population')

fig.show()
# Join main data with population data

data = data.merge(pop_data,how = 'left', on =['country'])

data.head()
data['confirmed_norm'] = data['confirmed']/data['pop']

data['deaths_norm'] = data['deaths']/data['pop']

data.describe()
# Locations having more than 1% cases?
data[data.confirmed_norm>0.01].tail()
# Italy

grouped_italy = data[data.country=='Italy']



fig = px.line(grouped_italy,'date','confirmed_norm',title='Italy confirmed cases')

fig.layout.yaxis.tickformat = ',.2%'

fig.update_layout(xaxis_title = 'Date',yaxis_title = 'Confirmed cases (%)')

fig.show()



fig = px.line(grouped_italy,'date','confirmed_norm',title='Italy confirmed cases (Log scale)',log_y = True)

fig.layout.yaxis.tickformat = ',.2%'

fig.update_layout(xaxis_title = 'Date',yaxis_title = 'Confirmed cases (%)')

fig.show()
# India

grouped_india = data[data.country=='India']



fig = px.line(grouped_india,'date','confirmed_norm',title='India confirmed cases')

fig.layout.yaxis.tickformat = ',.5%'

fig.update_layout(xaxis_title = 'Date',yaxis_title = 'Confirmed cases (%)')

fig.show()



fig = px.line(grouped_india,'date','confirmed_norm',title='India confirmed cases (Log scale)',log_y = True)

fig.layout.yaxis.tickformat = ',.5%'

fig.update_layout(xaxis_title = 'Date',yaxis_title = 'Confirmed cases (%)')

fig.show()
# India

grouped_china = data[data.country=='China']



fig = px.line(grouped_china,'date','confirmed_norm',title='China confirmed cases')

fig.layout.yaxis.tickformat = ',.3%'

fig.update_layout(xaxis_title = 'Date',yaxis_title = 'Confirmed cases (%)')

fig.show()



fig = px.line(grouped_china,'date','confirmed_norm',title='China confirmed cases (Log scale)',log_y = True)

fig.layout.yaxis.tickformat = ',.3%'

fig.update_layout(xaxis_title = 'Date',yaxis_title = 'Confirmed cases (%)')

fig.show()
data['max_date'] = data.groupby(['country'])['date'].transform('max')

data['max_date_flag'] = np.where(data['date'] == data['max_date'],True,False)

latest_data = data[data['max_date_flag']]

latest_data.tail()
# Bar plot of most affected countries (% wise)

fig = px.bar(latest_data.sort_values(by="confirmed_norm", ascending=False)[:10],'country','confirmed_norm',title = "Confirmed per 100 (Country wise)")

fig.layout.yaxis.tickformat = ',.1%'

fig.update_layout(xaxis_title = 'Country',yaxis_title = 'Confirmed cases (%)')

fig.show()
# Remove countries with population less than million and confirmed cases less than 100

fig = px.bar(latest_data[(latest_data['pop'] >= 1e6) & (latest_data['confirmed'] >= 100)].sort_values(by="confirmed_norm", ascending=False)[:10],'country','confirmed_norm',title = "Confirmed per 100 (Country wise)")

fig.layout.yaxis.tickformat = ',.2%'

fig.update_layout(xaxis_title = 'Country',yaxis_title = 'Confirmed cases (%)')

fig.show()
clean_data = latest_data[(latest_data['pop'] >= 1e6) & (latest_data['confirmed'] >= 100)]

clean_data.describe()
fig = px.scatter(clean_data,'medianage','confirmed_norm',title = "Confirmed % vs Median Age",

                hover_name="country")

fig.layout.yaxis.tickformat = ',.2%'

fig.update_layout(xaxis_title = 'Median Age',yaxis_title = 'Confirmed cases (%)')

fig.show()
# Correlation between smokers and confirmed norm

fig = px.scatter(clean_data,'smokers','confirmed_norm',title = "Confirmed % vs % Smokers",

                hover_name="country")

fig.layout.yaxis.tickformat = ',.2%'

fig.update_layout(xaxis_title = 'Smokers %',yaxis_title = 'Confirmed cases (%)')

fig.show()
# Correlation between urbanpop vs confirmed_norm

fig = px.scatter(clean_data,'urbanpop','confirmed_norm',title = "Confirmed % vs Urban population %",

                hover_name="country")

fig.layout.yaxis.tickformat = ',.2%'

fig.update_layout(xaxis_title = 'Urban population %',yaxis_title = 'Confirmed cases (%)')

fig.show()
# Correlation between Hospital bed vs confirmed_norm

fig = px.scatter(clean_data,'hospibed','confirmed_norm',title = "Confirmed % vs Hospital bed per 1k",

                                hover_name="country")

fig.layout.yaxis.tickformat = ',.2%'

fig.update_layout(xaxis_title = 'No. hospital beds per 1k',yaxis_title = 'Confirmed cases (%)')

fig.show()
# Correlation between Population density vs confirmed_norm

fig = px.scatter(clean_data,'density','confirmed_norm',title = "Confirmed % vs Population Density",

                                hover_name="country")

fig.layout.yaxis.tickformat = ',.2%'

fig.update_layout(xaxis_title = 'Population density (sq km)',yaxis_title = 'Confirmed cases (%)')

fig.show()
plot_data = data.copy()

plot_data['date'] = pd.to_datetime(plot_data['date']).dt.strftime("%Y-%b-%d")

plot_data['factor_size'] = plot_data['confirmed'].pow(0.5)
fig = px.scatter_geo(plot_data, locations="country", locationmode='country names', 

                     color="confirmed", size='factor_size', hover_name="country", 

                     range_color= [1, 1000], 

                     projection="natural earth", animation_frame="date", 

                     title='Coronavirus (COVID 19): Spread Over Time', color_continuous_scale="portland")

#fig.update(layout_coloraxis_showscale=False)

fig.show()
# Percentage wise

plot_data['factor_size_pc'] = plot_data['confirmed_norm'].fillna(0).pow(0.2)

plot_data['confirmed_pc'] = plot_data['confirmed_norm'].fillna(0)*100
fig = px.scatter_geo(plot_data, locations="country", locationmode='country names', 

                     color="confirmed_pc", size='factor_size_pc', hover_name="country", 

                     range_color= [1e-10, 0.001], 

                     projection="natural earth", animation_frame="date", 

                     title='Coronavirus (COVID 19): Spread Over Time (% of population as confirmed cases)', color_continuous_scale="portland")

# fig.update(layout_coloraxis_showscale=False)

fig.show()