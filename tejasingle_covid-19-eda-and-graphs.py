import numpy as np 

import pandas as pd 

import warnings



warnings.simplefilter("ignore")

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

train=pd.read_csv('../input/train.csv',parse_dates=['Date'])

train.head()
test=pd.read_csv('../input/test.csv')

test.head()
submission=pd.read_csv('../input/submission.csv')

submission.head()
train[['Country/Region', 'Province/State']] =train[['Country/Region', 'Province/State']].fillna('N/A')

test[['Country/Region', 'Province/State']] =test[['Country/Region', 'Province/State']].fillna('N/A')
clean_data = pd.read_csv('../input/covid_19_clean_complete.csv', parse_dates=['Date'])

clean_data .head()
clean_data['Country/Region'].value_counts()

clean_data.info()
clean_data.rename(columns={'Province/State':'state',

                     'Country/Region':'country',

                     'Last Update':'last_updated',

                     'Confirmed': 'confirmed',

                     'Deaths':'deaths',

                     'Recovered':'recovered'

                    }, inplace=True)

clean_data.head()


# # Active Case 

clean_data['active'] = clean_data['confirmed']-clean_data['deaths']-clean_data['recovered']



clean_data['country'] = clean_data['country'].replace('Mainland China', 'China')



# # filling missing values 

clean_data[['state']] = clean_data[['state']].fillna('')





data=clean_data
data.head()
print(f"Earliest Entry: {data['Date'].min()}")

print(f"Last Entry:     {data['Date'].max()}")

print(f"Total Days:     {data['Date'].max() - data['Date'].min()}")
grouped = data.groupby('Date')['Date', 'confirmed', 'deaths','recovered','active'].sum().reset_index()

grouped.head()
import plotly.express as px



grouped = data.groupby('Date')['Date', 'confirmed', 'deaths','recovered','active'].sum().reset_index()



fig = px.line(grouped, x="Date", y="confirmed", title="Worldwide Confirmed Cases Over Time")



fig.show()







fig = px.line(grouped, x="Date", y="deaths", title="Worldwide deaths Cases Over Time")



fig.show()
china = data[data['country'] == "China"].reset_index()

china_date =china.groupby('Date')['Date', 'confirmed', 'deaths','recovered','active'].sum().reset_index()



italy = data[data['country'] == "Italy"].reset_index()

italy_date =italy.groupby('Date')['Date', 'confirmed', 'deaths','recovered','active'].sum().reset_index()



us = data[data['country'] == "US"].reset_index()

us_date =us.groupby('Date')['Date', 'confirmed', 'deaths','recovered','active'].sum().reset_index()



rest = data[~data['country'].isin(['China', 'Italy', 'US','recovered','active'])].reset_index()

rest_date =rest.groupby('Date')['Date', 'confirmed', 'deaths'].sum().reset_index()


fig = px.line(china_date, x="Date", y="confirmed",title="Confirmed Cases in China",height=500)

fig.show()



fig = px.line(china_date, x="Date", y="deaths", title="Deaths Cases in China",color_discrete_sequence=['#F61067'],height=500)

fig.show()

# Confirmed Cases in Italy



fig = px.line(italy_date, x="Date", y="confirmed",title=f"Confirmed Cases in Italy",color_discrete_sequence=['#01C4F2'],height=500)

fig.show()



# Deaths Cases in Italy

fig = px.line(italy_date, x="Date", y="deaths",title="Deaths Cases in Italy",color_discrete_sequence=['#F61067'],height=500)

fig.show()





fig = px.line(us_date, x="Date", y="confirmed",title=f"Confirmed Cases in USA ",height=500)

fig.show()



fig = px.line(us_date, x="Date", y="deaths",title="Death Cases in USA",color_discrete_sequence=['#F61067'],height=500)

fig.show()



fig = px.line(rest_date, x="Date", y="confirmed",title=f"Confirmed Cases in Rest of the World",height=500)

fig.show()



fig = px.line(rest_date, x="Date", y="deaths",title=f"Death Cases in Rest of the World",color_discrete_sequence=['#F61067'],height=500)

fig.show()
data['state'] = data['state'].fillna('')



latest =data[data['Date'] == max(data['Date'])].reset_index()

latest_grouped = latest.groupby('country')['confirmed', 'deaths','recovered','active'].sum().reset_index()
fig = px.bar(latest_grouped.sort_values('confirmed', ascending=False)[:20][::-1],x='confirmed', y='country',

             title='Confirmed Cases Worldwide',text='confirmed', orientation='h')

fig.show()

fig = px.bar(latest_grouped.sort_values('deaths', ascending=False)[:20][::-1],x='deaths', y='country',

             title='Death Cases Worldwide',text='deaths', orientation='h')

fig.show()



fig = px.bar(latest_grouped.sort_values('active', ascending=False)[:20][::-1],x='active', y='country',

             title='Active Cases Worldwide', text='active', orientation='h')

fig.show()

fig = px.bar(latest_grouped.sort_values('recovered', ascending=False)[:10][::-1],x='recovered', y='country',

             title='Recovered Cases Worldwide', text='recovered', orientation='h')

fig.show()





cases_over_time=data.groupby('Date')['recovered', 'deaths', 'active'].sum().reset_index()

cases_over_time=cases_over_time.melt(id_vars="Date", value_vars=['recovered','deaths','active'],

                                     var_name='case', value_name='count')





fig = px.line(cases_over_time, x="Date",y="count", color='case',title='Cases over time')

fig.show()





rate=latest.groupby('country')['confirmed', 'deaths','recovered','active'].sum().reset_index()



rate['Death_Rate'] = round((rate['deaths']/rate['confirmed'])*100,2)

per_hundred = rate[rate['confirmed']>100]

per_hundred = per_hundred.sort_values('Death_Rate', ascending=False)



fig = px.bar(per_hundred.sort_values(by="Death_Rate", ascending=False)[:15][::-1],x = 'Death_Rate', y = 'country', 

             title='Deaths per 100 Confirmed Cases', text='Death_Rate', orientation='h',)

fig.show()
rate['Recovery_Rate'] = round((rate['recovered']/rate['confirmed'])*100, 2)

per_hundred= rate[rate['confirmed']>100]

per_hundred= per_hundred.sort_values('Recovery_Rate', ascending=False)



fig = px.bar(per_hundred.sort_values(by="Recovery_Rate",ascending=False)[:15][::-1],x='Recovery_Rate',y='country', 

             title='Recoveries per 100  Cases', text='Recovery_Rate', height=800, orientation='h')

fig.show()