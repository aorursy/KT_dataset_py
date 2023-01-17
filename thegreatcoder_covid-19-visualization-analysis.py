import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.dates as mdates



%matplotlib inline



plt.style.use('fivethirtyeight')



df  = pd.read_csv("https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv",parse_dates = ['Date'])

df['Total cases'] = df[['Confirmed','Recovered','Deaths']].sum(axis=1)
df.tail(10)
#Worldwide cases

worldwide_df = df.groupby(['Date']).sum()

worldwide_df.head()
w = worldwide_df.plot(figsize = (15,10))

w.set_xlabel('Date')

w.set_ylabel('Number of cases worldwide')

w.title.set_text("Worldwide COVID-19 Insights")
us_df = df[df['Country']=='US'].groupby(['Date']).sum()

us_df.to_csv("US COVID-19 Cases")
fig = plt.figure(figsize=(16,10))

ax = fig.add_subplot(111)



ax.plot(worldwide_df['Total cases'],label = 'Worldwide cases')

ax.plot(us_df['Total cases'],label = 'US cases')

ax.set_ylabel('Date')

ax.set_xlabel('Number of cases US')

ax.title.set_text("Worldwide vs US COVID-19 Insights")
df['Confirmed'].sub(df['Confirmed'].shift())
# United states daily cases and deaths

us_df = us_df.reset_index()

us_df['Daily Confirmed'] = us_df['Confirmed'].sub(us_df['Confirmed'].shift())

us_df['Daily Deaths'] = us_df['Deaths'].sub(us_df['Deaths'].shift())



fig = plt.figure(figsize=(16,10))

bx = fig.add_subplot(111)



bx.bar(us_df['Date'],us_df['Daily Confirmed'],color = 'b',label = 'US Daily Confirmed Cases')

bx.bar(us_df['Date'],us_df['Daily Deaths'],color = 'r',label = 'US Daily Deaths')

bx.set_ylabel('Date')

bx.set_xlabel('Number of people affected')

bx.title.set_text("Daily cases and deaths in US")
in_df = df[df['Country']=='India'].groupby(['Date']).sum()

in_df
fig = plt.figure(figsize=(15,10))

ax = fig.add_subplot(111)



ax.plot(worldwide_df['Total cases'],label = 'Worldwide cases')

ax.plot(in_df['Total cases'],label = 'India cases')

ax.set_ylabel('Date')

ax.set_xlabel('Number of cases India')

ax.title.set_text("Worldwide vs India COVID-19 Insights")
# India daily cases and deaths

in_df = in_df.reset_index()

in_df['Daily Confirmed'] = in_df['Confirmed'].sub(in_df['Confirmed'].shift())

in_df['Daily Deaths'] = in_df['Deaths'].sub(in_df['Deaths'].shift())



fig = plt.figure(figsize=(15,10))

ax = fig.add_subplot(111)



ax.bar(in_df['Date'],in_df['Daily Confirmed'],color = 'b',label = 'In Daily Confirmed Cases')

ax.bar(in_df['Date'],in_df['Daily Deaths'],color = 'r',label = 'In Daily Deaths')

ax.set_ylabel('Date')

ax.set_xlabel('Number of people affected')

ax.title.set_text("Daily cases and deaths in in")



plt.legend(loc = 'upper left')

plt.show()
country_df = df.groupby(['Date']).sum()

country_df = df.groupby(['Country']).sum()
country_df.sort_values(by=['Total cases'], inplace=True)

a = country_df.head(10)

a
country_df.sort_values(by=['Total cases'], inplace=True,ascending = False)

country_df.reset_index()

b = country_df.head(10)

b.head()
b['Total cases'].plot(figsize = (16,10),c = 'r',)

plt.xticks(fontsize = 25,rotation = 60)

plt.yticks(fontsize = 25)

plt.ylabel("Total cases ",fontsize = 25)

plt.xlabel("Country",fontsize = 25)

plt.title("Total cases by country",fontsize = 40)

plt.show()
b['Recovered'].plot(c = 'g',figsize = (16,10))

b['Total cases'].plot(c = 'b')

b['Deaths'].plot(c= 'r')

b['Confirmed'].plot(c = 'y')

plt.xticks(fontsize = 25,rotation = 60)

plt.yticks(fontsize = 25)

plt.ylabel("Cases ",fontsize = 25)

plt.xlabel("Country",fontsize = 25)

plt.title("Cases by country",fontsize = 50)

plt.legend()

plt.show()
b