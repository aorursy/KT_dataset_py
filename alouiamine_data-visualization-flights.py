import datetime, warnings, scipy 

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.patches as patches

from matplotlib.patches import ConnectionPatch

from matplotlib.gridspec import GridSpec

from mpl_toolkits.basemap import Basemap



%matplotlib inline

warnings.filterwarnings("ignore")

df_flights = pd.read_csv('../input/flight-delays/flights.csv', low_memory=False)

df_airports=pd.read_csv('../input/flight-delays/airports.csv', low_memory=False)

df_airlines=pd.read_csv('../input/flight-delays/airlines.csv', low_memory=False)

print('flights data dimensions:', df_flights.shape)

print('airports data dimensions:', df_airports.shape)

print('airlines data dimensions:', df_airlines.shape)



#some information about
df_flights.head()

# display the percentage of the null value for each column (feature)

df_flights.isnull().sum()/len(df_flights)
df_airports.info()
df_airlines.head()
#How many unique origin airports?

n_orig_arp=len(df_flights.ORIGIN_AIRPORT.unique())

#How many unique destination airports?

n_dest_arp=len(df_flights.DESTINATION_AIRPORT.unique())

print("Origin Airports: ", n_orig_arp)

print("Destination Airports: ", n_dest_arp)
#How many flights that have a scheduled departure time later than 18h00?

n_night_flight=len(df_flights.SCHEDULED_DEPARTURE[df_flights.SCHEDULED_DEPARTURE>1800])

print("Night Flights: ", n_night_flight)

print("Night Flights over Total: ", (n_night_flight/len(df_flights))*100, "%")
# How many flights in each month of the year?

import datetime as dt



months = []

for month in range(1, 13):

    months.append(dt.datetime(year=1994, month=month, day=1).strftime("%B"))



fl_per_month = list(df_flights.groupby('MONTH').count().YEAR

)

plt.xlabel('Month')

plt.ylabel('Night Flights')

plt.xticks(range(1,13), months, rotation='vertical')

plt.plot(range(1,13), np.array(fl_per_month), '.-')

plt.show()
flights_dayofweek = (

    df_flights.groupby(df_flights.DAY_OF_WEEK)

    .count()

    ).YEAR
import numpy as np

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

#Global aggregates

days=['monday','tuesday','wednesday','thursday','friday','saturday','sunday']

n_flight_week=list(df_flights.groupby('DAY_OF_WEEK').count().YEAR)

frequencies=[n_flight_week[i] for i in range(len(n_flight_week)) ]

plt.bar(range(0,7),frequencies)

plt.xlabel('Day of week ')

plt.ylabel('number of flights')

plt.xticks(range(0,7),days,rotation='vertical')

plt.show()


figure(num=None, figsize=(17, 6), dpi=80, facecolor='w', edgecolor='k');





n_flight_day_month=list(df_flights.groupby(['MONTH', 'DAY_OF_WEEK']).count().YEAR)

frequencies=[]

frequency=[]

for i in range(1,len(n_flight_day_month)+1):

    

    frequency.append(n_flight_day_month[i-1])

    if (i%7==0):

        frequencies.append(frequency)

        frequency=[]





# data to plot

n_groups = 12





colors = ['b',  'r', 'c',  'y', 'm','g','k']

# create plot

index = np.arange(0, n_groups * 5, 5)

bar_width = 0.55

opacity = 0.8



for i in range(7):

    plt.bar(index+bar_width*i,tuple([row[i] for row in frequencies]),align='edge',width=0.4,

    alpha=opacity,

    color=colors[i],

    label=days[i])



plt.xlabel('Day of week per month')

plt.ylabel('number of flights')

plt.xticks(index + bar_width, ('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'),rotation='vertical')

plt.legend()

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)



plt.figure(figsize=(17,5));

plt.show();
# use a heatmap to better visualize the data

pddf=pd.DataFrame({'count' : df_flights.groupby( ['MONTH', 'DAY_OF_WEEK'] ).size()}).reset_index()

pddf = pddf.pivot("MONTH", "DAY_OF_WEEK", "count")

fig, ax=plt.subplots(figsize=(10,8))

ax = sns.heatmap(pddf, 

                  linewidths=.5,

                  annot=True,

                  vmin=50000,

                  vmax=75000,

                  fmt='d',

                  cmap='Blues', ax=ax)

# set plot's labels

ax.set_xticklabels(days)

ax.set_yticklabels(list(reversed(months)), rotation=0)

ax.set(xlabel='Day of the week', ylabel='Month')

ax.set_title("Number of flights per day of the week for each month", fontweight="bold", size=15)



plt.show()
# How many flights in different days of months and in different hours of days?

# number of flights per day of the month

# create the pandas dataframe

pddf=pd.DataFrame({'count' : df_flights.groupby(df_flights.DAY).size()}).reset_index()



# plot the number of flights per day of the month

f, ax = plt.subplots(figsize=(18, 6))

sns.barplot(x="DAY",

            y="count",

            data=pddf,

            palette=sns.color_palette("GnBu_d", 31),

            ax=ax)



# set plot's labels

ax.set(xlabel='Day of the month', ylabel='Number of flights')

ax.set_title("Number of flights per day of the month during the year", fontweight="bold", size=15)



plt.plot()


pddf=pd.DataFrame({'count' : df_flights.groupby(['MONTH','DAY']).size()}).reset_index()



# use a heatmap to better visualize the data

pddf = pddf.pivot("MONTH", "DAY", "count")

f, ax = plt.subplots(figsize=(18, 6))

sns.heatmap(pddf,

            square=True,

            vmin=11400,

            vmax=15000,

            cmap='Blues')



# set plot's labels

ax.set_yticklabels(list(reversed(months)), rotation=0)

ax.set(xlabel='Day of the month', ylabel='Month')

ax.set_title("Number of flights per day of the month for each month", fontweight="bold", size=15)



plt.show()
# create the pandas dataframe

# number of flights per hour



pddf=pd.DataFrame({'count' : df_flights.groupby(((df_flights.SCHEDULED_DEPARTURE/100).astype(int))).size()}).reset_index()



# plot the number of flights per hour

f, ax = plt.subplots(figsize=(18, 6))

sns.barplot(x="SCHEDULED_DEPARTURE",

            y="count",

            data=pddf,

            palette=sns.color_palette("GnBu_d", 24),

            ax=ax)



# set plot's labels

ax.set(xlabel='Hour of the day', ylabel='Number of flights')

ax.set_title("Number of flights per hour during the year", fontweight="bold", size=15)



plt.plot()
df_flights['HOUR']=(df_flights.SCHEDULED_DEPARTURE/100).astype(int)
# create the pandas dataframe

# number of flights per hour per month



pddf=pd.DataFrame({'count' : df_flights.groupby(['MONTH','HOUR']).size()}).reset_index()



# use a heatmap to better visualize the data

pddf = pddf.pivot("MONTH", "HOUR", "count")

f, ax = plt.subplots(figsize=(18, 6))

sns.heatmap(pddf,

            square=True,

            cmap='Blues')



# set plot's labels

ax.set_yticklabels(list(reversed(months)), rotation=0)

ax.set(xlabel='Hour of the day', ylabel='Month')

ax.set_title("Number of flights per hour for each month", fontweight="bold", size=15)



plt.show() 
# Which are the **top 20** busiest airports? 

# number of outbound flights per airport

df_out_airport = pd.DataFrame({'count' : df_flights.groupby(df_flights.ORIGIN_AIRPORT).size()}).reset_index()

df_out_airport=df_out_airport.rename(columns={"ORIGIN_AIRPORT": "airport"})



# number of inbound flights per airport

df_in_airport = pd.DataFrame({'count' : df_flights.groupby(df_flights.DESTINATION_AIRPORT).size()}).reset_index()

df_in_airport=df_out_airport.rename(columns={"DESTINATION_AIRPORT": "airport"})

# number of flights per airport

df_airport=pd.DataFrame( pd.concat([df_out_airport,df_in_airport],ignore_index=True).groupby('airport').sum()).reset_index()

print("Top 20 busiest airports (outbound)")

print(df_out_airport.sort_values('count',ascending=False).head(20))



print("Top 20 busiest airports (inbound)")

print(df_in_airport.sort_values('count',ascending=False).head(20))



print("Top 20 busiest airports")

print(df_airport.sort_values('count',ascending=False).head(20))
# What is the percentage of delayed flights for different hours of the day?

# number of delayed flights per hour

df_flights_hour_delay = pd.DataFrame({'count' : df_flights[df_flights.ARRIVAL_DELAY > 15].groupby('HOUR').size()}).reset_index()



# number of flights per hour

df_flights_hour = pd.DataFrame({'count' : df_flights.groupby('HOUR').size()}).reset_index()





# percentage of flight in delay per hour

df=df_flights_hour.join(df_flights_hour_delay,on='HOUR',rsuffix='_d', how='inner')

df['percentage']=df['count_d']*100/df['count']

percentage_hour_delay = df[['HOUR','percentage']]
# create the pandas dataframe

# plot the percentage of flights in delay per hour

f, ax = plt.subplots(figsize=(12, 6))

sns.barplot(x="HOUR",

            y="percentage",

            data=percentage_hour_delay,

            palette=sns.color_palette("GnBu_d", 24),

            ax=ax)



# set plot's labels

ax.set(xlabel='Hour of the day', ylabel='Percentage of delayed flights')

ax.set_title("Percentage of delayed flights for different hours of the day", fontweight="bold", size=15)



plt.plot()
df = pd.DataFrame({'A': [1, 1, 2, 1, 2],

                    'B': [np.nan, 2, 3, 4, 5],

                    'C': [1, 2, 1, 1, 2]}, columns=['A', 'B', 'C'])
# create the pandas dataframe

# average delay per hour

hour_avg_delay = pd.DataFrame(data=df_flights.groupby('HOUR')['ARRIVAL_DELAY'].mean()).reset_index()



# plot the avg number of flights in delay per hour

f, ax = plt.subplots(figsize=(12, 6))

sns.barplot(x="HOUR",

            y="ARRIVAL_DELAY",

            data=hour_avg_delay,

            palette=sns.color_palette("GnBu_d", 24),

            ax=ax)



# set plot's labels

ax.set(xlabel='Hour of the day', ylabel='Average Delay')

ax.set_title("Average delay of the flights during the day", fontweight="bold", size=15)



plt.show()
import matplotlib.patches as mpatches

pdf_delay_ratio_per_hour = percentage_hour_delay

pdf_mean_delay_per_hour = hour_avg_delay

plt.xlabel("Hours")

plt.ylabel("Ratio of delay")

plt.title('The radio of delay over hours in day')

plt.grid(True,which="both",ls="-")

bars = plt.bar(pdf_delay_ratio_per_hour['HOUR'], pdf_delay_ratio_per_hour['percentage'], align='center', edgecolor = "black")

for i in range(0, len(bars)):

    color = 'red'

    if pdf_mean_delay_per_hour['ARRIVAL_DELAY'][i] < 0:

        color = 'lightgreen'

    elif pdf_mean_delay_per_hour['ARRIVAL_DELAY'][i] < 2:

        color = 'green'

    elif pdf_mean_delay_per_hour['ARRIVAL_DELAY'][i] < 4:

        color = 'yellow'

    elif pdf_mean_delay_per_hour['ARRIVAL_DELAY'][i] < 8:

        color = 'orange'



    bars[i].set_color(color)

        

patch1 = mpatches.Patch(color='lightgreen', label='Depart earlier')

patch2 = mpatches.Patch(color='green', label='delay < 2 minutes')

patch3 = mpatches.Patch(color='yellow', label='delay < 4 minutes')

patch4 = mpatches.Patch(color='orange', label='delay < 8 minutes')

patch5 = mpatches.Patch(color='red', label='delay >= 8 minutes')



plt.legend(handles=[patch1, patch2, patch3, patch4, patch5], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.margins(0.05, 0)

plt.show()
df_flights.columns
# create the pandas dataframe

df_daymonth_delayed =  pd.DataFrame({'count' : df_flights[df_flights.ARRIVAL_DELAY > 15].groupby('DAY').size()}).reset_index()

df_daymonth= pd.DataFrame({'count' : df_flights.groupby('DAY').size()}).reset_index()



df_daymonth_delayed['percentage']=df_daymonth_delayed['count']*100/df_daymonth['count']





# plot the number of delayed flights per day of the month

f, ax = plt.subplots(figsize=(12, 6))

sns.barplot(x="DAY",

            y="percentage",

            data=df_daymonth_delayed,

            palette=sns.color_palette("GnBu_d", 31),

            ax=ax)



# set plot's labels

ax.set(xlabel='Day of the month', ylabel='Percentage of delayed flight')

ax.set_title("Percentage of delayed flights over the days of the month", fontweight="bold", size=15)



plt.plot()
# create the pandas dataframe

df_dayweek_delayed =  pd.DataFrame({'count' : df_flights[df_flights.ARRIVAL_DELAY > 15].groupby('DAY_OF_WEEK').size()}).reset_index()

df_dayweek= pd.DataFrame({'count' : df_flights.groupby('DAY_OF_WEEK').size()}).reset_index()



df_dayweek_delayed['percentage']=df_dayweek_delayed['count']*100/df_dayweek['count']





# plot the number of delayed flights per day of the month

f, ax = plt.subplots(figsize=(12, 6))

sns.barplot(x="DAY_OF_WEEK",

            y="percentage",

            data=df_dayweek_delayed,

            palette=sns.color_palette("GnBu_d", 7),

            ax=ax)



# set plot's labels

ax.set(xlabel='Day of the week', ylabel='Percentage of delayed flight')

ax.set_xticklabels(days)

ax.set_title("Percentage of delayed flights over the days of the week", fontweight="bold", size=15)



plt.plot()
# create the pandas dataframe

df_month_delayed =  pd.DataFrame({'count' : df_flights[df_flights.ARRIVAL_DELAY > 15].groupby('MONTH').size()}).reset_index()

df_month= pd.DataFrame({'count' : df_flights.groupby('MONTH').size()}).reset_index()



df_month_delayed['percentage']=df_month_delayed['count']*100/df_month['count']







# plot the number of delayed flights per month

f, ax = plt.subplots(figsize=(12, 6))

sns.barplot(x="MONTH",

            y="percentage",

            data=df_month_delayed,

            palette=sns.color_palette("GnBu_d", 12),

            ax=ax)



# set plot's labels

ax.set(xlabel='Month', ylabel='Percentage of delayed flight')

ax.set_xticklabels(months)

ax.set_title("Percentage of delayed flights over the months of the year", fontweight="bold", size=15)



plt.plot()
# What is the delay probability for the top 20 busiest airports?

# top 20 busiest airports

df_busiest_airports = df_airport.sort_values('count',ascending=False).head(20).reset_index()

df_busiest_airports=df_busiest_airports.drop('index',axis=1)



# number of delayed flights in the top 20 busiest airport

df_delays_busiest_src_airports=pd.DataFrame({'count' : df_flights[(df_flights.ORIGIN_AIRPORT.isin([df_busiest_airports.iloc[i][0] for i in range(len(df_busiest_airports))])) & (df_flights.ARRIVAL_DELAY > 15)].groupby('ORIGIN_AIRPORT').size()}).reset_index()

df_delays_busiest_dest_airports =pd.DataFrame({'count' : df_flights[(df_flights.DESTINATION_AIRPORT.isin([df_busiest_airports.iloc[i][0] for i in range(len(df_busiest_airports))])) & (df_flights.ARRIVAL_DELAY > 15)].groupby('DESTINATION_AIRPORT').size()}).reset_index()

df_delays_busiest_src_airports=df_delays_busiest_src_airports.rename(columns={"ORIGIN_AIRPORT":"airport"})



# delay probability per source

df_prob_delay_busiest_src=df_busiest_airports.merge(df_delays_busiest_src_airports, on='airport',how='inner')

df_prob_delay_busiest_src['probability']=df_prob_delay_busiest_src['count_y']/df_prob_delay_busiest_src['count_x']

# delay probability per destination

df_delays_busiest_dest_airports=df_delays_busiest_dest_airports.rename(columns={"DESTINATION_AIRPORT":"airport"})



df_prob_delay_busiest_dest=df_busiest_airports.merge(df_delays_busiest_dest_airports, on='airport',how='inner')

df_prob_delay_busiest_dest['probability']=df_prob_delay_busiest_dest['count_y']/df_prob_delay_busiest_dest['count_x']



# delay propability per source and destination

prob_delay_busiest_any=df_prob_delay_busiest_src.merge(df_prob_delay_busiest_dest,on='airport',how='inner')

prob_delay_busiest_any=prob_delay_busiest_any.drop(['count_x_x','count_y_x','count_x_y','count_y_y'],axis=1)

prob_delay_busiest_any['probability']=prob_delay_busiest_any['probability_x']+prob_delay_busiest_any['probability_y']

prob_delay_busiest_any=prob_delay_busiest_any.drop(['probability_x','probability_y'],axis=1)
# plot 

f, ax = plt.subplots(figsize=(12, 6))

sns.barplot(x="airport",

            y="probability",

            data=prob_delay_busiest_any,

            palette=sns.color_palette("GnBu_d", 20),

            ax=ax)

ax.set(ylabel="All", xlabel="Airport")

ax.set(ylabel="Delay probability", xlabel="Airport")

ax.set_title("Delay probability of the top 20 busiest airports", fontweight="bold", size=15)



plt.plot()