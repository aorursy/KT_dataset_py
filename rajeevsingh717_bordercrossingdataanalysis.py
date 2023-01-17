import pandas as pd

import matplotlib.pyplot as plt

import os

import matplotlib.ticker as tick

%matplotlib inline
import pandas as pd

df = pd.read_csv("../input/border-crossing-entry-data/Border_Crossing_Entry_Data.csv")

df.head()
measure_count = df[['Measure','Value']].groupby('Measure').sum().reset_index()

count_per_state = df[['State','Value']].groupby('State').sum().reset_index()

count_per_state.head()
### converts  tick values in Bills/ Mills  - This function is borrowed from internet :)

    

def reformat_large_tick_values(tick_val, pos):

    

    if tick_val >= 1000000000:

        val = round(tick_val/1000000000, 1)

        new_tick_format = '{:}B'.format(val)

    elif tick_val >= 1000000:

        val = round(tick_val/1000000, 1)

        new_tick_format = '{:}M'.format(val)

    elif tick_val >= 1000:

        val = round(tick_val/1000, 1)

        new_tick_format = '{:}K'.format(val)

    elif tick_val < 1000:

        new_tick_format = round(tick_val, 1)

    else:

        new_tick_format = tick_val



    # convert into string value

    new_tick_format = str(new_tick_format)



    # code below will keep 4.5M as is but change values such as 4.0M to 4M since that zero after the decimal isn't needed

    index_of_decimal = new_tick_format.find(".")



    if index_of_decimal != -1:

        value_after_decimal = new_tick_format[index_of_decimal+1]

        if value_after_decimal == "0":

            # remove the 0 after the decimal point since it's not needed

            new_tick_format = new_tick_format[0:index_of_decimal] + new_tick_format[index_of_decimal+2:]



    return new_tick_format
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,5))

my_colors = 'rgbkymc'

xticklabel = measure_count['Measure']

ax1.bar(measure_count['Measure'],measure_count['Value']/10000, color=['black', 'red', 'green', 'blue', 'cyan'])

ax1.set_xlabel('Measuer')

ax1.set_ylabel('Number of Inbound * 10000')

ax1.set_xticklabels(xticklabel,rotation=90)



xticklabel = count_per_state['State']

ax2.bar(count_per_state['State'],count_per_state['Value']/10000, color=['black', 'red', 'green', 'blue', 'cyan'])

ax2.set_xlabel('State')

ax2.set_ylabel('Number of Inbound*10000')

ax2.set_xticklabels(xticklabel,rotation=90)







plt.show()
##### Aggregate counts based on Port - top 20

port_value_count=df[['Port Name','Value']].groupby('Port Name').sum().reset_index().sort_values(by='Value', ascending=False).head(20)

port_name = port_value_count['Port Name']

ax = port_value_count.plot.bar(figsize=(20,6), title = 'Count as per Port', legend=False, color='lightgreen' )

ax = plt.gca()

ax.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values));  ## Chaging the value format to Millions/Billions for better understanding

ax.set_xticklabels(port_name, rotation=75)

plt.show()
df[['Border','Value']].groupby('Border').sum().reset_index().sort_values(by='Value', ascending=False)
# df.head()

state_port_count = df[['State','Port Name','Value']].groupby(['State','Port Name']).sum().reset_index()

top_4_state_with_max_val = count_per_state.sort_values(by='Value', ascending=False).head(4)['State']

color = ['red', 'green', 'blue', 'cyan']

fig ,ax = plt.subplots(1,4, figsize=(20,5))

counter = 0

for item in top_4_state_with_max_val:

    tempdf = state_port_count[state_port_count['State'] == item]

    ax[counter].bar(tempdf['Port Name'],tempdf['Value'],color=color[counter])

    ax[counter].set_title(item)

    ax[counter].yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values));

    ax[counter].set_xticklabels(tempdf['Port Name'],rotation=90)

    counter = counter+1



plt.show()
### Creating dictionary for Measure Value - will use this one in pie chart as lable

counter =0

Measure_items = {}

for item in df['Measure'].unique():

    Measure_items.update({item:counter})

    counter =counter+1



df["Measure_code"] = df["Measure"].map(Measure_items)

print(Measure_items)
state_measure_count = df[['State','Measure_code','Value']].groupby(['State','Measure_code']).sum().reset_index()

#state_measure_count.head()

top_2_states= ['Texas','Arizona']



## Pick all the value above 20 Million

state_measure_count = state_measure_count[state_measure_count['Value']>20000000].sort_values(by='Value', ascending=False)



color = ['red', 'green']

fig ,ax = plt.subplots(1,2, figsize=(20,5))

counter = 0

print(Measure_items)

for item in top_2_states:

            

    tempdf = state_measure_count[state_measure_count['State'] == item]

    total = sum(state_measure_count[state_measure_count['State'] == item]['Value'])

    ax[counter].set_title(item)

    ax[counter].pie(tempdf['Value'],labels=tempdf['Measure_code'], startangle=90, autopct='%.1f%%', radius=1.5)

    ax[counter].legend(loc='upper right',  bbox_to_anchor=(0.0, 1) , labels=['%s, %1.1f%%' % 

                                                                             (l,(float(s) / total) * 100) 

                                                                             for l, s in zip(tempdf['Measure_code'], tempdf['Value'])])

    counter = counter+1

   

plt.show()
### Analysis on the Date Column - 

import datetime as dt

df['Date'] = df['Date'].apply(lambda x : dt.datetime.strptime(x,'%m/%d/%Y %H:%M:%S %p')) ## only to run the for the first time when chaging the str in timeformat

df['year'] = df['Date'].dt.year

df['month'] = df['Date'].dt.month

df['day'] = df['Date'].dt.day
year_count = df[['year','Value']].groupby('year').sum().reset_index()

year_count = year_count.sort_values(by='Value', ascending=False)

fig, ax = plt.subplots(figsize=(15,5))

ax.bar(year_count['year'],year_count['Value'], color=['pink','yellow','green'])

ax.set_title('Number of people per year')

ax.set_xlabel('year')

ax.set_xticks(year_count['year'])

ax.set_ylabel('Number of people')

ax.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values));

plt.show()
### which Month people travel a lot - Taking mean of all the months

per_month_mean = df[['month','Value']].groupby('month').mean().reset_index()

fig, ax = plt.subplots()

ax.pie(per_month_mean['Value'],labels=per_month_mean['month'], startangle=90, autopct='%.1f%%', radius=2)

plt.show()

### People cross border more on July, August
## Find the 30 Day average - Created a time series data frame from existing dataframe

days_average = df[['Date','Value']].set_index('Date')

days_30_average = days_average['Value'].rolling(30).mean()

days_60_average = days_average['Value'].rolling(60).mean()

days_120_average = days_average['Value'].rolling(120).mean()



fig, ax = plt.subplots(figsize=(20,7))

ax.plot(days_30_average.index , days_30_average, label='30 Days rolling average')

ax.plot(days_60_average.index , days_60_average, label='60 Days rolling average')

ax.plot(days_120_average.index , days_120_average, label='120 Days rolling average')

ax.set_xlabel('Date')

ax.set_ylabel('Value')

ax.set_title('30/60/120 Days rolling average')

ax.legend()

plt.show()


