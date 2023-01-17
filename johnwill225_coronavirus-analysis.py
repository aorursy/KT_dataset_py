import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.gridspec as gridspec
dataset = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")
dataset.head()
dataset['Date'] = pd.to_datetime(dataset['Date'])

dataset['Last Update'] = pd.to_datetime(dataset['Last Update'])
dataset['Day'] = dataset['Date'].dt.day

dataset['Month'] = dataset['Date'].dt.month

dataset['Week'] = dataset['Date'].dt.week

dataset['WeekDay'] = dataset['Date'].dt.weekday
# Layout Customization

displayed_cols = ['Confirmed','Deaths','Recovered']

def multi_plot():

    fig = plt.figure(constrained_layout=True, figsize=(15,8))

    grid = gridspec.GridSpec(ncols=4, nrows=2, figure=fig)



    ax1 = fig.add_subplot(grid[0, :2])

    ax1.set_title('Daily Reports')

    dataset.groupby(['Date']).sum()[displayed_cols].plot(ax=ax1)



    ax2 = fig.add_subplot(grid[1, :2])

    ax2.set_title('Monthly Reports')

    dataset.groupby(['Month']).sum()[displayed_cols].plot(kind='bar',ax=ax2)

    ax2.set_xticklabels(range(1,3))



    ax3= fig.add_subplot(grid[0, 2:])

    ax3.set_title('Weekly Reports')

    weekdays = dataset.groupby('Week').nth(-1)['Date']

    dataset[dataset['Date'].isin(weekdays)].groupby('Date')[displayed_cols].sum().plot(kind='bar',ax=ax3)

    ax3.set_xticklabels(range(1,len(weekdays)+1))



    ax4 = fig.add_subplot(grid[1, 2:])

    ax4.set_title('WeekDays Reports')

    dataset.groupby(['WeekDay']).sum()[displayed_cols].plot(ax=ax4)

    plt.show()

multi_plot()
recent_date = dataset['Date'].iloc[-1]

last_updated = dataset[dataset['Date'].dt.date == recent_date]
# Reports given for the total number of days

dataset['Date'].max() - dataset['Date'].min()
pd.DataFrame(dataset['Country'].value_counts()).style.set_table_styles(

[{'selector': 'tr:nth-of-type(-n+5)',

  'props': [('background', '#FFA500')]}, 

 {'selector': 'th',

  'props': [('background', '#606060'), 

            ('color', 'white'),

            ('font-family', 'verdana')]}

]

)
dataset['Country'].replace({'Mainland China':'China'},inplace=True)

last_updated['Country'].replace({'Mainland China':'China'},inplace=True)
#removing the zero confirmed case rows

zeroConfirmed = dataset[dataset['Confirmed'] == 0]

dataset = dataset[dataset['Confirmed'] != 0]
dataset.head()
dataset[dataset['Date'] != dataset['Last Update']]['Country'].value_counts()
dataset['Last Update'].max()
# missing values

dataset.isnull().sum()
confirmedCase = int(last_updated['Confirmed'].sum())

deathCase = int(last_updated['Deaths'].sum())

recoveredCase = int(last_updated['Recovered'].sum())

print("No of Confirmed cases globally {}".format(confirmedCase))

print("No of Recovered case globally {}".format(recoveredCase))

print("No of Death case globally {}".format(deathCase))
others = dataset[dataset['Country']=='Others']

dataset = dataset[dataset['Country']!='Others']

last_updated =last_updated[last_updated['Country']!='Others']
plt.figure(figsize=(15,6))

plt.title('Number of Province/State were affected in Each Country')

plt.xticks(rotation=90)

prv_lst = dataset.groupby(['Country'])['Province/State'].nunique().sort_values(ascending=False)

prv_lst.plot(kind='bar')

plt.tight_layout()
prv_lst.tail()
top5 =  last_updated.groupby(['Country']).sum().nlargest(5,['Confirmed'])[displayed_cols]

top5

print("Top 5 Countries were affected most")

print(top5)
plt.figure(figsize=(12,6))

plt.xticks(rotation=90)

plt.title("Top most 5 countries were affected by Coronavirus")

sns.barplot(x=top5.index,y='Confirmed',data=top5)

plt.show()
plt.figure(figsize=(15,6))

plt.title('Countries which has Confirmed cases')

plt.xticks(rotation=90)

sns.barplot(x='Country',y='Confirmed',data=last_updated)

plt.tight_layout()
plt.figure(figsize=(15,6))

plt.title('Province/State which reported more than 1000 Confirmed case')

plt.xticks(rotation=90)

prvinc = last_updated

prvincConfirmed = prvinc[prvinc['Confirmed']>100]

sns.barplot(data=prvincConfirmed, x='Province/State', y='Confirmed')
prvincConfirmed['Country'].value_counts()
plt.figure(figsize=(15,6))

plt.title('Province/State which reported Confirmed case between 100 to 1000')

plt.xticks(rotation=90)

prvinc = last_updated

prvincConfirmed = prvinc[(prvinc['Confirmed']>100)&(prvinc['Confirmed']<1000)]

sns.barplot(data=prvincConfirmed, x='Province/State', y='Confirmed')
prvincConfirmed['Country'].value_counts()
dataset.groupby(['Country']).nunique()
plt.figure(figsize=(15,6))

plt.xticks(rotation=90)

plt.title('Province/State has reported Deaths case')

sns.barplot(data=last_updated, x='Province/State', y='Deaths')
last_updated.groupby(['Country']).sum().nlargest(5,['Deaths'])['Deaths']
last_updated[last_updated['Country']!='China'].groupby(['Country']).sum()[displayed_cols].nlargest(5,['Confirmed'])
cruiseShip = others
cruiseShip['Confirmed'].plot()
print("{}% of people were affected in Cruise ship".format(round((cruiseShip.iloc[-1].Confirmed/3711 ) * 100,2)))