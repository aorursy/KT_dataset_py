import pandas as pd

import matplotlib.pyplot as plt

import datetime

#import gmaps

#import gmaps.datasets



#gmaps.configure(api_key="AI<your keys>")



dateparse = lambda x: datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')



data = pd.read_csv('../input/911.csv' ,header=0,names=['lat', 'lng','desc','zip','title','timeStamp','twp','addr','e'],

    dtype={'lat':float,'lng':float,'desc':str,'zip':str,

                  'title':str,'timeStamp':str,'twp':str,'addr':str,'e':int}, 

     parse_dates=['timeStamp'],date_parser=dateparse)



# Set index

data.index = pd.DatetimeIndex(data.timeStamp)

data=data[(data.timeStamp >= "2016-01-01 00:00:00")]



data.head()
# 1.Analyze the kind of 911 calls



totalSize = data.size



fireIncidents =data[data['title'].str.contains('^Fire:', na = 'NA')].size

emsIncidents = data[data['title'].str.contains('^EMS:', na = 'NA')].size

trafficIncidents = data[data['title'].str.contains('^Traffic:', na = 'NA')].size









labels = 'Fire','EMS','Traffic'

sizes = [fireIncidents,emsIncidents,trafficIncidents ]

colors = ['yellowgreen', 'mediumpurple', 'lightskyblue'] 

explode = (0, 0.1, 0)    # proportion with which to offset each wedge



plt.pie(sizes,              # data

        explode=explode,    # offset parameters 

        labels=labels,      # slice labels

        colors=colors,      # array of colours

        autopct='%1.1f%%',  # print the values inside the wedges

        startangle=70       # starting angle

        )

plt.axis('equal')



plt.show()



#maximum kind is of EMS type
# 2. Analyze the locations of incidents

#data.twp.unique().size



groupedByCity = data.groupby('twp',as_index = False).sum()

top10Townships = groupedByCity.sort_values('e',ascending=False).head(10)

top10Townships





#data[(data.twp == 'BERKS COUNTY')].size



print(data['timeStamp'].max())

data['month'] = data['timeStamp'].map(lambda x: x.month)





groupByMonth = data.groupby('month',as_index = False).sum()





y = groupByMonth['e'].values

labels  = ['Jan','Feb','Mar','Apr','May','June','July','Aug','Sep','Oct','Nov']

x = groupByMonth['month'].values

width = 1/1.5

plt.bar(x, y, width, color="blue",align='center')

plt.title('911 Calls each month')

plt.xticks(x, labels)

plt.show()
data['hour'] = data['timeStamp'].map(lambda x: x.hour)



groupByMonthDay = data[(data['hour'] >= 5) & (data['hour'] <= 17)].groupby('month',as_index = False).sum()



yy = groupByMonthDay['e'].values

labels  = ['Jan','Feb','Mar','Apr','May','June','July','Aug','Sep','Oct','Nov']

xx = groupByMonthDay['month'].values

width = 1/1.5

plt.bar(xx, yy, width, color="blue",align='center')

plt.title('911 Calls each month 5 am to 5 pm')

plt.xticks(xx, labels)

plt.show()



#type(data['timeStamp'])




groupByMonthNight = data[(data['hour'] > 17) | (data['hour'] < 5)].groupby('month',as_index = False).sum()



groupByMonthNight.head()



y = groupByMonthNight['e'].values

labels  = ['Jan','Feb','Mar','Apr','May','June','July','Aug','Sep','Oct','Nov']

x = groupByMonthNight['month'].values

width = 1/1.5

plt.bar(x, y, width, color="blue",align='center')

plt.title('911 Calls each month 5 pm to 5 am')

plt.xticks(x, labels)

plt.show()
ems = data[data['title'].str.contains('^EMS:', na = 'NA')]





ems = ems[['lat','lng']].values.tolist()





m = gmaps.Map()

m.add_layer(gmaps.Heatmap(data=ems))

m

traffic = data[data['title'].str.contains('^Traffic:', na = 'NA')]





traffic = traffic[['lat','lng']].values.tolist()





m = gmaps.Map()

m.add_layer(gmaps.Heatmap(data=traffic))

m
fire = data[data['title'].str.contains('^Fire:', na = 'NA')]





fire = fire[['lat','lng']].values.tolist()





m = gmaps.Map()

m.add_layer(gmaps.Heatmap(data=fire))

m
