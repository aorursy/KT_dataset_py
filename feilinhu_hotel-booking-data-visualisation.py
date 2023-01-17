import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

htl = pd.read_csv("/kaggle/input/hotel-booking-demand/hotel_bookings.csv")

pd.set_option('display.max_columns', None)

htl.head()
htl.columns
# a general overview of data



htl.describe(include="all").T
# How many numeric variables are actually more of categorical variables?



describe = []

for col in htl.columns:

    describe.append(len(htl[col].value_counts()))

describe
#form a description about all attributes we have, including col_name,data type,NA count and unique value count.



describe = pd.DataFrame(list(zip(htl.columns,htl.dtypes,htl.isnull().sum(),htl.describe(include="all").T['unique'],describe)),columns = ['col','type','NA','Ucount','count'])

describe

#Turn the following int variables to categoricals. 10 is more of an experimental number. We can change later based on the data we got after we get a deeper understanding about it.



to_cate = describe[describe['Ucount'].isna()][describe['count']<10]

to_cate
# Do the data manipulation on a copy of original data.



import copy

data = copy.deepcopy(htl)
# Creating new variables for better visualization



data['is_children'] = ['Y' if x > 0 else 'N' for x in data['children']]

data['is_baby'] = ['Y' if x > 0 else 'N' for x in data['babies']]

data['is_agent'] = ['Y' if x > 0 else 'N' for x in data['agent']]

data['is_company'] = ['Y' if x > 0 else 'N' for x in data['company']]

data['is_parking'] = ['Y' if x > 0 else 'N' for x in data['required_car_parking_spaces']]

data['is_request'] = ['Y' if x > 0 else 'N' for x in data['total_of_special_requests']]

data['is_canceled_before'] = ['Y' if x > 0 else 'N' for x in data['previous_cancellations']]

data['is_changed'] = ['Y' if x > 0 else 'N' for x in data['booking_changes']]

data['is_waited'] = ['Y' if x > 0 else 'N' for x in data['days_in_waiting_list']]

data['is_room_changed'] = data[['reserved_room_type','assigned_room_type']].apply(lambda x:x['reserved_room_type'] != x['assigned_room_type'], axis=1)



# Change data type

for i in to_cate['col']:

    data[i] = data[i].astype('str')

    

data['arrival_date_day_of_month'] = data['arrival_date_day_of_month'].astype('str')

data['is_room_changed'] = data['is_room_changed'].astype('str')



# And set different data types apart

cate = [var for var in data.columns if data[var].dtypes == 'object']

num = [var for var in data.columns if data[var].dtypes != 'object']
# Assign 0 to a negative Price data



data.loc[14969,'adr'] = 0
# Now we have 30 categorical variables



cate,len(cate)
# And 12 numeric variables



num,len(num)
# correlation for numeric variables



correlation = data.corr(method='pearson')

correlation[correlation >0.3]
# Distribution plots for all numeric variables.



from scipy import stats

fig = plt.figure()

#plt.figure(figsize=(120,60))

i = 0

for col in num:

    try:

        ax = fig.add_subplot(3,4,i+1)

        sns.distplot(data[col], kde=True, fit=stats.norm)

    except ValueError:

        print(str(i)+ ' '+ 'ValueError')

    except RuntimeError:

        print(str(i)+ ' '+ 'RuntimeError')

    i = i+1

fig.set_size_inches(24, 24)

# We get some Error Messages from the kde plot.Print the name and value_counts of the following attributes to see the cause.



num[4],num[5],num[6],num[7],num[10]
print(num[4] , ': ' , data.adults.value_counts().sort_index())

print('===========================================')

print(num[5] , ': ' , data.previous_cancellations.value_counts().sort_index())

print('===========================================')

print(num[6] , ': ' , data.previous_bookings_not_canceled.value_counts().sort_index())

print('===========================================')

print(num[7] , ': ' , data.booking_changes.value_counts().sort_index())

print('===========================================')

print(num[10] , ': ' , data.days_in_waiting_list.value_counts())
# plots for all numeric variables to Y



fig = plt.figure()

plt.figure(figsize=(120,60))

i = 1

for col in num:

    ax = fig.add_subplot(3,4,i)

    data.groupby(['is_canceled'])[col].mean().plot(kind = 'bar', ax = ax).set_title(col) 

    i = i+1

fig.set_size_inches(24, 24)

# Boxplots for all numeric variables to Y



fig = plt.figure()

plt.figure(figsize=(120,60))

i = 1

for col in num:

    ax = fig.add_subplot(3,4,i)

    sns.boxplot(x = 'is_canceled', y = col, data = data, ax = ax).set_title(col) 

    i = i+1

fig.set_size_inches(24, 24)
#To better check the longtail data:



fig = plt.figure()

fig.set_size_inches(12, 12)

fig.add_subplot(2,2,1)

sns.boxplot(x = 'is_canceled', y = 'stays_in_weekend_nights', data = data[data['stays_in_weekend_nights']<5]).set_title('stays_in_weekend_nights') 

fig.add_subplot(2,2,2)

sns.boxplot(x = 'is_canceled', y = 'stays_in_week_nights', data = data[data['stays_in_weekend_nights']<7]).set_title('stays_in_week_nights') 

fig.add_subplot(2,2,3)

sns.boxplot(x = 'is_canceled', y = 'adults', data = data[data['adults']<5]).set_title('adults') 

fig.add_subplot(2,2,4)

sns.boxplot(x = 'is_canceled', y = 'adr', data = data[data['adr']<5000]).set_title('adr') 
# plots for all categorical variables



fig = plt.figure()

#plt.figure(figsize=(140,60))

i = 0

for col in cate:

    if col == 'country' or col == 'reservation_status_date' or col == 'arrival_date_day_of_month' or col == 'arrival_date_month' or col == 'arrival_date_year':

        pass

    else:

        ax = fig.add_subplot(5,5,i+1)

        data[col].value_counts().plot(kind = 'bar', ax = ax).set_title(col)

        i = i+1

fig.set_size_inches(30, 30)


def stack2dim(raw, i, j, rotation = 0, location = 'upper left'):



#    plt.figure(figsize = (15, 10))

    import math

    data_raw = pd.crosstab(raw[i], raw[j])

    data = data_raw.div(data_raw.sum(1), axis=0)  

    

    createVar = locals()

    x = [0] 

    width = [] 

    k = 0

    for n in range(len(data)):

        

        createVar['width' + str(n)] = data_raw.sum(axis=1)[n] / sum(data_raw.sum(axis=1))

        width.append(createVar['width' + str(n)])  

        if n == 0:

            continue

        else:

            k += createVar['width' + str(n - 1)] / 2 + createVar['width' + str(n)] / 2 + 0.05

            x.append(k)  

    

    y_mat = []

    n = 0

    for p in range(data.shape[0]):

        for q in range(data.shape[1]):

            n += 1

            y_mat.append(data.iloc[p, q])

            if n == data.shape[0] * 2:

                break

            elif n % 2 == 1:

                y_mat.extend([0] * (len(data) - 1))

            elif n % 2 == 0:

                y_mat.extend([0] * len(data))



    y_mat = np.array(y_mat).reshape(len(data) * 2, len(data))

    y_mat = pd.DataFrame(y_mat) 

    

    createVar = locals()

    for row in range(len(y_mat)):

        createVar['a' + str(row)] = y_mat.iloc[row, :]

        if row % 2 == 0:

            if math.floor(row / 2) == 0:

                label = data.columns.name + ': ' + str(data.columns[row])

                plt.bar(x, createVar['a' + str(row)],

                        width=width[math.floor(row / 2)], label='0', color='#5F9EA0')

            else:

                plt.bar(x, createVar['a' + str(row)],

                        width=width[math.floor(row / 2)], color='#5F9EA0')

        elif row % 2 == 1:

            if math.floor(row / 2) == 0:

                label = data.columns.name + ': ' + str(data.columns[row])

                plt.bar(x, createVar['a' + str(row)], bottom=createVar['a' + str(row - 1)],

                        width=width[math.floor(row / 2)], label='1', color='#8FBC8F')

            else:

                plt.bar(x, createVar['a' + str(row)], bottom=createVar['a' + str(row - 1)],

                        width=width[math.floor(row / 2)], color='#8FBC8F')



    plt.title(j + ' vs ' + i)

#    group_labels = [data.index.name + ': ' + str(name) for name in data.index]

    group_labels = [str(name) for name in data.index]

    plt.xticks(x, group_labels, rotation = rotation)

    plt.ylabel(j)

    plt.legend(shadow=True, loc=location)

    plt.show()

# Plots for all categorical variables to Y



fig = plt.figure()

#plt.figure(figsize=(140,60))

for col in cate:

    if col == 'country' or col == 'reservation_status_date' or col == 'arrival_date_day_of_month' or col == 'arrival_date_month' or col == 'arrival_date_year':

        pass

    else:

        stack2dim(data, i=col, j="is_canceled", rotation = 40)

fig.set_size_inches(30, 30)

# Time series. It contains 3 years of data and we wonder if each year/month has some differences or strange outliers.



data['date'] = data['arrival_date_year'] + '-' + data['arrival_date_month'] + '-' + data['arrival_date_day_of_month']

data['date'] = pd.to_datetime(data['date'],  errors = 'coerce')

import datetime as dt

data['month'] = data['date'].dt.month

data['year'] = data['date'].dt.year

data.month.value_counts().sort_index().plot(kind="bar")
# What's the difference between city hotel and resort hotel?



pd.DataFrame(zip(list(data[data['hotel'] == 'City Hotel'].month.value_counts().sort_index()),list(data[data['hotel'] == 'Resort Hotel'].month.value_counts().sort_index())),

            columns=['City Hotel', 'Resort Hotel'], index = [1,2,3,4,5,6,7,8,9,10,11,12]).plot(kind="bar")
data.year.value_counts().sort_index().plot(kind="bar")
cross_table = pd.crosstab(data['month'],data['is_canceled'])

cross_table.div(cross_table.sum(1), axis = 0).plot(kind = 'bar', stacked = True)
cross_table_year = pd.crosstab(data['year'],data['is_canceled'])

cross_table_year.div(cross_table_year.sum(1), axis = 0).plot(kind = 'bar', stacked = True)
# assigned vs. reserved room and its cancelation rate. We want to check if the cancelation is related to the booking change caused by hotels.



roomtype = data.groupby(['reserved_room_type','assigned_room_type','is_canceled'], as_index=False)['adr'].agg(['count', 'mean']).reset_index()

roomtype.T
fig = plt.figure(figsize=(30,10))

sns.boxplot(x = 'reserved_room_type', y = 'adr' , hue="assigned_room_type",  data = data[data['is_canceled']== '1'][data['adr']<4000]).set_title('Canceled')

#fig.set_size_inches(100, 60)
sns.boxplot(x = 'reserved_room_type', y = 'adr' ,  data = data[data['adr']<4000])
sns.boxplot(x = 'assigned_room_type', y = 'adr' ,  data = data[data['adr']<4000])
# geo info. We want to check if our data is a global data.



def percConvert(ser):

    return ser/float(ser[-1])

cross_table_cty = pd.crosstab(data['country'],data['is_canceled'], margins=True)

cross_table_cty_p = cross_table_cty.apply(percConvert, axis=1)



cross_table_cty['cancel_rate']= cross_table_cty_p['1']

cross_table_cty.sort_values('cancel_rate',ascending=False).T
cross_table_cty = cross_table_cty.reset_index()

cross_table_cty[:-1].sort_values('All',ascending = False).T
import folium

from folium.features import GeoJson, GeoJsonTooltip



# Initialize the map:

m = folium.Map(zoom_start=1)

url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data'

country_geo = f'{url}/world-countries.json'



# Add the color for the chloropleth:

m.choropleth(

 geo_data=country_geo,

 name='choropleth',

 data=cross_table_cty[:-1],

 columns=['country', 'All'],

 key_on='feature.id',

 threshold_scale=[0, 50, 100, 1000, 10000,48591],

 fill_color='YlGn',

 fill_opacity=0.7,

 line_opacity=0.2,

 legend_name='Country Reservation:color range from [0, 50, 100, 1000, 10000, 48591]'

)

folium.LayerControl().add_to(m)

m
# And finally just for fun. We always wondered when is the best time to book the reservation.

# adr divided by total customer number, we get one night rate per person.



rate = data[['hotel','adults','children','babies','assigned_room_type','adr','month','is_canceled']]

rate['rate_p'] = rate['adr']/(rate['adults']+ rate['children'].astype(float)+ rate['babies'].astype(float))

rate.rate_p.describe()
rate['rate'] = [x if x <3000 else 0 for x in rate['rate_p'] ]

rate.rate.describe()
plt.figure(figsize=(30, 10))

sns.lineplot(x='month', y='rate', hue='hotel', data=rate, ci='sd')
sns.boxplot(x = 'is_canceled', y = 'rate', data = rate[rate['rate']<150]).set_title('rate') 