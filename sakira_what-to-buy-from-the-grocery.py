# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Load data

# Holiday events metadata



import pandas as pd

original_data = pd.read_csv('../input/holidays_events.csv')



original_data.head()

# Load data

# Oil meta data

oil_meta_data = pd.read_csv('../input/oil.csv')

print(oil_meta_data.head())

print('shape: ', oil_meta_data.shape)
oil_meta_data['dcoilwtico'].plot()
%pylab inline

#import matplotlib

#matplotlib.style.use('ggplot')

# plot of oil price over 5 years span

import pandas as pd

pd.set_option('display.mpl_style', 'default') # Make the graphs a bit prettier

figsize(15, 5)

oil_data = pd.read_csv('../input/oil.csv', parse_dates = ['date'], dayfirst=True, index_col='date')

oil_data['dcoilwtico'].plot()

# print the oil data

oils = oil_data[['dcoilwtico']]



oils[:5]



# Indexing the oil data with day and weekday

oils.index # index by date

oils.index.day # index by day

oils.index.weekday # index by weekday
# add the weekday indexing to the dataframe

oils['weekday'] = oils.index.weekday

oils[:5]
# whats the oil price throughout the week, not important for this analysis

# Just learning some pandas functionality

weekday_counts = oils.groupby('weekday').aggregate(sum)

print(weekday_counts)

weekday_counts.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

weekday_counts



# Plotting the aggregated weekdays oil price, Saturday, Sunday are excluded

weekday_counts.plot(kind='bar')



# Items meta data

import pandas as pd

items_meta_data = pd.read_csv('../input/items.csv')

print(items_meta_data.head())

print('shape: ', items_meta_data.shape)
# Number of unique class in item meta data

items_meta_data['class'].unique().shape
print(items_meta_data['family'].unique().shape) # Number of family = 33

items_family = items_meta_data.groupby('family') # group by family

items_class = items_meta_data.groupby('class') # group by class

items_family_class = items_meta_data.groupby(['family', 'class']) # group by family and class

# print the groupby information

print(items_family.count())

print(items_class.count())

print(items_family_class.count())
%pylab inline

#import matplotlib

#matplotlib.style.use('ggplot')

# which family has the highest number of items

import pandas as pd

pd.set_option('display.mpl_style', 'default') 

figsize(15, 5)

items_family['item_nbr'].count().plot(kind = 'bar')

# which class the highest number of items

items_class['item_nbr'].count().plot(kind = 'bar')

# which group has the highest number of items :)

items_family_class['item_nbr'].count().plot(kind = 'bar')
# which are perishable items

items_meta_data.groupby(['perishable', 'family', 'class']).count()
# which are perishable family

items_perishable_family = items_meta_data.groupby(['perishable', 'family'])

items_perishable_family.count()
# plot the number of items in perishable family group

items_perishable_family['item_nbr'].count().plot(kind = 'bar')
# Not so meaningful

#items_meta_data.groupby(['class', 'family']).count()
# transaction metda data

transactions_data = pd.read_csv('../input/transactions.csv')

print(transactions_data.head())

transactions_data['date'].unique()

# Transactions meta data

# Indexing the meta data

# Ploting 

# Find the number of unique stores involve in transactions 

transactions_meta_data = pd.read_csv('../input/transactions.csv', parse_dates = ['date'], dayfirst=True, index_col='date')

print(transactions_meta_data[1:10])

print('shape: ', transactions_meta_data.shape)

transactions_meta_data.plot()

store_nbr = transactions_meta_data['store_nbr'].unique()

print(store_nbr.shape)

store_nbr # 54



# transactions at store 1 and store 2

transactions_group_by_store = transactions_meta_data.groupby(['store_nbr'])

print(transactions_group_by_store.sum())

transactions_at_store_1 = transactions_group_by_store.get_group(1)

transactions_at_store_2 = transactions_group_by_store.get_group(2)

transactions_at_store_44 = transactions_group_by_store.get_group(44)



fig, axes = plt.subplots(nrows = 3, ncols = 1)



#start_date = '2013-01-01'

#end_date = '2013-01-31'

#train_data_2013 = train_data.loc[start_date:end_date]



#train_data_2013['unit_sales'].plot(ax = axes[0])

#axes[0].set_title('2013')



#start_date = '2014-01-01'

#end_date = '2014-01-31'

#train_data_2014 = train_data.loc[start_date:end_date]

#train_data_2014['unit_sales'].plot(ax = axes[1])

#axes[1].set_title('2014')

#figsize(15, 5)

transactions_at_store_1['transactions'].plot(ax = axes[0])

transactions_at_store_2['transactions'].plot(ax = axes[1])

transactions_at_store_44['transactions'].plot(ax = axes[2])

#transactions_meta_data = pd.read_csv('../input/transactions.csv')

store = transactions_meta_data.groupby(['store_nbr']).get_group(54)

store.groupby('date').sum()

# Train data of store 54 in comments

# Theres an anomaly between the number of transactions in transactions file and the number of unit_sales in the training file???

#           	id 	store_nbr 	item_nbr 	unit_sales 	onpromotion

#date 					

#2013-01-02 	606 	606 	606 	606 	0

#2013-01-03 	611 	611 	611 	611 	0

#2013-01-04 	568 	568 	568 	568 	0

#2013-01-05 	612 	612 	612 	612 	0

#2013-01-06 	629 	629 	629 	629 	0
# transactions at store 1 and store 2

# Considering all data

transactions_group_by_store = transactions_meta_data.groupby(['store_nbr'])

figsize(15, 70)

fig, axes = plt.subplots(nrows = 54, ncols = 1)

fig.tight_layout()

for i in range(1,55):

    try:

        transactions_at_store = transactions_group_by_store.get_group(i)

        transactions_at_store['transactions'].plot(ax = axes[i - 1])

        axes[i - 1].set_title('Store ' + str(i))

    except KeyError:

        pass

# Interesting phanomenon are observed in store 24, 25, 52 and 53 for example.

print(transactions_at_store.head())
# transactions at store 1 and store 2

# considering data from 2015

start_date = '2015-01-01'

end_date = '2015-12-31'

transactions_group_by_store = transactions_meta_data.loc[start_date:end_date].groupby(['store_nbr'])

print(transactions_meta_data.head())



figsize(10, 70)

fig, axes = plt.subplots(nrows = 54, ncols = 1)

fig.tight_layout()





for i in range(1,55):

    try:

        transactions_at_store = transactions_group_by_store.get_group(i)

        transactions_at_store['transactions'].plot(ax = axes[i - 1])

        axes[i - 1].set_title('Store ' + str(i))

    except KeyError:

        #print(str(KeyError))

        pass



#print(transactions_at_store.head())

# Interesting phanomenon are observed in store 24, 25, 52 and 53 for example.
# at store 12 transactions missing in April, 2015

figsize(15,5)

transactions_at_store_12 = transactions_group_by_store.get_group(12)

transactions_at_store_12['transactions'].plot()

transactions_at_store_12.loc['2015-03-01' : '2015-06-30']
# transactions at store 1 and store 2

# transactions from 2017

start_date = '2017-01-01'

end_date = '2017-12-31'

transactions_group_by_store = transactions_meta_data.loc[start_date:end_date].groupby(['store_nbr'])

transactions_group_by_store.head()

figsize(10, 70)

fig, axes = plt.subplots(nrows = 54, ncols = 1)

fig.tight_layout()

for i in range(1,55):

    transactions_at_store = transactions_group_by_store.get_group(i)

    transactions_at_store['transactions'].plot(ax = axes[i - 1],  subplots = True)

    axes[i - 1].set_title('Store ' + str(i))

figsize(15,5)

# group by by store_nbr

group_by_store_nbr = transactions_meta_data.groupby('store_nbr')

#print(group_by_store_nbr)

group_by_store_nbr.sum().plot(kind='bar')

#print(group_by_store_nbr.sum())
transactions_meta_data['store_nbr'].unique().shape
start_date = '2013-01-01'

end_date   = '2013-12-31'

transactions_2013 = transactions_meta_data.loc[start_date : end_date]

group_by_store_nbr_2013 = transactions_2013.groupby('store_nbr')



start_date = '2014-01-01'

end_date   = '2014-12-31'

transactions_2014 = transactions_meta_data.loc[start_date : end_date]

group_by_store_nbr_2014 = transactions_2014.groupby('store_nbr')



start_date = '2015-01-01'

end_date   = '2015-12-31'

transactions_2015 = transactions_meta_data.loc[start_date : end_date]

group_by_store_nbr_2015 = transactions_2015.groupby('store_nbr')



start_date = '2016-01-01'

end_date   = '2016-12-31'

transactions_2016 = transactions_meta_data.loc[start_date : end_date]

group_by_store_nbr_2016 = transactions_2016.groupby('store_nbr')



start_date = '2017-01-01'

end_date   = '2017-12-31'

transactions_2017 = transactions_meta_data.loc[start_date : end_date]

group_by_store_nbr_2017 = transactions_2017.groupby('store_nbr')



# plot

fig, ax = subplots()

group_by_store_nbr_2013.sum().plot(kind = 'bar', color = 'k', ax = ax)

group_by_store_nbr_2014.sum().plot(kind = 'bar', color = 'DarkGreen', ax = ax)

group_by_store_nbr_2015.sum().plot(kind = 'bar', color = 'DarkBlue', ax = ax)

group_by_store_nbr_2016.sum().plot(kind = 'bar', color = 'DarkOrange', ax = ax)

group_by_store_nbr_2017.sum().plot(kind = 'bar', color = 'Gray', ax = ax)

ax.legend (['Year 2013', 'Year 2014', 'Year 2015', 'Year 2016', 'Year 2017'])

transactions_at_store_44_year_2013 = group_by_store_nbr_2013.get_group(44)

transactions_at_store_44_year_2013['weekday'] = transactions_at_store_44_year_2013.index.weekday

weekday_counts = transactions_at_store_44_year_2013.groupby('weekday').aggregate(sum)

weekday_counts.index = ['Monday', 'Tuesday', 'WEDNESDAY', 'Thursday', 'Friday', 'Saturday',

                        'Sunday']

weekday_counts.plot(kind = 'bar')
# check the status of the one store for example the whole year

# store_nbr = 1

transactions_at_store_1_year_2013 = group_by_store_nbr_2014.get_group(1)

#print(transactions_at_store_1_2013)

#transactions_at_store_1_2013.plot(kind = 'bar')

transactions_at_store_1_year_2013['weekday'] = transactions_at_store_1_year_2013.index.weekday

weekday_counts = transactions_at_store_1_year_2013.groupby('weekday').aggregate(sum)

print(weekday_counts)

weekday_counts.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 

                        'Sunday']

weekday_counts

weekday_counts.plot(kind='bar')
# transactions and oils price 

p1 = oil_data['dcoilwtico'].plot()

transactions_meta_data['transactions'].plot(ax = p1, style = '.')
# Load stores meta data

stores_meta_data = pd.read_csv('../input/stores.csv')

print(stores_meta_data.head())

print('shape: ', stores_meta_data.shape)

print(stores_meta_data)

cities_states = stores_meta_data.groupby(['state', 'city'])

print(cities_states.count())
# Load data

# train data

#train_data = pd.read_csv('../input/train.csv', nrows=100)

#print(train_data.head())

#print('shape: ', train_data.shape)
# Shape of the train data

%pylab inline

#import matplotlib

#matplotlib.style.use('ggplot')



import pandas as pd

pd.set_option('display.mpl_style', 'default') # Make the graphs a bit prettier



train_data = pd.read_csv('../input/train.csv', nrows = 1000, parse_dates = ['date'], dayfirst = True, index_col = 'date')

print('shape: ', train_data.shape) # (125497040,6)

store_item_groupby = train_data.groupby(['store_nbr', 'item_nbr'])



#figsize(15, 5)

store_item_groupby['unit_sales'].sum().plot( kind = 'bar')

import pandas as pd



# read the train data

# parse the data based on year

# delete the variable after saving to csv file

#train_data_reader = pd.read_csv('../input/train.csv', 

 #                              parse_dates = ['date'], dayfirst = True, index_col = 'date')



#train_data_reader = pd.read_csv('../input/train.csv',  

#                               parse_dates = ['date'], dayfirst = True, index_col = 'date')

# 2013

#start_date = '2013-01-01'

#end_date = '2013-01-31'

#filename = 'train_2013.csv'

#train_data_2013 = train_data_reader.loc[start_date:end_date]

#train_data_2013.to_csv(filename)

#del train_data_2013





# 2014

#start_date = '2014-01-01'

#end_date = '2014-12-31'

#filename = 'train_2014.csv'

#train_data_2014 = train_data_reader.loc[start_date:end_date]

#train_data_2014.to_csv(filename)

#del train_data_2014



# 2015

#start_date = '2015-01-01'

#end_date = '2015-12-31'

#filename = 'train_2015.csv'

#train_data_2015 = train_data_reader.loc[start_date:end_date]

#train_data_2015.to_csv(filename)

#del train_data_2015





# 2016

#start_date = '2016-01-01'

#end_date = '2016-12-31'

#filename = 'train_2016.csv'

#train_data_2016 = train_data_reader.loc[start_date:end_date]

#train_data_2016.to_csv(filename)

#del train_data_2016

# 2017

#start_date = '2017-01-01'

#end_date = '2017-12-31'

#filename = 'train_2017.csv'

#train_data_2017 = train_data_reader.loc[start_date:end_date]

#train_data_2017.to_csv(filename)

#del train_data_2017

#del train_data_reader













#for train_data in train_data_reader:

 #   print(train_data)





#print('shape: ', train_data.shape) # (125497040,6)

#train_data_reader.get_chunk(125497040)



#store_groupby = train_data.groupby([ 'store_nbr'])



#figsize(15, 5)

#store_groupby['unit_sales'].count().plot()

#import pandas as pd

#train_1000 = pd.read_csv('train_1000.csv')

#print(train_1000)

# shape of the test data

#test_data = pd.read_csv('../input/test.csv')

#print('shape: ', test_data.shape) # (3370464,5)

#print(test_data.head(5)) 
import os.path

os.path.exists('train_2013.csv')
%pylab inline

#import matplotlib

#matplotlib.style.use('ggplot')



import pandas as pd

pd.set_option('display.mpl_style', 'default') # Make the graphs a bit prettier

figsize(15, 5)

train_data = pd.read_csv('../input/train.csv', parse_dates = ['date'], dayfirst=True, index_col='date')

# 2013

start_date = '2013-01-01'

end_date = '2013-01-31'

train_data_2013 = train_data.loc[start_date:end_date]

train_data_2013['unit_sales'].plot()

start_date = '2013-02-01'

end_date = '2013-02-28'

train_data_2013 = train_data.loc[start_date:end_date]

train_data_2013['unit_sales'].plot()
start_date = '2013-03-01'

end_date = '2013-03-31'

train_data_2013 = train_data.loc[start_date:end_date]

train_data_2013['unit_sales'].transpose().plot()
fig, axes = plt.subplots(nrows=1, ncols=2)



start_date = '2013-01-01'

end_date = '2013-01-31'

train_data_2013 = train_data.loc[start_date:end_date]



train_data_2013['unit_sales'].plot(ax = axes[0])

axes[0].set_title('2013')



start_date = '2014-01-01'

end_date = '2014-01-31'

train_data_2014 = train_data.loc[start_date:end_date]

train_data_2014['unit_sales'].plot(ax = axes[1])

axes[1].set_title('2014')
start_date = '2017-01-01'

end_date = '2017-08-15'

train_data_2017 = train_data.loc[start_date:end_date]

train_data_2017['unit_sales'].transpose().plot()

train_data_2017.groupby('item_nbr').get_group(96995).shape
import pandas as pd

test_data = pd.read_csv('../input/test.csv', parse_dates = ['date'], dayfirst = True, index_col = 'date')

test_data.head()

test_data.shape

test_data['dayofyear'] = test_data.index.dayofyear

test_data.head()
print(test_data['dayofyear'].unique())

print(test_data.index.unique())

print(test_data['store_nbr'].unique().shape)

print(test_data['item_nbr'].unique())

print(test_data['onpromotion'].unique())
%pylab inline

figsize(5,15)

test_data.plot(x = 'date', y = 'item_nbr')