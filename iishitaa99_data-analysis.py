import pandas as pd                                           #importing pandas library

import matplotlib.pyplot as mpt                    #importing python maths plot library

import seaborn as sb                                         #to work with more efficient statistical graph plots

import datetime as dt                                       #to import pandas timeStamp library

from decimal import Decimal                         #decimal library is used for more accuracy in arithmetic operations
sb.set()                                                                  #applying default seaborn scaling and themes
data = pd.read_csv("../input/911.csv")

data.head()                                                             #default value of head is 5, and it displays top 5 rows of data
data.shape
data.shape[0]                                        #gives the number of rows
data.shape[1]                                          #gives the number of columns
data.info()
column_names = list(data.columns)
column_names
data.title.head()                                         #now we seperate the type of call from the title column
def call_seperator(x):

    x= x.split(':')                         #here : is the delimeter

    return x[0]                          #this returns the value at 0th index after splitting, which is call type in this case
data['call_type'] = data['title'].apply(call_seperator)                               #inserting values in another column- call_type
data.head()
data['call_type'].nunique()                 #shows the number of unique type of calls 
data['call_type'].unique()                                      #shows the different values or calls
data['call_type'].value_counts()                           #counts all the unique value
data['timeStamp'] = pd.to_datetime(data['timeStamp'], infer_datetime_format = True)         #infer_datatime_format is used when we want the default format and not our own
data['timeStamp'].head()
data['year'] = data['timeStamp'].dt.year
data['month'] = data['timeStamp'].dt.month_name()
data['day'] = data['timeStamp'].dt.day_name()
data['hour'] = data['timeStamp'].dt.hour
def type_of_emergency(x):

    x = x.split(':')

    x= x[1]

    return x
data['emergecy_type'] = data['title'].apply(type_of_emergency)                        #creating a column named emergency_type to store fetched data
data.head()
call_types = data['call_type'].value_counts()              #counting different types of calls
call_types
mpt.figure(figsize = (12,8))

ax = call_types.plot.bar()

for p in ax.patches:                                                                                                         #patches is used to create whatever type of patch we want for eg. rectangular patch

    ax.annotate(Decimal(str(p.get_height())), (p.get_x(), p.get_height()))       #to annotate all the points on the plot

mpt.xticks(rotation=0)                                                                                                   #this is used to give label on the axis

mpt.savefig("Emergency type vs Frequency.png")
data.info()                            #fetching info again to visualize in some other way
calls_data = data.groupby(['month', 'call_type'])['call_type'].count()           #grouped acc. to the types of calls in diff. months
calls_data.head()
call_percentage = calls_data.groupby(level = 0).apply(lambda x: round(100*x/float(x.sum())))              #applying lambda funcs to calculate percentage
call_percentage.head()
font = {

    'size': 'x-large',

    'weight': 'bold'

}

month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
call_percentage = call_percentage.reindex(month_order,  level=0)
call_percentage = call_percentage.reindex(['EMS', 'Traffic', 'Fire'], level=1)
call_percentage.head()
sb.set(rc={'figure.figsize':(12, 8)})

call_percentage.unstack().plot(kind='bar')                                             #making an unstacked graph

mpt.xlabel('Name of the Month', fontdict=font)

mpt.ylabel('Percentage of Calls', fontdict=font)

mpt.xticks(rotation=0)

mpt.title('Calls/Month', fontdict=font)
call_percentage = call_percentage.sort_values(ascending=False)
mpt.figure(figsize=(12,8))

mpt.pie(call_percentage,  labels = call_percentage.index, autopct="%.2f")

mpt.savefig("call percentage pie chart.png")
hours_data = data.groupby(['hour', 'call_type'])['call_type'].count()
hours_data.head()
hours_percentage = hours_data.groupby(level=0).apply(lambda x: round(100*x/float(x.sum())))
hours_percentage.head()
hours_percentage = hours_percentage.reindex(['EMS', 'Traffic', 'Fire'], level=1)
sb.set(rc={'figure.figsize':(18, 8)})

hours_percentage.unstack().plot(kind='bar')

mpt.xlabel('Hour of the day', fontdict=font)

mpt.ylabel('Percentage of Calls', fontdict=font)

mpt.xticks(rotation=0)

mpt.title('Calls/Hour', fontdict=font)

mpt.savefig("Percentage of calls vs hours.png")