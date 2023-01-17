# Second Try 
import pandas as pd
data = pd.read_csv("../input/911.csv")
data.head(4)
def sepa(x):

    val = x.split(":")

    return val[0]
data['type'] = data['title'].apply(sepa)
data.head(3)
data['timeStamp'] = pd.to_datetime(data['timeStamp'] , infer_datetime_format = True)
data['timeStamp'].head(2)
data.head(2)
import datetime as dt

data['year'] = data['timeStamp'].dt.year
data['month'] = data['timeStamp'].dt.month_name()
data['day'] = data['timeStamp'].dt.day_name()
def emergency_type_separator(x):

    x = x.split(':')

    x = x[1]

    return x
data['emergency_type'] = data['title'].apply(emergency_type_separator)
data.head(2)
def emergency_type_separator(x):

    x = x.split(':')

    x = x[1]

    return x
data['emergency_type'] = data['title'].apply(emergency_type_separator)
call_types = data['type'].value_counts()

call_types
calls_data = data.groupby(['month', 'type'])['type'].count()
calls_data_percentage = calls_data.groupby(level=0).apply(lambda x: round(100*x/float(x.sum())))
calls_data_percentage.head()
font = {

    'size': 'x-large',

    'weight': 'bold'

}

month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
calls_data_percentage = calls_data_percentage.reindex(month_order, level=0)
calls_data_percentage.head()
import matplotlib.pyplot as plt

import seaborn as sns



sns.set()
sns.set(rc={'figure.figsize':(12, 8)})

calls_data_percentage.unstack().plot(kind='bar')

plt.xlabel('Name of the Month', fontdict=font)

plt.ylabel('Percentage of Calls', fontdict=font)

plt.xticks(rotation=0)

plt.title('Calls/Month', fontdict=font)
