import pandas as pd
data = pd.read_csv('../input/911.csv')
data.head()
data.info()
data.shape
data.title.nunique()
def call_type_seperate(x):

    x = x.split(':')

    return x[0]

def call_type_seperator(y):

    y = y.split(':')

    return y[1]
data['Call Type'] = data['title'].apply(call_type_seperate)

data['Call Purpose'] = data['title'].apply(call_type_seperator)
data.drop('title', axis=1, inplace = True)

data.head()
data['timeStamp'] = pd.to_datetime(data['timeStamp'], infer_datetime_format= True)
data.info()
data.head(4)
data['timeStamp'].head(3)
import datetime as dt
data['year'] = data['timeStamp'].dt.year

    
data['month_name'] = data['timeStamp'].dt.month_name()
data['day_name'] = data['timeStamp'].dt.day_name()
data['Hours'] = data['timeStamp'].dt.hour
data.head(4)
calls_data = data.groupby(['month_name'])['Call Type'].value_counts()

calls_data.head(3)
calls_percentage_data = calls_data.groupby(level = 0).apply(lambda x: round(100 * x/x.sum()))
calls_percentage_data
import matplotlib.pyplot as plt

import seaborn as sns
font = {

    'size': 'x-large',

    'weight': 'bold'

}
sns.set(rc={'figure.figsize':(12,8)})

calls_percentage_data.unstack().plot(kind = 'bar')

plt.xlabel('Name of the month', fontdict = font)

plt.ylabel('Percentage of calls', fontdict = font)

plt.xticks(rotation=0)

plt.title('Calls/Month', fontdict = font)