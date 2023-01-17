import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import plotly.graph_objs as go 
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
callsDataFrame = pd.read_csv('../input/911.csv')
callsDataFrame.info()
callsDataFrame.head()
callsDataFrame['zip'].value_counts().head(5)
callsDataFrame['twp'].value_counts().head(5)
callsDataFrame['lat'].value_counts().head(5)
callsDataFrame['lng'].value_counts().head(5)
callsDataFrame['title'].nunique()
len(callsDataFrame['title'].unique())
x = callsDataFrame['title'].iloc[0]
x
x.split(':')[0]
callsDataFrame['Reason'] = callsDataFrame['title'].apply(lambda title: title.split(':')[0])
callsDataFrame['Reason'].head()
callsDataFrame['Reason'].value_counts()
sns.countplot(x='Reason', data=callsDataFrame)
data = [go.Bar(
            x=['EMS', 'Fire', 'Traffic'],
            y=callsDataFrame['Reason'].value_counts()
    )]

iplot(data, filename='basic-bar')
type(callsDataFrame['timeStamp'].iloc[0])
callsDataFrame['timeStamp'] = pd.to_datetime(callsDataFrame['timeStamp'])
callsDataFrame['timeStamp'].iloc[0]
time = callsDataFrame['timeStamp'].iloc[0]
time.hour
time.month
time.dayofweek
callsDataFrame['Hour'] = callsDataFrame['timeStamp'].apply(lambda time: time.hour)
callsDataFrame['Month'] = callsDataFrame['timeStamp'].apply(lambda time: time.month)
callsDataFrame['Day Of Week'] = callsDataFrame['timeStamp'].apply(lambda time: time.dayofweek)
callsDataFrame.head(5)
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
callsDataFrame['Day Of Week'] = callsDataFrame['Day Of Week'].map(dmap)
callsDataFrame.head()
sns.countplot(x = 'Day Of Week', data = callsDataFrame, hue = 'Reason', palette="rocket")

# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns.countplot(x = 'Month', data = callsDataFrame, hue = 'Reason', palette="rocket")

# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
byMonth = callsDataFrame.groupby('Month').count()
byMonth.head()
byMonth['lat'].plot()
sns.countplot(x='Month', data=callsDataFrame, palette='rocket')

# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
## Resert Or convert the Month index to Column
byMonth.reset_index()
## Here I'm using seaborn liner plot passing Month Column as DATA
sns.lmplot(x='Month', y='twp', data=byMonth.reset_index())
callsDataFrame['timeStamp'].iloc[0]
callsDataFrame['timeStamp'].iloc[0].date()
# Creating new 'Date' column using timeStamp column
callsDataFrame['Date'] = callsDataFrame['timeStamp'].apply(lambda ts : ts.date())
callsDataFrame['Date'].head()
## Here, we can see Date Column inside DataFrame
callsDataFrame.head()
callsDataFrame.groupby('Date').count()['lat'].plot()
plt.tight_layout()
callsDataFrame[callsDataFrame['Reason'] == 'Traffic'].groupby('Date').count()['lat'].plot()
plt.title('Traffic')
plt.tight_layout()
callsDataFrame[callsDataFrame['Reason'] == 'Fire'].groupby('Date').count()['lat'].plot()
plt.title('Fire')
plt.tight_layout()
callsDataFrame[callsDataFrame['Reason'] == 'EMS'].groupby('Date').count()['lat'].plot()
plt.title('EMS')
plt.tight_layout()
# Multilevel index count
callsDataFrame.groupby(by=['Day Of Week', 'Hour']).count().head()
callsDataFrame.groupby(by=['Day Of Week', 'Hour']).count()['Reason'].head()
# Having matric level table (* There is an alternate ways like pivot table)
callsDataFrame.groupby(by=['Day Of Week', 'Hour']).count()['Reason'].unstack()
dayHour = callsDataFrame.groupby(by=['Day Of Week', 'Hour']).count()['Reason'].unstack()
## Day Of Week Vs Hour
sns.heatmap(dayHour)
sns.clustermap(dayHour)
dayMonth = callsDataFrame.groupby(by=['Day Of Week', 'Month']).count()['Reason'].unstack()
sns.heatmap(dayMonth)
sns.clustermap(dayMonth)


