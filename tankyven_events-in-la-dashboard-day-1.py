import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import ast, json
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import os
print(os.listdir("../input"))
df = pd.read_csv("../input/what's-happening-la-calendar-dataset/whats-happening-la-calendar-dataset.csv")
df.describe()
df.head()
# Date pre-processing - Only include events after 2010
df['Start Year'] = df['Event Date & Time Start'].astype(str).apply(lambda x: x.split('-')[0])
df['End Year'] = df['Event Date & Time Ends'].astype(str).apply(lambda x: x.split('-')[0])
df = df[df['Start Year'].astype(float) > 2010]
df = df[df['End Year'].astype(float) > 2010]
df['Event Date & Time Start'] = pd.to_datetime(df['Event Date & Time Start'])
df['Event Date & Time Ends'] = pd.to_datetime(df['Event Date & Time Ends'])

# Add new date columns
df['Month'] = df['Event Date & Time Start'].apply(lambda time: time.month)
df['Day of Week'] = df['Event Date & Time Start'].apply(lambda time: time.dayofweek)
dmap = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
df['Day of Week'] = df['Day of Week'].map(dmap)
df['Day of Week'] = pd.Categorical(df['Day of Week'], categories=['Mon','Tue','Wed','Thu','Fri','Sat', 'Sun'], ordered=True)

# Replace NAs in location address
df['Location Address'].fillna('{}', inplace = True)
# Normalize json in location address
def only_dict(d):
    '''
    Convert json string representation of dictionary to a python dict
    '''
    return ast.literal_eval(d)

location_data = json_normalize(data = df['Location Address'].apply(only_dict))
location_data['human_address'].fillna('{}', inplace = True)
human_address_data = json_normalize(data = location_data['human_address'].apply(only_dict))
human_address_data.head()
# Events started in each year - bar plot
plt.figure(figsize=(12,6))
sns.countplot(data = df, x = 'Start Year', palette = 'viridis')
plt.title('Number of Events Started each Year')
# Events started in each month - heatmap
byMonthDay = df.groupby(['Month','Day of Week']).count()['Event Date & Time Start'].unstack(level=0)
plt.figure(figsize=(12,6))
sns.heatmap(data = byMonthDay, cmap = 'coolwarm')
plt.title('Events across the Days/Months')
# Bar chart on age groupings of events
human_address_data['city'].value_counts().head()
# Clean human_address data
human_address_data['city'].replace('', np.NaN, inplace = True)
human_address_data = human_address_data.dropna()

# Find number of events in each county from 2014 - 2017
bycity = human_address_data.groupby('city').count()
bycity = bycity.reset_index().sort_values(by='address', ascending = False).rename(str.capitalize,axis = 'columns')
bycity = bycity.rename(index = str, columns = {'Address': 'Count'})
bycity
# Bar chart showing top 5 cities where events were held
plt.figure(figsize=(12,6))
sns.barplot(data= bycity.head(), x = 'City', y = 'Count',  palette='viridis')
plt.title('Top 5 Cities for Events in LA')

