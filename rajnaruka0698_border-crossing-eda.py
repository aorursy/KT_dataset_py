import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import datetime



import plotly.graph_objects as go

import plotly.express as px
df = pd.read_csv("../input/border-crossing-entry-data/Border_Crossing_Entry_Data.csv")
df.shape
df.duplicated().sum()
df.columns
df.isnull().sum()
df.info()
print('Attribute '+ 'Values')

for i in df.columns:

    print( i,len(df.loc[:,i].unique()) )
temp = df[['Port Name','Port Code']].drop_duplicates()

temp[temp['Port Name'].duplicated(keep=False)]
df.iloc[[29,217]]
indexes = df['Location'].drop_duplicates().index

temp = df.iloc[indexes].groupby(by='Port Code')['Location'].count()

temp.value_counts().plot(kind='pie', autopct='%1.1f%%', shadow=True, explode=[0,0.20],startangle=15)

del temp
df['Date'] = pd.to_datetime(df['Date'])



df['Year'] = df['Date'].apply(lambda x : x.year)



month_mapper = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun'

               ,7:'Jul', 8:'Aug', 9:'Sep' ,10:'Oct', 11:'Nov', 12:'Dec'}

df['Month'] = df['Date'].apply(lambda x : x.month).map(month_mapper)



del month_mapper
df.head()
temp = pd.DataFrame(df.groupby(by='Measure')['Value'].sum().sort_values(ascending=False)).reset_index()

fig = px.bar(temp, x='Measure', y='Value', height=400)

fig.show()

del temp
temp = df.groupby(by=['Border','Measure'])['Value'].sum().reset_index()

temp.fillna(0,inplace=True)

temp.sort_values(by='Value', inplace=True)

fig = px.bar(temp, x='Measure', y='Value', color='Border', barmode='group')

fig.show()

del temp
temp = df.groupby(by='Border')['Value'].sum()

fig = go.Figure(data=[go.Pie(labels = temp.index, values=temp.values)])

fig.update_traces(textfont_size=15,  marker=dict(line=dict(color='#000000', width=2)))

fig.show()

del temp
plt.figure(figsize=(10,6))

sns.lineplot(data=df, x='Year', y='Value', hue='Measure',legend='full')

plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

plt.title('Measure Values Through Years')
plt.figure(figsize=(10,6))

sns.lineplot(data=df, x='Month', y='Value',legend='full', hue='Measure')

plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

plt.title('Value by month')
temp = pd.DataFrame(df.groupby(by='Port Name')['Value'].sum().sort_values(ascending=False)).reset_index()

px.bar(temp, x='Port Name', y='Value')

del temp
measure_size = {'Trucks' : 'Mid_Size', 'Rail Containers Full' : 'Mid_Size', 'Trains' : 'Big_Size',

       'Personal Vehicle Passengers':'Small_Size', 'Bus Passengers':'Small_Size',

       'Truck Containers Empty':'Mid_Size', 'Rail Containers Empty':'Mid_Size',

       'Personal Vehicles' : 'Small_Size', 'Buses' : 'Mid_Size', 'Truck Containers Full' : 'Mid_Size',

       'Pedestrians':'Small_Size', 'Train Passengers':'Small_Size'}



df['Size'] = df['Measure'].map(measure_size)
temp = df.groupby(by=['Size','State'])['Value'].sum()

temp.fillna(0,inplace=True)

temp = temp.reset_index()

px.bar(temp, x='State', y='Value', facet_col='Size')
temp = df.groupby(by=['Size','State'])['Value'].sum().unstack()

temp.fillna(0,inplace=True)



plt.figure(figsize=(15,4))



plt.subplot(131)

temp.iloc[0].sort_values().plot(kind='bar')

plt.xticks(rotation=90)

plt.title('Big_Size')



plt.subplot(132)

temp.iloc[1].sort_values().plot(kind='bar')

plt.xticks(rotation=90)

plt.title('Mid_Size')



plt.subplot(133)

temp.iloc[2].sort_values().plot(kind='bar')

plt.xticks(rotation=90)

plt.title('Small_Size')



del temp
plt.figure(figsize=(15,6))

g = sns.FacetGrid(data=df, col='Size', sharey=False, height=5, aspect=1)

g.map(sns.lineplot, 'Month', 'Value')