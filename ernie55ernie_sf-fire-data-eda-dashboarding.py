import pandas as pd

import matplotlib.pyplot as plt
dtype = {

    'Call Number' : 'str', 

    'Incident Number' : 'str', 

    'Station Area' : 'str', 

    'Box' : 'str', 

    'Fire Prevention District' : 'str', 

    'Supervisor District' : 'str',

    'Call Type Group' : 'category'

}

df = pd.read_csv('../input/fire-department-calls-for-service.csv', dtype = dtype, na_values = ['NaN'])



date_time_cols = ['Call Date', 'Watch Date', 'Received DtTm', 'Entry DtTm', 'Dispatch DtTm', 'Response DtTm', 'On Scene DtTm', 'Transport DtTm', 'Transport DtTm', 'Hospital DtTm', 'Available DtTm']

for col in date_time_cols:

    df[col] =  pd.to_datetime(df[col], format='%Y-%m-%dT%H:%M:%S')
import folium

import json
m = folium.Map(location = [37.77, -122.42], zoom_start = 13)



df_call_date = df.set_index('Received DtTm')

lastest_12_hour = df_call_date.last('12h')



color_map = {'Potentially Life-Threatening': 'orange', 'Non Life-threatening': 'green', 'Alarm': 'purple', 'Fire': 'red'}

icon_map = {'Potentially Life-Threatening': 'heart', 'Non Life-threatening': 'info-sign', 'Alarm': 'exclamation-sign', 'Fire': 'fire'}



for index, row in lastest_12_hour.iterrows():

    json_str = row['Location'].replace('"', "*").replace('\'', '"').replace('False', '"False"')

    json_obj = json.loads(json_str)

    longitude = float(json_obj['longitude'])

    latitude = float(json_obj['latitude'])

    color = color_map[row['Call Type Group']]

    icon = icon_map[row['Call Type Group']]

    tooltip = index.strftime('%Y-%m-%dT%H:%M:%S')

    popup = 'Call Type Group: ' + row['Call Type Group'] + '<br>Call Type: ' + row['Call Type'] + '<br>Received Time: ' + index.strftime('%Y-%m-%dT%H:%M:%S')

    folium.Marker([latitude, longitude], popup = popup, tooltip = tooltip, icon = folium.Icon(color = color, icon = icon)).add_to(m)



legend_html = '''

<div style="position:fixed;bottom:50px;left:50px;width:250px;height:100px;border:2px solid grey;z-index:9999;font-size:14px;background-color:lightblue;">

&nbsp;Legend<br>

&nbsp; Potentially Life-Threatening &nbsp; <i class='fa-rotate-0 glyphicon glyphicon-heart  icon-white' style='color:orange'></i><br>

&nbsp; Non Life-threatening &nbsp; <i class='fa-rotate-0 glyphicon glyphicon-info-sign  icon-white' style='color:green'></i><br>

&nbsp; Fire &nbsp; <i class='fa-rotate-0 glyphicon glyphicon-fire  icon-white' style='color:red'></i><br>

&nbsp; Alarm &nbsp; <i class='fa-rotate-0 glyphicon glyphicon-exclamation-sign  icon-white' style='color:purple'></i>

</div>

'''



m.get_root().html.add_child(folium.Element(legend_html))

m
# count of calls per month

df_by_date = df['Call Date'].groupby([df['Call Date'].dt.year, df['Call Date'].dt.month, df['Call Date'].dt.day]).agg('count')



# convert to dataframe

df_by_date = df_by_date.to_frame()



# move date month from index to column

df_by_date['date'] = df_by_date.index



# rename column

df_by_date = df_by_date.rename(columns = {df_by_date.columns[0] : 'calls'})



# re-parse dates

df_by_date['date'] = pd.to_datetime(df_by_date['date'], format = '(%Y, %m, %d)')



# remove index

df_by_date = df_by_date.reset_index(drop = True)



# get date of meet

df_by_date['date'] = df_by_date.date.dt.date
import plotly.plotly as py

import plotly.graph_objs as go



from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()



# sepcify that we want a scatter plot with, with date on the x axis and meet on the y axis

data = [go.Scatter(x = df_by_date.date, y = df_by_date.calls)]



# specify the layout of our figure

layout = dict(title = "Number of Calls per Month",

              xaxis = dict(title = 'Date', ticklen =  5, zeroline = False))



# create and show our figure

fig = dict(data = data, layout = layout)

iplot(fig)
print('There are', df.shape[0], 'rows with', df.shape[1], 'columns')

print('The column names:')

df.columns.tolist()
print('The first 5 rows:')

df.head()
print('The statistic of numeric columns')

df.describe()
print('Data type for each column:')

df.dtypes
for col in df:

    print('Column', col, 'has', len(df[col].unique()), 'unique values')
print('Call Types:')

print(df['Call Type'].unique())

print('Call Type Groups:')

print(df['Call Type Group'].unique())

print('Unit Types:')

print(df['Unit Type'].unique())
df['Call Type'].value_counts().plot(kind = 'bar', title = 'Call Type', logy = True)

plt.show()



df['Call Type Group'].value_counts().plot(kind = 'bar', title = 'Call Type Group')

plt.show()



df['Unit Type'].value_counts().plot(kind = 'bar', title = 'Unit Type')

plt.show()