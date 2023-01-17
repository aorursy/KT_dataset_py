import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.offline as py
import plotly.figure_factory as ff
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly import tools

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('../input/flights.csv', low_memory=False)
df.head(2).T
df.info()
pd.concat([df.isnull().sum(), 100 * df.isnull().sum()/len(df)], 
              axis=1).rename(columns={0:'Missing Records', 1:'Percentage (%)'})
airlines = pd.read_csv('../input/airlines.csv')
df = pd.merge(df,airlines, left_on='AIRLINE', right_on = 'IATA_CODE')
df.insert(loc=5, column='AIRLINE', value=df.AIRLINE_y)
df = df.drop(['AIRLINE_y','IATA_CODE'], axis=1)
airport = pd.read_csv('../input/airports.csv')
df = pd.merge(df,airport[['IATA_CODE','AIRPORT','CITY']], left_on='ORIGIN_AIRPORT', right_on = 'IATA_CODE')
df = df.drop(['IATA_CODE'], axis=1)
df = pd.merge(df,airport[['IATA_CODE','AIRPORT','CITY']], left_on='DESTINATION_AIRPORT', right_on = 'IATA_CODE')
df = df.drop(['IATA_CODE'], axis=1)
dff = df['AIRPORT_x'].value_counts()[:10]
label = dff.index
size = dff.values

colors = ['skyblue', '#FEBFB3', '#96D38C', '#D0F9B1', 'gold', 'orange', 'lightgrey', 
          'lightblue','lightgreen','aqua']
trace = go.Pie(labels=label, values=size, marker=dict(colors=colors),hole = .2)

data = [trace]
layout = go.Layout(
    title='Origin Airport Distribution'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
dff = df.CITY_x.value_counts()[:10]

trace = go.Bar(
    x=dff.index,
    y=dff.values,
    marker=dict(
        color = dff.values,
        colorscale='Jet',
        showscale=True
    )
)

data = [trace]
layout = go.Layout(
    title='Origin City Distribution', 
    yaxis = dict(title = '# of Flights')
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
dff = df.AIRLINE.value_counts()[:10]

trace = go.Bar(
    x=dff.index,
    y=dff.values,
    marker=dict(
        color = dff.values,
        colorscale='Jet',
        showscale=True)
)

data = [trace]
layout = go.Layout(xaxis=dict(tickangle=15),
    title='Airline distribution', 
                   yaxis = dict(title = '# of Flights'))

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
dff = df.MONTH.value_counts().to_frame().reset_index().sort_values(by='index')
dff.columns = ['month', 'flight_num']
month = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May',
            6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
dff.month = dff.month.map(month)

trace = go.Bar(
    x=dff.month,
    y=dff.flight_num,
    marker=dict(
        color = dff.flight_num,
        colorscale='Reds',
        showscale=True)
)

data = [trace]
layout = go.Layout(
    title='# of Flights (monthly)', 
    yaxis = dict(title = '# of Flights'
                                                )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
df['dep_delay'] = np.where(df.DEPARTURE_DELAY>0,1,0)
df['arr_delay'] = np.where(df.ARRIVAL_DELAY>0,1,0)
dff = df.groupby('MONTH').dep_delay.mean().round(2)

dff.index = dff.index.map(month)
trace1 = go.Bar(
    x=dff.index,
    y=dff.values,
    name = 'Departure_delay',
    marker = dict(
        color = 'aqua'
    )
)

dff = df.groupby('MONTH').arr_delay.mean().round(2)
dff.index = dff.index.map(month)

trace2 = go.Bar(
    x=dff.index,
    y=dff.values,
    name='Arrival_delay',
    marker=dict(
        color = 'red'
    )
)

data = [trace1,trace2]
layout = go.Layout(
    title='% Delay (Months)', 
    yaxis = dict(title = '%')
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
dayOfWeek={1:'Monday', 2:'Tuesday', 3:'Wednesday', 4:'Thursday', 5:'Friday', 
                                           6:'Saturday', 7:'Sunday'}
dff = df.DAY_OF_WEEK.value_counts()
dff = dff.to_frame().sort_index()
dff.index = dff.index.map(dayOfWeek)

trace1 = go.Bar(
    x=dff.index,
    y=dff.DAY_OF_WEEK,
    name = 'Weather',
    marker=dict(
        color = dff.DAY_OF_WEEK,
        colorscale='Jet',
        showscale=True
    )
)

data = [trace1]
layout = go.Layout(
    title='# of Flights (Day of Week)', 
    yaxis = dict(title = '# of Flights'
                                                    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
flight_volume = df.pivot_table(index="CITY_x",columns="DAY_OF_WEEK",
                               values="DAY",aggfunc=lambda x:x.count())
fv = flight_volume.sort_values(by=1, ascending=False)[:7]
fv = fv.iloc[::-1]

fig = plt.figure(figsize=(16,9))
sns.heatmap(fv, cmap='RdBu',linecolor="w", linewidths=2)

plt.title('Air Traffic by Cities',size=16)
plt.ylabel('CITY')
plt.xticks(rotation=45)
plt.show()
flight_volume = df.pivot_table(index="CITY_x",columns="DAY_OF_WEEK",
                               values="DAY",aggfunc=lambda x:x.count())
fv = flight_volume.sort_values(by=1, ascending=False)[:8]
fv.index = np.where(fv.index=='Dallas-Fort Worth','Dallas', fv.index)

trace = go.Heatmap(z=[fv.values[1],fv.values[2],fv.values[3],fv.values[4],
                      fv.values[5],fv.values[6],fv.values[7]],
                   x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
                      'Saturday','Sunday'],
                   y=fv.index.values, colorscale='Reds'
                  )

data=[trace]
layout = go.Layout(
    title='Air Traffic by Cities')

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
dff = df.groupby('DAY_OF_WEEK').dep_delay.mean().round(2)
dff.index = dff.index.map(dayOfWeek)

trace1 = go.Bar(
    x=dff.index,
    y=dff.values,
    name = 'Departure_delay',
    marker=dict(
        color = 'cyan'
    )
)

dff = df.groupby('DAY_OF_WEEK').arr_delay.mean().round(2)
dff.index = dff.index.map(dayOfWeek)

trace2 = go.Bar(
    x=dff.index,
    y=dff.values,
    name='Arrival_delay',
    marker=dict(
        color = 'indigo'
    )
)

data = [trace1,trace2]
layout = go.Layout(
    title='% Delay (Day of Week)', 
    yaxis = dict(title = '%')
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
dff = df.groupby('AIRLINE').DEPARTURE_DELAY.mean().to_frame().sort_values(by='DEPARTURE_DELAY',
                                                    ascending=False).round(2)
trace1 = go.Bar(
    x=dff.index,
    y=dff.DEPARTURE_DELAY,
    name='departure_delay',
    marker=dict(
        color = 'navy'
    )
)

dff = df.groupby('AIRLINE').ARRIVAL_DELAY.mean().to_frame().sort_values(by='ARRIVAL_DELAY',
                                                    ascending=False).round(2)
trace2 = go.Bar(
    x=dff.index,
    y=dff.ARRIVAL_DELAY,
    name='arrival_delay',
    marker=dict(
        color = 'red'
    )
)

data = [trace1, trace2]
layout = go.Layout(xaxis=dict(tickangle=15), title='Mean Arrival & Departure Delay by Airlines',
    yaxis = dict(title = 'minute'), 
                   barmode='stack')

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
df['DEP_ARR_DIFF'] = df['DEPARTURE_DELAY'] - df['ARRIVAL_DELAY']
dff = df.groupby('AIRLINE').DEP_ARR_DIFF.mean().to_frame().sort_values(by='DEP_ARR_DIFF',
                                                    ascending=False).round(2)

trace = go.Bar(
    x=dff.index,
    y=dff.DEP_ARR_DIFF,
    marker=dict(
        color = dff.DEP_ARR_DIFF,
        colorscale='Jet',
        showscale=True
    )
)

data = [trace]
layout = go.Layout(xaxis=dict(tickangle=15),
    title='Mean (Departure Delay - Arrival Delay) by Airlines', 
                   yaxis = dict(title = 'minute')
                  )

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
dff = df.groupby('CITY_x').DEPARTURE_DELAY.mean().to_frame().sort_values(by='DEPARTURE_DELAY',
                                                        ascending=False)[:8].round(2)
trace1 = go.Bar(
    x=dff.index,
    y=dff.DEPARTURE_DELAY,
    marker=dict(
        color = 'red'
    )
)

dff = df.groupby('CITY_y').ARRIVAL_DELAY.mean().to_frame().sort_values(by='ARRIVAL_DELAY',
                                                        ascending=False)[:8].round(2)

trace2 = go.Bar(
    x=dff.index,
    y=dff.ARRIVAL_DELAY,
    marker=dict(
        color = 'navy'
    )
)

fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Mean Departure Delay by City', 
                                                          'Mean Arrival Delay by City'))
fig.append_trace(trace1, 1,1)
fig.append_trace(trace2, 1,2)

fig['layout'].update(yaxis = dict(title = 'minute'), height=500, width=850, 
                     title='Is it a systematic delay related to departure or arrival city?',  
                     showlegend=False)                    
py.iplot(fig)
arr = df.pivot_table(index="CITY_x",columns="DAY_OF_WEEK",values="ARRIVAL_DELAY",
                     aggfunc=lambda x:x.mean())
arr['sum'] = arr[1] + arr[2] +arr[3]+arr[4]+arr[5]+arr[6]+arr[7]

fv = arr.sort_values(by='sum')[:7]
fv = fv.iloc[::-1]
fv = fv.drop(['sum'], axis=1)
fig = plt.figure(figsize=(16,9))
sns.heatmap(fv, cmap='BuPu',linecolor="w", linewidths=2)

plt.title('Lowest Mean Arrival Delay by Cities', size=16)
plt.ylabel('CITY')
plt.xticks(rotation=45)
plt.show()
arr = df.pivot_table(index="CITY_x",columns="DAY_OF_WEEK",values="ARRIVAL_DELAY",
                     aggfunc=lambda x:x.mean())
arr['sum'] = arr[1] + arr[2] +arr[3]+arr[4]+arr[5]+arr[6]+arr[7]
fv = arr.sort_values(by='sum')[:8]
fv = fv.iloc[::-1]
fv = fv.drop(['sum'], axis=1)

trace = go.Heatmap(z=[fv.values[1],fv.values[2],fv.values[3],fv.values[4],fv.values[5],
                      fv.values[6],fv.values[7]],
                   x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday','Saturday','Sunday'],
                   y=fv.index.values, colorscale='Blues', 
                   reversescale = True
                  )

data=[trace]
layout = go.Layout(
    title='Lowest Arrival Delay by Cities')

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
dff = df.groupby('AIRLINE').TAXI_OUT.mean().to_frame().sort_values(by='TAXI_OUT',
                                                    ascending=False)[:8].round(2)

trace1 = go.Bar(
    x=dff.index,
    y=dff.TAXI_OUT,name='TAXI_OUT',
    marker=dict(
        color = 'aqua'
    )
)

dff = df.groupby('AIRLINE').TAXI_IN.mean().to_frame().sort_values(by='TAXI_IN',
                                                        ascending=False)[:8].round(2)

trace2 = go.Bar(
    x=dff.index,
    y=dff.TAXI_IN, name='TAXI_IN',
    marker=dict(
       color = 'indigo'
    )
)

fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Median Taxi Out', 'Median Taxi In'))

fig.append_trace(trace1, 1,1)
fig.append_trace(trace2, 1,2)

fig['layout'].update(yaxis = dict(title = 'minute'), height=500, width=850, 
                     title='Which is hard whell-off or whell-on?',  
                     showlegend=False)               
py.iplot(fig)
df['OUT_IN_DIFF'] = df['TAXI_OUT'] - df['TAXI_IN']
dff = df.groupby('AIRLINE').OUT_IN_DIFF.mean().to_frame().sort_values(by='OUT_IN_DIFF',
                                                    ascending=False).round(2)

trace = go.Bar(
    x=dff.index,
    y=dff.OUT_IN_DIFF,
    marker=dict(
        color = dff.OUT_IN_DIFF,
        colorscale='Jet',
        showscale=True
    )
)

data = [trace]
layout = go.Layout(xaxis=dict(tickangle=15),
    title='Mean (Taxi Out - Taxi In) by Airlines', 
                   yaxis = dict(title = 'minute'
                                                               )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
df['SPEED'] = 60*df['DISTANCE']/df['AIR_TIME']
dff = df.groupby('AIRLINE').SPEED.mean().to_frame().sort_values(by='SPEED',
                                                    ascending=False).round(2)

trace = go.Scatter(
    x=dff.index,
    y=dff.SPEED,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 30,
        color = dff.SPEED.values,
        colorscale='Jet',
        showscale=True
    )
)

data = [trace]
layout = go.Layout(xaxis=dict(tickangle=-20),
    title='Mean Speed by Airlines', 
                   yaxis = dict(title = 'Speed')
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
dff = df.groupby('AIRLINE')[['CANCELLED']].mean().sort_values(by='CANCELLED', 
                                                    ascending=False).round(3)

trace1 = go.Scatter(
    x=dff.index,
    y=dff.CANCELLED,
    mode='markers',
    marker=dict(
        symbol = 'star-square',
        sizemode = 'diameter',
        sizeref = 1,
        size = 30,
        color = dff.CANCELLED,
        colorscale='Portland',
        showscale=True
    )
)

data = [trace1]
layout = go.Layout(xaxis=dict(tickangle=20),
    title='Cancellation Rate by Airlines', yaxis = dict(title = 'Cancellation Rate'
                                                       )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="age")

dff = df.groupby('CITY_x')[['CANCELLED']].mean().sort_values(by='CANCELLED', 
                                            ascending=False)[:10].round(3)
trace2 = go.Scatter(
    x=dff.index,
    y=dff.CANCELLED,
    mode='markers',
    marker=dict(symbol = 'diamond',
        sizemode = 'diameter',
        sizeref = 1,
        size = 30,
        color = dff.CANCELLED,
        colorscale='Portland',
        showscale=True
    )
)

data = [trace2]
layout = go.Layout(xaxis=dict(tickangle=20),
    title='Cancellation Rate by Cities', 
                   yaxis = dict(title = 'Cancellation Rate'
                                                     )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
reason={'A':'Airline/Carrier', 'B':'Weather', 'C':'National Air System', 'D':'Security'}
df.CANCELLATION_REASON = df.CANCELLATION_REASON.map(reason)

dff = df[df.CANCELLED==1]['MONTH'].value_counts().reset_index().sort_values(by='index')
dff.columns = ['month', 'flight_num']
dff.month = dff.month.map(month)

trace = go.Bar(
    x=dff.month,
    y=dff.flight_num,
    marker=dict(
        color = dff.flight_num,
        colorscale='Reds',
        showscale=True
    )
)

data = [trace]
layout = go.Layout(
    title='# of Cancelled Flights (monthly)', 
    yaxis = dict(title = '# of Flights'
                                                          )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
dff = df[df.CANCELLATION_REASON=='Weather'].MONTH.value_counts()
dff = dff.to_frame().sort_index()
dff.index = dff.index.map(month)

trace1 = go.Bar(
    x=dff.index,
    y=dff.MONTH,
    name = 'Weather',
    marker=dict(
        color = 'aqua'
    )
)

dff = df[df.CANCELLATION_REASON=='Airline/Carrier'].MONTH.value_counts()
dff = dff.to_frame().sort_index()
dff.index = dff.index.map(month)

trace2 = go.Bar(
    x=dff.index,
    y=dff.MONTH,
    name='Airline/Carrier',
    marker=dict(
        color = 'red'
    )
)

dff = df[df.CANCELLATION_REASON=='National Air System'].MONTH.value_counts()
dff = dff.to_frame().sort_index()
dff.index = dff.index.map(month)

trace3 = go.Bar(
    x=dff.index,
    y=dff.MONTH,
    name='National Air System',
    marker=dict(
        color = 'navy'
    )
)

data = [trace1,trace2,trace3]
layout = go.Layout(
    title='Cancellation Reasons (Monthly)', 
    yaxis = dict(title = '# of Flights'
                                                        )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
dff = df[df.CANCELLATION_REASON == 'Weather'].DAY_OF_WEEK.value_counts()
dff = dff.to_frame().sort_index()
dff.index = dff.index.map(dayOfWeek)

trace1 = go.Bar(
    x=dff.index,
    y=dff.DAY_OF_WEEK,
    name = 'Weather',
    marker=dict(
        color = 'aqua'
    )
)

dff = df[df.CANCELLATION_REASON=='Airline/Carrier'].DAY_OF_WEEK.value_counts()
dff = dff.to_frame().sort_index()
dff.index = dff.index.map(dayOfWeek)

trace2 = go.Bar(
    x=dff.index,
    y=dff.DAY_OF_WEEK,
    name='Airline/Carrier',
    marker=dict(
        color = 'red'
    )
)

dff = df[df.CANCELLATION_REASON=='National Air System'].DAY_OF_WEEK.value_counts()
dff = dff.to_frame().sort_index()
dff.index = dff.index.map(dayOfWeek)

trace3 = go.Bar(
    x=dff.index,
    y=dff.DAY_OF_WEEK,
    name='National Air System',
    marker=dict(
        color = 'navy'
    )
)

data = [trace1,trace2,trace3]
layout = go.Layout(
    title='Cancellation Reasons (Day of Week)', 
    yaxis = dict(title = '# of Flights'
                                                            )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
df['Date'] = pd.to_datetime(df[['DAY','MONTH','YEAR']])
df = df[df.MONTH < 9]
df1dm = df.resample('D', on='Date').mean()
df1wm = df.resample('W', on='Date').mean()
df1mm = df.resample('M', on='Date').mean()
df1dc = df.resample('D', on='Date').count()
df1wc = df.resample('W', on='Date').count()
df1mc = df.resample('M', on='Date').count()
hist_data = [df1dm[df1dm.DAY_OF_WEEK<6].ARRIVAL_DELAY, df1dm[df1dm.DAY_OF_WEEK==6].ARRIVAL_DELAY,
            df1dm[df1dm.DAY_OF_WEEK==7].ARRIVAL_DELAY]

labels = ['Weekday', 'Saturday','Sunday']
colors = ['navy', 'green', 'red']

fig = ff.create_distplot(hist_data, labels, colors=colors,
                         show_hist=False, bin_size=.2)

fig['layout'].update(title='Mean Arrival Delay')
py.iplot(fig)
correlation = df[['DAY_OF_WEEK','MONTH','ELAPSED_TIME', 'AIR_TIME', 'DISTANCE',
       'WHEELS_ON', 'TAXI_IN', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME',
       'ARRIVAL_DELAY','SPEED']].fillna(0).corr()
cols = correlation.columns.values
corr  = correlation.values

trace = go.Heatmap(z = corr,
                   x = cols,
                   y = cols,
                   colorscale = "YlOrRd",reversescale = True
                                    ) 

data = [trace]
layout = go.Layout(dict(title = "Correlation Matrix for variables",
                        autosize = False,
                        height  = 600,
                        width   = 800,
                        margin  = dict(l = 200
                                      ),
                        yaxis   = dict(tickfont = dict(size = 8)),
                        xaxis   = dict(tickfont = dict(size = 8))
                       )
                  )

fig = go.Figure(data=data,layout=layout)
py.iplot(fig)