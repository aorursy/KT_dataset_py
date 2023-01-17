%matplotlib inline
# Import dependencies
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
import plotly.figure_factory as ff
from plotly import subplots
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from datetime import date
from fbprophet import Prophet
import math
import pandas_profiling
# Import dataset
covid = pd.read_csv("../input/updates2/time-series-19-covid-combined.csv")
covid.columns = ['DATE', 'COUNTRY', 'STATE','LAT','LONG','CONFIRMED', 'RECOVERED', 'DEATH',]
pandas_profiling.ProfileReport(covid)
# Produce quick summary with total numbers
summary_all = covid.groupby('DATE')['CONFIRMED', 'DEATH', 'RECOVERED'].sum().reset_index()
summary = summary_all.sort_values('DATE', ascending=False)
ct_sum = covid['COUNTRY'].unique().tolist()
print("As of now there is" +" " + str(len(ct_sum)) +" " + "countries to which virus has spread.")
summary.head(1).style.background_gradient(cmap='OrRd')

# Spread, death and recovered over the time outside of MainLand China
#covid_all =  covid[(covid['COUNTRY'] == 'Mainland China')]
#covid_all = covid_all.groupby('DATE')['CONFIRMED', 'DEATH', 'RECOVERED'].sum().reset_index()
covid_all = summary_all[summary_all['DATE'] > '2020-01-22']

fig = make_subplots(rows=1, cols=3, subplot_titles=(f"{int(covid_all.CONFIRMED.max()):,d}" +' ' + "CONFIRMED",
                                                    f"{int(covid_all.DEATH.max()):,d}" +' ' + "DEATHS",
                                                    f"{int(covid_all.RECOVERED.max()):,d}" +' ' + "RECOVERED"))

trace1 = go.Scatter(
                x=covid_all['DATE'],
                y=covid_all['CONFIRMED'],
                name="CONFIRMED",
                line_color='orange',
                opacity=0.8)
trace2 = go.Scatter(
                x=covid_all['DATE'],
                y=covid_all['DEATH'],
                name="DEATH",
                line_color='dimgray',
                opacity=0.8)

trace3 = go.Scatter(
                x=covid_all['DATE'],
                y=covid_all['RECOVERED'],
                name="RECOVERED",
                line_color='deepskyblue',
                opacity=0.8)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 1, 3)
fig.update_layout(template="ggplot2",title_text = '<b>Spread Vs. Death Vs Recovered around the world </b>',
                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'), showlegend=False)
fig.show()
# Bar plot for spread, death and recovered over the time around the world
covid_all = covid.groupby('DATE')['CONFIRMED', 'DEATH', 'RECOVERED'].sum().reset_index()
covid_all= covid_all[covid_all['DATE'] > '2020-01-22']

# Plotting Values for Confirmed, deaths and reocvered cases
fig = make_subplots(
    rows=2, cols=2,
    specs=[[{"colspan": 2}, None],[{}, {}]],
    subplot_titles=(f"{int(covid_all.CONFIRMED.max()):,d}" +' ' + "CONFIRMED",
                    f"{int(covid_all.RECOVERED.max()):,d}" +' ' +"RECOVERED",
                    f"{int(covid_all.DEATH.max()):,d}" +' ' +"DEATHS"))

fig.add_trace(go.Bar(x=covid_all['DATE'], y=covid_all['CONFIRMED'], text = covid_all['CONFIRMED'],
                     marker_color='Orange'), row=1, col=1)

fig.add_trace(go.Bar(x=covid_all['DATE'], y=covid_all['RECOVERED'], marker_color='Green'), row=2, col=1)

fig.add_trace(go.Bar(x=covid_all['DATE'], y=covid_all['DEATH'], marker_color='Red'), row=2, col=2)

fig.update_traces(marker_line_color='rgb(8,48,107)',marker_line_width=1.5, opacity=0.8,
                  texttemplate='%{text:.2s}', textposition='outside')

fig['layout']['yaxis1'].update(title='Count', range=[0, covid_all['CONFIRMED'].max() + 15000])
fig['layout']['yaxis2'].update(title='Count', range=[0, covid_all['RECOVERED'].max() + 10000])
fig['layout']['yaxis3'].update(title='Count', range=[0, covid_all['DEATH'].max() + 1000])
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.update_layout(template="ggplot2",title_text = '<b>CurrentConfirmed Vs. Death Vs Recovered Around The World </b>',
                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'), showlegend=False)
fig.show()
# Isolating the max values based on last date for china
ooc_df = covid[covid['DATE'] == covid['DATE'].max()]
ooc_df = ooc_df.groupby('COUNTRY')['CONFIRMED','DEATH','RECOVERED'].sum().reset_index()

# breakdown by state with heat map
import plotly.express as px
ooc_pl = ooc_df.sort_values(by='CONFIRMED', ascending=True).reset_index(drop=True)
fig = px.bar(ooc_pl, x='COUNTRY', y='CONFIRMED',
             hover_data=['COUNTRY', 'CONFIRMED'], color='CONFIRMED',text = ooc_pl.CONFIRMED,
             labels={'pop':'Confirmed Cases in US'}, height=600)
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(uniformtext_minsize=5, uniformtext_mode='hide')
fig.show()
# Producing daily data difference for Confirmed, Death, Recovered
sum_d = covid.groupby('DATE')['CONFIRMED', 'DEATH', 'RECOVERED'].sum().reset_index()
sum_d = sum_d.sort_values('DATE', ascending=True)
sum_d = pd.DataFrame(sum_d.set_index('DATE').diff()).reset_index()
sum_d = sum_d[sum_d['DATE'] > '2020-01-22']
sum_d.head()
# Describing daily data set
sum_d.describe()
# Ploting daily updtes for 
fig_d = go.Figure()
fig_d.add_trace(go.Scatter(x=sum_d.DATE, y=sum_d.CONFIRMED, mode="lines+markers", name=f"MAX. OF {int(sum_d.CONFIRMED.max()):,d}" + ' ' + "CONFIRMED",line_color='Orange'))
fig_d.add_trace(go.Scatter(x=sum_d.DATE, y=sum_d.RECOVERED, mode="lines+markers", name=f"MAX. OF {int(sum_d.RECOVERED.max()):,d}" + ' ' + "RECOVERED",line_color='deepskyblue'))
fig_d.add_trace(go.Scatter(x=sum_d.DATE, y=sum_d.DEATH, mode="lines+markers", name=f"MAX. OF {int(sum_d.DEATH.max()):,d}" + ' ' + "DEATHS",line_color='dimgray'))
fig_d.update_layout(template="ggplot2",title_text = '<b>Daily numbers for Confirmed, Death and Recovered </b>',
                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'), showlegend=True)
fig_d.update_layout(
    legend=dict(
        x=0.01,
        y=.98,
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=12,
            color="Black"
        ),
        bgcolor="LightSteelBlue",
        bordercolor="dimgray",
        borderwidth=2
    ))
fig_d.show()
# Death rate analysis on global level
D_vs_R = covid.copy()
D_vs_R['REC'] = 'REC'
D_vs_R['DTH'] = 'DTH'
recovered = pd.pivot_table(D_vs_R.dropna(subset=['RECOVERED']), index='DATE', 
                         columns='REC', values='RECOVERED', aggfunc=np.sum).fillna(method='ffill').reset_index()

death = pd.pivot_table(D_vs_R.dropna(subset=['DEATH']), index='DATE', 
                         columns='DTH', values='DEATH', aggfunc=np.sum).fillna(method='ffill').reset_index()

D_vs_R_df = pd.merge(recovered,death,on='DATE')
D_vs_R_df['RATIO'] = round(D_vs_R_df['DTH'] / (D_vs_R_df['DTH'] + D_vs_R_df['REC'])*100)

# ploting Current Deat Rate around the world
cur_ratio = D_vs_R_df[D_vs_R_df['DATE'] == D_vs_R_df['DATE'].max()]
fig_dr = go.Figure()
fig_dr.add_trace(go.Scatter(x=D_vs_R_df.DATE, y=D_vs_R_df.RATIO, mode="lines+markers", line_color='Red', name = 'Current Death Rate' + ' ' + f"{int(cur_ratio['RATIO']):,d}%"))
fig_dr.update_layout(template="ggplot2",title_text = '<b>Death Rate % Around The World </b>',
                  font=dict(family="Arial, Balto, Courier New, Droid Sans", color='black'), showlegend=True) 
fig_dr.update_layout(
    legend=dict(
        x=.65,
        y=0.95,
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=12,
            color="Black"
        ),
        bgcolor="LightSteelBlue",
        bordercolor="Red",
        borderwidth=2
    ))
fig_dr.show()


#Runing fbprophet algorythm on confirmed cases outside of MainLand China. Forecasting 7 days.
covid_nc =  covid.copy()
all_df = covid.groupby('DATE')['CONFIRMED', 'DEATH', 'RECOVERED'].sum().reset_index()
all_df = all_df[all_df['DATE'] > '2020-01-22']

df_prophet = all_df.loc[:,["DATE", 'CONFIRMED']]
df_prophet.columns = ['ds','y']
m_d = Prophet(
    yearly_seasonality= True,
    weekly_seasonality = True,
    daily_seasonality = True,
    seasonality_mode = 'additive')
m_d.fit(df_prophet)
future_d = m_d.make_future_dataframe(periods=14)
fcst_daily = m_d.predict(future_d)
fcst_daily[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# Plotting the predictions
fig_prpht = go.Figure()
trace1 = {
  "fill": None, 
  "mode": "markers",
  "marker_size": 10,
  "name": "Confirmed", 
  "type": "scatter", 
  "x": df_prophet.ds, 
  "y": df_prophet.y
}
trace2 = {
  "fill": "tonexty", 
  "line": {"color": "red"}, 
  "mode": "lines", 
  "name": "upper_band", 
  "type": "scatter", 
  "x": fcst_daily.ds, 
  "y": fcst_daily.yhat_upper
}
trace3 = {
  "fill": "tonexty", 
  "line": {"color": "dimgray"}, 
  "mode": "lines", 
  "name": "lower_band", 
  "type": "scatter", 
  "x": fcst_daily.ds, 
  "y": fcst_daily.yhat_lower
}
trace4 = {
  "line": {"color": "blue"}, 
  "mode": "lines+markers",
  "marker_size": 4,
  "name": "prediction", 
  "type": "scatter", 
  "x": fcst_daily.ds, 
  "y": fcst_daily.yhat
}
data = [trace1, trace2, trace3, trace4]
layout = {
  "title": "Confirmed Cases Time Series", 
  "xaxis": {
      "title": "Dates", 
    "ticklen": 5, 
    "gridcolor": "rgb(255, 255, 255)", 
    "gridwidth": 4, 
    "zerolinewidth": 2
  }, 
  "yaxis": {
    "title": "Confirmed Cases", 
    "ticklen": 5, 
    "gridcolor": "rgb(255, 255, 255)", 
    "gridwidth": 2, 
    "zerolinewidth": 1
  }, 
}
fig_prpht = go.Figure(data=data, layout=layout)
fig_prpht.update_layout(template="ggplot2",title_text = '<b>Forecastng of spread around the world </b>',
                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'), showlegend=True)
fig_prpht.update_layout(
    legend=dict(
        x=0.01,
        y=.99,
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=12,
            color="Black"
        ),
        bgcolor="LightSteelBlue",
        bordercolor="Orange",
        borderwidth=2
    ))
fig_prpht.show()
# Ploting cases on world map
import folium
#covid_geo = covid.dropna()
world_curr = covid[covid['DATE'] == covid['DATE'].max()]
map = folium.Map(location=[30, 30], tiles = "CartoDB dark_matter", zoom_start=2.2)
for i in range(0,len(world_curr)):
    folium.Circle(location=[world_curr.iloc[i]['LAT'],
                            world_curr.iloc[i]['LONG']],
                            radius=(math.sqrt(world_curr.iloc[i]['CONFIRMED'])*4000 ),
                            color='crimson',
                            fill=True,
                            fill_color='crimson').add_to(map)
map
# Produce quick summary for China with total numbers
covid_ch =  covid[(covid['COUNTRY'] == 'China')]
ch_df = covid_ch.groupby('DATE')['CONFIRMED', 'DEATH', 'RECOVERED'].sum().reset_index()

summary_ch = ch_df.sort_values('DATE', ascending=False)
summary_ch.head(1).style.background_gradient(cmap='OrRd')
# Spread, death and recovered over the time outside of MainLand China

ch_df = ch_df[ch_df['DATE'] > '2020-01-22']

fig = make_subplots(rows=1, cols=3, subplot_titles=(f"{int(ch_df.CONFIRMED.max()):,d}" +' ' + "CONFIRMED",
                                                    f"{int(ch_df.DEATH.max()):,d}" +' ' + "DEATHS",
                                                    f"{int(ch_df.RECOVERED.max()):,d}" +' ' + "RECOVERED"))

trace1 = go.Scatter(
                x=ch_df['DATE'],
                y=ch_df['CONFIRMED'],
                name="CONFIRMED",
                line_color='orange',
                opacity=0.8)
trace2 = go.Scatter(
                x=ch_df['DATE'],
                y=ch_df['DEATH'],
                name="DEATH",
                line_color='dimgray',
                opacity=0.8)

trace3 = go.Scatter(
                x=ch_df['DATE'],
                y=ch_df['RECOVERED'],
                name="RECOVERED",
                line_color='deepskyblue',
                opacity=0.8)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 1, 3)
fig.update_layout(template="ggplot2",title_text = '<b>Spread Vs. Death Vs Recovered within Mainland China </b>',
                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'), showlegend=False)
fig.show()
# Producing daily data difference for Confirmed, Death, Recovered
sum_cd = covid[(covid['COUNTRY'] == 'China')]
sum_cd = sum_cd.groupby('DATE')['CONFIRMED', 'DEATH', 'RECOVERED'].sum().reset_index()
sum_cd = sum_cd.sort_values('DATE', ascending=True)
sum_cd = pd.DataFrame(sum_cd.set_index('DATE').diff()).reset_index()
#sum_d = pd.DataFrame(round(sum_d.set_index('DATE').pct_change()*100)).reset_index()
sum_cd = sum_cd[sum_cd['DATE'] > '2020-01-22']
sum_cd.tail()
# Ploting daily updtes for 
fig_d = go.Figure()
fig_d.add_trace(go.Scatter(x=sum_cd.DATE, y=sum_cd.CONFIRMED, mode="lines+markers", name=f"MAX. OF {int(sum_cd.CONFIRMED.max()):,d}" + ' ' + "CONFIRMED",line_color='Orange'))
fig_d.add_trace(go.Scatter(x=sum_cd.DATE, y=sum_cd.RECOVERED, mode="lines+markers", name=f"MAX. OF {int(sum_cd.RECOVERED.max()):,d}" + ' ' + "RECOVERED",line_color='deepskyblue'))
fig_d.add_trace(go.Scatter(x=sum_cd.DATE, y=sum_cd.DEATH, mode="lines+markers", name=f"MAX. OF {int(sum_cd.DEATH.max()):,d}" + ' ' + "DEATHS",line_color='dimgray'))
fig_d.update_layout(template="ggplot2",title_text = '<b>Daily numbers for Confirmed, Death and Recovered </b>',
                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'), showlegend=True)
fig_d.update_layout(
    legend=dict(
        x=0.01,
        y=.98,
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=12,
            color="Black"
        ),
        bgcolor="LightSteelBlue",
        bordercolor="dimgray",
        borderwidth=2
    ))
fig_d.show()
# Death rate analysis on global level
china_drr = covid[(covid['COUNTRY'] == 'China')]
china_drr['REC'] = 'REC'
china_drr['DTH'] = 'DTH'
ch_recovered = pd.pivot_table(china_drr.dropna(subset=['RECOVERED']), index='DATE', 
                         columns='REC', values='RECOVERED', aggfunc=np.sum).fillna(method='ffill').reset_index()

ch_death = pd.pivot_table(china_drr.dropna(subset=['DEATH']), index='DATE', 
                         columns='DTH', values='DEATH', aggfunc=np.sum).fillna(method='ffill').reset_index()
china_drr_df = pd.merge(ch_recovered,ch_death,on='DATE')
china_drr_df['RATIO'] = round(china_drr_df['DTH'] / (china_drr_df['DTH'] + china_drr_df['REC'])*100)
china_drr_df.head()

cur_ch_ratio = china_drr_df[china_drr_df['DATE'] == china_drr_df['DATE'].max()]
# ploting Current Deat Rate around the world
fig_dr = go.Figure()
fig_dr.add_trace(go.Scatter(x=china_drr_df.DATE, y=china_drr_df.RATIO, mode="lines+markers", line_color='Red', name = 'Current Death Rate' + ' ' + f"{int(cur_ch_ratio['RATIO']):,d}%"))
fig_dr.update_layout(template="ggplot2",title_text = '<b>Death Rate % In Mainland China </b>',
                  font=dict(family="Arial, Balto, Courier New, Droid Sans", color='black'), showlegend=True) 
fig_dr.update_layout(
    legend=dict(
        x=.65,
        y=0.95,
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=12,
            color="Black"
        ),
        bgcolor="LightSteelBlue",
        bordercolor="Red",
        borderwidth=2
    ))
fig_dr.show()
# Isolating the max values based on last date for china
china_cur  = covid[(covid['COUNTRY'] == 'China')]
china_cur_st = china_cur[china_cur['DATE'] == china_cur['DATE'].max()]
china_cur_st = china_cur_st.groupby('STATE')['CONFIRMED','DEATH','RECOVERED'].sum().reset_index()
china_cur_st.head()

# Ploting distribution between provinces in China
fig = px.treemap(china_cur_st.sort_values(by='CONFIRMED', ascending=False).reset_index(drop=True), 
                 path=["STATE"], values="CONFIRMED", 
                 title='Number of Confirmed Cases in US Cities',
                 color_discrete_sequence = px.colors.qualitative.Prism)
fig.update_layout(template="ggplot2",title_text = '<b>Current confirmed cases within China`s Provincies </b>',
                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'), showlegend=True)
fig.show()

# Ploting Cases in China
map = folium.Map(location=[30, 100], tiles = "CartoDB dark_matter", zoom_start=3.5)
for i in range(0,len(china_drr)):
    folium.Circle(location=[china_drr.iloc[i]['LAT'],
                            china_drr.iloc[i]['LONG']],
                            radius=(math.sqrt(china_drr.iloc[i]['CONFIRMED'])*1000),
                           color='crimson',
                            fill=True,
                            fill_color='crimson').add_to(map)
map
# Breaking up by state
us_cur = covid[(covid['COUNTRY'] == 'US')]
us_cur_st = us_cur[us_cur['DATE'] == us_cur['DATE'].max()]

# Produce quick summary for China with total numbers
covid_us =  covid[(covid['COUNTRY'] == 'US')]
covid_us = covid_us.groupby('DATE')['CONFIRMED', 'DEATH', 'RECOVERED'].sum().reset_index()

covid_us = covid_us.sort_values('DATE', ascending=False)
covid_us.head(1).style.background_gradient(cmap='OrRd')
# Spread, death and recovered over the time for US
us_cur = us_cur.groupby('DATE')['CONFIRMED', 'DEATH', 'RECOVERED'].sum().reset_index()

fig = make_subplots(rows=1, cols=3, subplot_titles=(f"{int(us_cur.CONFIRMED.max()):,d}" +' ' + "CONFIRMED",
                                                    f"{int(us_cur.DEATH.max()):,d}" +' ' + "DEATHS",
                                                    f"{int(us_cur.RECOVERED.max()):,d}" +' ' + "RECOVERED"))

trace1 = go.Scatter(
                x=us_cur['DATE'],
                y=us_cur['CONFIRMED'],
                name="CONFIRMED",
                line_color='orange',
                opacity=0.8)
trace2 = go.Scatter(
                x=us_cur['DATE'],
                y=us_cur['DEATH'],
                name="DEATH",
                line_color='dimgray',
                opacity=0.8)

trace3 = go.Scatter(
                x=us_cur['DATE'],
                y=us_cur['RECOVERED'],
                name="RECOVERED",
                line_color='deepskyblue',
                opacity=0.8)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 1, 3)
fig.update_layout(template="ggplot2",title_text = '<b>Spread Vs. Death Vs Recovered in US </b>',
                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'), showlegend=False)
fig.show()
# Producing daily data difference for Confirmed, Death, Recovered

sum_us = covid_us.sort_values('DATE', ascending=True)
sum_us = pd.DataFrame(sum_us.set_index('DATE').diff()).reset_index()
#sum_d = pd.DataFrame(round(sum_d.set_index('DATE').pct_change()*100)).reset_index()
sum_us = sum_us[sum_us['DATE'] > '2020-01-22']
sum_us.head()
# Ploting daily updtes for 
fig_d = go.Figure()
fig_d.add_trace(go.Scatter(x=sum_us.DATE, y=sum_us.CONFIRMED, mode="lines+markers", name=f"MAX. OF {int(sum_us.CONFIRMED.max()):,d}" + ' ' + "CONFIRMED",line_color='Orange'))
fig_d.add_trace(go.Scatter(x=sum_us.DATE, y=sum_us.RECOVERED, mode="lines+markers", name=f"MAX. OF {int(sum_us.RECOVERED.max()):,d}" + ' ' + "RECOVERED",line_color='deepskyblue'))
fig_d.add_trace(go.Scatter(x=sum_us.DATE, y=sum_us.DEATH, mode="lines+markers", name=f"MAX. OF {int(sum_us.DEATH.max()):,d}" + ' ' + "DEATHS",line_color='dimgray'))
fig_d.update_layout(template="ggplot2",title_text = '<b>Daily numbers for Confirmed, Death and Recovered in US </b>',
                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'), showlegend=True)
fig_d.update_layout(
    legend=dict(
        x=0.01,
        y=.98,
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=12,
            color="Black"
        ),
        bgcolor="LightSteelBlue",
        bordercolor="dimgray",
        borderwidth=2
    ))
fig_d.show()
# Death rate analysis on global level
us_drr = covid[(covid['COUNTRY'] == 'US')]
us_drr['REC'] = 'REC'
us_drr['DTH'] = 'DTH'
us_recovered = pd.pivot_table(us_drr.dropna(subset=['RECOVERED']), index='DATE', 
                         columns='REC', values='RECOVERED', aggfunc=np.sum).fillna(method='ffill').reset_index()

us_death = pd.pivot_table(us_drr.dropna(subset=['DEATH']), index='DATE', 
                         columns='DTH', values='DEATH', aggfunc=np.sum).fillna(method='ffill').reset_index()
us_drr_df = pd.merge(us_recovered,us_death,on='DATE')
us_drr_df['RATIO'] = round(us_drr_df['DTH'] / (us_drr_df['DTH'] + us_drr_df['REC'])*100)

us_ratio = us_drr_df[us_drr_df['DATE'] == us_drr_df['DATE'].max()]
# ploting Current Deat Rate around the world
fig_us = go.Figure()
fig_us.add_trace(go.Scatter(x=us_drr_df.DATE, y=us_drr_df.RATIO, mode="lines+markers", line_color='Red', name = 'Current Death Rate' + ' ' + f"{int(us_ratio['RATIO']):,d}%"))
fig_us.update_layout(template="ggplot2",title_text = '<b>Death Rate % in US</b>',
                  font=dict(family="Arial, Balto, Courier New, Droid Sans", color='black'), showlegend=True) 
fig_us.update_layout(
    legend=dict(
        x=.02,
        y=0.95,
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=12,
            color="Black"
        ),
        bgcolor="LightSteelBlue",
        bordercolor="Red",
        borderwidth=2
    ))
fig_us.show()
# Plotting US data on map

map = folium.Map(location=[35, -100], tiles = "CartoDB dark_matter", zoom_start=4.4)
for i in range(0,len(us_drr)):
    folium.Circle(location=[us_drr.iloc[i]['LAT'],
                            us_drr.iloc[i]['LONG']],
                            radius=(math.sqrt(us_drr.iloc[i]['CONFIRMED'])*200),
                            color='crimson',
                            fill=True,
                            fill_color='crimson').add_to(map)
folium.LayerControl().add_to(map)
map
us_drr
# Analyzing US
# Breaking up by state

#us_cur = covid[(covid['COUNTRY'] == 'US')]
#us_cur_st = us_cur[us_cur['DATE'] == us_cur['DATE'].max()]
#us_cur_st['STATE_'] = us_cur_st['STATE'].str.rsplit(',').str[-1] 
#us_cur_st = us_cur_st.groupby('STATE_')['CONFIRMED','DEATH','RECOVERED'].sum().reset_index()


#us_cur_st = us_cur_st.sort_values(by='CONFIRMED', ascending=False).reset_index(drop=True)
#fig = px.bar(us_cur_st, x='STATE_', y='CONFIRMED',
#             hover_data=['STATE_', 'CONFIRMED'], color='CONFIRMED',text = us_cur_st.CONFIRMED,
#             labels={'pop':'Confirmed Cases in US'}, height=600)
#fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
#fig.update_layout(uniformtext_minsize=5, uniformtext_mode='hide')
#fig.show()
#Runing fbprophet algorythm on confirmed cases outside of MainLand China. Forecasting 7 days.
# Analyzing US
# Breaking up by state

from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.plot import plot_plotly

# restrict to one country
df_us = covid[covid['COUNTRY']=='US']
total_us = df_us.groupby(['DATE']).sum().loc[:,['CONFIRMED','DEATH','RECOVERED']].reset_index()
#total_us = total_us[total_us['DATE'] > '2020-03-8']

us_prophet= total_us.rename(columns={'DATE': 'ds', 'CONFIRMED': 'y'})
# Make a future dataframe for X days
m_us = Prophet(
    changepoint_prior_scale=20,
    seasonality_prior_scale=20,
    n_changepoints=19,
    changepoint_range=0.9,
    yearly_seasonality=False,
    weekly_seasonality = False,
    daily_seasonality = True,
    seasonality_mode = 'additive')
# Add seasonlity
#m_us.add_seasonality(name='yearly', period=365, fourier_order=5)
m_us.fit(us_prophet)

# Make predictions
future_us = m_us.make_future_dataframe(periods=7)

forecast_us = m_us.predict(future_us)
trace1 = {
  "fill": None, 
  "mode": "markers", 
  "name": "actual no. of Confirmed", 
  "type": "scatter", 
  "x": us_prophet.ds, 
  "y": us_prophet.y
}
trace2 = {
  "fill": "tonexty", 
  "line": {"color": "#57b8ff"}, 
  "mode": "lines", 
  "name": "upper_band", 
  "type": "scatter", 
  "x": forecast_us.ds, 
  "y": forecast_us.yhat_upper
}
trace3 = {
  "fill": "tonexty", 
  "line": {"color": "#57b8ff"}, 
  "mode": "lines", 
  "name": "lower_band", 
  "type": "scatter", 
  "x": forecast_us.ds, 
  "y": forecast_us.yhat_lower
}
trace4 = {
  "line": {"color": "#eb0e0e"}, 
  "mode": "lines+markers", 
  "name": "prediction", 
  "type": "scatter", 
  "x": forecast_us.ds, 
  "y": forecast_us.yhat
}

data = [trace1, trace2, trace3, trace4]
layout = {
  "title": "Confirmed - Time Series Forecast - Daily Trend", 
  "xaxis": {
    "title": "", 
    "ticklen": 5, 
    "gridcolor": "rgb(255, 255, 255)", 
    "gridwidth": 2, 
    "zerolinewidth": 1
  }, 
  "yaxis": {
    "title": "Confirmed nCov - US", 
    "ticklen": 5, 
    "gridcolor": "rgb(255, 255, 255)", 
    "gridwidth": 2, 
    "zerolinewidth": 1
  }, 
}
fig = go.Figure(data=data, layout=layout)
iplot(fig)
# Runing data for summary on Italy as of now
italy_cur = covid[(covid['COUNTRY'] == 'Italy')]
italy_cur = italy_cur.groupby('DATE')['CONFIRMED', 'DEATH', 'RECOVERED'].sum().reset_index()
summary_it = italy_cur.sort_values('DATE', ascending=False)
summary_it.head(1).style.background_gradient(cmap='OrRd')
# Spread, death and recovered over the time for Italy
fig = make_subplots(rows=1, cols=3, subplot_titles=(f"{int(italy_cur.CONFIRMED.max()):,d}" +' ' + "CONFIRMED",
                                                    f"{int(italy_cur.DEATH.max()):,d}" +' ' + "DEATHS",
                                                    f"{int(italy_cur.RECOVERED.max()):,d}" +' ' + "RECOVERED"))

trace1 = go.Scatter(
                x=italy_cur['DATE'],
                y=italy_cur['CONFIRMED'],
                name="CONFIRMED",
                line_color='orange',
                opacity=0.8)
trace2 = go.Scatter(
                x=italy_cur['DATE'],
                y=italy_cur['DEATH'],
                name="DEATH",
                line_color='dimgray',
                opacity=0.8)

trace3 = go.Scatter(
                x=italy_cur['DATE'],
                y=italy_cur['RECOVERED'],
                name="RECOVERED",
                line_color='deepskyblue',
                opacity=0.8)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 1, 3)
fig.update_layout(template="ggplot2",title_text = '<b>Spread Vs. Death Vs Recovered in Italy </b>',
                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'), showlegend=False)
fig.show()
# Producing daily data difference for Confirmed, Death, Recovered

sum_it = italy_cur.sort_values('DATE', ascending=True)
sum_it = pd.DataFrame(sum_it.set_index('DATE').diff()).reset_index()
sum_it = sum_it[sum_it['DATE'] > '2020-01-22']
sum_it.head()
# Ploting daily updtes for 
fig_it = go.Figure()
fig_it.add_trace(go.Scatter(x=sum_it.DATE, y=sum_it.CONFIRMED, mode="lines+markers", name=f"MAX. OF {int(sum_it.CONFIRMED.max()):,d}" + ' ' + "CONFIRMED",line_color='Orange'))
fig_it.add_trace(go.Scatter(x=sum_it.DATE, y=sum_it.RECOVERED, mode="lines+markers", name=f"MAX. OF {int(sum_it.RECOVERED.max()):,d}" + ' ' + "RECOVERED",line_color='deepskyblue'))
fig_it.add_trace(go.Scatter(x=sum_it.DATE, y=sum_it.DEATH, mode="lines+markers", name=f"MAX. OF {int(sum_it.DEATH.max()):,d}" + ' ' + "DEATHS",line_color='dimgray'))
fig_it.update_layout(template="ggplot2",title_text = '<b>Daily numbers for Confirmed, Death and Recovered In Itally </b>',
                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'), showlegend=True)
fig_it.update_layout(
    legend=dict(
        x=0.01,
        y=.98,
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=12,
            color="Black"
        ),
        bgcolor="LightSteelBlue",
        bordercolor="dimgray",
        borderwidth=2
    ))
fig_it.show()
# Death rate analysis on global level
it_drr = covid[(covid['COUNTRY'] == 'Italy')]
it_drr['REC'] = 'REC'
it_drr['DTH'] = 'DTH'
it_recovered = pd.pivot_table(it_drr.dropna(subset=['RECOVERED']), index='DATE', 
                         columns='REC', values='RECOVERED', aggfunc=np.sum).fillna(method='ffill').reset_index()

it_death = pd.pivot_table(it_drr.dropna(subset=['DEATH']), index='DATE', 
                         columns='DTH', values='DEATH', aggfunc=np.sum).fillna(method='ffill').reset_index()
it_drr_df = pd.merge(it_recovered,it_death,on='DATE')
it_drr_df['RATIO'] = round(it_drr_df['DTH'] / (it_drr_df['DTH'] + it_drr_df['REC'])*100)

it_ratio = it_drr_df[it_drr_df['DATE'] == it_drr_df['DATE'].max()]
# ploting Current Deat Rate around the world
fig_it = go.Figure()
fig_it.add_trace(go.Scatter(x=it_drr_df.DATE, y=it_drr_df.RATIO, mode="lines+markers", line_color='Red', name = 'Current Death Rate' + ' ' + f"{int(it_ratio['RATIO']):,d}%"))
fig_it.update_layout(template="ggplot2",title_text = '<b>Death Rate % In Italy </b>',
                  font=dict(family="Arial, Balto, Courier New, Droid Sans", color='black'), showlegend=True) 
fig_it.update_layout(
    legend=dict(
        x=.02,
        y=0.95,
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=12,
            color="Black"
        ),
        bgcolor="LightSteelBlue",
        bordercolor="Red",
        borderwidth=2
    ))
fig_it.show()