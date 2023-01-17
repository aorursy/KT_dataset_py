import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats

%matplotlib inline

plt.style.use('ggplot')





import plotly.express as px

import plotly.graph_objs as go

from plotly.offline import iplot, init_notebook_mode

import plotly.figure_factory as ff

from plotly import subplots

from plotly.subplots import make_subplots

import ipywidgets as widgets

init_notebook_mode(connected=True)





from datetime import datetime, date, timedelta



from fbprophet import Prophet



import warnings

warnings.filterwarnings('ignore')





pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)
def get_data(path):

    tab_name = ['Deaths','Recovered']

    jc = ['Province/State','Country/Region']

    path_c = path+'Confirmed.csv'

    df = pd.read_csv(path_c)

    df = pd.melt(df, id_vars=['Province/State','Country/Region','Lat','Long'],

            var_name='Date', value_name= 'Confirmed')

    df['Date'] = pd.to_datetime(df['Date'])

    for name in tab_name:

        path_ = path+name+'.csv'

        data = pd.read_csv(path_)

        data = pd.melt(data, id_vars=['Province/State','Country/Region','Lat','Long'], 

                    var_name='Date', value_name= name)

        data['Date'] = pd.to_datetime(data['Date'])

        df[name] = data[name].values

        

        

    return(df)









def prepare_data(df):

    num_col = ['Confirmed','Deaths','Recovered'] 

    new_col = ['PS','Country','Lat','Long','Date','Confirmed','Deaths','Recovered']

    df.columns = new_col

    df[num_col] = df[num_col].apply(lambda x: x.fillna(value = 0))

    df[num_col] = df[num_col].astype(np.int32)

    df['Country'] = np.where(df['Country'] == 'Mainland China','China',df['Country'])

    df['PS'] = np.where(df['PS'].isnull(), df['Country'],df['PS'])

    

    return(df)

 

def check_anomalies(df):

    count_c = df.loc[(df['Confirmed_'] <0)].shape[0]

    count_d = df.loc[(df['Deaths_'] <0)].shape[0]

    count_r = df.loc[(df['Recovered_'] <0)].shape[0]

    

    print("Number of negative Confirmed_: {}\n".format(count_c))

    print("Number of negative Deaths_: {}\n".format(count_d))

    print("Number of negative Recovered_: {}\n".format(count_r))

    

def rebinnable_interactive_histogram(series, title,initial_bin_width=10):

    trace = go.Histogram(

        x=series,

        xbins={"size": initial_bin_width},

        marker_color = 'rgb(55, 83, 109)',

    )

    figure_widget = go.FigureWidget(

        data=[trace],

        layout=go.Layout(yaxis={"title": "Count"}, xaxis={"title": "x"}, bargap=0.05,

                        title = 'Histogram of Corfirmed Case - {}'.format(title)),

    )



    bin_slider = widgets.FloatSlider(

        value=initial_bin_width,

        min=5,

        max=24,

        step=2,

        description="Bin width:",

        readout_format=".0f", 

    )



    histogram_object = figure_widget.data[0]



    def set_bin_size(change):

        histogram_object.xbins = {"size": change["new"]}



    bin_slider.observe(set_bin_size, names="value")



    output_widget = widgets.VBox([figure_widget, bin_slider])

    return output_widget

#path = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-'

df = pd.read_csv('/kaggle/input/timeseries/nCov_daily.csv',index_col = 0)
df.head(10)
df = prepare_data(df)
df.head(10)
df.isnull().sum()
print("Number of rows in the dataset: {}".format(df.shape[0]))

print("Number of Columns in the dataset: {}".format(df.shape[1]))
sorted_df = df.sort_values(['Country','PS', 'Date'])

cols = ['Country', 'PS']

sorted_df['Confirmed_'] = np.where(np.all(sorted_df.shift()[cols]==sorted_df[cols], axis=1)

                                   , sorted_df['Confirmed'].diff(), sorted_df['Confirmed'])

sorted_df['Deaths_'] = np.where(np.all(sorted_df.shift()[cols]==sorted_df[cols], axis=1),

                                sorted_df['Deaths'].diff(), sorted_df['Deaths'])

sorted_df['Recovered_'] = np.where(np.all(sorted_df.shift()[cols]==sorted_df[cols], axis=1),

                                   sorted_df['Recovered'].diff(), sorted_df['Recovered'])
check_anomalies(sorted_df)
sorted_df.loc[sorted_df['Confirmed_']<0]
# Queensland

df.loc[[651,801],'Confirmed'] = 2

# Japan

df.loc[107,'Confirmed'] = 2

df.loc[1157,'Confirmed'] = 25
sorted_df.loc[sorted_df['Recovered_']<0]
# Guangxi

df.loc[1506,'Recovered'] = 32

# Guizhou

df.loc[1057,'Recovered'] = 6

# Hainan

df.loc[1733,'Recovered'] = 37

# Heilongjiang

df.loc[1435,'Recovered'] = 21

# Ningxia

df.loc[1294,'Recovered'] = 9

# Shanxi

df.loc[849,'Recovered'] = 2
sorted_df = df.sort_values(['Country','PS', 'Date'])

cols = ['Country', 'PS']

sorted_df['Confirmed_'] = np.where(np.all(sorted_df.shift()[cols]==sorted_df[cols], axis=1)

                                   , sorted_df['Confirmed'].diff(), sorted_df['Confirmed'])

sorted_df['Deaths_'] = np.where(np.all(sorted_df.shift()[cols]==sorted_df[cols], axis=1),

                                sorted_df['Deaths'].diff(), sorted_df['Deaths'])

sorted_df['Recovered_'] = np.where(np.all(sorted_df.shift()[cols]==sorted_df[cols], axis=1),

                                   sorted_df['Recovered'].diff(), sorted_df['Recovered'])
check_anomalies(sorted_df)
df_world = df.loc[df['Country'] != 'China'].groupby(['Country'])[['PS','Long','Lat','Confirmed']].max().reset_index()

rebinnable_interactive_histogram(df_world.Confirmed,'Rest of the World')
limits = [0,13,27,41,83,df_world.Confirmed.max()+1]

df_world['text'] = 'Country: ' + df_world['Country'].astype(str) + '<br>Province/State ' + (df_world['PS']).astype(str) + '<br>Confirmed: ' + (df_world['Confirmed']).astype(str)

fig = go.Figure()



for i in range(len(limits)):

    lim = limits[i]

    df_sub = df_world.loc[(df_world['Confirmed'] < lim) & (df_world['Confirmed'] >= limits[i-1])]

    fig.add_trace(go.Scattergeo(

        locationmode = 'country names',

        lon = df_sub.Long,

        lat = df_sub.Lat,

        text = df_sub['text'],

        marker = dict(

            reversescale = True,

            size = df_sub.Confirmed*1.1,

            color = df_sub.Confirmed,

            colorscale = 'geyser',

            line_color='rgb(40,40,40)',

            line_width=0.5,

            sizemode = 'area'

        ),

        name = '{0}-{1}'.format(limits[i-1],limits[i])

    )

                 )



fig.update_layout(

        title_text = 'Confirmed cases Rest of the World',

        showlegend = True,

        geo = dict(

            scope = 'world',

            projection_type = 'natural earth',

            showcountries = True,

            showocean = False,

        )

    )



fig.show()
df_china = df.loc[df['Country'] == 'China'].groupby(['PS'])[['Country','Long','Lat','Confirmed']].max().reset_index()

rebinnable_interactive_histogram(df_china.loc[df_china['PS'] != 'Hubei'].Confirmed,'China (not Hubei)',initial_bin_width=24)
limits = [0,100,200,300,400,500,600,1000,1500,df_china.Confirmed.max()+1]



df_china['text'] = 'Province/State: ' + (df_china['PS']).astype(str) + '<br>Confirmed: ' + (df_china['Confirmed']).astype(str)

fig = go.Figure()



for i in range(len(limits)):

    df_sub = df_china.loc[(df_china['Confirmed'] < limits[i]) & (df_china['Confirmed'] >= limits[i-1])]

    fig.add_trace(go.Scattergeo(

        locationmode = 'country names',

        lon = df_sub.Long,

        lat = df_sub.Lat,

        text = df_sub['text'],

        marker = dict(

            opacity = .7,

            size = df_sub.Confirmed/10,

            color = df_sub.Confirmed.max(),

            colorscale = 'geyser',

            line_color='rgb(40,40,40)',

            line_width=0.5,

            sizemode = 'area'

        ),

        name = '{0}-{1}'.format(limits[i-1],limits[i])

    )

                 )



fig.update_layout(

        title = {'text': 'Corona Virus spreading in Asia',

                                'y':0.98,

                                'x':0.5,

                                'xanchor': 'center',

                                'yanchor': 'top'},

        showlegend = True,

        geo = dict(

            scope = 'asia',

            projection = go.layout.geo.Projection(

            type = 'kavrayskiy7',

            scale=1.2

            ),

            showcountries = True,

            

        )

    )



fig.show()
fig = go.Figure()

fig.add_trace(

    go.Bar(

        x=sorted_df.Country.loc[sorted_df['Country'] != 'China'],

        y=sorted_df.Confirmed_.loc[sorted_df['Country'] != 'China'],

        name='Rest of the world',

        marker_color='rgb(55, 83, 109)',

        text = sorted_df.Date.astype(str),

        hovertemplate =

        '<br><b>Country</b>: %{x} <br>' +

        '<b>Confirmed Cases:</b> %{y}<br>' +

        '<b>Date:</b> %{text}<br>'

    )

)



fig.update_layout(

    title={'text': 'Confirmed case all over the world',

           'y':0.95,

           'x':0.5,

           'xanchor': 'center',

           'yanchor': 'top'},

    xaxis_tickfont_size=14,

    xaxis=dict(tickangle=45),

    yaxis=dict(

        title='',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.1,

    bargroupgap=0.1,

    hoverlabel_align = 'left'

)

fig.show()
fig = go.Figure()

fig.add_trace(

    go.Bar(

        x=sorted_df.PS.loc[sorted_df['Country'] == 'China'],

        y=sorted_df.Confirmed_.loc[sorted_df['Country'] == 'China'],

        name='China',

        marker_color='rgb(26, 118, 255)',

        text = sorted_df.loc[sorted_df['Country'] == 'China'].Date.astype(str),

        hovertemplate =

        '<br><b>Province</b>: %{x} <br>' +

        '<b>Confirmed Cases:</b> %{y}<br>' +

        '<b>Date:</b> %{text}<br>'

    )

)



fig.update_layout(

    title={'text': 'Confirmed case in China',

           'y':0.95,

           'x':0.5,

           'xanchor': 'center',

           'yanchor': 'top'},

    xaxis_tickfont_size=14,

    xaxis=dict(tickangle=45),

    yaxis=dict(

        title='',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.15,

    bargroupgap=0.1,

    hoverlabel_align = 'left',

)

fig.show()



fig2 = go.Figure()

fig2.add_trace(

    go.Bar(

        x=sorted_df.PS.loc[(sorted_df['Country'] == 'China') & (sorted_df['PS'] != 'Hubei')],

        y=sorted_df.Confirmed_.loc[(sorted_df['Country'] == 'China') & (sorted_df['PS'] != 'Hubei')],

        name='China',

        marker_color='rgb(26, 118, 255)',

        text = df.Date.loc[(sorted_df['Country'] == 'China') & (sorted_df['PS'] != 'Hubei')].astype(str),

        hovertemplate =

        '<br><b>Province</b>: %{x} <br>' +

        '<b>Confirmed Cases:</b> %{y}<br>' +

        '<b>Date:</b> %{text}<br>'

    )

)



fig2.update_layout(

    title={'text': 'Confirmed case in China (not Hubei)',

           'y':0.95,

           'x':0.5,

           'xanchor': 'center',

           'yanchor': 'top'},

    xaxis_tickfont_size=14,

    xaxis=dict(tickangle=45),

    yaxis=dict(

        title='',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.15,

    bargroupgap=0.1,

    hoverlabel_align = 'left',

)

fig2.show()

fig = go.Figure()

fig.add_trace(

    go.Bar(

        x=sorted_df.Country.loc[sorted_df['Country'] != 'China'],

        y=sorted_df.Deaths_.loc[sorted_df['Country'] != 'China'],

        name='Deaths',

        marker_color='rgb(55, 83, 109)',

        text = sorted_df.Date.astype(str),

        hovertemplate =

        '<br><b>Country</b>: %{x} <br>' +

        '<b>Death Cases:</b> %{y}<br>' +

        '<b>Date:</b> %{text}<br>'

    )

)

fig.add_trace(

    go.Bar(

        x=sorted_df.Country.loc[sorted_df['Country'] != 'China'],

        y=sorted_df.Recovered_.loc[sorted_df['Country'] != 'China'],

        name='Recovered',

        marker_color='rgb(26, 118, 255)',

        text = sorted_df.Date.astype(str),

        hovertemplate =

        '<br><b>Country</b>: %{x} <br>' +

        '<b>Recovered Cases:</b> %{y}<br>' +

        '<b>Date:</b> %{text}<br>'

    )

)

fig.update_layout(

    title={'text': 'Deaths & Recovered case all over the world',

           'y':0.95,

           'x':0.5,

           'xanchor': 'center',

           'yanchor': 'top'},

    xaxis_tickfont_size=14,

    xaxis=dict(tickangle=45),

    yaxis=dict(

        title='',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=1,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.15, 

    bargroupgap=0.1,

    hoverlabel_align = 'left',

)

fig.show()



df_not_hubei = sorted_df.loc[sorted_df['PS'] != 'Hubei']





fig2 = go.Figure()

fig2.add_trace(

    go.Bar(

        x=df_not_hubei.PS.loc[df_not_hubei['Country'] == 'China'],

        y=df_not_hubei.Deaths_.loc[df_not_hubei['Country'] == 'China'],

        name='Deaths',

        marker_color='rgb(55, 83, 109)',

        text = sorted_df.Date.astype(str),

        hovertemplate =

        '<br><b>Country</b>: %{x} <br>' +

        '<b>Death Cases:</b> %{y}<br>' +

        '<b>Date:</b> %{text}<br>'

    )

)

fig2.add_trace(

    go.Bar(

        x=df_not_hubei.PS.loc[df_not_hubei['Country'] == 'China'],

        y=df_not_hubei.Recovered_.loc[df_not_hubei['Country'] == 'China'],

        name='Recovered',

        marker_color='rgb(26, 118, 255)',

        text = sorted_df.Date.astype(str),

        hovertemplate =

        '<br><b>Country</b>: %{x} <br>' +

        '<b>Recovered Cases:</b> %{y}<br>' +

        '<b>Date:</b> %{text}<br>'

    )

)

fig2.update_layout(

    title={'text': 'Deaths & Recovered case in China (not Hubei)',

           'y':0.95,

           'x':0.5,

           'xanchor': 'center',

           'yanchor': 'top'},

    xaxis_tickfont_size=14,

    xaxis=dict(tickangle=45),

    yaxis=dict(

        title='',

        titlefont_size=16,

        tickfont_size=14

        ,range = [0, df_not_hubei['Recovered'].max() + 10]

    ),

    legend=dict(

        x=1,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.15, 

    bargroupgap=0.1 ,

    hoverlabel_align = 'left',

)

fig2.show()
df_hubei = sorted_df.loc[sorted_df['PS'] == 'Hubei']



fig2 = go.Figure()



fig2.add_trace(

    go.Scatter(

        x=df_hubei.Date,

        y=df_hubei.Deaths,

        name='Deaths',

        mode='lines+markers',

        marker_color='rgb(55, 83, 109)',

         hovertemplate =

        '<br><b>Date</b>: %{x} <br>' +

        '<b>Death Cases:</b> %{y}<br>'

    )

)



fig2.add_trace(

    go.Scatter(

        x=df_hubei.Date,

        y=df_hubei.Recovered,

        name='Recovered',

        marker_color='rgb(26, 118, 255)',

         hovertemplate =

        '<br><b>Date</b>: %{x} <br>' +

        '<b>Recovered Cases:</b> %{y}<br>'

    )

)



fig2.update_traces(

    mode='lines+markers',

    marker_line_width=2,

    marker_size=5

)



fig2.update_layout(

    title={'text': 'Deaths and Recovered in Hubei (China)',

           'y':0.95,

           'x':0.5,

           'xanchor': 'center',

           'yanchor': 'top'},

    yaxis_zeroline=False,

    xaxis_zeroline=False,

    hoverlabel_align= 'left',

)



fig2.show()
df_hubei['confirmed_case_world'] = df_not_hubei.groupby('Date').sum()['Confirmed'].values



fig = go.Figure()

fig.add_trace(

    go.Scatter(

        x = df_hubei.Date,

        y = df_hubei.Confirmed,

        name = 'Hubei',

        mode = 'lines+markers',

        marker_color = 'rgb(55,83,109)',

        hovertemplate =

        '<br><b>Date</b>: %{x} <br>' +

        '<b>Confirmed Cases:</b> %{y}<br>'

    )

)



fig.add_trace(

    go.Scatter(

        x=df_hubei.Date,

        y=df_hubei.confirmed_case_world,

        name='Other',

        marker_color='rgb(26, 118, 255)',

        hovertemplate =

        '<b>Date</b>: %{x} <br>' +

        '<b>Confirmed Cases:</b> %{y}<br>'

    )

)



fig.update_traces(mode='lines+markers',

                  marker_line_width=2,

                  marker_size=5)

fig.update_layout(

    title={'text': 'Confermed case in Hubei vs Rest of World',

           'y':0.95,

           'x':0.5,

           'xanchor': 'center',

           'yanchor': 'top'},

    yaxis_zeroline=False,

    xaxis_zeroline=False,

    hoverlabel_align = 'left',

)



fig.show()
from fbprophet import Prophet

from fbprophet.diagnostics import cross_validation, performance_metrics

from fbprophet.plot import plot_cross_validation_metric, add_changepoints_to_plot, plot_plotly
df_prophet = df_hubei[['Date','Confirmed']]

df_prophet.columns = ['ds','y']
m_d = Prophet(

    yearly_seasonality=False,

    weekly_seasonality = False,

    daily_seasonality = True,

    seasonality_mode = 'additive')

m_d.fit(df_prophet)

future_d = m_d.make_future_dataframe(periods=7)

fcst_daily = m_d.predict(future_d)
trace1 = {

  "fill": None, 

  "mode": "markers", 

  "name": "actual no. of Confirmed", 

  "type": "scatter", 

  "x": df_prophet.ds, 

  "y": df_prophet.y

}

trace2 = {

  "fill": "tonexty", 

  "line": {"color": "#57b8ff"}, 

  "mode": "lines", 

  "name": "upper_band", 

  "type": "scatter", 

  "x": fcst_daily.ds, 

  "y": fcst_daily.yhat_upper

}

trace3 = {

  "fill": "tonexty", 

  "line": {"color": "#57b8ff"}, 

  "mode": "lines", 

  "name": "lower_band", 

  "type": "scatter", 

  "x": fcst_daily.ds, 

  "y": fcst_daily.yhat_lower

}

trace4 = {

  "line": {"color": "#eb0e0e"}, 

  "mode": "lines+markers", 

  "name": "prediction", 

  "type": "scatter", 

  "x": fcst_daily.ds, 

  "y": fcst_daily.yhat

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

    "title": "Confirmed nCov - Hubei", 

    "ticklen": 5, 

    "gridcolor": "rgb(255, 255, 255)", 

    "gridwidth": 2, 

    "zerolinewidth": 1

  }, 

}

fig = go.Figure(data=data, layout=layout)

iplot(fig)
m_nd = Prophet(

    yearly_seasonality=False,

    weekly_seasonality = False,

    daily_seasonality = False,

    seasonality_mode = 'additive')

m_nd.fit(df_prophet)

future_nd = m_nd.make_future_dataframe(periods=7)

fcst_no_daily = m_nd.predict(future_nd)
trace1 = {

  "fill": None, 

  "mode": "markers", 

  "name": "actual no. of Confirmed", 

  "type": "scatter", 

  "x": df_prophet.ds, 

  "y": df_prophet.y

}

trace2 = {

  "fill": "tonexty", 

  "line": {"color": "#57b8ff"}, 

  "mode": "lines", 

  "name": "upper_band", 

  "type": "scatter", 

  "x": fcst_no_daily.ds, 

  "y": fcst_no_daily.yhat_upper

}

trace3 = {

  "fill": "tonexty", 

  "line": {"color": "#57b8ff"}, 

  "mode": "lines", 

  "name": "lower_band", 

  "type": "scatter", 

  "x": fcst_no_daily.ds, 

  "y": fcst_no_daily.yhat_lower

}

trace4 = {

  "line": {"color": "#eb0e0e"}, 

  "mode": "lines+markers", 

  "name": "prediction", 

  "type": "scatter", 

  "x": fcst_no_daily.ds, 

  "y": fcst_no_daily.yhat

}

data = [trace1, trace2, trace3, trace4]

layout = {

  "title": "Confirmed - Time Series Forecast", 

  "xaxis": {

    "title": "", 

    "ticklen": 5, 

    "gridcolor": "rgb(255, 255, 255)", 

    "gridwidth": 2, 

    "zerolinewidth": 1

  }, 

  "yaxis": {

    "title": "Confirmed nCov - Hubei", 

    "ticklen": 5, 

    "gridcolor": "rgb(255, 255, 255)", 

    "gridwidth": 2, 

    "zerolinewidth": 1

  }, 

}

fig = go.Figure(data=data, layout=layout)

iplot(fig)
def mean_absolute_percentage_error(y_true, y_pred): 

    """Calculates MAPE given y_true and y_pred"""

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
max_date = df_prophet.ds.max()

y_true = df_prophet.y.values

y_pred_daily = fcst_daily.loc[fcst_daily['ds'] <= max_date].yhat.values

y_pred_no_daily = fcst_no_daily.loc[fcst_no_daily['ds'] <= max_date].yhat.values
print('MAPE with daily seasonality: {}'.format(mean_absolute_percentage_error(y_true,y_pred_daily)))

print('MAPE without daily seasonality: {}'.format(mean_absolute_percentage_error(y_true,y_pred_no_daily)))
m_d = Prophet(

    changepoint_prior_scale=20,

    seasonality_prior_scale=20,

    n_changepoints=19,

    changepoint_range=0.9,

    yearly_seasonality=False,

    weekly_seasonality = False,

    daily_seasonality = True,

    seasonality_mode = 'additive')

m_d.fit(df_prophet)

future_d = m_d.make_future_dataframe(periods=7)

fcst_daily = m_d.predict(future_d)
trace1 = {

  "fill": None, 

  "mode": "markers", 

  "name": "actual no. of Confirmed", 

  "type": "scatter", 

  "x": df_prophet.ds, 

  "y": df_prophet.y

}

trace2 = {

  "fill": "tonexty", 

  "line": {"color": "#57b8ff"}, 

  "mode": "lines", 

  "name": "upper_band", 

  "type": "scatter", 

  "x": fcst_daily.ds, 

  "y": fcst_daily.yhat_upper

}

trace3 = {

  "fill": "tonexty", 

  "line": {"color": "#57b8ff"}, 

  "mode": "lines", 

  "name": "lower_band", 

  "type": "scatter", 

  "x": fcst_daily.ds, 

  "y": fcst_daily.yhat_lower

}

trace4 = {

  "line": {"color": "#eb0e0e"}, 

  "mode": "lines+markers", 

  "name": "prediction", 

  "type": "scatter", 

  "x": fcst_daily.ds, 

  "y": fcst_daily.yhat

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

    "title": "Confirmed nCov - Hubei", 

    "ticklen": 5, 

    "gridcolor": "rgb(255, 255, 255)", 

    "gridwidth": 2, 

    "zerolinewidth": 1

  }, 

}

fig = go.Figure(data=data, layout=layout)

iplot(fig)
m_nd = Prophet(

    changepoint_range=0.90,

    changepoint_prior_scale=20,

    n_changepoints=19,

    yearly_seasonality=False,

    weekly_seasonality = False,

    daily_seasonality = False,

    seasonality_mode = 'additive')

m_nd.fit(df_prophet)

future_nd = m_nd.make_future_dataframe(periods=7)

fcst_no_daily = m_nd.predict(future_nd)
trace1 = {

  "fill": None, 

  "mode": "markers", 

  "name": "actual no. of Confirmed", 

  "type": "scatter", 

  "x": df_prophet.ds, 

  "y": df_prophet.y

}

trace2 = {

  "fill": "tonexty", 

  "line": {"color": "#57b8ff"}, 

  "mode": "lines", 

  "name": "upper_band", 

  "type": "scatter", 

  "x": fcst_no_daily.ds, 

  "y": fcst_no_daily.yhat_upper

}

trace3 = {

  "fill": "tonexty", 

  "line": {"color": "#57b8ff"}, 

  "mode": "lines", 

  "name": "lower_band", 

  "type": "scatter", 

  "x": fcst_no_daily.ds, 

  "y": fcst_no_daily.yhat_lower

}

trace4 = {

  "line": {"color": "#eb0e0e"}, 

  "mode": "lines+markers", 

  "name": "prediction", 

  "type": "scatter", 

  "x": fcst_no_daily.ds, 

  "y": fcst_no_daily.yhat

}

data = [trace1, trace2, trace3, trace4]

layout = {

  "title": "Confirmed - Time Series Forecast", 

  "xaxis": {

    "title": "", 

    "ticklen": 5, 

    "gridcolor": "rgb(255, 255, 255)", 

    "gridwidth": 2, 

    "zerolinewidth": 1

  }, 

  "yaxis": {

    "title": "Confirmed nCov - Hubei", 

    "ticklen": 5, 

    "gridcolor": "rgb(255, 255, 255)", 

    "gridwidth": 2, 

    "zerolinewidth": 1

  }, 

}

fig = go.Figure(data=data, layout=layout)

iplot(fig)
y_true = df_prophet.y.values

y_pred_daily = fcst_daily.loc[fcst_daily['ds'] <= max_date].yhat.values

y_pred_no_daily = fcst_no_daily.loc[fcst_no_daily['ds'] <= max_date].yhat.values
print('MAPE with daily seasonality: {}'.format(mean_absolute_percentage_error(y_true,y_pred_daily)))

print('MAPE without daily seasonality: {}'.format(mean_absolute_percentage_error(y_true,y_pred_no_daily)))
df_ch_d = pd.DataFrame()

df_ch_nd = pd.DataFrame()



df_ch_d['deltas'] = m_d.params['delta'].mean(0)

df_ch_d['x'] = [x for x in range(19)]



df_ch_nd['deltas'] = m_nd.params['delta'].mean(0)

df_ch_nd['x'] = [x for x in range(19)]



fig = go.Figure()

fig2 = go.Figure()



fig.add_trace(

    go.Bar(

        x=df_ch_d.x,

        y=df_ch_d.deltas,

        name='# of changepoints',

        marker_color='rgb(55, 83, 109)',

        hovertemplate ="Change Rate: %{y: .2f}<extra></extra>",

        

    )

)



fig.update_layout(

    title={'text': 'Barplot of ChangePoints - Daily Model',

           'y':0.95,

           'x':0.5,

           'xanchor': 'center',

           'yanchor': 'top'},

    xaxis_tickfont_size=14,

    xaxis=dict(

        title = 'Potential ChangePoint'),

    yaxis=dict(

        title='Rate Change',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.1,

    bargroupgap=0.1

)





fig2.add_trace(

    go.Bar(

        x=df_ch_nd.x,

        y=df_ch_nd.deltas,

        name='# of changepoints',

        marker_color='rgb(55, 83, 109)',

        hovertemplate ="Change Rate: %{y: .2f}<extra></extra>",

    )

)



fig2.update_layout(

    title={'text': 'Barplot of ChangePoints - Non Daily Model',

           'y':0.95,

           'x':0.5,

           'xanchor': 'center',

           'yanchor': 'top'},

    xaxis_tickfont_size=14,

    xaxis=dict(

        title = 'Potential ChangePoint'),

    yaxis=dict(

        title='Rate Change',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.1,

    bargroupgap=0.1

)



fig.show()

fig2.show()
fig = plot_plotly(m_nd, fcst_no_daily) 

fig.update_layout(

    title={'text': 'Prediction Confermed cases in Hubei',

           'y':0.95,

           'x':0.5,

           'xanchor': 'center',

           'yanchor': 'top'},

    yaxis=dict(

        title='Confirmed Cases',

        titlefont_size=16,

        tickfont_size=14,

    )

)

fig.show()
df_death = df_hubei[['Date','Deaths']]

df_death.columns = ['ds','y']
m_death = Prophet(

    changepoint_range=0.90,

    changepoint_prior_scale=20,

    n_changepoints=17,

    yearly_seasonality=False,

    weekly_seasonality = False,

    daily_seasonality = False,

    seasonality_mode = 'additive')

m_death.fit(df_death)

future_death = m_death.make_future_dataframe(periods=7)

fcst_death = m_death.predict(future_death)
trace1 = {

  "fill": None, 

  "mode": "markers",

  "marker_size": 10,

  "name": "actual no. of Confirmed", 

  "type": "scatter", 

  "x": df_death.ds, 

  "y": df_death.y

}

trace2 = {

  "fill": "tonexty", 

  "line": {"color": "#57b8ff"}, 

  "mode": "lines", 

  "name": "upper_band", 

  "type": "scatter", 

  "x": fcst_death.ds, 

  "y": fcst_death.yhat_upper

}

trace3 = {

  "fill": "tonexty", 

  "line": {"color": "#57b8ff"}, 

  "mode": "lines", 

  "name": "lower_band", 

  "type": "scatter", 

  "x": fcst_death.ds, 

  "y": fcst_death.yhat_lower

}

trace4 = {

  "line": {"color": "#eb0e0e"}, 

  "mode": "lines+markers",

  "marker_size": 4,

  "name": "prediction", 

  "type": "scatter", 

  "x": fcst_death.ds, 

  "y": fcst_death.yhat

}

data = [trace1, trace2, trace3, trace4]

layout = {

  "title": "Deaths - Time Series Forecast", 

  "xaxis": {

    "title": "Monthly Dates", 

    "ticklen": 5, 

    "gridcolor": "rgb(255, 255, 255)", 

    "gridwidth": 2, 

    "zerolinewidth": 1

  }, 

  "yaxis": {

    "title": "Deaths nCov - Hubei", 

    "ticklen": 5, 

    "gridcolor": "rgb(255, 255, 255)", 

    "gridwidth": 2, 

    "zerolinewidth": 1

  }, 

}

fig = go.Figure(data=data, layout=layout)

iplot(fig)
max_date = df_death.ds.max()

y_true = df_death.y.values

y_pred_death = fcst_death.loc[fcst_death['ds'] <= max_date].yhat.values
print('MAPE with daily seasonality: {}'.format(mean_absolute_percentage_error(y_true,y_pred_death)))