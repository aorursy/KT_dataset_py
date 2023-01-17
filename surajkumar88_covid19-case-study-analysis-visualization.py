import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#Plotly Libraris

import plotly.express as px

import plotly.graph_objects as go

#import plotly.figure_factory as ff

#from plotly.colors import n_colors

from plotly.subplots import make_subplots

# Minmax scaler

from sklearn.preprocessing import MinMaxScaler



#itertools

import itertools



#dataframe display settings

pd.set_option('display.max_columns', 5000000)

pd.set_option('display.max_rows', 50000000)



#to suppress un-necessary warnings

import warnings  

warnings.filterwarnings('ignore')
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

latest_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/08-22-2020.csv')

us_medical_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/08-22-2020.csv')

#apple_mobility = pd.read_csv('https://covid19-static.cdn-apple.com/covid19-mobility-data/2015HotfixDev7/v3/en-us/applemobilitytrends-2020-08-21.csv')
confirmed_df.head()
confirmed_group_df = confirmed_df.groupby(by='Country/Region',as_index=False).sum()

deaths_group_df = deaths_df.groupby(by='Country/Region',as_index=False).sum()

recoveries_group_df = recoveries_df.groupby(by='Country/Region',as_index=False).sum()



active_group_df = pd.DataFrame(columns=[confirmed_group_df.columns])

active_group_df = deaths_group_df.copy()

for i in range(confirmed_group_df.shape[0]):

    for j in range(3, confirmed_group_df.shape[1]):

        active_group_df.iloc[i,j] = confirmed_group_df.iloc[i,j]-(recoveries_group_df.iloc[i,j]+deaths_group_df.iloc[i,j])
confirmed_df.describe()
base_stats = pd.DataFrame(columns=['Dates','Confirmed','Deaths','Recovered','Active'])

base_stats['Dates'] = confirmed_df.columns[4:]



base_stats['Confirmed'] = base_stats['Dates'].apply(lambda x: confirmed_df[x].sum())

base_stats['Deaths'] = base_stats['Dates'].apply(lambda x: deaths_df[x].sum())

base_stats['Recovered'] = base_stats['Dates'].apply(lambda x: recoveries_df[x].sum())

base_stats.reset_index(drop=False, inplace=True)

base_stats['Active'] = base_stats['index'].apply(lambda x: (base_stats['Confirmed'][x]-(base_stats['Deaths'][x]+base_stats['Recovered'][x])))

base_stats.head()
latest_stats_fig = go.Figure()

latest_stats_fig.add_trace(go.Treemap(labels = ['Confirmed','Active','Recovered','Deaths'],

                                     parents = ['','Confirmed','Confirmed','Confirmed'],

                                     values = [base_stats['Confirmed'].sum(), base_stats['Active'].sum(), base_stats['Recovered'].sum(), base_stats['Deaths'].sum()],

                                      branchvalues="total", marker_colors = ['#118ab2','#ef476f','#06d6a0','#073b4c'],

                                      textinfo = "label+text+value",

                                      outsidetextfont = {"size": 30, "color": "darkblue"},

                                      marker = {"line": {"width": 2}},

                                        pathbar = {"visible": False}

                                     ))

latest_stats_fig.update_layout(#width=1000, 

                               height=300)

latest_stats_fig.show()
base_stats_fig = go.Figure()



for column in base_stats.columns.to_list()[2:6]:

    color_dict = {

      "Confirmed": "#118ab2",

      "Active": "#ef476f",

      "Recovered": "#06d6a0",

      "Deaths": "#073b4c"

        }

    base_stats_fig.add_trace(

        go.Scatter(

            x = base_stats['Dates'],

            y = base_stats[column],

            name = column,

            line = dict(color=color_dict[column]),

            hovertemplate ='<br><b>Date</b>: %{x}'+'<br><i>Count</i>:'+'%{y}',

        )

    )

    

for column in base_stats.columns.to_list()[2:6]:

    color_dict = {

      "Confirmed": "#149ECC",

      "Active": "#F47C98",

      "Recovered": "#24F9C1",

      "Deaths": "#0C6583"

        }

    base_stats_fig.add_trace(

        go.Scatter(

            x = base_stats['Dates'],

            y = base_stats['index'].apply(lambda x: (base_stats[column][x-7:x].sum())/7 if x>7 else (base_stats[column][0:x].sum())/7),

            name = column+" 7-day Moving Avg.",

            line = dict(dash="dash", color=color_dict[column]), showlegend=False,

            hovertemplate = '<br><b>Date</b>: %{x}'+'<br><i>7-day moving avg.</i>: %{y}'

        )

    )

    

base_stats_fig.update_layout(

    updatemenus=[

        dict(

        buttons=list(

            [dict(label = 'All Cases',

                  method = 'update',

                  args = [{'visible': [True, True, True, True, True, True, True, True]},

                          {'title': 'All Cases',

                           'showlegend':True}]),

             dict(label = 'Confirmed',

                  method = 'update',

                  args = [{'visible': [True, False, False, False, True, False, False, False]},

                          {'title': 'Confirmed',

                           'showlegend':True}]),

             dict(label = 'Active',

                  method = 'update',

                  args = [{'visible': [False, False, False, True, False, False, False, True]},

                          {'title': 'Active',

                           'showlegend':True}]),

             dict(label = 'Recovered',

                  method = 'update',

                  args = [{'visible': [False, False, True, False, False, False, True, False]},

                          {'title': 'Recovered',

                           'showlegend':True}]),

             dict(label = 'Deaths',

                  method = 'update',

                  args = [{'visible': [False, True, False, False, False, True, False, False]},

                          {'title': 'Deaths',

                           'showlegend':True}]),

            ]),

             type = "dropdown",

             direction="down",

#             pad={"r": 10, "t": 40},

             showactive=True,

             x=0,

             xanchor="left",

             y=1.25,

             yanchor="top"

        ),

        dict(

        buttons=list(

            [dict(label = 'Linear Scale',

                  method = 'relayout',

                  args = [{'yaxis': {'type': 'linear'}},

                          {'title': 'All Cases',

                           'showlegend':True}]),

             dict(label = 'Log Scale',

                  method = 'relayout',

                  args = [{'yaxis': {'type': 'log'}},

                          {'title': 'Confirmed',

                           'showlegend':True}]),

            ]),

             type = "dropdown",

             direction="down",

#             pad={"r": 10, "t": 10},

             showactive=True,

             x=0,

             xanchor="left",

             y=1.36,

             yanchor="top"

        )

    ])



# Add range slider

# base_stats_fig.update_layout(

#     xaxis=dict(

#         rangeselector=dict(

#             buttons=list([

#                 dict(count=10,

#                      label="10y",

#                      step="day",

#                      stepmode="backward"),

#                 dict(count=20,

#                      label="20y",

#                      step="day",

#                      stepmode="backward"),

#                 dict(count=50,

#                      label="50y",

#                      step="day",

#                      stepmode="todate"),

#                 dict(count=100,

#                      label="100y",

#                      step="day",

#                      stepmode="backward"),

#                 dict(step="all")

#             ])

#         ),

#         rangeslider=dict(

#             visible=True

#         ),

#         type="date"

#     )

# )



base_stats_fig.update_xaxes(showticklabels=False)

base_stats_fig.update_layout(

    #height=600, width=600, 

    title_text="Basic Statistics for Covid19", title_x=0.5, title_font_size=20,

                            legend=dict(orientation='h',yanchor='top',y=1.15,xanchor='right',x=1), paper_bgcolor="mintcream",

                            xaxis_title="Date", yaxis_title="# of Cases")

base_stats_fig.show()
daily_case_fig = make_subplots(rows=2, cols=2, vertical_spacing=0.05, horizontal_spacing=0.04, # shared_yaxes=True,

                           subplot_titles=('Confirmed','Active','Recovered','Deaths'),

                            x_title='Dates', y_title='# of Cases',)



daily_case_fig.add_trace(go.Bar(x=base_stats['Dates'], y=base_stats['index'].apply(lambda x: base_stats['Confirmed'][x]-base_stats['Confirmed'][x-1:x].sum()),

                              name='Confirmed',hovertemplate = '<br><b>Date</b>: %{x}'+'<br><i>Confirmed Count</i>: %{y}',

                                marker=dict(color='#118ab2')),row=1, col=1)

daily_case_fig.add_trace(go.Scatter(x=base_stats['Dates'], y=base_stats['index'].apply(lambda x: (base_stats['Confirmed'][x-7:x].sum()-base_stats['Confirmed'][x-8:x-1].sum())/7 if x>0 else 0),

                             name='7-day moving average', hovertemplate = '<br><b>Date</b>: %{x}'+'<br><i>7-day average</i>: %{y}', showlegend=False,

                                    line=dict(dash="dash", color='#149ECC')),row=1, col=1)



daily_case_fig.add_trace(go.Bar(x=base_stats['Dates'], y=base_stats['index'].apply(lambda x: base_stats['Active'][x]-base_stats['Active'][x-1:x].sum()), 

                             name='Active',hovertemplate = '<br><b>Date</b>: %{x}'+'<br><i>Active Count</i>: %{y}',

                               marker=dict(color='#ef476f')),row=1, col=2)

daily_case_fig.add_trace(go.Scatter(x=base_stats['Dates'], y=base_stats['index'].apply(lambda x: (base_stats['Active'][x-7:x].sum()-base_stats['Active'][x-8:x-1].sum())/7 if x>0 else 0),

                             name='7-day moving average', hovertemplate = '<br><b>Date</b>: %{x}'+'<br><i>7-day average</i>: %{y}', showlegend=False,

                                    line=dict(dash="dash", color='#F47C98')),row=1, col=2)



daily_case_fig.add_trace(go.Bar(x=base_stats['Dates'], y=base_stats['index'].apply(lambda x: base_stats['Recovered'][x]-base_stats['Recovered'][x-1:x].sum()), 

                              name='Recovered',hovertemplate = '<br><b>Date</b>: %{x}'+'<br><i>Recovered Count</i>: %{y}',

                               marker=dict(color='#06d6a0')),row=2, col=1)

daily_case_fig.add_trace(go.Scatter(x=base_stats['Dates'], y=base_stats['index'].apply(lambda x: (base_stats['Recovered'][x-7:x].sum()-base_stats['Recovered'][x-8:x-1].sum())/7 if x>0 else 0),

                             name='7-day moving average', hovertemplate = '<br><b>Date</b>: %{x}'+'<br><i>7-day average</i>: %{y}', showlegend=False,

                                    line=dict(dash="dash", color='#24F9C1')),row=2, col=1)



daily_case_fig.add_trace(go.Bar(x=base_stats['Dates'], y=base_stats['index'].apply(lambda x: base_stats['Deaths'][x]-base_stats['Deaths'][x-1:x].sum()), 

                              name='Deaths',hovertemplate = '<br><b>Date</b>: %{x}'+'<br><i>Death Count</i>: %{y}',

                               marker=dict(color='#073b4c')),row=2, col=2)

daily_case_fig.add_trace(go.Scatter(x=base_stats['Dates'], y=base_stats['index'].apply(lambda x: (base_stats['Deaths'][x-7:x].sum()-base_stats['Deaths'][x-8:x-1].sum())/7 if x>0 else 0),

                             name='7-day moving average', hovertemplate = '<br><b>Date</b>: %{x}'+'<br><i>7-day average</i>: %{y}', line=dict(dash="dash", color='#0C6583')),row=2, col=2)









daily_case_fig.update_xaxes(showticklabels=False)

daily_case_fig.update_layout(

    #height=600, width=1100, 

    title_text="Daily change in cases of Covid19", title_x=0.5, title_font_size=20,

                            legend=dict(orientation='h',yanchor='top',y=1.1,xanchor='right',x=1), paper_bgcolor="mintcream")





daily_case_fig.show()                    
country_data = go.Figure()

country_data.add_trace(go.Table(

    header=dict(values=['Country','Confirmed','Active','Recovered','Deaths','Daily Increase','Mortality Rate'],

                fill = dict(color='#A5B3F3'),

                line_color='darkslategray',

                align = ['left'] * 5),

    cells=dict(values=[confirmed_group_df.sort_values(by=confirmed_group_df.columns[-1], ascending=False)['Country/Region'], 

                      confirmed_group_df.sort_values(by=confirmed_group_df.columns[-1], ascending=False)['Country/Region'].apply(lambda x: confirmed_group_df[confirmed_group_df['Country/Region']==x][confirmed_group_df.columns[4:]].values.tolist()[0][-1]),

                      confirmed_group_df.sort_values(by=confirmed_group_df.columns[-1], ascending=False)['Country/Region'].apply(lambda x: active_group_df[active_group_df['Country/Region']==x][active_group_df.columns[4:]].values.tolist()[0][-1]),

                      confirmed_group_df.sort_values(by=confirmed_group_df.columns[-1], ascending=False)['Country/Region'].apply(lambda x: recoveries_group_df[recoveries_group_df['Country/Region']==x][recoveries_group_df.columns[4:]].values.tolist()[0][-1]),

                      confirmed_group_df.sort_values(by=confirmed_group_df.columns[-1], ascending=False)['Country/Region'].apply(lambda x: deaths_group_df[deaths_group_df['Country/Region']==x][deaths_group_df.columns[4:]].values.tolist()[0][-1]),

                      confirmed_group_df.sort_values(by=confirmed_group_df.columns[-1], ascending=False)['Country/Region'].apply(lambda x: confirmed_group_df[confirmed_group_df['Country/Region']==x][confirmed_group_df.columns[4:]].values.tolist()[0][-1]-confirmed_group_df[confirmed_group_df['Country/Region']==x][confirmed_group_df.columns[4:]].values.tolist()[0][-2]),

                      confirmed_group_df.sort_values(by=confirmed_group_df.columns[-1], ascending=False)['Country/Region'].apply(lambda x: (deaths_group_df[deaths_group_df['Country/Region']==x][deaths_group_df.columns[4:]].values.tolist()[0][-1]/confirmed_group_df[confirmed_group_df['Country/Region']==x][confirmed_group_df.columns[4:]].values.tolist()[0][-1])*100).round(decimals=3)

                      ],

               fill = dict(color='#F0FCFD'),

               line_color='darkslategray',

               align = ['left'] * 5)))



country_data.update_layout(

    #height=600, width=1100, 

    title_text="Countries with no active cases",

                                     title_x=0.5, title_font_size=20,

                                     paper_bgcolor="mintcream")

country_data.show()
base_stats_map_fig = go.Figure()

df_dict={

  "Confirmed": [confirmed_group_df,"blues",True],

  "Active": [active_group_df,"reds",False],

  "Recovered": [recoveries_group_df,"greens",False],

  "Deaths": [deaths_group_df,"gray_r",False],

  "Daily_inc": [None, "oranges", False]

}

for filter_name in ['Confirmed','Active','Recovered','Deaths']:



    base_stats_map_fig.add_trace(go.Choropleth(locations=df_dict[filter_name][0]['Country/Region'],

                                       z=df_dict[filter_name][0][confirmed_group_df.columns[-1]],

                                       locationmode='country names', name=filter_name,

                                       colorscale=df_dict[filter_name][1], showscale=False,

                                       colorbar_title="# of Cases World wide", visible=df_dict[filter_name][2],

                                               hoverinfo = 'all',

                                       ))

    



base_stats_map_fig.add_trace(go.Choropleth(locations=confirmed_group_df.sort_values(by=confirmed_group_df.columns[-1], ascending=False)['Country/Region'],

                                       z=confirmed_group_df.sort_values(by=confirmed_group_df.columns[-1], ascending=False)['Country/Region'].apply(lambda x: confirmed_group_df[confirmed_group_df['Country/Region']==x][confirmed_group_df.columns[4:]].values.tolist()[0][-1]-confirmed_group_df[confirmed_group_df['Country/Region']==x][confirmed_group_df.columns[4:]].values.tolist()[0][-2]),

                                       locationmode='country names', name='Daily increase',

                                       colorscale=df_dict['Daily_inc'][1], showscale=False,

                                       colorbar_title="# of new Cases World wide", visible=df_dict['Daily_inc'][2],

                                               hoverinfo = 'all',

                                       ))    

    

    

    

base_stats_map_fig.update_layout(

    updatemenus=[

        dict(

        buttons=list(

            [dict(label = 'Confirmed',

                  method = 'update',

                  args = [{'visible': [True, False, False, False, False]},

                          {'title': 'Confirmed',

                           'showlegend':True}]),

             dict(label = 'Active',

                  method = 'update',

                  args = [{'visible': [False, True, False, False, False]},

                          {'title': 'Active',

                           'showlegend':True}]),

             dict(label = 'Recovered',

                  method = 'update',

                  args = [{'visible': [False, False, True, False, False]},

                          {'title': 'Recovered',

                           'showlegend':True}]),

             dict(label = 'Deaths',

                  method = 'update',

                  args = [{'visible': [False, False, False, True, False]},

                          {'title': 'Deaths',

                           'showlegend':True}]),

             dict(label = 'Daily Increase',

                  method = 'update',

                  args = [{'visible': [False, False, False, False, True]},

                          {'title': 'Daily Increase',

                           'showlegend':True}]),

            ]),

             type = "buttons",

             direction="right",

#             pad={"r": 10, "t": 40},

             showactive=True,

             x=-0.1,

             xanchor="left",

             y=1.1,

             yanchor="top"

        )

    ])



base_stats_map_fig.update_xaxes(showticklabels=False)

base_stats_map_fig.update_layout(

    #height=600, width=1100, 

    title_text="# of Cases World wide", title_x=0.5, title_font_size=20,

                            legend=dict(orientation='h',yanchor='top',y=1.12,xanchor='right',x=1), paper_bgcolor="mintcream")

base_stats_map_fig.show()
imp_ratios_fig = go.Figure()

color_dict = {

  "Confirmed": ["#118ab2",(base_stats['Deaths']/base_stats['Confirmed'])*100, 'Deaths/100 Cases','#149ECC'],

  "Active": ["#ef476f",(base_stats['Deaths']/base_stats['Recovered'])*100, 'Deaths/100 Recovered','#F47C98'],

  "Recovered": ["#06d6a0",(base_stats['Recovered']/base_stats['Confirmed'])*100, 'Recovered/100 cases','#24F9C1'],

  "Deaths": ["#073b4c",(base_stats['Recovered']/base_stats['Deaths'])*100, 'Recovered/100 Deaths','#0C6583']

    }

for column in base_stats.columns.to_list()[2:6]:



    imp_ratios_fig.add_trace(go.Scatter(x = base_stats['Dates'],y = color_dict[column][1],

            name = color_dict[column][2],line = dict(color=color_dict[column][0]),

            hovertemplate ='<br><b>Date</b>: %{x}'+'<br><i>Ratio </i>:'+'%{y}'))

    imp_ratios_fig.add_trace(go.Scatter(x = base_stats['Dates'],y = [color_dict[column][1].mean()]*base_stats['Dates'].shape[0],

            name = "Mean value",line = dict(dash="dash", color=color_dict[column][3]),

            hovertemplate ='<br><i>Mean value </i>:'+'%{y}', visible=False))



imp_ratios_fig.update_layout(

    updatemenus=[

        dict(

        buttons=list(

            [dict(label = 'All Ratios',

                  method = 'update',

                  args = [{'visible': [True, False, True, False, True, False, True, False]},

                          {'title': 'All Cases',

                           'showlegend':True}]),

             dict(label = 'Deaths/100 Cases<br>(Mortality rate)',

                  method = 'update',

                  args = [{'visible': [True, True, False, False, False, False, False, False]},

                          {'title': 'Confirmed',

                           'showlegend':True}]),

             dict(label = 'Deaths/100 Recovered',

                  method = 'update',

                  args = [{'visible': [False, False, True, True, False, False, False, False]},

                          {'title': 'Active',

                           'showlegend':True}]),

             dict(label = 'Recovered/100 cases<br>(Recovery rate)',

                  method = 'update',

                  args = [{'visible': [False, False, False, False, True, True, False, False]},

                          {'title': 'Recovered',

                           'showlegend':True}]),

             dict(label = 'Recovered/100 Deaths',

                  method = 'update',

                  args = [{'visible': [False, False, False, False, False, False, True, True]},

                          {'title': 'Deaths',

                           'showlegend':True}]),

            ]),

             type = "buttons",

             direction="down",

#             pad={"r": 10, "t": 40},

             showactive=True,

             x=1.01,

             xanchor="left",

             y=1,

             yanchor="top"

        )

    ])



imp_ratios_fig.update_xaxes(showticklabels=False)

imp_ratios_fig.update_layout(

    #height=500, width=1100, 

    title_text="Important Ratios for Covid19", title_x=0.5, title_font_size=20,

                            legend=dict(orientation='h',yanchor='top',y=1.12,xanchor='right',x=1), paper_bgcolor="mintcream",

                            xaxis_title="Date", yaxis_title="Ratio")

imp_ratios_fig.show()
treemap_fig = go.Figure()

df_dict={

  "Confirmed": [confirmed_group_df,True],

  "Active": [active_group_df,False],

  "Recovered": [recoveries_group_df,False],

  "Deaths": [deaths_group_df,False],

  "Daily_inc": [None,False]

}

for column in ['Confirmed','Active','Recovered','Deaths']:



    treemap_fig.add_trace(go.Treemap(labels = confirmed_group_df['Country/Region'], name="Treemap",

                                     parents = ['']*confirmed_group_df.shape[0],

                                     values = df_dict[column][0][confirmed_group_df.columns[-1]],

                                     branchvalues="total",

                                     textinfo = "percent root+label+value+text", outsidetextfont = {"size": 30, "color": "darkblue"},

                                     marker = {"line": {"width": 2}}, pathbar = {"visible": False}, visible = df_dict[column][1], 

                                     hovertemplate='<b>%{label} </b> <br> Count: %{value}<br>'

                                     )) 

    

treemap_fig.add_trace(go.Treemap(labels = confirmed_group_df.sort_values(by=confirmed_group_df.columns[-1], ascending=False)['Country/Region'], name="Treemap",

                                 parents = ['']*confirmed_group_df.shape[0],

                                 values = confirmed_group_df.sort_values(by=confirmed_group_df.columns[-1], ascending=False)['Country/Region'].apply(lambda x: confirmed_group_df[confirmed_group_df['Country/Region']==x][confirmed_group_df.columns[4:]].values.tolist()[0][-1]-confirmed_group_df[confirmed_group_df['Country/Region']==x][confirmed_group_df.columns[4:]].values.tolist()[0][-2]),

                                 branchvalues="total",

                                 textinfo = "percent root+label+value+text", outsidetextfont = {"size": 30, "color": "darkblue"},

                                 marker = {"line": {"width": 2}}, pathbar = {"visible": False}, visible = df_dict['Daily_inc'][1], 

                                 hovertemplate='<b>%{label} </b> <br> Count: %{value}<br>'

                                 )) 



treemap_fig.update_layout(

    updatemenus=[

        dict(

        buttons=list([

             dict(label = 'Confirmed',

                  method = 'update',

                  args = [{'visible': [True, False, False, False, False]},

                          {'title': 'Confirmed',

                           'showlegend':True}]),

             dict(label = 'Active',

                  method = 'update',

                  args = [{'visible': [False, True, False, False, False]},

                          {'title': 'Active',

                           'showlegend':True}]),

             dict(label = 'Recovered',

                  method = 'update',

                  args = [{'visible': [False, False, True, False, False]},

                          {'title': 'Recovered',

                           'showlegend':True}]),

             dict(label = 'Deaths',

                  method = 'update',

                  args = [{'visible': [False, False, False, True, False]},

                          {'title': 'Deaths',

                           'showlegend':True}]),

            dict(label = 'Daily Increase',

                  method = 'update',

                  args = [{'visible': [False, False, False, False, True]},

                          {'title': 'Daily Increase',

                           'showlegend':True}]),

            ]),

             type = "buttons",

             direction="down",

#             pad={"r": 10, "t": 40},

             showactive=True,

             x=1.01,

             xanchor="left",

             y=0.8,

             yanchor="top"

        )

    ])



treemap_fig.update_layout(

    #height=600, width=1100, 

    title_text="Treemap of Countries <br> The Treemap shows the number of Cases in Different coutries <br> and their percent of total cases worldwide",

                          title_x=0.5, title_font_size=15,

                          legend=dict(orientation='h',yanchor='top',y=1.12,xanchor='right',x=1), paper_bgcolor="mintcream")

treemap_fig.show()
base_stats['Dates'] = pd.to_datetime(base_stats["Dates"])

base_stats.set_index(base_stats["Dates"],inplace=True)
week_month_fig = make_subplots(rows=1, cols=3, vertical_spacing=0.05, horizontal_spacing=0.04, # shared_yaxes=True,

                           subplot_titles=('Daily Statistics','Weekly Statistics','Monthly Statistics'),y_title='# of Cases',)



for column in ['Confirmed','Active','Recovered','Deaths']:

    df_dict={

      "Confirmed": [confirmed_group_df,"#118ab2",True],

      "Active": [active_group_df,"#ef476f",False],

      "Recovered": [recoveries_group_df,"#06d6a0",False],

      "Deaths": [deaths_group_df,"#073b4c",False]        

    }

    week_month_fig.add_trace(go.Bar(x=list(range(len(base_stats[column].resample('D').sum()))),

                            y=base_stats[column].resample('D').sum(), visible = df_dict[column][2],

                            name='Daily '+column,hovertemplate = '<br><b>day</b>: %{x}'+'<br><i>Confirmed Count</i>: %{y}',

                            marker=dict(color=df_dict[column][1]), showlegend=False) ,row=1, col=1)

    week_month_fig.add_trace(go.Bar(x=list(range(len(base_stats[column].resample('W').sum()))),

                            y=base_stats[column].resample('W').sum(), visible = df_dict[column][2],

                            name='Weekly '+column,hovertemplate = '<br><b>Week</b>: %{x}'+'<br><i>Confirmed Count</i>: %{y}',

                            marker=dict(color=df_dict[column][1]), showlegend=False) ,row=1, col=2)

    week_month_fig.add_trace(go.Bar(x=list(range(len(base_stats[column].resample('M').sum()))),

                            y=base_stats[column].resample('M').sum(), visible = df_dict[column][2],

                            name='Monthly '+column,hovertemplate = '<br><b>Month</b>: %{x}'+'<br><i>Confirmed Count</i>: %{y}',

                            marker=dict(color=df_dict[column][1]), showlegend=False) ,row=1, col=3)





week_month_fig.update_layout(

    updatemenus=[

        dict(

        buttons=list([

             dict(label = 'Confirmed',

                  method = 'update',

                  args = [{'visible': [True, True, True, False, False, False, False, False, False, False, False, False]},

                          {'title': 'Confirmed',

                           'showlegend':True}]),

             dict(label = 'Active',

                  method = 'update',

                  args = [{'visible': [False, False, False, True, True, True, False, False, False, False, False, False]},

                          {'title': 'Active',

                           'showlegend':True}]),

             dict(label = 'Recovered',

                  method = 'update',

                  args = [{'visible': [False, False, False, False, False, False, True, True, True, False, False, False]},

                          {'title': 'Recovered',

                           'showlegend':True}]),

             dict(label = 'Deaths',

                  method = 'update',

                  args = [{'visible': [False, False, False, False, False, False, False, False, False, True, True, True]},

                          {'title': 'Deaths',

                           'showlegend':True}]),

            ]),

             type = "buttons",

            direction="right",

#             pad={"r": 10, "t": 40},

             showactive=True,

             x=-0.05,

             xanchor="left",

             y=1.2,

             yanchor="top"

        )

    ])



week_month_fig.update_layout(

    #height=500, width=1150, 

    title_text="Weekly/Monthly Statistics", title_x=0.5, title_font_size=20,

                             paper_bgcolor="mintcream")

week_month_fig.update_xaxes(title_text="Days", row=1, col=1)

week_month_fig.update_xaxes(title_text="Weeks", row=1, col=2)

week_month_fig.update_xaxes(title_text="Months", row=1, col=3)

week_month_fig.show()
confirmed_group_sorted_df = confirmed_group_df.sort_values(by=confirmed_group_df.columns[-1], ascending=False)


world_ani = pd.DataFrame(columns=['Dates','Count', 'Country'])

Count, Dates, Country = [],[],[]

for i in range(20):

        tree = []

        Count.extend(confirmed_group_sorted_df[confirmed_group_sorted_df.columns[3:]][i:i+1].T.values.tolist())

        Dates.extend(confirmed_group_sorted_df.columns[3:])

        tree.append(confirmed_group_sorted_df.iloc[i,0])

        tree = tree*(confirmed_group_sorted_df.shape[1]-3)

        Country.extend(tree)

world_ani['Count'] = pd.DataFrame(Count)[0]

world_ani['Dates'] = pd.DataFrame(Dates)[0]

world_ani['Country'] = pd.DataFrame(Country)[0]



#confirmed_group_df.shape[0]


cases_over_time_fig = px.scatter_geo(world_ani, locations='Country', color="Country",locationmode='country names',

                     hover_name="Country", size="Count", size_max=50,

                     animation_frame="Dates", projection="natural earth")



#Increasing the speed of animation

#cases_over_time_fig.update_layout(transition = {'duration': 1000})

# cases_over_time_fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = pow(10, -1000)

# cases_over_time_fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = pow(10, -1000)



cases_over_time_fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 2

cases_over_time_fig.show()
confirmed_group_sorted_T_df = confirmed_group_sorted_df[0:20].drop(columns=['Lat','Long']).T

confirmed_group_sorted_T_df.columns = confirmed_group_sorted_T_df[0:1].values.tolist()[0]

confirmed_group_sorted_T_df.drop(['Country/Region'], inplace=True)

#confirmed_group_sorted_T_df.head()
layout = go.Layout(

                   hovermode='x unified',

                   updatemenus=[

                        dict(

                            type='buttons', showactive=False,

                            y=0.8,

                            x=-0.07,

                            xanchor='right',

                            yanchor='top',

                            pad=dict(t=0, r=10),

                            buttons=[dict(label='Play',

                            method='animate',

                            args=[None, 

                                  dict(frame=dict(duration=1, 

                                                  redraw=False),

                                                  transition=dict(duration=0),

                                                  fromcurrent=True,

                                                  mode='immediate')]

                            )]

                        ),

                        dict(

                            type = "buttons",

                            direction = "left",

                            buttons=list([

                                dict(

                                    args=[{"yaxis.type": "linear"}],

                                    label="LINEAR",

                                    method="relayout"

                                ),

                                dict(

                                    args=[{"yaxis.type": "log"}],

                                    label="LOG",

                                    method="relayout"

                                ),

#                                 dict(label = 'Log Scale',

#                                       method = 'relayout',

#                                       args = [{'yaxis': {'type': 'log'}},]),

                            ]),

                        ),

                    ]              

                  )



frames = [dict(data= [dict(type='scatter',

                           x=confirmed_group_sorted_T_df.index[:k+1],

                           y=confirmed_group_sorted_T_df['US'][:k+1]),

                      dict(type='scatter',

                           x=confirmed_group_sorted_T_df.index[:k+1],

                           y=confirmed_group_sorted_T_df['India'][:k+1]),

                      dict(type='scatter',

                           x=confirmed_group_sorted_T_df.index[:k+1],

                           y=confirmed_group_sorted_T_df["Brazil"][:k+1]),

                      dict(type='scatter',

                           x=confirmed_group_sorted_T_df.index[:k+1],

                           y=confirmed_group_sorted_T_df['Russia'][:k+1]),

                      dict(type='scatter',

                           x=confirmed_group_sorted_T_df.index[:k+1],

                           y=confirmed_group_sorted_T_df['Peru'][:k+1]),

                      dict(type='scatter',

                           x=confirmed_group_sorted_T_df.index[:k+1],

                           y=confirmed_group_sorted_T_df['Colombia'][:k+1]),

                      dict(type='scatter',

                           x=confirmed_group_sorted_T_df.index[:k+1],

                           y=confirmed_group_sorted_T_df["South Africa"][:k+1]),

                      dict(type='scatter',

                           x=confirmed_group_sorted_T_df.index[:k+1],

                           y=confirmed_group_sorted_T_df['Mexico'][:k+1]),

                      dict(type='scatter',

                           x=confirmed_group_sorted_T_df.index[:k+1],

                           y=confirmed_group_sorted_T_df["Spain"][:k+1]),

                      dict(type='scatter',

                           x=confirmed_group_sorted_T_df.index[:k+1],

                           y=confirmed_group_sorted_T_df['Argentina'][:k+1]),

                     ],

               traces= list(range(10)),  

              )for k  in  range(1, confirmed_group_sorted_T_df.shape[0])]



top_5_country_fig = go.Figure(frames=frames, layout=layout)

lockdown_list = ['3/19/20','3/17/20','3/25/20','3/28/20','3/16/20','3/26/20','3/25/20','3/23/20','3/14/20','3/19/20']

for country in confirmed_group_sorted_T_df.columns[:10]:

    top_5_country_fig.add_trace(go.Scatter(x =confirmed_group_sorted_T_df.index,

                y = confirmed_group_sorted_T_df[country],

                name = country,

                hovertemplate ='%{y}',

            )

        )

for i in range(10):

    top_5_country_fig.add_trace(go.Scatter(x =[lockdown_list[i]],

                y = [confirmed_group_sorted_T_df[confirmed_group_sorted_T_df.columns[i]][lockdown_list[i]]],

                name = confirmed_group_sorted_T_df.columns[i],

                mode='markers',

                marker_symbol='circle',

                hovertemplate ='%{y}',

                showlegend=False,

            )

        )



top_5_country_fig.update_xaxes(showticklabels=False)

top_5_country_fig.update_layout(

    #height=500, width=1100, 

    title_text="Top 10 Countries by Confirmed Cases",

                                title_x=0.5, title_font_size=20,paper_bgcolor="mintcream",yaxis_title="Confirmed Count",

                                xaxis=dict(title='Dates <br> The Circle in the plots represents the First Lockdown implemented by the Country <br> These are top 10 countries by number of confirmed Cases.'))

top_5_country_fig.show(config={"displayModeBar": False, "showTips": False})

# #############Not currently used############################

# top_20_countries_df = pd.DataFrame(columns=['Country','Confirmed','Active','Recovered','Deaths'])

# Confirmed, Active, Recovered, Deaths = [], [], [], []



# top_20_countries_df['Country'] = confirmed_group_sorted_df['Country/Region'][:20].reset_index(drop=True)

# for i in top_20_countries_df['Country']:

#     Confirmed.append(confirmed_group_df.set_index("Country/Region").loc[i, confirmed_group_df.columns[-1]])

#     Active.append(confirmed_group_df.set_index("Country/Region").loc[i, confirmed_group_df.columns[-1]])

#     Recovered.append(recoveries_group_df.set_index("Country/Region").loc[i, recoveries_group_df.columns[-1]])

#     Deaths.append(deaths_group_df.set_index("Country/Region").loc[i, deaths_group_df.columns[-1]])



# top_20_countries_df['Confirmed'] = pd.DataFrame(Confirmed)

# top_20_countries_df['Active'] = pd.DataFrame(Active)

# top_20_countries_df['Recovered'] = pd.DataFrame(Recovered)

# top_20_countries_df['Deaths'] = pd.DataFrame(Deaths)

# top_20_countries_df.head()
scatter_ani_df = pd.DataFrame(columns=['Dates', 'Country','Confirmed','Recovered','Deaths'])

Dates, Country, Confirmed, Deaths, Recovered = [],[],[],[],[]

for i in range(20):

        temp1 = []

        Confirmed.extend(confirmed_group_sorted_df[confirmed_group_sorted_df.columns[3:]][i:i+1].T.values.tolist())

        Dates.extend(confirmed_group_sorted_df.columns[3:])

        temp1.append(confirmed_group_sorted_df.iloc[i,0])

        temp = temp1*(confirmed_group_sorted_df.shape[1]-3)

        Country.extend(temp)

        

        Recovered.extend(recoveries_group_df.set_index(recoveries_group_df["Country/Region"], drop=True)[confirmed_group_sorted_df.columns[3:]].loc[temp1].values.tolist()[0])

        Deaths.extend(deaths_group_df.set_index(deaths_group_df["Country/Region"], drop=True)[confirmed_group_sorted_df.columns[3:]].loc[temp1].values.tolist()[0])

        

        

        

scatter_ani_df['Confirmed'] = pd.DataFrame(Confirmed)[0]

scatter_ani_df['Dates'] = pd.DataFrame(Dates)[0]

scatter_ani_df['Country'] = pd.DataFrame(Country)[0]

scatter_ani_df['Recovered'] = pd.DataFrame(Recovered)[0]

scatter_ani_df['Deaths'] = pd.DataFrame(Deaths)[0]

fig = px.scatter(scatter_ani_df, x="Confirmed", y="Deaths", animation_frame="Dates", animation_group="Country",

           size="Confirmed", color="Country", hover_name="Country",

           #log_x=True, 

           size_max=50, range_x=[-10000,8000000], range_y=[-10000,400000])



fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 50

fig.show()
affected_countries_df = confirmed_df.groupby("Country/Region").sum().drop(['Lat','Long'],axis =1).apply(lambda x: x[x > 0].count(), axis =0)
affected_countries_fig = go.Figure()

affected_countries_fig.add_trace(go.Scatter(x = base_stats['Dates'],

            y = affected_countries_df,

            name = 'Affected Countries',

            mode='lines',

            line = dict(color='#118ab2'),

            hovertemplate ='<br><b>Date</b>: %{x}'+'<br><i>No. of Countries </i>:'+'%{y}',

        )

    ) 



affected_countries_fig.update_xaxes(showticklabels=False)

affected_countries_fig.update_layout(

    #height=500, width=1100, 

    title_text="Number of Countries Affected With COVID19",

                                     title_x=0.5, title_font_size=20, legend=dict(orientation='h',yanchor='top',y=1.12,xanchor='right',x=1),

                                     paper_bgcolor="mintcream",

                                    xaxis_title="Date", yaxis_title="Number of Countries")

affected_countries_fig.show()
confirmed_group_melted_df = pd.melt(confirmed_group_df, id_vars=['Country/Region'], value_vars=confirmed_group_df.columns[3:])

confirmed_group_melted_df.rename(columns={"variable": "Dates", "value": "Confirmed"}, inplace=True)



active_group_melted_df = pd.melt(active_group_df, id_vars=['Country/Region'], value_vars=active_group_df.columns[3:])

active_group_melted_df.rename(columns={"variable": "Dates", "value": "Count"}, inplace=True)



recovered_group_melted_df = pd.melt(recoveries_group_df, id_vars=['Country/Region'], value_vars=recoveries_group_df.columns[3:])

recovered_group_melted_df.rename(columns={"variable": "Dates", "value": "Count"}, inplace=True)



deaths_group_melted_df = pd.melt(deaths_group_df, id_vars=['Country/Region'], value_vars=deaths_group_df.columns[3:])

deaths_group_melted_df.rename(columns={"variable": "Dates", "value": "Count"}, inplace=True)
country_specific_fig = make_subplots(specs=[[{"secondary_y": True}]])



df_dict={

  "Confirmed": [confirmed_group_melted_df ,"#118ab2"],

  "Active": [active_group_melted_df ,"#ef476f"],

  "Recovered": [recovered_group_melted_df ,"#06d6a0"],

  "Deaths": [deaths_group_melted_df ,"#073b4c"]        

}



for country in confirmed_group_df.sort_values(by=confirmed_group_df.columns[-1], ascending=False)['Country/Region'].values.tolist()[:20]:

    country_specific_fig.add_trace(go.Scatter(y=confirmed_group_df[confirmed_group_df['Country/Region']==country][confirmed_group_df.columns[4:]].values.tolist()[0],

                                             x=confirmed_group_df.columns[4:],

                                             mode='lines', visible=(lambda x: True if x=="US" else False)(country), 

                                              name="Confirmed", showlegend=True,

                                             line = dict(dash="solid", color=df_dict['Confirmed'][1])

                                             ))

    

    

    

    

    country_specific_fig.add_trace(go.Bar(y=confirmed_group_df[confirmed_group_df['Country/Region']==country][confirmed_group_df.columns[4:]].T.reset_index().reset_index().rename(columns={confirmed_group_df[confirmed_group_df['Country/Region']==country][confirmed_group_df.columns[4:]].T.reset_index().reset_index().columns[-1]:'count','index':'dates','level_0':'index'})['index'].apply(lambda x: confirmed_group_df[confirmed_group_df['Country/Region']==country][confirmed_group_df.columns[4:]].T.reset_index().reset_index().rename(columns={confirmed_group_df[confirmed_group_df['Country/Region']==country][confirmed_group_df.columns[4:]].T.reset_index().reset_index().columns[-1]:'count','index':'dates','level_0':'index'})['count'][x]-confirmed_group_df[confirmed_group_df['Country/Region']==country][confirmed_group_df.columns[4:]].T.reset_index().reset_index().rename(columns={confirmed_group_df[confirmed_group_df['Country/Region']==country][confirmed_group_df.columns[4:]].T.reset_index().reset_index().columns[-1]:'count','index':'dates','level_0':'index'})['count'][x-1:x].sum()),

                                         x=confirmed_group_df.columns[4:],

                                          name="Daily Confirmed", showlegend=True,

                                          visible=(lambda x: True if x=="US" else False)(country),

                                          yaxis='y2', opacity=0.2))

    

    

    

    

    

    

#     Trace for average moving

#     country_specific_fig.add_trace(go.Scatter(y=confirmed_group_melted_df[confirmed_group_melted_df['Country/Region']==country].reset_index(drop=True).reset_index()['index'].apply(lambda x: (confirmed_group_melted_df[confirmed_group_melted_df['Country/Region']==country].reset_index(drop=True).reset_index()['Confirmed'][x-7:x].sum())/7 if x>7 else (confirmed_group_melted_df[confirmed_group_melted_df['Country/Region']==country].reset_index(drop=True).reset_index()['Confirmed'][0:x].sum())/7),

#                                              x=confirmed_group_df.columns[4:],

#                                              mode='lines', visible=False, name=country, showlegend=False,

#                                              line = dict(dash="dash"),

#                                              hovertemplate = '<br><b>Date</b>: %{x}'+'<br><i>7-day moving avg.</i>: %{y}',

#                                              ))

    for i in ["Active", "Recovered", "Deaths"]:

        country_specific_fig.add_trace(go.Scatter(y=df_dict[i][0][df_dict[i][0]['Country/Region']==country].reset_index(drop=True).reset_index()['index'].apply(lambda x: (df_dict[i][0][df_dict[i][0]['Country/Region']==country].reset_index(drop=True).reset_index()['Count'][x-7:x].sum())/7 if x>7 else (df_dict[i][0][df_dict[i][0]['Country/Region']==country].reset_index(drop=True).reset_index()['Count'][0:x].sum())/7),

                                             x=confirmed_group_df.columns[4:],

                                             mode='lines', visible=(lambda x: True if x=="US" else False)(country),

                                            name=i, showlegend=True,

                                             line = dict(dash="solid", color=df_dict[i][1]),

                                             #hovertemplate = '<br><i>'+i+'</i>: %{y:.2f}',

                                             ))





country_specific_fig.update_layout(

updatemenus=[

        dict(

        buttons=list(

            [dict(label = country,

                  method = 'update',

                  args = [{'visible': list(map(lambda x: True if 5*index<=x<=5*index+4 else False, list(range(100))))},

                          {'title': "Country :"+country+"<br>Position :"+str(index+1),

                           'showlegend':True}]) for index, country in enumerate(confirmed_group_df.sort_values(by=confirmed_group_df.columns[-1], ascending=False)['Country/Region'].values.tolist()[:20])

            ]),

             type = "dropdown",

             direction="down",

             pad={"r": 0, "t": 0},

             showactive=True,

             x=0,

             xanchor="left",

             y=1.2,

             yanchor="top"

        )

])





country_specific_fig.update_xaxes(showticklabels=False)

country_specific_fig.update_layout(

    #height=500, width=1100, 

                                     title_text="Number of Cases in top 50 Countries",

                                     title_x=0.5, title_font_size=15, paper_bgcolor="mintcream",

                                     legend=dict(orientation='h',yanchor='top',y=1.12,xanchor='right',x=1),

                                     yaxis_title="Number of Cases", hovermode='x unified',

                                     xaxis=dict(title='Dates <br> The Position of countries is solely based on No. of Confirmed Cases<br>Please use the dropdown to select the country of choice'))

country_specific_fig.show()
corr_mat_fig = go.Figure()



corr_mat_fig = go.Figure(data=go.Splom(

                dimensions=[dict(label='Confirmed',

                                 values=base_stats['Confirmed']),

                            dict(label='Active',

                                 values=base_stats['Active']),

                            dict(label='Recovered',

                                 values=base_stats['Recovered']),

                            dict(label='Deaths',

                                 values=base_stats['Deaths'])],

                text=base_stats['Dates'],

    diagonal_visible=False,

    marker=dict(color='red',

                showscale=False, # colors encode categorical variables

                line_color='white', line_width=0.5)

                ))





corr_mat_fig.update_layout(

    #height=600, width=600, 

                                     title_text="Correlation Matrix for types of Cases",

                                     title_x=0.5, title_font_size=15, paper_bgcolor="mintcream",

                                     legend=dict(orientation='h',yanchor='top',y=1.12,xanchor='right',x=1))



corr_mat_fig.show()

country_data = go.Figure()

country_data.add_trace(go.Table(

    header=dict(values=['Country','Confirmed','Active','Recovered','Deaths','Mortality Rate'],

                fill = dict(color='#BDF6A9'),

                align = ['left'] * 5),

    cells=dict(values=[active_group_df[active_group_df[active_group_df.columns[-1]]==0]['Country/Region'].values.tolist(),

                       active_group_df[active_group_df[active_group_df.columns[-1]]==0]['Country/Region'].apply(lambda x: confirmed_group_df[confirmed_group_df['Country/Region']==x][confirmed_group_df.columns[-1]].values.tolist()[0]).values.tolist(),

                       active_group_df[active_group_df[active_group_df.columns[-1]]==0]['Country/Region'].apply(lambda x: active_group_df[active_group_df['Country/Region']==x][active_group_df.columns[-1]].values.tolist()[0]).values.tolist(),

                       active_group_df[active_group_df[active_group_df.columns[-1]]==0]['Country/Region'].apply(lambda x: recoveries_group_df[recoveries_group_df['Country/Region']==x][recoveries_group_df.columns[-1]].values.tolist()[0]).values.tolist(),

                       active_group_df[active_group_df[active_group_df.columns[-1]]==0]['Country/Region'].apply(lambda x: deaths_group_df[deaths_group_df['Country/Region']==x][deaths_group_df.columns[-1]].values.tolist()[0]).values.tolist(),

                       active_group_df[active_group_df[active_group_df.columns[-1]]==0]['Country/Region'].apply(lambda x: deaths_group_df[deaths_group_df['Country/Region']==x][deaths_group_df.columns[-1]].values.tolist()[0]/confirmed_group_df[confirmed_group_df['Country/Region']==x][confirmed_group_df.columns[-1]].values.tolist()[0]).values.tolist()

                      ],

               fill = dict(color='#DAFACE'),

               align = ['left'] * 5)))



country_data.update_layout(

    #height=300, width=1100, 

    title_text="Countries with no active cases",

                                     title_x=0.5, title_font_size=20,

                                     paper_bgcolor="mintcream")

country_data.show()
# corr = []

# for i in base_stats.columns[2:]:

#     temp = []

#     for j in base_stats.columns[2:]:

#         temp.append(base_stats[j].sum()/base_stats[i].sum())

#     corr.append(temp)



# ff_heatmap = go.Figure(data=go.Heatmap(

#         z=corr,

#         x=base_stats.columns[2:],

#         y=base_stats.columns[2:],

#         colorscale='reds'))



# ff_heatmap.update_layout(title_text='title', title_x=0.5, 

#                    width=600, height=600,

#                    xaxis_showgrid=False,

#                    yaxis_showgrid=False,

#                    yaxis_autorange='reversed'

#                         )

# ff_heatmap.show()