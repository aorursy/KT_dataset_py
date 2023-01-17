import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
confirmed_cases_file_link = "https://raw.githubusercontent.com/AbdelfattahMohamed/COVID-20/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
death_cases_file_link = "https://raw.githubusercontent.com/AbdelfattahMohamed/COVID-20/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
recovered_cases_file_link = "https://raw.githubusercontent.com/AbdelfattahMohamed/COVID-20/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"
#datasets/covid-19-master/data/countries-aggregated.csv
country_cases_file_link = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv"
confirmed_df = pd.read_csv(confirmed_cases_file_link)
print(confirmed_df.shape)
death_df = pd.read_csv(death_cases_file_link)
print(death_df.shape)
recovered_df = pd.read_csv(recovered_cases_file_link)
print(recovered_df.shape)
cases_country_df = pd.read_csv(country_cases_file_link)
print(cases_country_df.shape)
confirmed_df.columns
confirmed_df[confirmed_df["Country/Region"] == "Egypt"]
confirmed_df["Country/Region"].nunique()
confirmed_df = confirmed_df.replace(np.nan, '', regex = True)
death_df = death_df.replace(np.nan, '', regex = True)
recovered_df = recovered_df.replace(np.nan, '', regex = True)
cases_country_df = cases_country_df.replace(np.nan, '', regex = True)
confirmed_df.columns
cases_country_df.columns
global_data = cases_country_df.copy().drop(['Lat', 'Long_', 'Country_Region', 'Last_Update'], axis = 1)
global_summary = pd.DataFrame(global_data.sum()).transpose()
global_summary.style.format("{:,.0f}")
global_summary.style.format({"": "${:20,.0f}", 
                          "": "${:20,.0f}", 
                          "": "${:20,.0f}",
                          "":"${:20,.0f}"})\
                 .format({"": lambda x:x.lower(),
                          "": lambda x:x.lower()})\
                 .hide_index()\
                 .background_gradient(cmap='Blues')

confirmed_SouthAfrica = cases_country_df[cases_country_df['Country_Region'] == "South Africa"]
confirmed_SouthAfrica = confirmed_SouthAfrica.drop(['Lat', 'Long_', 'Country_Region', 'Last_Update'], axis = 1)
SA_Summary = pd.DataFrame(confirmed_SouthAfrica.sum()).transpose()
SA_Summary.style.format({"": "${:20,.0f}", 
                          "": "${:20,.0f}", 
                          "": "${:20,.0f}",
                          "":"${:20,.0f}"})\
                 .format({"": lambda x:x.lower(),
                          "": lambda x:x.lower()})\
                 .hide_index()\
                 .background_gradient(cmap='Blues')

confirmed_cp = confirmed_df.copy().drop(['Province/State','Long','Lat','Country/Region'], axis = 1)
#confirmed_cp.head()
confirmed_ts_summary = confirmed_cp.sum()
#confirmed_ts_summary.head()
fig = go.Figure(data=go.Scatter(x=confirmed_ts_summary.index,
                                y=confirmed_ts_summary.values,
                                mode='lines+markers')) # hover text goes here

fig.update_layout(title='Total COVID_19 Confirmed Cases (Globally)',
                  yaxis_title = 'Confirmed Cases',
                  
                  xaxis_tickangle = 300)
fig.show()
# Intialize color array to be used across the analysis
color_arr = px.colors.qualitative.Dark24
def draw_plot(ts_array,
              ts_label,
              title,
              colors,
              mode_size,
              line_size,
              x_axis_title,
              y_axis_title,
              tickangle = 0,
              yaxis_type="",
              additional_annotations=""
             ):

    # Intialize figure
    fig = go.Figure()
    #add all traces
    for index, ts in enumerate(ts_array):
        fig.add_trace(go.Scatter(
                                x = ts.index,
                                y = ts.values,
                                name = ts_label[index],
                                line = dict(color = colors[index], width = line_size[index]),
                                connectgaps=True
                                ))
    # base x_axis prop.
    x_axis_dict = dict(
                       showline = True,
                       showgrid = True,
                       showticklabels=True,
                       linecolor='rgb(204,204,204)',
                       linewidth=2,
                       ticks='outside',
                       tickfont = dict(family = 'Arial',size = 12,color='rgb(204,204,204)')
                      )
    # Setting x_axis params
    if x_axis_title:
        x_axis_dict['title'] = x_axis_title
        
    if tickangle > 0:
        x_axis_dict['tickangle'] = tickangle
        
    # Base y_axis prop 
    y_axis_dict = dict(
                        showline = True,
                        showgrid = True,
                        showticklabels=True,
                        linecolor = 'rgb(204,204,204)',
                        linewidth=2
    )
    # Setting y_axis prop
    if yaxis_type != "":
        y_axis_dict['type'] = yaxis_type
    
    if y_axis_title:
        y_axis_dict['title'] = y_axis_title
        
    # Updating the layout
    fig.update_layout(xaxis = x_axis_dict,
                      yaxis = y_axis_dict,
                      autosize=True,
                      margin = dict(autoexpand=True,l=100,r=20,t=110),
                      showlegend = True,
                      
                        )
    # Base annotations for any graph
    annotations = []
    # Title
    annotations.append(dict(xref='paper',yref='paper',x=0.0,y=1.05,xanchor='left',yanchor='bottom',
                            text=title,
                            font=dict(family='Impact',size=18,color='#072F39'),
                            showarrow=False,
                           ))
    # Adding annotations in params
    if len(additional_annotations) > 0:
        annotations.append(additional_annotations)
    # Updating the layout
    fig.update_layout(annotations=annotations)
    
    return fig
confirmed_agg_ts = confirmed_df.copy().drop(['Province/State','Long','Lat','Country/Region'],
                                            axis = 1).sum()
death_agg_ts = death_df.copy().drop(['Province/State','Long','Lat','Country/Region'],
                                             axis = 1).sum()
recovered_agg_ts = recovered_df.copy().drop(['Province/State','Long','Lat','Country/Region'],
                                             axis = 1).sum ()
# There is no timeseries data for Active cases, therefore it needs to be engineered separately
active_agg_ts = pd.Series(
                  data=np.array(
                        [x1 - x2 - x3 for (x1,x2,x3) in zip(confirmed_agg_ts.values,
                                                                death_agg_ts.values,
                                                                recovered_agg_ts.values)]),
                        index=confirmed_agg_ts.index)
#plot and add traces for all the aggregrated timeseries
ts_array=[confirmed_agg_ts,active_agg_ts,recovered_agg_ts, death_agg_ts]
labels = ['Confirmed','Active','Recovered','Deaths']
colors = [color_arr[5],color_arr[0], color_arr[2], color_arr[3]]
mode_size = [8,8,12,8]
line_size = [2,2,4,2]
# Calling the draw plot function defined above
fig_2 = draw_plot(
                ts_array=ts_array,
                ts_label=labels,
                title = "(COVID_19) case status from 22/1/2020 to 21/4/2020",
                colors=colors,
                mode_size=mode_size,
                line_size=line_size,
                x_axis_title = "Date",
                y_axis_title = "Case Count",
                tickangle = 315,
                yaxis_type = "",
                additional_annotations=[]
                )

fig_2.show()
from IPython.display import HTML
HTML('<img src="India-States.gif" height="600" width="400">')
cases_country_df.copy().drop(
    ['Lat','Long_','Last_Update'],axis = 1).sort_values('Confirmed', ascending=False).reset_index(drop=True).style.bar(
    align="left",width=98,color="#000")
                
cases_country_df.copy().drop(
    ['Lat','Long_','Last_Update','People_Tested','People_Hospitalized'],axis = 1).sort_values('Recovered', ascending=False).reset_index(drop=True).style.bar(
    align="left",width=98,color="#FF3333")

# Confirmed Cases
confirmed_Egypt_ts = confirmed_df[confirmed_df['Country/Region'] == "Egypt"]
confirmed_Egypt_ts = confirmed_Egypt_ts.drop(
    ['Lat','Long','Country/Region','Province/State'],axis = 1).reset_index(drop=True).sum()
# Deaths Casse
deaths_Egypt_ts = death_df[death_df['Country/Region'] == "Egypt"]
deaths_Egypt_ts = deaths_Egypt_ts.drop(['Lat','Long','Country/Region','Province/State'],
                                             axis = 1).reset_index(drop=True).sum()
# Recovered Cases
recovered_Egypt_ts = recovered_df[recovered_df['Country/Region'] == "Egypt"]
recovered_Egypt_ts = recovered_Egypt_ts.drop(['Lat','Long','Country/Region','Province/State'],
                                             axis = 1).reset_index(drop=True).sum()
# Active Cases
active_Egypt_ts = pd.Series(
                  data=np.array(
                        [x1 - x2 - x3 for (x1,x2,x3) in zip(confirmed_agg_ts.values,
                                                                death_agg_ts.values,
                                                                recovered_agg_ts.values)]),
                        index=confirmed_agg_ts.index)
ts_array=[confirmed_Egypt_ts,active_Egypt_ts,recovered_Egypt_ts, deaths_Egypt_ts]
labels = ['Confirmed','Active','Recovered','Deaths']
colors = [color_arr[5],color_arr[0], color_arr[2], color_arr[3]]
mode_size = [8,8,12,8]
line_size = [2,2,4,2]
# Calling the draw plot function defined above
fig_2 = draw_plot(
                ts_array=ts_array,
                ts_label=labels,
                title = "(COVID_19) In EGYPT status from 22/1/2020 to 21/4/2020 (مصر)",
                colors=colors,
                mode_size=mode_size,
                line_size=line_size,
                x_axis_title = "Date",
                y_axis_title = "Case Count",
                tickangle = 315,
                yaxis_type = "",
                additional_annotations=[]
                )

fig_2.show()
ts_array=[confirmed_Egypt_ts[39:],active_Egypt_ts[39:],recovered_Egypt_ts[39:], deaths_Egypt_ts]
labels = ['Confirmed','Active','Recovered','Deaths']
colors = [color_arr[5],color_arr[0], color_arr[2], color_arr[3]]
mode_size = [8,8,12,8]
line_size = [2,2,4,2]
# Calling the draw plot function defined above
fig_2 = draw_plot(
                ts_array=ts_array,
                ts_label=labels,
                title = "(COVID_19) In EGYPT status from 22/1/2020 to 21/4/2020 (مصر)",
                colors=colors,
                mode_size=mode_size,
                line_size=line_size,
                x_axis_title = "Date",
                y_axis_title = "Case Count",
                tickangle = 315,
                yaxis_type = "",
                additional_annotations=[]
                )

fig_2.show()
