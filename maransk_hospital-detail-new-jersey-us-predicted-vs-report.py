# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import plotly.graph_objects as go



import plotly.offline as ply

ply.init_notebook_mode(connected=True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
base_path = "/kaggle/input/"

ihme_path_04_09 = base_path + "ihme-covid/ihme_covid/2020_04_09/" + "Hospitalization_all_locs.csv"

ihme_path_03_30 = base_path + "ihme-covid/ihme_covid/2020_03_30/" + "Hospitalization_all_locs.csv"



ihme_df_04_09 = pd.read_csv(ihme_path_04_09)

ihme_df_03_30 = pd.read_csv(ihme_path_03_30)

ihme_df_04_09.head(2)
ihme_df_04_09_nj=ihme_df_04_09[ihme_df_04_09.location_name=='New Jersey']

ihme_df_03_30_nj=ihme_df_03_30[ihme_df_03_30.location_name=='New Jersey']

ihme_df_04_09_nj.loc[:,'ICUbed_available']=465.0
ihme_df_04_09_nj.loc[ihme_df_04_09_nj['date']=='2020-04-09'].T[3:12].T
ihme_df_04_09_nj.loc[ihme_df_04_09_nj['date']=='2020-04-09'].T[12:21].T
ihme_df_04_09_nj.loc[ihme_df_04_09_nj['date']=='2020-04-09'].T[21:].T
fig = go.Figure()

fig.add_trace(go.Scatter(

    x=ihme_df_04_09_nj["date"], y=ihme_df_04_09_nj["allbed_mean"],

    line_color='rgb(0,100,80)',

    name='04_09',

))

fig.add_trace(go.Scatter(

    x=ihme_df_03_30_nj["date"], y=ihme_df_03_30_nj["allbed_mean"],

    line_color='rgb(231,107,243)',

    name='03_30',

))

fig.update_traces(mode='lines')

fig.show()
# fig = go.Figure()



# fig.add_trace(go.Scatter(

#     x=ihme_df_04_09_nj["date"], y=ihme_df_04_09_nj["newICU_mean"],

#     line_color='rgb(0,100,80)',

# #     name='ICUbed mean',

# ))

# fig.add_trace(go.Scatter(

#     x=ihme_df_04_09_nj["date"], y=ihme_df_04_09_nj["ICUbed_mean"],

#     line_color='rgba(0,100,80,0.5)',

# #     name='ICUbed available',

# ))

# fig.update_traces(mode='lines')

# fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(

    x=ihme_df_04_09_nj["date"].tolist()+ihme_df_04_09_nj["date"][::-1].tolist(),

    y=ihme_df_04_09_nj['ICUbed_upper'].tolist()+(ihme_df_04_09_nj['ICUbed_lower'][::-1]).tolist(),

    fill='toself',

    fillcolor='rgba(0,100,80,0.2)',

    line_color='rgba(255,255,255,0)',

    showlegend=False,

    name='ICUbed',

))

fig.add_trace(go.Scatter(

    x=ihme_df_04_09_nj["date"], y=ihme_df_04_09_nj["ICUbed_mean"],

    line_color='rgb(0,100,80)',

    name='ICUbed mean',

))

fig.add_trace(go.Scatter(

    x=ihme_df_04_09_nj["date"], y=ihme_df_04_09_nj["ICUbed_available"],

    line_color='rgba(0,100,80,0.5)',

    name='ICUbed available',

))

fig.update_traces(mode='lines')

fig.show()
def plot_eleven_columns_using_plotly_regular(dataframe,

                                             column_one,

                                             column_two,

                                             column_three,

                                             column_four,

                                             column_five,

                                             column_six,

                                             column_seven,

                                             column_eight,

                                             column_nine,

                                             column_ten,

                                             column_eleven,

                                             title):    

    '''

    This function plots four numerical columns against a date column.

    It using the regular plotly library instead of plotly express.

    '''

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=dataframe.date, y=dataframe[column_one],

                        mode='lines+markers',name=column_one))

    fig.add_trace(go.Scatter(x=dataframe.date, y=dataframe[column_two],

                        mode='lines+markers',name=column_two))

    fig.add_trace(go.Scatter(x=dataframe.date, y=dataframe[column_three],

                        mode='lines+markers',name=column_three))

    fig.add_trace(go.Scatter(x=dataframe.date, y=dataframe[column_four],

                        mode='lines+markers',name=column_four))

    fig.add_trace(go.Scatter(x=dataframe.date, y=dataframe[column_five],

                        mode='lines+markers',name=column_five))

    fig.add_trace(go.Scatter(x=dataframe.date, y=dataframe[column_six],

                        mode='lines+markers',name=column_six))

    fig.add_trace(go.Scatter(x=dataframe.date, y=dataframe[column_seven],

                        mode='lines+markers',name=column_seven))

    fig.add_trace(go.Scatter(x=dataframe.date, y=dataframe[column_eight],

                        mode='lines+markers',name=column_eight))

    fig.add_trace(go.Scatter(x=dataframe.date, y=dataframe[column_nine],

                        mode='lines+markers',name=column_nine))

    fig.add_trace(go.Scatter(x=dataframe.date, y=dataframe[column_ten],

                        mode='lines+markers',name=column_ten))

    fig.add_trace(go.Scatter(x=dataframe.date, y=dataframe[column_eleven],

                        mode='lines+markers',name=column_eleven))

    fig.update_layout(title={'text':title},

                      xaxis_title='Date',yaxis_title='Average of upper and lower predictions',

                      legend_orientation="h",showlegend=True)

    #fig.update_layout(xaxis=dict(range=[lower_axis_limit,upper_axis_limit]))

    fig.show()    

    

def plot_eleven_columns_using_plotly_express(dataframe,

                                             column_one,

                                             column_two,

                                             column_three,

                                             column_four,

                                             column_five,

                                             column_six,

                                             column_seven,

                                             column_eight,

                                             column_nine,

                                             column_ten,

                                             column_eleven,

                                             title):

    '''

    This function plots four numerical columns against a date column.

    It using the plotly express library instead of the normal plotly library.

    '''

    df_melt = dataframe.melt(id_vars='date', value_vars=[column_one,

                                                         column_two,

                                                         column_three,

                                                         column_four,

                                                         column_five,

                                                         column_six,

                                                         column_seven,

                                                         column_eight,

                                                         column_nine,

                                                         column_ten,

                                                         column_eleven])

    fig = px.line(df_melt, x="date", y="value", color="variable",title=title).update(layout=dict(xaxis_title='date',yaxis_title='Average of upper and lower predictions',legend_orientation="h",showlegend=True))

    #fig.update_xaxes(range=[lower_axis_limit,upper_axis_limit])

    fig.show()
todays_date = '4/09/2020'
plot_eleven_columns_using_plotly_regular(dataframe=ihme_df_04_09_nj,

                                        column_one='allbed_mean',

                                        column_two='ICUbed_mean',

                                        column_three='InvVen_mean',

                                        column_four='deaths_mean',

                                        column_five='admis_mean',

                                        column_six='newICU_mean',

                                        column_seven='newICU_lower',

                                        column_eight='newICU_upper',

                                        column_nine='totdea_mean',

                                        column_ten='bedover_mean',

                                        column_eleven='bedover_mean',

                                        title='Mean of Upper and Lower Predictions from IHME for New Jersey as of '+todays_date)
plot_eleven_columns_using_plotly_express(dataframe=ihme_df_04_09_nj,

                                        column_one='allbed_mean',

                                        column_two='ICUbed_mean',

                                        column_three='InvVen_mean',

                                        column_four='deaths_mean',

                                        column_five='admis_mean',

                                        column_six='newICU_mean',

                                        column_seven='newICU_lower',

                                        column_eight='newICU_upper',

                                        column_nine='totdea_mean',

                                        column_ten='bedover_mean',

                                        column_eleven='bedover_mean',

                                        title='Mean of Upper and Lower Predictions from IHME for New Jersey as of '+todays_date)