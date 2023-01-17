import pandas as pd

import numpy as np

import plotly.express as px

import plotly.graph_objects as go

import warnings 

warnings.filterwarnings('ignore')



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

    

#import os

#os.listdir('/kaggle/input/ihmes-covid19-projections/')



todays_date = '4/09/2020' # Update this line every time that you rerun the notebook

df = pd.read_csv('/kaggle/input/ihmes-covid19-projections/2020_04_09.04/Hospitalization_all_locs.csv')

df_colorado = df[df.location_name=='Colorado']

df_colorado = df_colorado[['date', 

                           'allbed_mean',

                           'ICUbed_mean', 

                           'InvVen_mean',  

                           'deaths_mean',

                           'admis_mean', 

                           'newICU_mean', 

                           'newICU_lower', 

                           'newICU_upper',

                           'totdea_mean', 

                           'bedover_mean',

                           'icuover_mean']]

df_colorado[40:50].head()
title = 'Mean of Upper and Lower Predictions from IHME for Colorado as of '+todays_date

df_colorado.plot(title=title,figsize=(12,9), grid=True)
plot_eleven_columns_using_plotly_regular(dataframe=df_colorado,

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

                                        title='Mean of Upper and Lower Predictions from IHME for Colorado as of '+todays_date)
plot_eleven_columns_using_plotly_express(dataframe=df_colorado,

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

                                        title='Mean of Upper and Lower Predictions from IHME for Colorado as of '+todays_date)