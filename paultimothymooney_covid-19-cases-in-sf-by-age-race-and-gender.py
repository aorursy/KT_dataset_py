import numpy as np 

import pandas as pd 

import plotly.express as px

import plotly.graph_objects as go



def plot_three_columns_using_plotly_regular(dataframe,column_one,column_two,column_three,title):    

    '''

    This function plots four numerical columns against a date column.

    It using the regular plotly library instead of plotly express.

    '''

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=dataframe.result_date, y=dataframe[column_one],

                        mode='lines+markers',name=column_one))

    fig.add_trace(go.Scatter(x=dataframe.result_date, y=dataframe[column_two],

                        mode='lines+markers',name=column_two))

    fig.add_trace(go.Scatter(x=dataframe.result_date, y=dataframe[column_three],

                        mode='lines+markers',name=column_three))

    fig.update_layout(title={'text':title},

                      xaxis_title='result_date',yaxis_title='COVID-19 Cases in SF',

                      #legend_orientation="h",

                      showlegend=True)

    fig.show()    
tests = pd.read_csv('/kaggle/input/san-francisco-covid19-data/rows_4.csv')

cases = pd.read_csv('/kaggle/input/san-francisco-covid19-data/rows_1.csv')

hospitalizations = pd.read_csv('/kaggle/input/san-francisco-covid19-data/rows.csv')

age_and_gender = pd.read_csv('/kaggle/input/san-francisco-covid19-data/rows_3.csv')

race_and_ethnicity = pd.read_csv('/kaggle/input/san-francisco-covid19-data/rows_2.csv')
tests['pct'] = tests['pct']*100

tests = tests.sort_values('result_date',ascending=True)

todays_date = '2020-04-26' 
plot_three_columns_using_plotly_regular(tests,'tests','pos','pct','Percent and Number of COVID-19 Tests that are Positive')
fig = px.bar(age_and_gender.sort_values('Age Group',ascending=True)[0:20], 

             x="Age Group", 

             y="Confirmed Cases",

             color='Gender',

             barmode='group',

             title='COVID-19 Infections in SF as of '+todays_date)

fig.show()
fig = px.bar(race_and_ethnicity.sort_values('Confirmed Cases',ascending=True)[0:20], 

             x="Race", 

             y="Confirmed Cases",

             color='Ethnicity',

             barmode='group',

             title='COVID-19 Infections in SF as of '+todays_date)

fig.show()