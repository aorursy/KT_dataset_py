import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
os.listdir("../input")
multiple_df = pd.read_csv('../input/multipleChoiceResponses.csv', low_memory=False)
free_df = pd.read_csv('../input/freeFormResponses.csv', low_memory=False)
schema_df = pd.read_csv('../input/SurveySchema.csv', low_memory=False)
print("Multiple choice response - rows: {} columns: {}".format(multiple_df.shape[0], multiple_df.shape[1]))
print("Free form response - rows: {} columns: {}".format(free_df.shape[0], free_df.shape[1]))
print("Survey schema - rows: {} columns: {}".format(schema_df.shape[0], schema_df.shape[1]))
multiple_df.head(3)
free_df.head(3)
def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
df = missing_data(multiple_df)
def plot_percent_of_available_data(title):
    trace = go.Box(
        x = df['Percent'],
        name="Percent",
         marker=dict(
                    color='rgba(238,23,11,0.5)',
                    line=dict(
                        color='tomato',
                        width=0.9),
                ),
         orientation='h')
    data = [trace]
    layout = dict(title = 'Percent of available data  - all columns ({})'.format(title),
              xaxis = dict(title = 'Percent', showticklabels=True), 
              yaxis = dict(title = 'All columns'),
              hovermode = 'closest',
             )
    fig = dict(data=data, layout=layout)
    iplot(fig, filename='percent')
plot_percent_of_available_data('multiple_df')
plot_percent_of_available_data('free_df')
tmp = pd.DataFrame(multiple_df.columns.values)
columns = []
for i in range(1,50):
    var = "Q{}".format(i)
    l = len(list(tmp[tmp[0].str.contains(var)][0]))
    if(l == 1):
        columns.append(var)

print("The columns with only one item in the column group are:\n",columns)
def get_categories(data, val):
    tmp = data[1::][val].value_counts()
    return pd.DataFrame(data={'Number': tmp.values}, index=tmp.index).reset_index()
df = get_categories(multiple_df, 'Q1')
def draw_trace_bar(data, title, xlab, ylab,color='Blue'):
    trace = go.Bar(
            x = data['index'],
            y = data['Number'],
            marker=dict(color=color),
            text=data['index']
        )
    data = [trace]

    layout = dict(title = title,
              xaxis = dict(title = xlab, showticklabels=True, tickangle=15,
                          tickfont=dict(
                            size=9,
                            color='black'),), 
              yaxis = dict(title = ylab),
              hovermode = 'closest'
             )
    fig = dict(data = data, layout = layout)
    iplot(fig, filename='draw_trace')
draw_trace_bar(df, 'Number of people', 'Gender', 'Number of people' )
draw_trace_bar(get_categories(multiple_df,'Q2'), "Number of people in each age range", "Age range", "Number of people", "Green")
df = get_categories(multiple_df, 'Q3')
df.head()
trace = go.Choropleth(
            locations = df['index'],
            locationmode='country names',
            z = df['Number'],
            text = df['index'],
            autocolorscale =False,
            reversescale = True,
            colorscale = 'rainbow',
            marker = dict(
                line = dict(
                    color = 'rgb(0,0,0)',
                    width = 0.5)
            ),
            colorbar = dict(
                title = 'Respondents',
                tickprefix = '')
        )

data = [trace]
layout = go.Layout(
    title = 'Number of respondents per country',
    geo = dict(
        showframe = True,
        showlakes = False,
        showcoastlines = True,
        projection = dict(
            type = 'natural earth'
        )
    )
)

fig = dict( data=data, layout=layout )
iplot(fig)
draw_trace_bar(get_categories(multiple_df,'Q4'), "Highest level of formal education", "Education", "Number of people", "Magenta")
draw_trace_bar(get_categories(multiple_df,'Q5'), "Undergraduate major (best descrition)", "Undergraduate major", "Number of respondents", "Orange")
draw_trace_bar(get_categories(multiple_df,'Q6'), "Current title", "Current title", "Number of respondents", "Red")
draw_trace_bar(get_categories(multiple_df,'Q7'), "Current employer industry", "Current employer industry", "Number of respondents", "Tomato")
draw_trace_bar(get_categories(multiple_df,'Q8'), "Years of experience in the current employer industry", "Years of experience", "Number of respondents", "Lightblue")
draw_trace_bar(get_categories(multiple_df,'Q9'), "Current yearly compensation", "Current yearly compensation", "Number of respondents", "Gold")
draw_trace_bar(get_categories(multiple_df,'Q10'), "Does the current employer uses machine learning", "Use of machine learning", "Number of respondents", "Brown")
draw_trace_bar(get_categories(multiple_df,'Q23'), multiple_df['Q23'][0], "Option", "Number of respondents", "Yellow")
draw_trace_bar(get_categories(multiple_df,'Q24'), multiple_df['Q24'][0], "Option", "Number of respondents", "Lightgreen")
draw_trace_bar(get_categories(multiple_df,'Q25'), multiple_df['Q25'][0], "Option", "Number of respondents", "Orange")
draw_trace_bar(get_categories(multiple_df,'Q26'), multiple_df['Q26'][0], "Option", "Number of respondents", "Lightblue")
draw_trace_bar(get_categories(multiple_df,'Q40'), multiple_df['Q40'][0], "Option", "Number of respondents", "Magenta")
draw_trace_bar(get_categories(multiple_df,'Q43'), multiple_df['Q43'][0], "Option", "Number of respondents", "Tomato")
draw_trace_bar(get_categories(multiple_df,'Q46'), multiple_df['Q46'][0], "Option", "Number of respondents", "Green")
draw_trace_bar(get_categories(multiple_df,'Q48'), "Are ML models `black boxes`?", "Are ML models `black boxes`?", "Number of respondents", "Black")
def get_categories_group(data, val_group, val):
    tmp = data[1::].groupby(val_group)[val].value_counts()
    return pd.DataFrame(data={'Number': tmp.values}, index=tmp.index).reset_index()
def draw_trace_group_bar(data_df, val_group, val, title, xlab, ylab,color='Blue'):
    data = list()
    groups = (data_df.groupby([val_group])[val_group].nunique()).index
    for group in groups:
        data_group_df = data_df[data_df[val_group]==group]
        trace = go.Bar(
                x = data_group_df[val],
                y = data_group_df['Number'],
                name = group,
                #marker=dict(color=color),
                text=data_group_df[val]
            )
        data.append(trace)

    layout = dict(title = title,
              xaxis = dict(title = xlab, showticklabels=True, tickangle=15,
                          tickfont=dict(
                            size=9,
                            color='black'),), 
              yaxis = dict(title = ylab),
              hovermode = 'closest'
             )
    fig = dict(data = data, layout = layout)
    iplot(fig, filename='draw_trace')
df = get_categories_group(multiple_df, 'Q1', 'Q2')
draw_trace_group_bar(df, 'Q1', 'Q2', 'Number of respondents by Sex and age', 'Age', 'Number of respondents')
df = get_categories_group(multiple_df, 'Q2', 'Q4')
draw_trace_group_bar(df, 'Q2', 'Q4', 'Number of respondents by Age and Highest level of formal education', 'Highest level of formal education', 'Number of respondents')
df = get_categories_group(multiple_df, 'Q2', 'Q9')
draw_trace_group_bar(df, 'Q2', 'Q9', 'Number of respondents by Age and Current yearly compensation', 'Current yearly compensation', 'Number of respondents')
df = get_categories_group(multiple_df, 'Q4', 'Q9')
draw_trace_group_bar(df, 'Q4', 'Q9', 'Number of respondents by Highest level of formal education and Current yearly compensation', 'Current yearly compensation', 'Number of respondents')
df = get_categories_group(multiple_df, 'Q8', 'Q9')
draw_trace_group_bar(df, 'Q8', 'Q9', 'Number of respondents by Years of experience and Current yearly compensation', 'Current yearly compensation', 'Number of respondents')
df = get_categories_group(multiple_df, 'Q6', 'Q4')
draw_trace_group_bar(df, 'Q6', 'Q4', 'Number of respondents by Current title and Highest level of education', 'Current title', 'Number of respondents')