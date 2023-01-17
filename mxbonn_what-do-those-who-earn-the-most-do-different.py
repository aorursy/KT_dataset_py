import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.figure_factory as ff

import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
# Load the data

multiple_choice = pd.read_csv('../input/multipleChoiceResponses.csv', encoding="ISO-8859-1")

conversion_rates_df = pd.read_csv('../input/conversionRates.csv', encoding="ISO-8859-1")

conversion_rates_df.drop('Unnamed: 0', axis=1, inplace=True)

conversion_rates_df.set_index('originCountry', drop=True, inplace=True)
# Functions to convert all currencies to USD

def isFloat(x):

    try:

        float(x)

        return True

    except:

        return False

    

def convertCurrency(row):

    compensation = row['CompensationAmount']

    currency = row['CompensationCurrency']

    try:

        compensation = compensation * conversion_rates_df.loc[currency]['exchangeRate']

    except KeyError:

        compensation = -1

    row['CompensationAmount'] = compensation

    return row
original_mc = pd.read_csv('../input/multipleChoiceResponses.csv', encoding="ISO-8859-1")

multiple_choice.dropna(subset=['CompensationAmount', 'CompensationCurrency'],inplace=True)

multiple_choice['CompensationAmount'] = multiple_choice['CompensationAmount'].str.replace(',', '')

multiple_choice = multiple_choice[multiple_choice.CompensationAmount.apply(lambda x : isFloat(x))]

multiple_choice['CompensationAmount'] = pd.to_numeric(multiple_choice.CompensationAmount)

multiple_choice = multiple_choice.apply(lambda row: convertCurrency(row), axis=1)

multiple_choice = multiple_choice[(multiple_choice.CompensationAmount > 5000) & (multiple_choice.CompensationAmount < 2000000)]

multiple_choice = multiple_choice[~multiple_choice.EmploymentStatus.str.contains('Employed part-time')]
HIGH_INCOME = 125000

high_earners = multiple_choice[multiple_choice.CompensationAmount > HIGH_INCOME]
x = ['All Respondents', 'Respondents with income info', 'High Income Earners']

y = [original_mc.shape[0], multiple_choice.shape[0], high_earners.shape[0]]



data = [go.Bar(

            x=x,

            y=y,

            text=y,

            textposition = 'auto',

            marker=dict(

                color='rgba(50, 171, 96, 0.6)',

                line=dict(

                    color='rgba(50, 171, 96, 1.0)',

                    width=1.5),

            ),

        )]



layout = go.Layout(

    title='Number of respondents')



fig= go.Figure(data=data, layout=layout)

py.iplot(fig)
labels_all = multiple_choice.GenderSelect.value_counts().index.values

values_all = multiple_choice.GenderSelect.value_counts().values



labels_high = high_earners.GenderSelect.value_counts().index.values

values_high = high_earners.GenderSelect.value_counts().values



colors = ['#84d2ff','#ff83b7','#fff083', '#83ff93']



trace_all = go.Pie(labels=labels_all, values=values_all, marker=dict(colors=colors) , hoverinfo='label+value', name='All')

trace_high = go.Pie(labels=labels_high, values=values_high, marker=dict(colors=colors) , hoverinfo='label+value', name='High')



data = [trace_all, trace_high]



updatemenus = list([

    dict(active=1,

         buttons=list([   

            dict(label = 'All',

                 method = 'update',

                 args = [{'visible': [True, False]},

                         {'title': 'Gender distribution: Everyone'}]),

            dict(label = 'High',

                 method = 'update',

                 args = [{'visible': [False, True]},

                         {'title': 'Gender distribution: High Earners'}])

        ]),

    )

])



legend = dict(

    x = 0.9,

    y = 1)

layout = dict(updatemenus=updatemenus, title='Gender distribution', legend=legend)



fig = dict(data=data, layout=layout)

py.iplot(fig)
x_all = multiple_choice[multiple_choice.Age > 0].Age

x_high = high_earners[high_earners.Age > 0].Age 



hist_data = [x_all, x_high]

group_labels = ['Everyone', 'High Earners']



fig = ff.create_distplot(hist_data, group_labels, bin_size=1, colors=['rgba(66, 134, 244, 0.8)','rgba(255, 215, 0, 0.8)'])

fig['layout'].update(title='Age distribution')



py.iplot(fig)
scale=[[0, '#f7fcf5'], [0.05, '#e5f5e0'], [0.15, '#c7e9c0'], 

              [0.2, '#a1d99b'], [0.25, '#74c476'], [0.35, '#41ab5d'], 

              [0.45, '#238b45'], [0.55, '#006d2c'], [1.0, '#00441b']]



data_all = dict(

        type = 'choropleth',

        autocolorscale = False,

        colorscale = scale,

        showscale = True,

        locations = multiple_choice.groupby('Country').median()[['CompensationAmount']].index,

        z = multiple_choice.groupby('Country').median()[['CompensationAmount']].CompensationAmount.values,

        locationmode = 'country names',

        text = multiple_choice.groupby('Country').median()[['CompensationAmount']].index,

        marker = dict(

            line = dict(color = 'rgb(250,250,225)', width = 0.5)),

            colorbar = dict(autotick = True, tickprefix = '', 

            title = 'Average Salary')

            )



data_high = dict(

        type = 'choropleth',

        autocolorscale = False,

        colorscale = scale,

        showscale = True,

        locations = high_earners.groupby('Country').median()[['CompensationAmount']].index,

        z = high_earners.groupby('Country').median()[['CompensationAmount']].CompensationAmount.values,

        locationmode = 'country names',

        text = high_earners.groupby('Country').median()[['CompensationAmount']].index,

        marker = dict(

            line = dict(color = 'rgb(250,250,225)', width = 0.5)),

            colorbar = dict(autotick = True, tickprefix = '', 

            title = 'Average Salary')

            )



layout = dict(

    title = 'Average Salary\n (Countries in white represent no survey participation)',

    geo = dict(

        showframe = True,

        showocean = True,

        projection = dict(

        type = 'orthographic',

            rotation = dict(

                    lon = 10,

                    lat = 30),

        ),

        lonaxis =  dict(

                showgrid = False,

                gridcolor = 'rgb(102, 102, 102)'

            ),

        lataxis = dict(

                showgrid = False,

                gridcolor = 'rgb(102, 102, 102)'

                )

            ),

        )

data = [data_all]

fig = dict(data=data, layout=layout)

py.iplot(fig, validate=False)



data = [data_high]

layout['title'] = 'Average Salary of High Income Earners\n (Countries in white represent no survey participation)'

fig = dict(data=data, layout=layout)

py.iplot(fig, validate=False)
data_all = multiple_choice.dropna(subset=['FormalEducation']).FormalEducation.value_counts()

labels = list(reversed(high_earners.dropna(subset=['FormalEducation']).FormalEducation.value_counts().index))

values_all = [data_all[labels[0]], data_all[labels[1]], data_all[labels[2]], data_all[labels[3]], data_all[labels[4]], data_all[labels[5]], data_all[labels[6]]]



values_all = np.asarray(values_all / np.sum(values_all) * 100)



data_high = high_earners.dropna(subset=['FormalEducation']).FormalEducation.value_counts()

values_high = [data_high[labels[0]], data_high[labels[1]], data_high[labels[2]], data_high[labels[3]], data_high[labels[4]], data_high[labels[5]], data_high[labels[6]]]

values_high = np.asarray(values_high / np.sum(values_high) * 100)



trace_all = go.Bar(y=labels, x=values_all, hoverinfo='x', name='All', marker=dict(color='rgba(66, 134, 244, 0.8)'), orientation='h')

trace_high = go.Bar(y=labels, x=values_high, hoverinfo='x', name='High', marker = dict(

        color = 'rgba(255, 215, 0, 0.8)'), orientation='h')



data = [trace_all, trace_high]

margin = dict(l=450,r=33)



layout = dict(title='Formal Education Distribution', barmode='grouped', hovermode='closest', margin=margin)



fig = dict(data=data, layout=layout)

py.iplot(fig)
labels_all = multiple_choice.EmploymentStatus.value_counts().index.values

values_all = multiple_choice.EmploymentStatus.value_counts().values



labels_high = high_earners.EmploymentStatus.value_counts().index.values

values_high = high_earners.EmploymentStatus.value_counts().values



colors = ['#96D38C','#ff9563']



trace_all = go.Pie(labels=labels_all, values=values_all, marker=dict(colors=colors) , hoverinfo='label+value', name='All')

trace_high = go.Pie(labels=labels_high, values=values_high, marker=dict(colors=colors) , hoverinfo='label+value', name='High')



data = [trace_all, trace_high]



updatemenus = list([

    dict(active=1,

         buttons=list([   

            dict(label = 'All',

                 method = 'update',

                 args = [{'visible': [True, False]},

                         {'title': 'Employment Status: Everyone'}]),

            dict(label = 'High',

                 method = 'update',

                 args = [{'visible': [False, True]},

                         {'title': 'Employment Status: High Earners'}])

        ]),

    )

])

legend = dict(

    x = 0.9,

    y = 1)



layout = dict(updatemenus=updatemenus, title='Employment Status distribution', legend=legend)



fig = dict(data=data, layout=layout)

py.iplot(fig)
data_all = multiple_choice.dropna(subset=['RemoteWork']).RemoteWork.value_counts()

labels = ['Never', 'Rarely', 'Sometimes', 'Most of the time', 'Always']

values_all = [data_all[labels[0]], data_all[labels[1]], data_all[labels[2]], data_all[labels[3]], data_all[labels[4]]]



values_all = np.asarray(values_all / np.sum(values_all) * 100)



data_high = high_earners.dropna(subset=['RemoteWork']).RemoteWork.value_counts()

values_high = [data_high[labels[0]], data_high[labels[1]], data_high[labels[2]], data_high[labels[3]], data_high[labels[4]]]

values_high = np.asarray(values_high / np.sum(values_high) * 100)



trace_all = go.Bar(x=labels, y=values_all, hoverinfo='y', name='All', marker=dict(color='rgba(66, 134, 244, 0.8)'))

trace_high = go.Bar(x=labels, y=values_high, hoverinfo='y', name='High', marker = dict(

        color = 'rgba(255, 215, 0, 0.8)'))



data = [trace_all, trace_high]



layout = dict(title='How often do you work remotely?', barmode='grouped', hovermode='closest')



fig = dict(data=data, layout=layout)

py.iplot(fig)
data_all = multiple_choice.CurrentJobTitleSelect.value_counts().sort_index()

labels_all = data_all.index

values_all = data_all.values

values_all = np.asarray(values_all / np.sum(values_all) * 100)



data_high = high_earners.CurrentJobTitleSelect.value_counts().sort_index()

labels_high = data_high.index

values_high = data_high.values

values_high = np.asarray(values_high / np.sum(values_high) * 100)



trace_all = go.Bar(y=labels_all, x=values_all, hoverinfo='x', name='All', marker=dict(color='rgba(66, 134, 244, 0.8)'), orientation='h')

trace_high = go.Bar(y=labels_high, x=values_high, hoverinfo='x', name='High', marker = dict(

        color = 'rgba(255, 215, 0, 0.8)'), orientation='h')



data = [trace_all, trace_high]

margin = dict(l=250,r=33)



layout = dict(title='Current job title distribution', barmode='grouped', hovermode='closest', margin=margin)



fig = dict(data=data, layout=layout)

py.iplot(fig)
data_all = multiple_choice.dropna(subset=['SalaryChange']).SalaryChange.value_counts()

labels = ['Increased 20%+', 'Increased 6% - 19%', 'Stayed the same', 'Decreased 6% - 19%', 'Decreased 20%+']

values_all = [data_all['Has increased 20% or more'], data_all['Has increased between 6% and 19%'], data_all['Has stayed about the same (has not increased or decreased more than 5%)'], data_all['Has decreased between 6% and 19%'], data_all['Has decreased 20% or more']]



values_all = np.asarray(values_all / np.sum(values_all) * 100)



data_high = high_earners.dropna(subset=['SalaryChange']).SalaryChange.value_counts()

values_high = [data_high['Has increased 20% or more'], data_high['Has increased between 6% and 19%'], data_high['Has stayed about the same (has not increased or decreased more than 5%)'], data_high['Has decreased between 6% and 19%'], data_high['Has decreased 20% or more']]

values_high = np.asarray(values_high / np.sum(values_high) * 100)



trace_all = go.Bar(x=labels, y=values_all, hoverinfo='y', name='All', marker=dict(color='rgba(66, 134, 244, 0.8)'))

trace_high = go.Bar(x=labels, y=values_high, hoverinfo='y', name='High', marker = dict(

        color = 'rgba(255, 215, 0, 0.8)'))



data = [trace_all, trace_high]



layout = dict(title='How has your salary changed in the past 3 years?', barmode='grouped', hovermode='closest')



fig = dict(data=data, layout=layout)

py.iplot(fig)
data_all = multiple_choice.dropna(subset=['JobSatisfaction']).JobSatisfaction.value_counts()

labels = list(range(1,11))

values_all = [data_all['1 - Highly Dissatisfied'], data_all['2'], data_all['3'], data_all['4'], data_all['5'], data_all['6'], data_all['7'], data_all['8'], data_all['9'], data_all['10 - Highly Satisfied']]



values_all = np.asarray(values_all / np.sum(values_all) * 100)



data_high = high_earners.dropna(subset=['JobSatisfaction']).JobSatisfaction.value_counts()

values_high = [data_high['1 - Highly Dissatisfied'], data_high['2'], data_high['3'], data_high['4'], data_high['5'], data_high['6'], data_high['7'], data_high['8'], data_high['9'], data_high['10 - Highly Satisfied']]

values_high = np.asarray(values_high / np.sum(values_high) * 100)



trace_all = go.Bar(x=labels, y=values_all, hoverinfo='y', name='All', marker=dict(color='rgba(66, 134, 244, 0.8)'))

trace_high = go.Bar(x=labels, y=values_high, hoverinfo='y', name='High', marker = dict(

        color = 'rgba(255, 215, 0, 0.8)'))



data = [trace_all, trace_high]



layout = dict(title='On a scale from 1 (Highly Dissatisfied) - 10 (Highly Satisfied), how satisfied are you with your current job?', barmode='grouped', hovermode='closest', xaxis = dict(autotick = False, dtick = 1))



fig = dict(data=data, layout=layout)

py.iplot(fig)