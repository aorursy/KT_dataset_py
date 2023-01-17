!pip install ipywidgets

!jupyter nbextension enable widgetsnbextension
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.linear_model import LinearRegression

from subprocess import check_output



from ipywidgets import interact, interactive, fixed, interact_manual

import ipywidgets as widgets

from ipywidgets import *



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv('../input/multipleChoiceResponses.csv', encoding="ISO-8859-1", low_memory=False)

exchange = pd.read_csv('../input/conversionRates.csv', encoding="ISO-8859-1", low_memory=False)



df = pd.merge(left=df, right=exchange, how='left', 

              left_on='CompensationCurrency', right_on='originCountry')
df['exchangeRate'] = df['exchangeRate'].fillna(0)

df['CompensationAmount'] = df['CompensationAmount'].fillna(0)

df['CompensationAmount'] = df.CompensationAmount.apply(lambda x: 0 if (pd.isnull(x) or (x=='-') or (x==0))

                                                       else float(x.replace(',','')))

df['CompensationAmount'] = df['CompensationAmount']*df['exchangeRate']



df = df[df['CompensationAmount']>0]
f = {'CompensationAmount':['median','count']}





temp_df = df.groupby('GenderSelect').agg(f).sort_values(by=[('CompensationAmount','median')], ascending=False)

temp_df = temp_df[temp_df[('CompensationAmount','count')]>50]



trace1 = go.Bar(

    x = temp_df.index,

    y = temp_df[('CompensationAmount','median')],

    text = ['Total Users ' + str(x) for x in temp_df[('CompensationAmount','count')]])



layout = go.Layout(

    title = '<b>Gender<b>',

    titlefont = dict(

        size = 30,

        color = 'rgb(130,130,130)'),

    yaxis = dict(

        title = 'Median Yearly Income ($)'))

fig = go.Figure(data=[trace1], layout=layout)

py.iplot(fig)
temp_df = df.groupby('Age').agg(f).sort_values(by=[('CompensationAmount','median')], ascending=False)

temp_df = temp_df[temp_df[('CompensationAmount','count')]>50]



trace1 = go.Bar(

    x = temp_df.index,

    y = temp_df[('CompensationAmount','median')],

    text = ['Total Users ' + str(x) for x in temp_df[('CompensationAmount','count')]])



layout = go.Layout(

    title = '<b>Age<b>',

    titlefont = dict(

        size = 30,

        color = 'rgb(130,130,130)'),

    yaxis = dict(

        title = 'Median Yearly Income ($)'))

fig = go.Figure(data=[trace1], layout=layout)

py.iplot(fig)
temp_df = df.groupby('CurrentJobTitleSelect').agg(f).sort_values(by=[('CompensationAmount','median')], ascending=False)

temp_df = temp_df[temp_df[('CompensationAmount','count')]>50]



trace1 = go.Bar(

    x = temp_df.index,

    y = temp_df[('CompensationAmount','median')],

    text = ['Total Users ' + str(x) for x in temp_df[('CompensationAmount','count')]])



layout = go.Layout(

    title = '<b>Job Title<b>',

    titlefont = dict(

        size = 30,

        color = 'rgb(130,130,130)'),

    yaxis = dict(

        title = 'Median Yearly Income ($)'))

fig = go.Figure(data=[trace1], layout=layout)

py.iplot(fig)
temp_df = df.groupby('Country').agg(f).sort_values(by=[('CompensationAmount','median')], ascending=False)

temp_df = temp_df[temp_df[('CompensationAmount','count')]>5]



trace1 = go.Bar(

    x = temp_df.index,

    y = temp_df[('CompensationAmount','median')],

    text = ['Total Users ' + str(x) for x in temp_df[('CompensationAmount','count')]])



layout = go.Layout(

    title = '<b>Country<b>',

    titlefont = dict(

        size = 30,

        color = 'rgb(130,130,130)'),

    yaxis = dict(

        title = 'Median Yearly Income ($)'))

fig = go.Figure(data=[trace1], layout=layout)

py.iplot(fig)
temp_df = df.groupby('Tenure').agg(f).sort_values(by=[('CompensationAmount','median')], ascending=False)

temp_df = temp_df[temp_df[('CompensationAmount','count')]>50]



trace1 = go.Bar(

    x = temp_df.index,

    y = temp_df[('CompensationAmount','median')],

    text = ['Total Users ' + str(x) for x in temp_df[('CompensationAmount','count')]])



layout = go.Layout(

    title = '<b>Tenure<b>',

    titlefont = dict(

        size = 30,

        color = 'rgb(130,130,130)'),

    yaxis = dict(

        title = 'Median Yearly Income ($)'))

fig = go.Figure(data=[trace1], layout=layout)

py.iplot(fig)
temp_df = df.groupby('FormalEducation').agg(f).sort_values(by=[('CompensationAmount','median')], ascending=False)

temp_df[temp_df[('CompensationAmount','count')]>50]



trace1 = go.Bar(

    x = temp_df.index,

    y = temp_df[('CompensationAmount','median')],

    text = ['Total Users ' + str(x) for x in temp_df[('CompensationAmount','count')]])



layout = go.Layout(

    title = '<b>Education<b>',

    titlefont = dict(

        size = 30,

        color = 'rgb(130,130,130)'),

    yaxis = dict(

        title = 'Median Yearly Income ($)'))

fig = go.Figure(data=[trace1], layout=layout)

py.iplot(fig)
temp_df = df.groupby('ParentsEducation').agg(f).sort_values(by=[('CompensationAmount','median')], ascending=False)

temp_df[temp_df[('CompensationAmount','count')]>50]



trace1 = go.Bar(

    x = temp_df.index,

    y = temp_df[('CompensationAmount','median')],

    text = ['Total Users ' + str(x) for x in temp_df[('CompensationAmount','count')]])



layout = go.Layout(

    title = '<b>Parents Education<b>',

    titlefont = dict(

        size = 30,

        color = 'rgb(130,130,130)'),

    yaxis = dict(

        title = 'Median Yearly Income ($)'))

fig = go.Figure(data=[trace1], layout=layout)

py.iplot(fig)
temp_df = df.groupby('MajorSelect').agg(f).sort_values(by=[('CompensationAmount','median')], ascending=False)

temp_df[temp_df[('CompensationAmount','count')]>50]



trace1 = go.Bar(

    x = temp_df.index,

    y = temp_df[('CompensationAmount','median')],

    text = ['Total Users ' + str(x) for x in temp_df[('CompensationAmount','count')]])



layout = go.Layout(

    title = '<b>Major<b>',

    titlefont = dict(

        size = 30,

        color = 'rgb(130,130,130)'),

    yaxis = dict(

        title = 'Median Yearly Income ($)'))

fig = go.Figure(data=[trace1], layout=layout)

py.iplot(fig)
temp_df = df.groupby(['MajorSelect','Country']).agg(f).sort_values(by=[('CompensationAmount','median')], ascending=False)

temp_df[temp_df[('CompensationAmount','count')]>30]
temp_df = df.groupby('JobSatisfaction').agg(f)

temp_df = temp_df[temp_df[('CompensationAmount','count')]>30]



trace1 = go.Bar(

    x = temp_df.index,

    y = temp_df[('CompensationAmount','median')],

    text = ['Total Users ' + str(x) for x in temp_df[('CompensationAmount','count')]])



layout = go.Layout(

    title = '<b>Jon Satisfaction<b>',

    titlefont = dict(

        size = 30,

        color = 'rgb(130,130,130)'),

    yaxis = dict(

        title = 'Median Yearly Income ($)'))

fig = go.Figure(data=[trace1], layout=layout)

py.iplot(fig)
temp_df = df.groupby('EmployerIndustry').agg(f).sort_values(by=[('CompensationAmount','median')], ascending=False)

temp_df = temp_df[temp_df[('CompensationAmount','count')]>30]



trace1 = go.Bar(

    x = temp_df.index,

    y = temp_df[('CompensationAmount','median')],

    text = ['Total Users ' + str(x) for x in temp_df[('CompensationAmount','count')]])



layout = go.Layout(

    title = '<b>Industry<b>',

    titlefont = dict(

        size = 30,

        color = 'rgb(130,130,130)'),

    yaxis = dict(

        title = 'Median Yearly Income ($)'))

fig = go.Figure(data=[trace1], layout=layout)

py.iplot(fig)
def plot_fun(country, limit):

    if country == 'All':

        temp_df = df.groupby('MLSkillsSelect').agg(f).sort_values(by=[('CompensationAmount','median')], ascending=False)

    else:

        temp_df = df[(df.Country==country)].groupby('MLSkillsSelect').agg(f).sort_values(by=[('CompensationAmount','median')], ascending=False)

        

    temp_df = temp_df[temp_df[('CompensationAmount','count')]>limit]



    trace1 = go.Bar(

        x = temp_df.index,

        y = temp_df[('CompensationAmount','median')],

        text = ['Total Users ' + str(x) for x in temp_df[('CompensationAmount','count')]])



    layout = go.Layout(

        title = '<b>Methods used<b>',

        titlefont = dict(

            size = 30,

            color = 'rgb(130,130,130)'),

        yaxis = dict(

            title = 'Median Yearly Income ($)'))

    fig = go.Figure(data=[trace1], layout=layout)

    py.iplot(fig)



interact(plot_fun, country=['All'] + df.Country.unique().tolist(), limit =IntSlider(

    value=30,

    min=1,

    max=100,

    step=1,

    description='Minimal number of users'))

# cat_features = ['EmployerIndustry', 'MajorSelect', 'ParentsEducation', 

#         'FormalEducation', 'Tenure', 'Country', 'CurrentJobTitleSelect', 'GenderSelect']
# cat_df = df[cat_features]

# cat_df = pd.get_dummies(cat_df, drop_first=True)

# cat_df = cat_df.drop('GenderSelect_Male', axis = 1)

# cat_df['Age'] = df['Age']

# cat_df['Age'] = cat_df['Age'].fillna(cat_df['Age'].mean)
# reg =  LinearRegression()

# from sklearn.linear_model import Ridge

# reg = Ridge()



# X_train, X_test, y_train, y_test = train_test_split(cat_df, df.CompensationAmount, test_size=0.33, random_state=42)
# from sklearn.metrics import r2_score

# r2_score(reg.predict(X_train),y_train)
# scores = pd.DataFrame()

# scores['features'] = cat_df.columns

# scores['value'] = reg.coef_
# scores.sort_values(by='value', ascending = False)