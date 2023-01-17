import pandas as pd
import numpy as np

#Import plotly in offline mode.
import plotly.offline as offline  
import plotly.graph_objs as go 

offline.init_notebook_mode()
pd.set_option('display.max_columns', 150)

application_train = pd.read_csv('../input/application_train.csv')
application_train.head()
len(application_train)
x = application_train['TARGET'].value_counts().index
y = application_train['TARGET'].value_counts().values
data = [go.Bar(
            x = x,
            y = y,
            opacity=0.6,
            width = 0.3
    )]
layout = go.Layout(title = "Distribution of data between -> (1) and ->(0)",
                  font=dict(family='Courier New, monospace', size=18, color='#7f7f7f'))
fig = go.Figure(data = data, layout = layout)
offline.iplot(fig)
gen_name = {'F':'Female', 'M':'Male', 'XNA':'Others'}
gen_index1 = application_train[application_train['TARGET'] == 1]['CODE_GENDER'].value_counts().index
x = [gen_name[ind] for ind in gen_index1]
x.append('Others')
y = application_train[application_train['TARGET'] == 1]['CODE_GENDER'].value_counts().values
y = np.append(y, 0)


bar1 = go.Bar(
    x = x,
    y = y,
    opacity=0.6,
    #width = 0.2,
    name = "Defaulters"
)

gen_index0 = application_train[application_train['TARGET'] == 0]['CODE_GENDER'].value_counts().index
x = [gen_name[ind] for ind in gen_index0]
y = application_train[application_train['TARGET'] == 0]['CODE_GENDER'].value_counts().values

bar2 = go.Bar(
    x = x,
    y = y,
    opacity=0.6,
    #width = 0.2,
    name = "Good Clients"
)

data = [bar1, bar2]
layout = go.Layout(
    barmode='group',
    title = 'Gender wise distribution - Defaulters vs Good Clients',
    font=dict(family='Courier New, monospace', size=18, color='#7f7f7f')
)

fig = go.Figure(data=data, layout=layout)
offline.iplot(fig)
application_train[application_train['TARGET'] == 0]['FLAG_OWN_CAR'].value_counts()
application_train[application_train['TARGET'] == 1]['FLAG_OWN_CAR'].value_counts()
application_train[application_train['TARGET'] == 1]['FLAG_OWN_REALTY'].value_counts()
application_train[application_train['TARGET'] == 0]['FLAG_OWN_REALTY'].value_counts()
fig = {
  "data": [
    {
      "values": application_train[application_train['TARGET'] == 1]['CNT_CHILDREN'].value_counts().values,
      "labels": application_train[application_train['TARGET'] == 1]['CNT_CHILDREN'].value_counts().index,
      "domain": {"x": [0, .48]},
      "name": "Good Client",
      #"hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    },
    {
      "values": application_train[application_train['TARGET'] == 0]['CNT_CHILDREN'].value_counts().values,
      "labels": application_train[application_train['TARGET'] == 0]['CNT_CHILDREN'].value_counts().index,
      "text":["Defaulters"],
      "textposition":"inside",
      "domain": {"x": [.52, 1]},
      "name": "Defaulters",
      #"hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    }],
  "layout": {
        "title":"Distribution of client based on number of children in family",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Good Client",
                "x": 0.17,
                "y": 0.5
            },
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Defaulters",
                "x": 0.8,
                "y": 0.5
            }
        ]
    }
}
offline.iplot(fig, filename='donut')
application_train['NAME_CONTRACT_TYPE'].value_counts()
l = len(application_train['AMT_INCOME_TOTAL'])

x = np.arange(l)
y = application_train['AMT_INCOME_TOTAL']

# Create a trace
trace = go.Scattergl(
    x = x,
    y = y,
    mode = 'markers',
    marker = dict(
    color = '#FFBAD2',
    line = dict(width = 1)
    )
)

layout = go.Layout(
    title ='Income of the client',
    font = dict(family='Courier New, monospace', size=18, color='#7f7f7f'))

fig = go.Figure(data = [trace], layout = layout)

# Plot and embed in ipython notebook!
offline.iplot(fig)
l = len(application_train['AMT_CREDIT'])

x = np.arange(l)
y = application_train['AMT_CREDIT']

# Create a trace
trace = go.Scattergl(
    x = x,
    y = y,
    mode = 'markers',
    marker = dict(
    color = '#FFBAD3',
    line = dict(width = 1)
    )
)

layout = go.Layout(title = 'Credit amount of the loan', 
                  font = dict(family='Courier New, monospace', size=18, color='#7f7f7f'))
figure = go.Figure(data = [trace], layout = layout)

# Plot and embed in ipython notebook!
offline.iplot(figure)
l = len(application_train['AMT_ANNUITY'])

x = np.arange(l)
y = application_train['AMT_ANNUITY']

# Create a trace
trace = go.Scattergl(
    x = x,
    y = y,
    mode = 'markers',
    marker = dict(
    color = '#FFBAD3',
    line = dict(width = 1)
    )
)

layout = go.Layout(title = 'Loan annuity', 
                  font = dict(family='Courier New, monospace', size=18, color='#7f7f7f'))
figure = go.Figure(data = [trace], layout = layout)

# Plot and embed in ipython notebook!
offline.iplot(figure)
l = len(application_train['AMT_GOODS_PRICE'])

x = np.arange(l)
y = application_train['AMT_GOODS_PRICE']

# Create a trace
trace = go.Scattergl(
    x = x,
    y = y,
    mode = 'markers',
    marker = dict(
    color = '#FFBAD3',
    line = dict(width = 1)
    )
)

layout = go.Layout(title = 'Price of the goods for which the loan is given', 
                  font = dict(family='Courier New, monospace', size=18, color='#7f7f7f'))
figure = go.Figure(data = [trace], layout = layout)

# Plot and embed in ipython notebook!
offline.iplot(figure)
pie = {
    'data': [
        {
            'labels': application_train['NAME_TYPE_SUITE'].value_counts().index,
            'values': application_train['NAME_TYPE_SUITE'].value_counts().values,
            'type': 'pie',
            'name': 'Accompained by client',
            'marker': {'colors': ['rgb(56, 75, 126)',
                                  'rgb(18, 36, 37)',
                                  'rgb(34, 53, 101)',
                                  'rgb(36, 55, 57)',
                                  'rgb(6, 4, 4)',
                                  'rgb(16, 23, 24)',
                                  'rgb(116, 28, 42)']},
            'domain': {'x': [.0, .33],
                       'y': [.0, .40]},
            'hoverinfo':'label+percent+name',
            'textinfo':'Accompanying Client'
        },
        {
            'labels': application_train['NAME_INCOME_TYPE'].value_counts().index,
            'values': application_train['NAME_INCOME_TYPE'].value_counts().values,
            'marker': {'colors': ['rgb(177, 127, 38)',
                                  'rgb(205, 152, 36)',
                                  'rgb(99, 79, 37)',
                                  'rgb(129, 180, 179)',
                                  'rgb(124, 103, 37)',
                                  'rgb(142, 113, 57)',
                                  'rgb(200, 105, 67)',
                                  'rgb(153, 123, 137)']},
            'type': 'pie',
            'name': 'Clients income type',
            'domain': {'x': [.34, .67],
                       'y': [.0, .40]},
            'hoverinfo':'label+percent+name',
            'textinfo':'Income Type'
        },
        {
            'labels': application_train['NAME_EDUCATION_TYPE'].value_counts().index,
            'values': application_train['NAME_EDUCATION_TYPE'].value_counts().values,
            'marker': {'colors': ['rgb(33, 75, 99)',
                                  'rgb(79, 129, 102)',
                                  'rgb(151, 179, 100)',
                                  'rgb(175, 49, 35)',
                                  'rgb(36, 73, 147)']},
            'type': 'pie',
            'name': 'Education',
            'domain': {'x': [.0, .33],
                       'y': [0.50, .90]},
            'hoverinfo':'label+percent+name',
            'textinfo':'Education Type'
        },
        {
            'labels': application_train['NAME_FAMILY_STATUS'].value_counts().index,
            'values': application_train['NAME_FAMILY_STATUS'].value_counts().values,
            'marker': {'colors': ['rgb(146, 123, 21)',
                                  'rgb(177, 180, 34)',
                                  'rgb(206, 206, 40)',
                                  'rgb(175, 51, 21)',
                                  'rgb(35, 36, 21)',
                                  'rgb(135, 33, 121)']},
            'type': 'pie',
            'name':'Family status',
            'domain': {'x': [.32, .67],
                       'y': [.60, 1]},
            'hoverinfo':'label+percent+name',
            'textinfo':'Family Status'
        },
        {
            'labels': application_train['NAME_HOUSING_TYPE'].value_counts().index,
            'values': application_train['NAME_HOUSING_TYPE'].value_counts().values,
            'marker': {'colors': ['rgb(141, 123, 21)',
                                  'rgb(157, 180, 34)',
                                  'rgb(206, 286, 40)',
                                  'rgb(175, 151, 21)',
                                  'rgb(35, 26, 21)',
                                  'rgb(135, 83, 121)']},
            'type': 'pie',
            'name':'Housing situation',
            'domain': {'x': [0.65, 1],
                       'y': [.60, 1]},
            'hoverinfo':'label+percent+name',
            'textinfo':'Housing Type'
        }
    ],
    'layout': {'title': '1. Education Type 2. Family Status 3. Housing Situation 4. Client Accompained, 5. Income Type',
                'showlegend': False} 
    
}
offline.iplot(pie)
application_train[['TARGET','NAME_TYPE_SUITE']].groupby(['TARGET','NAME_TYPE_SUITE']).size()
clAcc = application_train[['TARGET','NAME_TYPE_SUITE']].groupby(['TARGET','NAME_TYPE_SUITE']).size().reset_index().rename(
columns={0:'SumTotal'})

subject = clAcc['NAME_TYPE_SUITE'].tolist()
total = clAcc['SumTotal'].values.tolist()

data = [dict(
  type = 'scatter',
  x = subject,
  y = total,
  mode = 'markers',
  transforms = [dict(
    type = 'groupby',
    groups = subject,
    styles = [
        dict(target = 'Children', value = dict(marker = dict(color = 'blue'))),
        dict(target = 'Family', value = dict(marker = dict(color = 'red'))),
        dict(target = 'Group of people', value = dict(marker = dict(color = 'black'))),
        dict(target = 'Other_A', value = dict(marker = dict(color = 'green'))),
        dict(target = 'Other_B', value = dict(marker = dict(color = 'yellow'))),
        dict(target = 'Spouse, partner', value = dict(marker = dict(color = 'magenta'))),
        dict(target = 'Unaccompanied', value = dict(marker = dict(color = 'cyan')))
    ]
  )]
)]

offline.iplot({'data': data}, validate=False)
application_train[['TARGET','NAME_INCOME_TYPE']].groupby(['TARGET','NAME_INCOME_TYPE']).size()
InType = application_train[['TARGET','NAME_INCOME_TYPE']].groupby(['TARGET','NAME_INCOME_TYPE']).size().reset_index().rename(
columns={0:'SumTotal'})

subject = InType['NAME_INCOME_TYPE'].tolist()
total = InType['SumTotal'].values.tolist()

data = [dict(
  type = 'scatter',
  x = subject,
  y = total,
  mode = 'markers',
  transforms = [dict(
    type = 'groupby',
    groups = subject,
    styles = [
        dict(target = 'Businessman', value = dict(marker = dict(color = 'blue'))),
        dict(target = 'Commercial associate', value = dict(marker = dict(color = 'red'))),
        dict(target = 'Maternity leave', value = dict(marker = dict(color = 'black'))),
        dict(target = 'Pensioner', value = dict(marker = dict(color = 'green'))),
        dict(target = 'State servant', value = dict(marker = dict(color = 'yellow'))),
        dict(target = 'Student', value = dict(marker = dict(color = 'magenta'))),
        dict(target = 'Unemployed', value = dict(marker = dict(color = 'cyan'))),
        dict(target = 'Working', value = dict(marker = dict(color = 'indigo')))
    ]
  )]
)]

offline.iplot({'data': data}, validate=False)
application_train[['TARGET','NAME_EDUCATION_TYPE']].groupby(['TARGET','NAME_EDUCATION_TYPE']).size()
EduType = application_train[['TARGET','NAME_EDUCATION_TYPE']].groupby(['TARGET','NAME_EDUCATION_TYPE']).size().reset_index().rename(
columns={0:'SumTotal'})

subject = EduType['NAME_EDUCATION_TYPE'].tolist()
total = EduType['SumTotal'].values.tolist()

data = [dict(
  type = 'scatter',
  x = subject,
  y = total,
  mode = 'markers',
  transforms = [dict(
    type = 'groupby',
    groups = subject,
    styles = [
        dict(target = 'Academic degree', value = dict(marker = dict(color = 'blue'))),
        dict(target = 'Higher education', value = dict(marker = dict(color = 'red'))),
        dict(target = 'Incomplete higher', value = dict(marker = dict(color = 'black'))),
        dict(target = 'Lower secondary', value = dict(marker = dict(color = 'green'))),
        dict(target = 'Secondary / secondary special', value = dict(marker = dict(color = 'yellow')))
    ]
  )]
)]

offline.iplot({'data': data}, validate=False)
application_train[['TARGET','NAME_FAMILY_STATUS']].groupby(['TARGET','NAME_FAMILY_STATUS']).size()
EduType = application_train[['TARGET','NAME_FAMILY_STATUS']].groupby(['TARGET','NAME_FAMILY_STATUS']).size().reset_index().rename(
columns={0:'SumTotal'})

subject = EduType['NAME_FAMILY_STATUS'].tolist()
total = EduType['SumTotal'].values.tolist()

data = [dict(
  type = 'scatter',
  x = subject,
  y = total,
  mode = 'markers',
  transforms = [dict(
    type = 'groupby',
    groups = subject,
    styles = [
        dict(target = 'Civil marriage', value = dict(marker = dict(color = 'blue'))),
        dict(target = 'Married', value = dict(marker = dict(color = 'red'))),
        dict(target = 'Separated', value = dict(marker = dict(color = 'black'))),
        dict(target = 'Single / not married', value = dict(marker = dict(color = 'green'))),
        dict(target = 'Unknown', value = dict(marker = dict(color = 'yellow'))),
        dict(target = 'Widow', value = dict(marker = dict(color = 'cyan')))
    ]
  )]
)]

offline.iplot({'data': data}, validate=False)
application_train[['TARGET','NAME_HOUSING_TYPE']].groupby(['TARGET','NAME_HOUSING_TYPE']).size()
HType = application_train[['TARGET','NAME_HOUSING_TYPE']].groupby(['TARGET','NAME_HOUSING_TYPE']).size().reset_index().rename(
columns={0:'SumTotal'})

subject = HType['NAME_HOUSING_TYPE'].tolist()
total = HType['SumTotal'].values.tolist()

data = [dict(
  type = 'scatter',
  x = subject,
  y = total,
  mode = 'markers',
  transforms = [dict(
    type = 'groupby',
    groups = subject,
    styles = [
        dict(target = 'Co-op apartment', value = dict(marker = dict(color = 'blue'))),
        dict(target = 'House / apartment', value = dict(marker = dict(color = 'red'))),
        dict(target = 'Municipal apartment', value = dict(marker = dict(color = 'black'))),
        dict(target = 'Office apartment', value = dict(marker = dict(color = 'green'))),
        dict(target = 'Rented apartment', value = dict(marker = dict(color = 'yellow'))),
        dict(target = 'With parents', value = dict(marker = dict(color = 'cyan')))
    ]
  )]
)]

offline.iplot({'data': data}, validate=False)
x = application_train['REGION_POPULATION_RELATIVE'].value_counts().sort_index(ascending = False)[:.0192].index
y = application_train['REGION_POPULATION_RELATIVE'].value_counts().sort_index(ascending = False)[:.0192].values
data = [go.Bar(
            x = x,
            y = y,
            marker=dict(color='rgb(100,18,20)'),
            width = 0.0003
    )]
layout = go.Layout(title = "Top 15, most populated regions where client lives", 
                  font=dict(family='Courier New, monospace', size=18, color='#7f7f7f'))
fig = go.Figure(data = data, layout = layout)
offline.iplot(fig)
x = application_train['REGION_POPULATION_RELATIVE'].value_counts().sort_index(ascending = True)[:.004].index
y = application_train['REGION_POPULATION_RELATIVE'].value_counts().sort_index(ascending = True)[:.004].values
data = [go.Bar(
            x = x,
            y = y,
            marker=dict(color='rgb(100,18,20)'),
            width = 0.00002
    )]
layout = go.Layout(title = "Top 15, least populated regions where client lives", 
                  font=dict(family='Courier New, monospace', size=18, color='#7f7f7f'))
fig = go.Figure(data = data, layout = layout)
offline.iplot(fig)
temp = round((application_train['DAYS_BIRTH']/ 365) * -1).astype(int).value_counts()[:10]

labels = temp.index
values = temp.values

trace = go.Pie(labels=labels, values=values,
               hoverinfo='label+percent', textinfo='value', 
               textfont=dict(size=20),
               marker=dict(line=dict(color='#000000', width=2))
               )

layout = go.Layout(title = "Top 10 Age Group Who Took Maximum Loan",
                  font=dict(family='Courier New, monospace', size=18, color='#8f8k8k'))
fig = go.Figure(data = [trace], layout = layout)
offline.iplot(fig)
application_train['Age'] = round((application_train['DAYS_BIRTH']/ 365) * -1).astype(int)

application_train['Age Group'] = application_train['Age'].apply(lambda x: 1 if x > 20 and x <=30 else 
                                  (2 if x > 30 and x <= 40 else 
                                   (3 if x > 40 and x <= 50 else 
                                    (4 if x > 50 and x <= 60 else 5))))
gen_name = {1:'21 Years - 30 Years', 2:'31 Years - 40 Years', 3:'41 Years - 50 Years', 3:'41 Years - 50 Years',
           4: '51 Years - 60 Years', 5: '61 Years - 70 Years'}
gen_index1 = application_train[application_train['TARGET'] == 1]['Age Group'].value_counts().index
x = [gen_name[ind] for ind in gen_index1]

y = application_train[application_train['TARGET'] == 1]['Age Group'].value_counts().values

bar1 = go.Bar(
    x = x,
    y = y,
    opacity=0.6,
    #width = 0.2,
    name = "Defaulters"
)

gen_index0 = application_train[application_train['TARGET'] == 0]['Age Group'].value_counts().index
x = [gen_name[ind] for ind in gen_index0]
y = application_train[application_train['TARGET'] == 0]['Age Group'].value_counts().values

bar2 = go.Bar(
    x = x,
    y = y,
    opacity=0.6,
    #width = 0.2,
    name = "Good Clients"
)

data = [bar1, bar2]
layout = go.Layout(
    barmode='group',
    title = 'Age Range - Defaulters vs Good Clients',
    font=dict(family='Courier New, monospace', size=18, color='#7f7f7f')
)

fig = go.Figure(data=data, layout=layout)
offline.iplot(fig)