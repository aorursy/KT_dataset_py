import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime

import matplotlib.pyplot as plt

import seaborn as sns



from plotly.offline import init_notebook_mode, iplot 

import plotly.figure_factory as ff

import plotly.graph_objs as go

import plotly.offline as py

import pycountry

py.init_notebook_mode(connected=True)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/glassdoor-analyze-gender-pay-gap/Glassdoor Gender Pay Gap.csv')

df['TotalPay'] = df['BasePay'] + df['Bonus']

df.head()
title = pd.get_dummies(df, columns=['Gender']).groupby('JobTitle').count().sort_values(by='Age')



fig = go.Figure(data=[go.Bar(

            x = title.index,

            y = title['Age'],

            #text=y,

            width=0.4,

            textposition='auto',

            marker=dict(color=["steelblue","dodgerblue","lightskyblue","powderblue","cyan","deepskyblue","cyan","darkturquoise","paleturquoise","turquoise"])

 )])



fig.data[0].marker.line.width = 1

fig.data[0].marker.line.color = "black"

fig.update_layout(yaxis=dict(title=''),width=700,height=500,

                  xaxis=dict(title='Roles'), title='Job Titles with Number of Entries')

fig.show()
title = pd.get_dummies(df, columns=['Gender']).groupby('JobTitle').sum()



female = go.Pie(labels=title.index,values=title['Gender_Female'],name="Female",hole=0.5,domain={'x': [0,0.46]})

male = go.Pie(labels=title.index,values=title['Gender_Male'],name="Male",hole=0.5,domain={'x': [0.52,1]})



layout = dict(title = 'Job Title Distribution', font=dict(size=14), legend=dict(orientation="h"),

              annotations = [dict(x=0.2, y=0.5, text='Female', showarrow=False, font=dict(size=20)),

                             dict(x=0.8, y=0.5, text='Male', showarrow=False, font=dict(size=20)) ])



fig = dict(data=[female, male], layout=layout)

py.iplot(fig)
edu = pd.get_dummies(df, columns=['Gender']).groupby('Education').sum()



female = go.Pie(labels=edu.index,values=edu['Gender_Female'],name="Female",hole=0.5,domain={'x': [0,0.46]})

male = go.Pie(labels=edu.index,values=edu['Gender_Male'],name="Male",hole=0.5,domain={'x': [0.52,1]})



layout = dict(title = 'Education Level Distribution', font=dict(size=14), legend=dict(orientation="h"),

              annotations = [dict(x=0.2, y=0.5, text='Female', showarrow=False, font=dict(size=20)),

                             dict(x=0.8, y=0.5, text='Male', showarrow=False, font=dict(size=20)) ])



fig = dict(data=[female, male], layout=layout)

py.iplot(fig)
seniority = pd.get_dummies(df, columns=['Gender']).groupby('Seniority').sum()



female = go.Pie(labels=seniority.index,values=seniority['Gender_Female'],name="Female",hole=0.5,domain={'x': [0,0.46]})

male = go.Pie(labels=seniority.index,values=seniority['Gender_Male'],name="Male",hole=0.5,domain={'x': [0.52,1]})



layout = dict(title = 'Seniority Level Distribution', font=dict(size=14), legend=dict(orientation="h"),

              annotations = [dict(x=0.2, y=0.5, text='Female', showarrow=False, font=dict(size=20)),

                             dict(x=0.8, y=0.5, text='Male', showarrow=False, font=dict(size=20)) ])



fig = dict(data=[female, male], layout=layout)

py.iplot(fig)
age_female = []

age_male = []

for i in range(len(df)):

    if df.iloc[i,]['Gender'] == 'Male':

        age_male.append(df.iloc[i,]['Age'])

    else:

        age_female.append(df.iloc[i,]['Age'])



hist_data = [age_female, age_male]



group_labels = ['Female', 'Male']

colors = ['#835AF1', '#333F44']



fig = ff.create_distplot(hist_data, group_labels, colors=colors,

                         show_curve=True, show_hist=False)



# Add title

fig.update(layout_title_text='Distribution of Age')

fig.show()
gender = df.groupby('Gender').count()



fig = go.Figure(data=[go.Bar(

            x = gender.index,

            y = gender['JobTitle'],

            #text=y,

            width=0.4,

            textposition='auto',

            marker=dict(color='dodgerblue')

 )])



fig.data[0].marker.line.width = 1

fig.data[0].marker.line.color = "black"

fig.update_layout(yaxis=dict(title=''),width=700,height=500,

                  title= 'No of Male and Female Job Entries on the Dataset',

                  xaxis=dict(title='Gender'))

fig.show()
female = df[df['Gender'] == 'Female'].groupby('JobTitle').sum()

male = df[df['Gender'] == 'Male'].groupby('JobTitle').sum()



female['BasePay'] /= title['Gender_Female'].tolist()

female['TotalPay'] /= title['Gender_Female'].tolist()

female['Bonus'] /= title['Gender_Female'].tolist()

male['BasePay'] /= title['Gender_Male'].tolist()

male['TotalPay'] /= title['Gender_Male'].tolist()

male['Bonus'] /= title['Gender_Male'].tolist()
fig = go.Figure(data=[

    go.Bar(name='Female', x=female.index, y=female['BasePay']),

    go.Bar(name='Male', x=male.index, y=male['BasePay'])

])

# Change the bar mode

fig.update_layout(barmode='group', title='BasePay Gap by JobTitle')

fig.show()
fig = go.Figure(data=[

    go.Bar(name='Female', x=female.index, y=female['Bonus']),

    go.Bar(name='Male', x=male.index, y=male['Bonus'])

])

# Change the bar mode

fig.update_layout(barmode='group', title='Bonus Pay Gap by JobTitle')

fig.show()
fig = go.Figure(data=[

    go.Bar(name='Female', x=female.index, y=female['TotalPay']),

    go.Bar(name='Male', x=male.index, y=male['TotalPay'])

])

# Change the bar mode

fig.update_layout(barmode='group', title='TotalPay Gap by JobTitle')

fig.show()
fig = go.Figure(data=[

    go.Bar(name='Difference', x=female.index, y= male['TotalPay'] - female['TotalPay'])

])

# Change the bar mode

fig.update_layout(barmode='group', title='Total Pay [Male - Female]')

fig.show()
diff = (male['TotalPay'] - female['TotalPay']).tolist()

titles = male.index.tolist()



for i in range(len(diff)):

    if diff[i] > 0:

        print('Men make ' + str(int(diff[i])) + ' more than Women as a ' + titles[i])

    else:

        print('Men make ' + str(int(-diff[i])) + ' less than Women as a ' + titles[i])