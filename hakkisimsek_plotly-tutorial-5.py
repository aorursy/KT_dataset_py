import numpy as np
import pandas as pd
import datetime
import random

from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff

import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))
df = pd.read_csv('../input/multipleChoiceResponses.csv')
df = df.iloc[:,~df.columns.str.contains('_')]
df.columns = df.iloc[0]
df = df.iloc[1:]
df.head()
df['Duration (in seconds)'] = df['Duration (in seconds)'].astype('int')
df['For how many years have you used machine learning methods (at work or in school)?'] = np.where(df['For how many years have you used machine learning methods (at work or in school)?'] =='I have never studied machine learning but plan to learn in the future', 
                                        'I have never studied ML but plan to learn',
                                       df['For how many years have you used machine learning methods (at work or in school)?'] )

male = df[(df['What is your gender? - Selected Choice'] == 'Male')]
female = df[(df['What is your gender? - Selected Choice'] == 'Female')]

arr_order1 =  [      
               '< 1 year', '1-2 years', '3-5 years', '5-10 years', 
            '10-20 years', '20-30 years','30-40 years', '40+ years'
]

arr_order2 =  [
   '< 1 year', '1-2 years', 
'2-3 years', '3-4 years', '4-5 years','5-10 years', '10-15 years', '20+ years'
    ]

arr_order3 = [
    '0','0-10', '10-20','20-30', '30-40', '40-50', '50-60', '60-70',
     '70-80', '80-90','90-100'
]
def HistChart(column, title, limit, angle, order):
    
    count_male = round((male[column].value_counts(normalize=True) * 100),3) 
    count_male = count_male.reindex_axis(order).to_frame()[:limit]
    count_female = round((female[column].value_counts(normalize=True) * 100),3)
    count_female = count_female.reindex_axis(order).to_frame()[:limit]
    
    color1 = random.choice(['red',  'navy', 'pink','orange', 'indigo', 'tomato' 
                            ])
    color2 = random.choice([ 'lightgreen',  'aqua','skyblue', 'lightgrey',  
                            'cyan','yellow'
                           ])
    
    trace1 = go.Bar(
        x=count_male.index,
        y=count_male[column],
        name='male',
        marker=dict(
            color = color1
        )
    )

    trace2 = go.Bar(
        x=count_female.index,
        y=count_female[column],
        name='female',
        marker=dict(
            color = color2
        )
    )

    data = [trace1,trace2]
    layout = go.Layout(xaxis=dict(tickangle=angle),titlefont=dict(size=13),
        title=title, yaxis = dict(title = '%')
    )
    
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)
def PieChart(column, title, limit):
    
    color = ['red',  'navy',  'cyan', 'lightgrey','orange', 'gold','lightgreen', 
                            '#D0F9B1','tomato', 'tan']
    
    count_male = male[column].value_counts()[:limit].reset_index()
    count_female = female[column].value_counts()[:limit].reset_index()
    
    trace1 = go.Pie(labels=count_male['index'], 
                    values=count_male[column], 
                    name= "male", 
                    hole= .5, 
                    domain= {'x': [0, .48]},
                   marker=dict(colors=color))

    trace2 = go.Pie(labels=count_female['index'], 
                    values=count_female[column], 
                    name="female", 
                    hole= .5,  
                    domain= {'x': [.52, 1]})

    layout = dict(title= title, font=dict(size=12), legend=dict(orientation="h"),
                  annotations = [
                      dict(
                          x=.20, y=.5,
                          text='Male', 
                          showarrow=False,
                          font=dict(size=20)
                      ),
                      dict(
                          x=.81, y=.5,
                          text='Female', 
                          showarrow=False,
                          font=dict(size=20)
                      )
        ])
    
    fig = dict(data=[trace1, trace2], layout=layout)
    py.iplot(fig)
def major_comparison(comp, eng, math_stat, business, column,title, angle,limit, order=None):
    
    df1 = round(comp[column]\
             .value_counts(normalize=True), 4).to_frame()[:limit]
    df1 = df1.reindex_axis(order)

    df2 = round(eng[column]\
             .value_counts(normalize=True), 4).to_frame()[:limit]
    df2 = df2.reindex_axis(order)
    
    df3 = round(math_stat[column]\
             .value_counts(normalize=True), 4).to_frame()[:limit]
    df3 = df3.reindex_axis(order)
    
    df4 = round(business[column]\
             .value_counts(normalize=True), 4).to_frame()[:limit]
    df4 = df4.reindex_axis(order)
    
    trace1 = go.Bar(
        x=df1.index,
        y=df1[column],
        name='CS Engineer',
        marker=dict(
         color = 'red'
        )
    )

    trace2 = go.Bar(
        x=df2.index,
        y=df2[column],
         name='Non-CS Engineer',
        marker=dict(
            color = 'navy'
        )
    )

    trace3 = go.Bar(
        x=df3.index,
        y=df3[column],
         name='Math & Stats',
        marker=dict(
            color = 'grey'
        )
    )

    trace4 = go.Bar(
        x=df4.index,
        y=df4[column],
         name='Business Majors',
        marker=dict(
            color = 'aqua'
        )
    )

    fig = tools.make_subplots(rows=2, cols=2, subplot_titles=('Computer Engineer', 'Non-CS Engineer',
                                                              'Math & Stats','Business Majors'))
    fig.append_trace(trace1, 1,1)
    fig.append_trace(trace2, 1,2)
    fig.append_trace(trace3, 2,1)
    fig.append_trace(trace4, 2,2)
    
    fig['layout'].update( height=500, width=850, 
                         title=title,  font=dict(size=10),
                         showlegend=False)  
    fig['layout']['xaxis1'].update(dict(tickangle=angle,tickfont = dict(size = 10)))
    fig['layout']['xaxis2'].update(dict(tickangle=angle,tickfont = dict(size = 10)))
    fig['layout']['xaxis3'].update(dict(tickangle=angle,tickfont = dict(size = 10)))
    fig['layout']['xaxis4'].update(dict(tickangle=angle, tickfont = dict(size = 10)))
    py.iplot(fig)
colors = ['aqua', 'lightgrey', 'lightgreen', '#D0F9B1', 'khaki', 'grey']

def PieChart2(column, title, limit):
    count_trace = df[column].value_counts()[:limit].reset_index()
    trace1 = go.Pie(labels=count_trace['index'], 
                    values=count_trace[column], 
                    name= "count", 
                    hole= .5, 
                    textfont=dict(size=10),
                   marker=dict(colors=colors))
    layout = dict(title= title, font=dict(size=15))
    
    fig = dict(data=[trace1], layout=layout)
    py.iplot(fig)
count_geo = df.groupby('In which country do you currently reside?')['In which country do you currently reside?'].count()

data = [dict(
        type = 'choropleth',
        locations = count_geo.index,
        locationmode = 'country names',
        z = count_geo.values,
        text = count_geo.index,
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],
                      [0.5,"rgb(70, 100, 245)"],
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],
                      [1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = ''),
      ) ]

layout = dict(
    title = 'Number of Participants by Country',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)
PieChart2('What is your gender? - Selected Choice', '', 5)
df = df[~((df['What is your current yearly compensation (approximate $USD)?'] == 'I do not wish to disclose my approximate yearly compensation') |
   (df['How long have you been writing code to analyze data?'] == 'I have never written code and I do not want to learn') |
           (df['How long have you been writing code to analyze data?'] == 'I have never written code but I want to learn') |
      (df['For how many years have you used machine learning methods (at work or in school)?'] == 'I have never studied ML but plan to learn') |
        (df['For how many years have you used machine learning methods (at work or in school)?'] == 'I have never studied machine learning and I do not plan to'))]
  
labels = ['male', 'female']
colors = ['navy', 'deeppink']

dur1 = df[(df['What is your gender? - Selected Choice'] == 'Male') & 
          (df['Duration (in seconds)'] < 3600)]['Duration (in seconds)']
dur2 = df[(df['What is your gender? - Selected Choice'] == 'Female') & 
          (df['Duration (in seconds)'] < 3600)]['Duration (in seconds)']

hist_data = [dur1, dur2]
fig = ff.create_distplot(hist_data, labels, colors=colors, 
                         show_hist=False)

fig['layout'].update(title='')
py.iplot(fig)
trace0 = go.Box(x=male[male['Duration (in seconds)'] < 3600]\
                ['Duration (in seconds)'], name="Male",fillcolor='navy')
trace1 = go.Box(x=female[female['Duration (in seconds)'] < 3600]\
                ['Duration (in seconds)'],name="Female",fillcolor='deeppink')

data = [trace0, trace1]
layout = dict(title='')

fig = dict(data=[trace0, trace1], layout=layout)
py.iplot(fig)
age1 = round(df['What is your age (# years)?'].value_counts(normalize=True).\
             to_frame().sort_index(), 4)
age2 = round(df[df['What is your gender? - Selected Choice'] == 'Male']\
                    ['What is your age (# years)?'].value_counts(normalize=True).\
             to_frame().sort_index(), 4)
age3 = round(df[df['What is your gender? - Selected Choice'] == 'Female']\
                    ['What is your age (# years)?'].value_counts(normalize=True).\
             to_frame().sort_index(), 4)

trace = [
    go.Bar(x=age1.index,
    y=age1['What is your age (# years)?'],
                opacity = 0.8,
                 name="total",
                 hoverinfo="y",
                 marker=dict(
        color = age1['What is your age (# years)?'],
        colorscale='Reds',
        showscale=True)
                ),
    
    go.Bar(x=age2.index,
    y=age2['What is your age (# years)?'],
                 visible=False,
                 opacity = 0.8,
                 name = "male",
                 hoverinfo="y",
                 marker=dict(
        color = age2['What is your age (# years)?'],
        colorscale='Blues',
        reversescale = True,
        showscale=True)
                ),
    
    go.Bar(x=age3.index,
    y=age3['What is your age (# years)?'],
                 visible=False,
                opacity = 0.8,
                 name = "female",
                 hoverinfo="y",
                marker=dict(
        color = age3['What is your age (# years)?'],
        colorscale='Bluered',
        reversescale = True,
        showscale=True)    
                )
]

layout = go.Layout(title = '',
    paper_bgcolor = 'rgb(240, 240, 240)',
     plot_bgcolor = 'rgb(240, 240, 240)',
    autosize=True,
                   xaxis=dict(title="",
                             titlefont=dict(size=20),
                             tickmode="linear"),
                   yaxis=dict(title="%",
                             titlefont=dict(size=17)),
                  )

updatemenus = list([
    dict(
    buttons=list([
        dict(
            args = [{'visible': [True, False, False, False, False, False]}],
            label="Total",
            method='update',
        ),
        dict(
            args = [{'visible': [False, True, False, False, False, False]}],
            label="Male",
            method='update',
        ),
        dict(
            args = [{'visible': [False, False, True, False, False, False]}],
            label="Female",
            method='update',
        ),
        
    ]),
        direction="down",
        pad = {'r':10, "t":10},
        x=0.1,
        y=1.25,
        yanchor='top',
    ),
])
layout['updatemenus'] = updatemenus

fig = dict(data=trace, layout=layout)
py.iplot(fig)
PieChart("What is the highest level of formal education that you have attained or plan to attain within the next 2 years?", "", 5)
PieChart("Which best describes your undergraduate major? - Selected Choice", "", 6)
comp = df[df["Which best describes your undergraduate major? - Selected Choice"] == 'Computer science (software engineering, etc.)']
eng = df[df["Which best describes your undergraduate major? - Selected Choice"] == 'Engineering (non-computer focused)']
math_stat = df[df["Which best describes your undergraduate major? - Selected Choice"] == 'Mathematics or statistics']
business = df[df["Which best describes your undergraduate major? - Selected Choice"] == 'A business discipline (accounting, economics, finance, etc.)']
arr_order4 =  [
       '0-10,000', '10-20,000', '20-30,000', '30-40,000', '40-50,000',
       '50-60,000', '60-70,000', '70-80,000', '80-90,000', '90-100,000',
        '100-125,000','125-150,000','150-200,000', '200-250,000', 
        '250-300,000','300-400,000', '400-500,000','500,000+'
]

age1 = round(df['What is your current yearly compensation (approximate $USD)?']\
             .value_counts(normalize=True).to_frame().sort_index(), 4)
age1 = age1.reindex_axis(arr_order4)

age2 = round(df[df['What is your gender? - Selected Choice'] == 'Male']\
                    ['What is your current yearly compensation (approximate $USD)?'].\
             value_counts(normalize=True).to_frame().sort_index(), 4)
age2 = age2.reindex_axis(arr_order4)

age3 = round(df[df['What is your gender? - Selected Choice'] == 'Female']\
                    ['What is your current yearly compensation (approximate $USD)?'].\
             value_counts(normalize=True).to_frame().sort_index(), 4)
age3 = age3.reindex_axis(arr_order4)

trace = [
    go.Bar(x=age1.index,
    y=age1['What is your current yearly compensation (approximate $USD)?'],
                opacity = 0.7,
                 name="total",
                 hoverinfo="y",
                 marker=dict(
        color = age1['What is your current yearly compensation (approximate $USD)?'],
        colorscale='Blues',
        reversescale = True,
        showscale=True)
                ),
    
    go.Bar(x=age2.index,
    y=age2['What is your current yearly compensation (approximate $USD)?'],
                 visible=False,
                 opacity = 0.7,
                 name = "male",
                 hoverinfo="y",
                 marker=dict(
        color = age2['What is your current yearly compensation (approximate $USD)?'],
        colorscale='Reds',
        showscale=True)
                ),
    
    go.Bar(x=age3.index,
    y=age3['What is your current yearly compensation (approximate $USD)?'],
                 visible=False,
                opacity = 0.7,
                 name = "female",
                 hoverinfo="y",
                marker=dict(
        color = age3['What is your current yearly compensation (approximate $USD)?'],
        colorscale='Bluered',
        reversescale = True,
        showscale=True)    
                )
]

layout = go.Layout(title = '',
    paper_bgcolor = 'rgb(240, 240, 240)',
     plot_bgcolor = 'rgb(240, 240, 240)',
    autosize=True,
                   xaxis=dict(title="", tickangle=30,
                             titlefont=dict(size=12),
                             tickmode="linear"),
                   yaxis=dict(title="%",
                             titlefont=dict(size=17)),
                  )

updatemenus = list([
    dict(
    buttons=list([
        dict(
            args = [{'visible': [True, False, False, False, False, False]}],
            label="Total",
            method='update',
        ),
        dict(
            args = [{'visible': [False, True, False, False, False, False]}],
            label="Male",
            method='update',
        ),
        dict(
            args = [{'visible': [False, False, True, False, False, False]}],
            label="Female",
            method='update',
        ),
        
    ]),
        direction="down",
        pad = {'r':10, "t":10},
        x=0.1,
        y=1.25,
        yanchor='top',
    ),
])
layout['updatemenus'] = updatemenus

fig = dict(data=trace, layout=layout)
py.iplot(fig)
major_comparison(comp, eng, math_stat, business,'What is your current yearly compensation (approximate $USD)?','Salary distribution by majors',40, 20,arr_order4 )
HistChart("Select the title most similar to your current role (or most recent title if retired): - Selected Choice", "Job titles by gender",10,15, df["Select the title most similar to your current role (or most recent title if retired): - Selected Choice"].unique())
order = df["Select the title most similar to your current role (or most recent title if retired): - Selected Choice"].value_counts()[:10].index
major_comparison(comp, eng, math_stat, business,"Select the title most similar to your current role (or most recent title if retired): - Selected Choice",'Job titles by majors',20, 10,order )
HistChart('In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice', 'Industries by gender', 10,15, df['In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice'].unique())
order = df['In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice'].value_counts()[:10].index
major_comparison(comp, eng, math_stat, business,'In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice','Industries by majors',20, 10,order )
HistChart('What specific programming language do you use most often? - Selected Choice', 'Programs most often used by gender', 10, 15,
         df['What specific programming language do you use most often? - Selected Choice'].unique())
order = df['What specific programming language do you use most often? - Selected Choice'].value_counts()[:10].index
major_comparison(comp, eng, math_stat, business,'What specific programming language do you use most often? - Selected Choice','Programs most often used by majors',20, 10,order )
HistChart('What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice', 'Recommended programs by gender', 10, 0,
         df['What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice'].unique())
order = df['What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice'].value_counts()[:10].index
major_comparison(comp, eng, math_stat, business,'What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice','Recommended programs by majors',20, 10,order )
HistChart('Of the choices that you selected in the previous question, which ML library have you used the most? - Selected Choice', 'Top ML libraries by gender', 12,20,
         df['Of the choices that you selected in the previous question, which ML library have you used the most? - Selected Choice'].unique())
order = df['Of the choices that you selected in the previous question, which ML library have you used the most? - Selected Choice'].value_counts()[:10].index
major_comparison(comp, eng, math_stat, business,'Of the choices that you selected in the previous question, which ML library have you used the most? - Selected Choice','ML libraries by majors',20, 10,order )
HistChart('Of the choices that you selected in the previous question, which specific data visualization library or tool have you used the most? - Selected Choice', 'Top visualization libraries by gender', 10, 0,
         df['Of the choices that you selected in the previous question, which specific data visualization library or tool have you used the most? - Selected Choice'].unique())
order = df['Of the choices that you selected in the previous question, which specific data visualization library or tool have you used the most? - Selected Choice'].value_counts()[:10].index
major_comparison(comp, eng, math_stat, business,'Of the choices that you selected in the previous question, which specific data visualization library or tool have you used the most? - Selected Choice','Top visualization libraries by majors',20, 10,order )
HistChart('Do you consider yourself to be a data scientist?', 'Are you a data scientist?', 10,0,
         df['Do you consider yourself to be a data scientist?'].unique())
order = df['Do you consider yourself to be a data scientist?'].value_counts()[:10].index
major_comparison(comp, eng, math_stat, business,'Do you consider yourself to be a data scientist?','Are you a data scientist?',20, 10,order )
HistChart('On which online platform have you spent the most amount of time? - Selected Choice', 'Top online platforms by gender', 8, 10,
         df['On which online platform have you spent the most amount of time? - Selected Choice'].unique())
order = df['On which online platform have you spent the most amount of time? - Selected Choice'].value_counts()[:10].index
major_comparison(comp, eng, math_stat, business,'On which online platform have you spent the most amount of time? - Selected Choice','Top online platforms by majors',20, 10,order )
HistChart('How long have you been writing code to analyze data?', 
          'Active coding by gender', 20, 13, arr_order1)
major_comparison(comp, eng, math_stat, business,'How long have you been writing code to analyze data?','Active coding by majors',20, 10,arr_order1 )
HistChart('For how many years have you used machine learning methods (at work or in school)?', 'Years spent in ML by gender', 10,13, arr_order2)
major_comparison(comp, eng, math_stat, business,'For how many years have you used machine learning methods (at work or in school)?', 'Years spent in ML by majors',20, 10, arr_order2 )
HistChart('What is the type of data that you currently interact with most often at work or school? - Selected Choice', 'Data types by gender', 9, 10,
         df['What is the type of data that you currently interact with most often at work or school? - Selected Choice'].unique())
order = df['What is the type of data that you currently interact with most often at work or school? - Selected Choice'].value_counts()[:10].index
major_comparison(comp, eng, math_stat, business,'What is the type of data that you currently interact with most often at work or school? - Selected Choice','Data types by majors',20, 10,order )
HistChart('Approximately what percent of your data projects involved exploring unfair bias in the dataset and/or algorithm?', 'Time spent on unfair bias by gender', 10,0,
         arr_order3)
major_comparison(comp, eng, math_stat, business,'Approximately what percent of your data projects involved exploring unfair bias in the dataset and/or algorithm?', 'Time spent on unfair bias by majors',40, 20,arr_order3 )
HistChart('Approximately what percent of your data projects involve exploring model insights?', '% of projects involve exploring model insights by gender', 10, 0, arr_order3)
major_comparison(comp, eng, math_stat, business,'Approximately what percent of your data projects involve exploring model insights?', '% of projects involve exploring model insights by majors',40, 20,arr_order3 )
PieChart('Which better demonstrates expertise in data science: academic achievements or independent projects? - Your views:', "Which demonsrates expertise in Data Science?", 6)
PieChart('Do you consider ML models to be "black boxes" with outputs that are difficult or impossible to explain?', 'Are ML models black boxes?', 10)