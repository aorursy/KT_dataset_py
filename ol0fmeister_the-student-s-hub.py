import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import folium
import folium.plugins
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
import plotly.plotly as py
from sklearn.cluster import KMeans
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from bubbly.bubbly import bubbleplot 
from plotly.graph_objs import Scatter, Figure, Layout
from __future__ import division
import datetime
import squarify
#Importing the datasets
free = pd.read_csv('../input/freeFormResponses.csv', low_memory=False, header=[0,1])
choice = pd.read_csv('../input/multipleChoiceResponses.csv', low_memory=False, header=[0,1])
schema = pd.read_csv('../input/SurveySchema.csv', low_memory=False, header=[0,1])

#Formatting the datasets
free.columns = ['_'.join(col) for col in free.columns]
choice.columns = ['_'.join(col) for col in choice.columns]
schema.columns = ['_'.join(col) for col in schema.columns]
pd.set_option('display.max_columns', None)
choice.head(2)
# Gender distribution
choice['Q1_What is your gender? - Selected Choice'].value_counts().plot(kind='bar', rot=15, title='Gender distribution of survey respondents', 
                                       figsize=(15,5))
color_brewer = ['#57B8FF','#B66D0D','#009FB7','#FBB13C','#FE6847','#4FB5A5','#8C9376','#F29F60','#8E1C4A','#85809B','#515B5D','#9EC2BE','#808080','#9BB58E','#5C0029','#151515','#A63D40','#E9B872','#56AA53','#CE6786','#449339','#2176FF','#348427','#671A31','#106B26','008DD5','#034213','#BC2F59','#939C44','#ACFCD9','#1D3950','#9C5414','#5DD9C1','#7B6D49','#8120FF','#F224F2','#C16D45','#8A4F3D','#616B82','#443431','#340F09']
def treemap(v):
    x = 0.
    y = 0.
    width = 50.
    height = 50.
    type_list = v.index
    values = v.values

    normed = squarify.normalize_sizes(values, width, height)
    rects = squarify.squarify(normed, x, y, width, height)

    shapes = []
    annotations = []
    counter = 0

    for r in rects:
        shapes.append( 
            dict(
                type = 'rect', 
                x0 = r['x'], 
                y0 = r['y'], 
                x1 = r['x']+r['dx'], 
                y1 = r['y']+r['dy'],
                line = dict( width = 1 ),
                fillcolor = color_brewer[counter]
            ) 
        )
        annotations.append(
            dict(
                x = r['x']+(r['dx']/2),
                y = r['y']+(r['dy']/2),
                text = "{}".format(type_list[counter]),
                showarrow = False
            )
        )
        counter = counter + 1
        if counter >= len(color_brewer):
            counter = 0

    # For hover text
    trace0 = go.Scatter(
        x = [ r['x']+(r['dx']/2) for r in rects ], 
        y = [ r['y']+(r['dy']/2) for r in rects ],
        text = [ str(v) for v in values ], 
        mode = 'text',
    )

    layout = dict(
        height=600, 
        width=800,
        xaxis=dict(showgrid=False,zeroline=False),
        yaxis=dict(showgrid=False,zeroline=False),
        shapes=shapes,
        annotations=annotations,
        hovermode='closest',
        font=dict(color="#FFFFFF")
    )

    # With hovertext
    figure = dict(data=[trace0], layout=layout)
    iplot(figure, filename='squarify-treemap')
country = choice["Q3_In which country do you currently reside?"].dropna()
for i in country.unique():
    if country[country == i].count() < 200:
        country[country == i] = 'Others'
x = 0.
y = 0.
width = 50.
height = 50.
type_list = country.value_counts().index
values = country.value_counts().values

normed = squarify.normalize_sizes(values, width, height)
rects = squarify.squarify(normed, x, y, width, height)

color_brewer = color_brewer
shapes = []
annotations = []
counter = 0

for r in rects:
    shapes.append( 
        dict(
            type = 'rect', 
            x0 = r['x'], 
            y0 = r['y'], 
            x1 = r['x']+r['dx'], 
            y1 = r['y']+r['dy'],
            line = dict( width = 1 ),
            fillcolor = color_brewer[counter]
        ) 
    )
    annotations.append(
        dict(
            x = r['x']+(r['dx']/2),
            y = r['y']+(r['dy']/2),
            text = "{}".format(type_list[counter]),
            showarrow = False
        )
    )
    counter = counter + 1
    if counter >= len(color_brewer):
        counter = 0

# For hover text
trace0 = go.Scatter(
    x = [ r['x']+(r['dx']/2) for r in rects ], 
    y = [ r['y']+(r['dy']/2) for r in rects ],
    text = [ str(v) for v in values ], 
    mode = 'text',
)

layout = dict(
    height=600, 
    width=850,
    xaxis=dict(showgrid=False,zeroline=False),
    yaxis=dict(showgrid=False,zeroline=False),
    shapes=shapes,
    annotations=annotations,
    hovermode='closest',
    font=dict(color="#FFFFFF"),
    margin = go.Margin(
            l=0,
            r=0,
            pad=0
        )
)

# With hovertext
figure = dict(data=[trace0], layout=layout)
iplot(figure, filename='treemap')
choice['Q2_What is your age (# years)?'].value_counts().plot(kind='bar', title='Age distribution of survey respondents', rot=15,
                                       figsize=(20,5))
counts = choice['Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice'].value_counts()
trace1 = go.Bar(
                x = counts.index,
                y = counts.values,
                name = "Current_role",
                marker = dict(color = 'red'),
                text = counts.index)
data = [trace1]
layout = go.Layout(barmode = "group",title='Frequency of current roles of respondents', yaxis= dict(title='Counts'),showlegend=False)
fig = go.Figure(data = data, layout = layout)
iplot(fig)
a = choice.loc[choice['Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice'] == 'Student']
df = pd.DataFrame(a)
counts = df['Q1_What is your gender? - Selected Choice'].value_counts()
trace1 = go.Bar(
                x = counts.index,
                y = counts.values,
                name = "student_gender",
                marker = dict(color = 'gold'),
                text = counts.index)
data = [trace1]
layout = go.Layout(barmode = "group",title='Count of male and female students', yaxis= dict(title='Counts'),showlegend=False)
fig = go.Figure(data = data, layout = layout)
iplot(fig)
counts = df['Q2_What is your age (# years)?'].value_counts()
trace1 = go.Bar(
                x = counts.index,
                y = counts.values,
                name = "age_spread",
                marker = dict(color = 'cyan'),
                text = counts.index)
data = [trace1]
layout = go.Layout(barmode = "group",title='Age distribution of students', yaxis= dict(title='Counts'),showlegend=False)
fig = go.Figure(data = data, layout = layout)
iplot(fig)
country = df["Q3_In which country do you currently reside?"].dropna()
for i in country.unique():
    if country[country == i].count() < 50:
        country[country == i] = 'Others'
x = 0.
y = 0.
width = 50.
height = 50.
type_list = country.value_counts().index
values = country.value_counts().values

normed = squarify.normalize_sizes(values, width, height)
rects = squarify.squarify(normed, x, y, width, height)

color_brewer = color_brewer
shapes = []
annotations = []
counter = 0

for r in rects:
    shapes.append( 
        dict(
            type = 'rect', 
            x0 = r['x'], 
            y0 = r['y'], 
            x1 = r['x']+r['dx'], 
            y1 = r['y']+r['dy'],
            line = dict( width = 1 ),
            fillcolor = color_brewer[counter]
        ) 
    )
    annotations.append(
        dict(
            x = r['x']+(r['dx']/2),
            y = r['y']+(r['dy']/2),
            text = "{}".format(type_list[counter]),
            showarrow = False
        )
    )
    counter = counter + 1
    if counter >= len(color_brewer):
        counter = 0

# For hover text
trace0 = go.Scatter(
    x = [ r['x']+(r['dx']/2) for r in rects ], 
    y = [ r['y']+(r['dy']/2) for r in rects ],
    text = [ str(v) for v in values ], 
    mode = 'text',
)

layout = dict(
    height=600, 
    width=850,
    xaxis=dict(showgrid=False,zeroline=False),
    yaxis=dict(showgrid=False,zeroline=False),
    shapes=shapes,
    annotations=annotations,
    hovermode='closest',
    font=dict(color="#FFFFFF"),
    margin = go.Margin(
            l=0,
            r=0,
            pad=0
        )
)

# With hovertext
figure = dict(data=[trace0], layout=layout)
iplot(figure, filename='student-only-treemap')
counts = df['Q5_Which best describes your undergraduate major? - Selected Choice'].value_counts()
trace1 = go.Bar(
                x = counts.index,
                y = counts.values,
                name = "Majors",
                marker = dict(color = 'purple'),
                text = counts.index)
data = [trace1]
layout = go.Layout(barmode = "group",title='Undergraduate majors of students', yaxis= dict(title='Counts'),showlegend=False)
fig = go.Figure(data = data, layout = layout)
iplot(fig)
counts = df['Q4_What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'].value_counts()
trace1 = go.Bar(
                x = counts.index,
                y = counts.values,
                name = "formal_peak",
                marker = dict(color = 'skyblue'),
                text = counts.index)
data = [trace1]
layout = go.Layout(barmode = "group",title='Formal education attained/planning to attain', yaxis= dict(title='Counts'),showlegend=False)
fig = go.Figure(data = data, layout = layout)
iplot(fig)
counts = df['Q20_Of the choices that you selected in the previous question, which ML library have you used the most? - Selected Choice'].value_counts()
trace1 = go.Bar(
                x = counts.index,
                y = counts.values,
                name = "ml_lib",
                marker = dict(color = 'pink'),
                text = counts.index)
data = [trace1]
layout = go.Layout(barmode = "group",title='Most used ML library', yaxis= dict(title='Counts'),showlegend=False)
fig = go.Figure(data = data, layout = layout)
iplot(fig)
counts = df['Q17_What specific programming language do you use most often? - Selected Choice'].value_counts()
trace1 = go.Bar(
                x = counts.index,
                y = counts.values,
                name = "Lang_used",
                marker = dict(color = 'lightgreen'),
                text = counts.index)
data = [trace1]
layout = go.Layout(barmode = "group",title='Programming Language most used', yaxis= dict(title='Counts'),showlegend=False)
fig = go.Figure(data = data, layout = layout)
iplot(fig)
counts = df['Q18_What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice'].value_counts()
trace1 = go.Bar(
                x = counts.index,
                y = counts.values,
                name = "Lang_recommended",
                marker = dict(color = 'yellow'),
                text = counts.index)
data = [trace1]
layout = go.Layout(barmode = "group",title='Programming Language recommended by students', yaxis= dict(title='Counts'),showlegend=False)
fig = go.Figure(data = data, layout = layout)
iplot(fig)
counts = df['Q22_Of the choices that you selected in the previous question, which specific data visualization library or tool have you used the most? - Selected Choice'].value_counts()
trace1 = go.Bar(
                x = counts.index,
                y = counts.values,
                name = "Current_role",
                marker = dict(color = 'violet'),
                text = counts.index)
data = [trace1]
layout = go.Layout(barmode = "group",title='Most used data visualization libraries used by students', yaxis= dict(title='Counts'),showlegend=False)
fig = go.Figure(data = data, layout = layout)
iplot(fig)
counts = df['Q23_Approximately what percent of your time at work or school is spent actively coding?'].value_counts()
trace1 = go.Bar(
                x = counts.index,
                y = counts.values,
                name = "coding_time",
                marker = dict(color = 'lightgreen'),
                text = counts.index)
data = [trace1]
layout = go.Layout(barmode = "group",title='Time spent by students actively coding', yaxis= dict(title='Counts'),showlegend=False)
fig = go.Figure(data = data, layout = layout)
iplot(fig)
counts = df['Q24_How long have you been writing code to analyze data?'].value_counts()
trace1 = go.Bar(
                x = counts.index,
                y = counts.values,
                name = "analyze_time",
                marker = dict(color = 'blue'),
                text = counts.index)
data = [trace1]
layout = go.Layout(barmode = "group",title='Time invested by students via coding to analyze data', yaxis= dict(title='Counts'),showlegend=False)
fig = go.Figure(data = data, layout = layout)
iplot(fig)
cnt_dict = {
    'Self-taught' : (df['Q35_Part_1_What percentage of your current machine learning/data science training falls under each category? (Answers must add up to 100%) - Self-taught'].astype(float)>0).sum(),
    'Online Courses/MOOCS/E Learning sites' : (df['Q35_Part_2_What percentage of your current machine learning/data science training falls under each category? (Answers must add up to 100%) - Online courses (Coursera, Udemy, edX, etc.)'].astype(float)>0).sum(),
    'Work' : (df['Q35_Part_3_What percentage of your current machine learning/data science training falls under each category? (Answers must add up to 100%) - Work'].astype(float)>0).sum(),
    'University' : (df['Q35_Part_4_What percentage of your current machine learning/data science training falls under each category? (Answers must add up to 100%) - University'].astype(float)>0).sum(),
    'Kaggle competitions' : (df['Q35_Part_5_What percentage of your current machine learning/data science training falls under each category? (Answers must add up to 100%) - Kaggle competitions'].astype(float)>0).sum(),
    'Other' : (df['Q35_Part_6_What percentage of your current machine learning/data science training falls under each category? (Answers must add up to 100%) - Other'].astype(float)>0).sum()
}

cnt_srs = pd.Series(cnt_dict)

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color='orange',
    ),
)

layout = go.Layout(
    title='Number of students corresponding to each learning methodology'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="learning_sources_of_students")
cnt_dict = {
    'Udacity' : int(df['Q36_Part_1_On which online platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Udacity'].value_counts()),
    'Coursera' : int(df['Q36_Part_2_On which online platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Coursera'].value_counts()),
    'edX' : int(df['Q36_Part_3_On which online platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - edX'].value_counts()),
    'DataCamp' : int(df['Q36_Part_4_On which online platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - DataCamp'].value_counts()),
    'DataQuest' : int(df['Q36_Part_5_On which online platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - DataQuest'].value_counts()),
    'Fast.AI' : int(df['Q36_Part_7_On which online platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Fast.AI'].value_counts()),
    'Kaggle_learn' : int(df['Q36_Part_6_On which online platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Kaggle Learn'].value_counts()),
    'developers.google.com' : int(df['Q36_Part_8_On which online platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - developers.google.com'].value_counts()),
    'Udemy' : int(df['Q36_Part_9_On which online platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Udemy'].value_counts()),
    'TheSchoole.AI' : int(df['Q36_Part_10_On which online platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - TheSchool.AI'].value_counts()),
    'Online University Courses' : int(df['Q36_Part_11_On which online platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Online University Courses'].value_counts()),
    'None' : int(df['Q36_Part_12_On which online platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - None'].value_counts())
    }

cnt_srs = pd.Series(cnt_dict)

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color='red',
    ),
)

layout = go.Layout(
    title='Number of students opting for each E-learning site'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="learning_platforms_of_students")

