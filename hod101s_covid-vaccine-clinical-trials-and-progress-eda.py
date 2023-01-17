!pip install calmap
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go

import plotly.figure_factory as ff

import plotly.express as px

from plotly.subplots import make_subplots

from collections import defaultdict 

import calmap

plt.rcParams['figure.figsize'] = 8, 5

plt.style.use("fivethirtyeight")

pd.options.plotting.backend = "plotly"

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import re
data = pd.read_csv('../input/covid19-clinical-trials-dataset/COVID clinical trials.csv')

data.head()
data['Start Date'] = pd.to_datetime(data['Start Date'])

data['Completion Date'] = pd.to_datetime(data['Completion Date'])

data['Primary Completion Date'] = pd.to_datetime(data['Primary Completion Date'])

data['Start Date'] = pd.to_datetime(data['First Posted'])

data['Start Date'] = pd.to_datetime(data['Results First Posted'])

data['Start Date'] = pd.to_datetime(data['Last Update Posted'])
def NullUnique(df):

    dic = defaultdict(list)

    for col in df.columns:

        dic['Feature'].append(col)

        dic['NumUnique'].append(len(df[col].unique()))

        dic['NumNull'].append(df[col].isnull().sum())

        dic['%Null'].append(round(df[col].isnull().sum()/df.shape[0] * 100,2))

    return pd.DataFrame(dict(dic)).sort_values(['%Null'],ascending=False).style.background_gradient()
NullUnique(data)
# Returns list of Series index and its count where count > threshold

def popularity(col,threshold):

    idx = []

    counts = []

    other = 0

    for index,vcount in zip(data[col].value_counts().index,data[col].value_counts().values):

        if vcount < threshold:

            other+=1

            continue

        idx.append(index)

        counts.append(vcount)

    idx.append('Others')

    counts.append(other)

    return idx,counts
fig = px.pie(data,'Study Results')

fig.update_layout(title='Do we have any results to study?')

fig.show()
fig = go.Figure(go.Bar(

    x= data.groupby('Phases').agg('count')['Rank'].sort_values(ascending=False).index, 

    y= data.groupby('Phases').agg('count')['Rank'].sort_values(ascending=False).values,  

    text=data.groupby('Phases').agg('count')['Rank'].sort_values(ascending=False).index,

    textposition='outside',

    marker_color=data.groupby('Phases').agg('count')['Rank'].sort_values(ascending=False).values

))

fig.update_layout(title='Phases across Studies')

fig.show()
data.Status.hist()
idx , counts = popularity('Interventions',8)

fig = go.Figure([go.Pie(labels=idx,values=counts,textinfo='label+percent')])

fig.update_layout(title='What are the top Interventions?')

fig.show()
conditions=list(data['Conditions'].dropna().unique())

fig, (ax2) = plt.subplots(1,1,figsize=[17, 10])

wordcloud2 = WordCloud(width=1000,height=400).generate(" ".join(conditions))

ax2.imshow(wordcloud2,interpolation='bilinear')

ax2.axis('off')

ax2.set_title('What Conditions are we trying to treat',fontsize=20)
def cleanAge(age):

    if len(re.findall(r'\(.*\)',age)):

        return re.findall(r'\(.*\)',age)[0]

    return '('+age+')'
ageData = data.Age.apply(lambda x : cleanAge(x))

ageData.hist()
data['AgeBrackets'] = ageData
i = 0

fig = make_subplots(rows=3, cols=2, subplot_titles=list(pd.DataFrame(data.groupby(['AgeBrackets'])['Gender'].value_counts()).unstack().index))

for row in range(1,4):

    for col in range(1,3):

        dt = pd.DataFrame(data.groupby(['AgeBrackets'])['Gender'].value_counts()).unstack().iloc[i]

        fig.add_trace(go.Bar(x=dt.Gender.index,y=dt.Gender.values),row = row, col = col)        

        i+=1

fig.show()
data.Enrollment.hist()
data['Study Type'].hist()
def splitLoc(loc):

    return loc.split(',')[-1].strip()
data['Loc'] = data.Locations.apply(lambda x:splitLoc(str(x)))
fig = go.Figure([go.Choropleth(

    locations=data.groupby(['Loc']).agg('count')['Rank'].index,

    z=data.groupby(['Loc']).agg('count')['Rank'].values.astype(float),

    locationmode='country names',

    colorscale='Blues',

    autocolorscale=False,

    marker_line_color='white',

    showscale = True,

)])

fig.update_layout(title='Study Locations')

fig.show()
idx , counts = popularity('Funded Bys',0)

fig = go.Figure([go.Pie(labels=idx,values=counts,textinfo='label+percent')])

fig.update_layout(title='Who are the top Funders?')

fig.show()
fig,ax = calmap.calendarplot(data.groupby(['Start Date']).Rank.count(), monthticks=1, daylabels='MTWTFSS',cmap='YlGn',

                    linewidth=0, fig_kws=dict(figsize=(20,5)))

fig.suptitle('Start Date of Studies' )

fig.colorbar(ax[0].get_children()[1], ax=ax.ravel().tolist())

fig.show()
data['Completion Date'].dt.year.hist()