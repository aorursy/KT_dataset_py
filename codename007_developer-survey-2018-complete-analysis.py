import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
from pandas import Series

import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
import plotly.plotly as py1
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
from mpl_toolkits.basemap import Basemap
from numpy import array
from matplotlib import cm

# import cufflinks and offline mode
import cufflinks as cf
cf.go_offline()

from wordcloud import WordCloud, STOPWORDS
from scipy.misc import imread
import base64

from sklearn import preprocessing
# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings("ignore")

# Print all rows and columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
stack_data = pd.read_csv('../input/survey_results_public.csv')
schema = pd.read_csv('../input/survey_results_schema.csv')
print("Size of data", stack_data.shape)
stack_data.head()
pd.options.display.max_colwidth = 300
schema
# checking missing data in stack data 
total = stack_data.isnull().sum().sort_values(ascending = False)
percent = (stack_data.isnull().sum()/stack_data.isnull().count()*100).sort_values(ascending = False)
missing_stack_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_stack_data
temp = stack_data['Hobby'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='% of Developers who codes as their hobby', hole = 0.8, color = ['#00FFFF','#CDC0B0'])
temp = stack_data['OpenSource'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='% of Developers who contributes to open source projects', hole = 0.5, color = ['#FAEBD7','#7FFFD4'])
temp = stack_data["Country"].dropna().value_counts().head(10)
temp.iplot(kind='bar', xTitle = 'Country name', yTitle = "Count", title = 'Which Country having highest number of respondents', color='#8A360F')
temp = stack_data['Student'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='% of Developers who are students', hole = 0.6, color = ['#7FFF00','#FF6103','#8A360F'])
temp = stack_data['Employment'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Employment Status of Developers', hole = 0.8, color = ['#8B7355','#FF6103','#8EE5EE','#458B00','#FFF8DC','#68228B'])
cnt_srs = stack_data["FormalEducation"].value_counts()
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=(cnt_srs/cnt_srs.sum() * 100)[::-1],
    orientation = 'h',
    marker=dict(
        color=['#8B7355','#BF3EFF','#FF6103','#8EE5EE','#458B00','#FFF8DC','#68228B','#1E90FF','#FFC125'],
        colorscale = 'Blues',
        reversescale = True
    ),
)

layout = dict(
    title='highest level of formal education of Developers in %',
    margin=dict(
    l=600,
)
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
cnt_srs = stack_data["UndergradMajor"].value_counts()
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=(cnt_srs/cnt_srs.sum() * 100)[::-1],
    orientation = 'h',
    marker=dict(
        color=['#8B7355','#BF3EFF','#FF6103','#8EE5EE','#458B00','#FFF8DC','#68228B','#1E90FF','#FFC125','#8B8B00','#FF3E96'],
        colorscale = 'Blues',
        reversescale = True
    ),
)

layout = dict(
    title='highest level of formal education of Developers in %',
    margin=dict(
    l=500,
)
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
cnt_srs = stack_data["CompanySize"].value_counts()
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=(cnt_srs/cnt_srs.sum() * 100)[::-1],
    orientation = 'h',
    marker=dict(
        color=['#8B7355','#BF3EFF','#FF6103','#8EE5EE','#458B00','#FFF8DC','#68228B','#1E90FF','#FFC125'],
        colorscale = 'Blues',
        reversescale = True
    ),
)

layout = dict(
    title='Number of people are employed by the company or organization they work for in %',
    margin=dict(
    l=200,
)
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
temp1 = pd.DataFrame(stack_data['DevType'].dropna().str.split(';').tolist()).stack()
cnt_srs = temp1.value_counts().sort_values(ascending=False)
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=(cnt_srs/cnt_srs.sum() * 100)[::-1],
    orientation = 'h',
)

layout = dict(
    title='Description of people who participated in survey (%)',
    margin=dict(
    l=400,
)
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
temp = stack_data["JobSatisfaction"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Satisfaction of peoples with their current job', hole = 0.8, color = ['#8B7355','#FFFF00','#FF6103','#8EE5EE','#458B00','#FFF8DC','#68228B'])
temp = stack_data["CareerSatisfaction"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Satisfaction of peoples with their career thus far', hole = 0.8, color = ['#FFF8DC','#68228B','#1E90FF','#8B7355','#FFC125'])
temp = stack_data["SurveyEasy"].value_counts()
#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
    marker=dict(
        color=['#FFF8DC','#68228B','#1E90FF','#FFC125'],
        colorscale = 'Blues',
        reversescale = True
    ),
)
data = [trace]
layout = go.Layout(
    title = "# of peoples with Difficulty of the survey in % ",
    xaxis=dict(
        title='Survey was easy or difficult',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')
cnt_srs = stack_data["SurveyTooLong"].value_counts()
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=(cnt_srs/cnt_srs.sum() * 100)[::-1],
    orientation = 'h',
    marker=dict(
        color=['#8B7355','#BF3EFF','#FF6103','#FFC125'],
        colorscale = 'Blues',
        reversescale = True
    ),
)

layout = dict(
    title='# of people thinking about length of the survey (%)',
    margin=dict(
    l=300,
)
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
temp = stack_data["YearsCoding"].value_counts()
#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
    marker=dict(
        color=['#FF3E96','#00E5EE','#FFF8DC','#68228B','#1E90FF','#FFC125','#FF6103','#8EE5EE','#458B00','#FFF8DC'],
        colorscale = 'Blues',
        reversescale = True
    ),
)
data = [trace]
layout = go.Layout(
    title = "for how many years have peoples been coding (%) ",
    xaxis=dict(
        title='Years',
        tickfont=dict(
            size=11,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')
temp = stack_data["YearsCodingProf"].value_counts()
#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
    marker=dict(
        color=['#FF3E96','#00E5EE','#FFF8DC','#68228B','#1E90FF','#FFC125','#FF6103','#8EE5EE','#458B00','#FFF8DC'],
        colorscale = 'Blues',
        reversescale = True
    ),
)
data = [trace]
layout = go.Layout(
    title = "for how many years have peoples been coding (%) ",
    xaxis=dict(
        title='Years',
        tickfont=dict(
            size=11,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')
cnt_srs = stack_data["HopeFiveYears"].value_counts()
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=(cnt_srs/cnt_srs.sum() * 100)[::-1],
    orientation = 'h',
    marker=dict(
        color=['#FF6103','#8EE5EE','#458B00','#FFF8DC' ,'#BF3EFF','#FF6103','#FFC125'],
        colorscale = 'Blues',
        reversescale = True
    ),
)

layout = dict(
    title='Peoples hope to be doing in the next five years (%)',
    margin=dict(
    l=500,
)
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
cnt_srs = stack_data["JobSearchStatus"].value_counts()
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=(cnt_srs/cnt_srs.sum() * 100)[::-1],
    orientation = 'h',
    marker=dict(
        color=['#BF3EFF','#FF6103','#FFC125'],
        colorscale = 'Blues',
        reversescale = True
    ),
)

layout = dict(
    title='Peoples current job-seeking status (%)',
    margin=dict(
    l=500,
)
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
cnt_srs = stack_data["LastNewJob"].value_counts()
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=(cnt_srs/cnt_srs.sum() * 100)[::-1],
    orientation = 'h',
    marker=dict(
        color=['#FF6103','#8EE5EE','#458B00','#FFF8DC'],
        colorscale = 'Blues',
        reversescale = True
    ),
)

layout = dict(
    title='When was the last time that peoples took a job with a new employer (%)',
    margin=dict(
    l=300,
)
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
temp1 = pd.DataFrame(stack_data['CommunicationTools'].dropna().str.split(';').tolist()).stack()
temp = temp1.value_counts().sort_values(ascending=False).head(20)
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
)
data = [trace]
layout = go.Layout(
    title = "Most popular communication tools (%) ",
    xaxis=dict(
        title='Tool Name',
        tickfont=dict(
            size=12,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
temp1 = pd.DataFrame(stack_data['LanguageWorkedWith'].dropna().str.split(';').tolist()).stack()
temp1 = temp1.value_counts().sort_values(ascending=False).head(20)
temp2 = pd.DataFrame(stack_data['LanguageDesireNextYear'].dropna().str.split(';').tolist()).stack()
temp2 = temp2.value_counts().sort_values(ascending=False).head(20)
trace1 = go.Bar(
    y=temp1.index[::-1],
    x=temp1.values[::-1],
    orientation = 'h',
    #name = ''
)
trace2 = go.Bar(
    y=temp2.index[::-1],
    x=temp2.values[::-1],
    orientation = 'h',
    #name = ''
)

fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('On which developers worked with ', 'On which developers want to work in over the next year'))
                                                          

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig['layout'].update(height=500, width=1000, title='Most popular languages')
iplot(fig, filename='simple-subplot')
temp1 = pd.DataFrame(stack_data['DatabaseWorkedWith'].dropna().str.split(';').tolist()).stack()
temp1 = temp1.value_counts().sort_values(ascending=False).head(20)
temp2 = pd.DataFrame(stack_data['DatabaseDesireNextYear'].dropna().str.split(';').tolist()).stack()
temp2 = temp2.value_counts().sort_values(ascending=False).head(20)
trace1 = go.Bar(
    y=temp1.index[::-1],
    x=temp1.values[::-1],
    orientation = 'h',
    marker=dict(
        color=temp2.values[::-1],
        colorscale = 'red',
#         reversescale = True
    ),
    #name = ''
)
trace2 = go.Bar(
    y=temp2.index[::-1],
    x=temp2.values[::-1],
    orientation = 'h',
    #name = ''
)

fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('On which developers worked with ', 'On which developers want to work in over the next year '))
                                                          

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
                          
fig['layout'].update(height=500, width=1100, title='Most popular Databases', margin=dict(l=285,))
iplot(fig, filename='simple-subplot')
temp1 = pd.DataFrame(stack_data['PlatformWorkedWith'].dropna().str.split(';').tolist()).stack()
temp1 = temp1.value_counts().sort_values(ascending=False).head(20)
temp2 = pd.DataFrame(stack_data['PlatformDesireNextYear'].dropna().str.split(';').tolist()).stack()
temp2 = temp2.value_counts().sort_values(ascending=False).head(20)
trace1 = go.Bar(
    y=temp1.index[::-1],
    x=temp1.values[::-1],
    orientation = 'h',
    #name = ''
)
trace2 = go.Bar(
    y=temp2.index[::-1],
    x=temp2.values[::-1],
    orientation = 'h',
    #name = ''
)

fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('On which developers worked with ', 'On which developers want to work in over the next year'))
                                                          

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
                          
fig['layout'].update(height=500, width=1100, title='Most popular platforms', margin=dict(l=250,))
iplot(fig, filename='simple-subplot')
temp1 = pd.DataFrame(stack_data['FrameworkWorkedWith'].dropna().str.split(';').tolist()).stack()
temp1 = temp1.value_counts().sort_values(ascending=False).head(20)
temp2 = pd.DataFrame(stack_data['FrameworkDesireNextYear'].dropna().str.split(';').tolist()).stack()
temp2 = temp2.value_counts().sort_values(ascending=False).head(20)
trace1 = go.Bar(
    y=temp1.index[::-1],
    x=temp1.values[::-1],
    orientation = 'h',
    marker=dict(
        color=temp2.values[::-1],
        colorscale = 'red',
#         reversescale = True
    ),
    #name = ''
)
trace2 = go.Bar(
    y=temp2.index[::-1],
    x=temp2.values[::-1],
    orientation = 'h',
    #name = ''
)

fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('On which developers worked with', 'On which developers want to work in over the next year'))
                                                          

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
                          
fig['layout'].update(height=500, width=1100, title='Most popular Frameworks', margin=dict(l=100,))
iplot(fig, filename='simple-subplot')
temp = stack_data['StackOverflowRecommend'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='StackOverflow Recommendation', hole = 0.8,)
temp = stack_data['StackOverflowVisit'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Peoples visiting StackOverflow', hole = 0.6, color =['#FAEBD7','#00FFFF','#458B74','#C1FFC1','#7FFF00','#FF7F24'])
temp = stack_data['StackOverflowHasAccount'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Has StackOverflow account', hole = 0.6, color =['#FAEBD7','#00FFFF','#458B74'])
cnt_srs = stack_data["StackOverflowParticipate"].value_counts()
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=(cnt_srs/cnt_srs.sum() * 100)[::-1],
    orientation = 'h',
    marker=dict(
        color=['#FFF8DC','#68228B','#1E90FF','#FFC125','#C1FFC1','#7FFF00','#FF7F24'],
        colorscale = 'Blues',
        reversescale = True
    ),
)

layout = dict(
    title='Participation on StackOverflow (%)',
    margin=dict(
    l=400,
)
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
temp = stack_data['StackOverflowJobs'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Number of peoples visited StackOverflow jobs board', hole = 0.6, color =['#1E90FF','#FFC125','#C1FFC1'])
cnt_srs = stack_data["StackOverflowDevStory"].value_counts()
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=(cnt_srs/cnt_srs.sum() * 100)[::-1],
    orientation = 'h',
    marker=dict(
        color=['#FFF8DC','#68228B','#1E90FF','#FFC125','#C1FFC1','#7FFF00','#FF7F24'],
        colorscale = 'Blues',
        reversescale = True
    ),
)

layout = dict(
    title='Up-to-date developer story on StackOverflow (%)',
    margin=dict(
    l=300,
)
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
temp = stack_data['StackOverflowJobsRecommend'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='StackOverflow Jobs Board Recommendation', hole = 0.7, color =['#1E90FF','#FFC125','#C1FFC1','#8B7355','#BF3EFF','#FF6103','#FFF8DC','#68228B','#FF1493','#8B0A50'])
temp = stack_data['StackOverflowConsiderMember'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Members of the Stack Overflow community', hole = 0.7, color =['#1E90FF','#FFC125','#C1FFC1',])
so_comm = stack_data[stack_data['StackOverflowConsiderMember'] == 'Yes']
temp1 = so_comm.set_index('Gender').DevType.str.split(';', expand=True).stack().reset_index('Gender')
temp1.columns = ['Gender','job']
temp = temp1['job'].value_counts()
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(temp1["Gender"][temp1["job"] == val] == 'Male'))
    temp_y0.append(np.sum(temp1["Gender"][temp1["job"] == val] == 'Female'))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Male'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Female'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Top DevType who consider themselves part of the Stack Overflow community (%)",
    barmode='stack',
    width = 1000,
    xaxis=dict(
        title='DevType',
        tickfont=dict(
            size=8,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
print('ht1 sum', stack_data['HypotheticalTools1'].map({'Not at all interested' : 1, 'A little bit interested' : 2,'Somewhat interested' : 3, 'Very interested' : 4, 'Extremely interested' : 5}).sum())
print('ht2 sum', stack_data['HypotheticalTools2'].map({'Not at all interested' : 1, 'A little bit interested' : 2,'Somewhat interested' : 3, 'Very interested' : 4, 'Extremely interested' : 5}).sum())
print('ht3 sum', stack_data['HypotheticalTools3'].map({'Not at all interested' : 1, 'A little bit interested' : 2,'Somewhat interested' : 3, 'Very interested' : 4, 'Extremely interested' : 5}).sum())
print('ht4 sum', stack_data['HypotheticalTools4'].map({'Not at all interested' : 1, 'A little bit interested' : 2,'Somewhat interested' : 3, 'Very interested' : 4, 'Extremely interested' : 5}).sum())
print('ht5 sum', stack_data['HypotheticalTools5'].map({'Not at all interested' : 1, 'A little bit interested' : 2,'Somewhat interested' : 3, 'Very interested' : 4, 'Extremely interested' : 5}).sum())

temp = stack_data[['StackOverflowVisit', 'StackOverflowRecommend']]
temp.columns = ['StackOverflowVisit', 'StackOverflowRecommend']
cm = sns.light_palette("red", as_cmap=True)
pd.crosstab(temp['StackOverflowVisit'], temp['StackOverflowRecommend']).style.background_gradient(cmap = cm)
temp = stack_data[['StackOverflowVisit', 'StackOverflowParticipate']]
cm = sns.light_palette("red", as_cmap=True)
pd.crosstab(temp['StackOverflowVisit'], temp['StackOverflowParticipate']).style.background_gradient(cmap = cm)
temp = stack_data[['StackOverflowVisit', 'StackOverflowJobs']]
cm = sns.light_palette("red", as_cmap=True)
pd.crosstab(temp['StackOverflowVisit'], temp['StackOverflowJobs']).style.background_gradient(cmap = cm)
temp = stack_data[['StackOverflowVisit', 'StackOverflowDevStory']]
cm = sns.light_palette("red", as_cmap=True)
pd.crosstab(temp['StackOverflowVisit'], temp['StackOverflowDevStory']).style.background_gradient(cmap = cm)
temp = stack_data[['StackOverflowJobs', 'StackOverflowJobsRecommend']]
cm = sns.light_palette("red", as_cmap=True)
pd.crosstab(temp['StackOverflowJobs'], temp['StackOverflowJobsRecommend']).style.background_gradient(cmap = cm)
cnt_srs = stack_data["UpdateCV"].value_counts()
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=(cnt_srs/cnt_srs.sum() * 100)[::-1],
    orientation = 'h',
    marker=dict(
        color=['#BF3EFF','#FFF8DC','#68228B','#1E90FF','#FFC125','#C1FFC1','#7FFF00','#FF7F24'],
        colorscale = 'Blues',
        reversescale = True
    ),
)

layout = dict(
    title='Reasons of upadating a CV  (%)',
    margin=dict(
    l=450,
)
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
temp = pd.DataFrame(stack_data['EducationTypes'].dropna().str.split(';').tolist()).stack()
cnt_srs =  temp.value_counts().sort_values(ascending=False)
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=(cnt_srs/cnt_srs.sum() * 100)[::-1],
    orientation = 'h',
    marker=dict(
        color=['#FF1493','#00BFFF','#FCE6C9','#1E90FF','#FFC125','#FFD700','#C1FFC1','#7FFF00','#FF7F24'],
        colorscale = 'Blues',
        reversescale = True
    ),
)

layout = dict(
    title='Types of Non-degree education in which people participated (%)',
    margin=dict(
    l=550,
)
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
temp = pd.DataFrame(stack_data['SelfTaughtTypes'].dropna().str.split(';').tolist()).stack()
cnt_srs =  temp.value_counts().sort_values(ascending=False)
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=(cnt_srs/cnt_srs.sum() * 100)[::-1],
    orientation = 'h',
    marker=dict(
        color=['#FFD700','#C1FFC1','#7FFF00','#FF7F24','#FF1493','#00BFFF','#FCE6C9','#1E90FF','#FFC125',],
        colorscale = 'Blues',
        reversescale = True
    ),
)

layout = dict(
    title='Top resources used by peoples who taught yourself without taking a course (%)',
    margin=dict(
    l=610,
)
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
temp = pd.DataFrame(stack_data['HackathonReasons'].dropna().str.split(';').tolist()).stack()
cnt_srs =  temp.value_counts().sort_values(ascending=False)
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=(cnt_srs/cnt_srs.sum() * 100)[::-1],
    orientation = 'h',
    marker=dict(
        color=['#FFD700','#C1FFC1','#FCE6C9','#1E90FF','#FFC125','#7FFF00','#FF7F24','#FF1493','#00BFFF'],
        colorscale = 'Blues',
        reversescale = True
    ),
)

layout = dict(
    title='Top reasons who participated in online coding compitition or hackathon (%)',
    margin=dict(
    l=610,
)
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
temp = stack_data['EthicsChoice'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='What peoples think about writing a code that is extremely unethical ?', hole = 0.5, color = ['#FAEBD7','#7FFFD4','#1E90FF'])
temp = stack_data['EthicsReport'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Do peoples want to report out unethical code in question ?', hole = 0.5, color = ['#8EE5EE','#458B00','#FFF8DC','#1E90FF'])
temp = stack_data['EthicsResponsible'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='who is most responsible for code that accomplishes something unethical?', hole = 0.5, color = ['#458B00','#FFF8DC','#1E90FF'])
temp = stack_data['EthicalImplications'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='There can be obligation to consider the ethical implications of the code that you write?', hole = 0.5, color = ['#FF6103','#8EE5EE','#458B00'])
temp = pd.DataFrame(stack_data['IDE'].dropna().str.split(';').tolist()).stack()
temp =  temp.value_counts().sort_values(ascending=False)
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
    marker=dict(
        color=['#FFF8DC','#68228B','#1E90FF','#FFC125','#FF6103','#8EE5EE','#458B00','#FFF8DC','#8B7355','#BF3EFF','#FF6103','#FFC125'],
        colorscale = 'Blues',
        reversescale = True
    ),
)
data = [trace]
layout = go.Layout(
    title = "Top most used IDE  by the developers (%) ",
    xaxis=dict(
        title='IDE Name',
        tickfont=dict(
            size=10,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
temp = pd.DataFrame(stack_data['OperatingSystem'].dropna().str.split(';').tolist()).stack()
temp =  temp.value_counts().sort_values(ascending=False)
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
    marker=dict(
        color=['#FFF8DC','#68228B','#1E90FF'],
        colorscale = 'Blues',
        reversescale = True
    ),
)
data = [trace]
layout = go.Layout(
    title = "Top Used Operating system by the developers (%) ",
    xaxis=dict(
        title='Operating System',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
temp = pd.DataFrame(stack_data['VersionControl'].dropna().str.split(';').tolist()).stack()
cnt_srs =  temp.value_counts().sort_values(ascending=False)
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=(cnt_srs/cnt_srs.sum() * 100)[::-1],
    orientation = 'h',
    marker=dict(
        color=['#8B7355','#BF3EFF','#FF6103','#FFC125','#8B7355','#BF3EFF','#FF6103','#FFC125'],
        colorscale = 'Blues',
        reversescale = True
    ),
)

layout = dict(
    title='Top version control system developers regularly use (%)',
    margin=dict(
    l=300,
)
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
temp = pd.DataFrame(stack_data['Methodology'].dropna().str.split(';').tolist()).stack()
temp =  temp.value_counts().sort_values(ascending=False)
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
    marker=dict(
        color=['#FFF8DC','#68228B','#1E90FF','#FFC125','#FF6103','#8EE5EE','#458B00','#FFF8DC','#8B7355','#BF3EFF','#FF6103','#FFC125'],
        colorscale = 'Blues',
        reversescale = True
    ),
)
data = [trace]
layout = go.Layout(
    title = "Top methodologies developers have experience working in (%) ",
    xaxis=dict(
        title='Methodology',
        tickfont=dict(
            size=10,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
temp = stack_data['WakeTime'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Developers wake up time on working days', hole = 0.5, color = ['#8EE5EE','#458B00','#1E90FF','#030303','#FFC125','#FFF8DC','#8B7355','#BF3EFF','#FF6103','#FFC125'])
temp = stack_data['HoursComputer'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Developers spend their time in front of desktop or computer on a typical day', hole = 0.5, color = ['#FFC125','#FFF8DC','#8B7355','#BF3EFF','#FF6103','#FFC125'])
temp = stack_data['HoursOutside'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Developers spend their time outside on a typical day', hole = 0.5, color = ['#BF3EFF','#FF6103','#FFC125','#FFF8DC','#8B7355',])
temp = stack_data['SkipMeals'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='In a typical week, how many times Developers skip a meal in order to be more productive?', hole = 0.5, color = ['#BF3EFF','#FF6103','#FFC125','#FFF8DC','#8B7355',])
temp = pd.DataFrame(stack_data['ErgonomicDevices'].dropna().str.split(';').tolist()).stack()
temp =  temp.value_counts().sort_values(ascending=False)
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
    marker=dict(
        color=['#FFF8DC','#68228B','#1E90FF','#FFC125',],
        colorscale = 'Blues',
        reversescale = True
    ),
)
data = [trace]
layout = go.Layout(
    title = "Top ergonomic furniture or devices developers use on a regular basis (%) ",
    xaxis=dict(
        title='Ergonomic furniture or device',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
temp = pd.DataFrame(stack_data['Exercise'].dropna().str.split(';').tolist()).stack()
temp =  temp.value_counts().sort_values(ascending=False)
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
    marker=dict(
        color=['#BF3EFF','#FF6103','#FFC125','#FFF8DC','#8B7355'],
        colorscale = 'Blues',
        reversescale = True
    ),
)
data = [trace]
layout = go.Layout(
    title = "In a typical week, how many times do developers exercise? (%) ",
    xaxis=dict(
        title='Number of times',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
temp = pd.DataFrame(stack_data['Age'].dropna().str.split(';').tolist()).stack()
temp =  temp.value_counts().sort_values(ascending=False)
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
    marker=dict(
        color=['#68228B','#1E90FF','#FFC125','#BF3EFF','#FF6103','#FFC125','#FFF8DC','#8B7355'],
        colorscale = 'Blues',
        reversescale = True
    ),
)
data = [trace]
layout = go.Layout(
    title = "Age of the developers of participated in the survey (%) ",
    xaxis=dict(
        title='Age of the developers',
        tickfont=dict(
            size=12,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
temp = stack_data["NumberMonitors"].dropna().value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Number of monitors are set up at developers workstation', hole = 0.5, color = ['#BF3EFF','#FF6103','#FFC125','#FFF8DC','#8B7355',])
temp = stack_data["CheckInCode"].dropna().value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Over the last year, how often have developers checked-in or committed code?', hole = 0.5, color = ['#FFF8DC','#8B7355','#BF3EFF','#FF6103','#FFC125','#0000FF'])
temp = pd.DataFrame(stack_data['Gender'].dropna().str.split(';').tolist()).stack()
temp =  temp.value_counts().sort_values(ascending=False)
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Gender', hole = 0.5, color = ['#FFF8DC','#8B7355','#BF3EFF','#FF6103','#FFC125','#0000FF'])
temp = pd.DataFrame(stack_data['SexualOrientation'].dropna().str.split(';').tolist()).stack()
temp =  temp.value_counts().sort_values(ascending=False)
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Sexual Orientation', hole = 0.5, color = ['#FF6103','#FFC125','#0000FF','#FFF8DC','#8B7355','#BF3EFF',])
temp = stack_data["Exercise"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(stack_data["Gender"][stack_data["Exercise"]==val] == 'Male'))
    temp_y0.append(np.sum(stack_data["Gender"][stack_data["Exercise"]==val] == 'Female'))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Male'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Female'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Number of times developers excercise(Male V.S. Female) (%)",
    barmode='stack',
    width = 1000,
    xaxis=dict(
        title='Number of times',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = stack_data["Age"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(stack_data["Gender"][stack_data["Age"]==val] == 'Male'))
    temp_y0.append(np.sum(stack_data["Gender"][stack_data["Age"]==val] == 'Female'))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Male'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Female'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Age of the developers who participated in the survey(Male V.S. Female) (%)",
    barmode='stack',
    width = 1000,
    xaxis=dict(
        title='Age',
        tickfont=dict(
            size=12,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = stack_data["FormalEducation"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(stack_data["Gender"][stack_data["FormalEducation"]==val] == 'Male'))
    temp_y0.append(np.sum(stack_data["Gender"][stack_data["FormalEducation"]==val] == 'Female'))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Male'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Female'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Fromal Education of developers(Male V.S. Female) (%)",
    barmode='stack',
    width = 1000,
    xaxis=dict(
        title='Fromal Education of developers',
        tickfont=dict(
            size=8,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = stack_data.set_index(['ConvertedSalary','Gender']).DevType.str.split(';', expand=True).stack().reset_index(['ConvertedSalary','Gender'])
temp.columns = ['ConvertedSalary','Gender','job']
temp = temp.set_index(['ConvertedSalary','job']).Gender.str.split(';', expand=True).stack().reset_index(['ConvertedSalary','job'])
temp.columns = ['MedianSalary','job','Gender']
temp = temp.groupby(['Gender','job'])['MedianSalary'].median().sort_values(ascending = False).reset_index()
temp1 = temp[temp.Gender == 'Male']
temp2 = temp[temp.Gender == 'Female']
trace1 = go.Bar(
    x = temp1.job,
    y = temp1.MedianSalary,
    name='Male'
)
trace2 = go.Bar(
    x = temp2.job,
    y = temp2.MedianSalary,
    name='Female'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Top DevType with Median Salary(Male V.S. Female)",
    barmode='stack',
    width = 1000,
    xaxis=dict(
        title='DevType',
        tickfont=dict(
            size=8,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Median Salary ($)',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = stack_data.groupby('Country').count().reset_index()
respondent_less_than_500 = list(temp[temp['Respondent']<=500]['Country'])
def func(x):
    if x not in respondent_less_than_500:
        return True
    return False
copy = stack_data[stack_data['Country'].apply(func)]
temp = copy.set_index(['ConvertedSalary','Gender']).Country.str.split(';', expand=True).stack().reset_index(['ConvertedSalary','Gender'])
temp.columns = ['ConvertedSalary','Gender','Country']
temp = temp.set_index(['ConvertedSalary','Country']).Gender.str.split(';', expand=True).stack().reset_index(['ConvertedSalary','Country'])
temp.columns = ['MedianSalary','Country','Gender']
temp = temp.groupby(['Gender','Country'])['MedianSalary'].median().sort_values(ascending = False).reset_index()
#temp.head()
temp1 = temp[temp.Gender == 'Male']
temp2 = temp[temp.Gender == 'Female']
trace1 = go.Bar(
    x = temp1.Country,
    y = temp1.MedianSalary,
    name='Male'
)
trace2 = go.Bar(
    x = temp2.Country,
    y = temp2.MedianSalary,
    name='Female'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Top Countries where respondents are > 500 with Median Salary(Male V.S. Female)",
    barmode='stack',
    width = 1000,
    xaxis=dict(
        title='Country Name',
        tickfont=dict(
            size=10,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Median Salary ($)',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
print("Overall Developers Annual Salary\n",stack_data['ConvertedSalary'].dropna().describe())
print("Male Developers Annual Salary\n",stack_data[stack_data.Gender == 'Male']['ConvertedSalary'].dropna().describe())
print("Female Developers Annual Salary\n",stack_data[stack_data.Gender == 'Female']['ConvertedSalary'].dropna().describe())
print("Overall Developers Annual Median Salary\n",stack_data['ConvertedSalary'].dropna().median())
print("Male Developers Annual Median Salary\n",stack_data[stack_data.Gender == 'Male']['ConvertedSalary'].dropna().median())
print("Female Developers Annual Median Salary\n",stack_data[stack_data.Gender == 'Female']['ConvertedSalary'].dropna().median())
temp = stack_data.groupby('Country').count().reset_index()
respondent_less_than_500 = list(temp[temp['Respondent']<=500]['Country'])
def func(x):
    if x not in respondent_less_than_500:
        return True
    return False
copy = stack_data[stack_data['Country'].apply(func)]
#copy['Country'].head()
temp = copy[['Country','ConvertedSalary']].groupby('Country')['ConvertedSalary'].median().sort_values(ascending = False)
temp1 = temp.head(20)
#print(temp)
temp1.iplot(kind='bar', xTitle = 'Country name', yTitle = "Median Salary ($)", title = 'Top countries where respondents are > 500 with highest Median Salary in USD ', color='green')
temp = copy[['Country','ConvertedSalary']].groupby('Country')['ConvertedSalary'].median().sort_values(ascending = False)
data = [dict(
        type='choropleth',
        locations= temp.index,
        locationmode='country names',
        z=temp.values,
        text=temp.index,
        colorscale='Red',
        marker=dict(line=dict(width=0.7)),
        colorbar=dict(autotick=False, tickprefix='', title='Median Salary ($)'),
)]
layout = dict(title = 'Top countries where respondents are > 500 with highest Median Salary in USD',
             geo = dict(
            showframe = False,
            #showcoastlines = False,
            projection = dict(
                type = 'Mercatorodes'
            )
        ),)
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)
india = stack_data[stack_data['Country'] == "India"]
india = india.set_index('ConvertedSalary').DevType.str.split(';', expand=True).stack().reset_index('ConvertedSalary')
india.columns = ['ConvertedSalary','job']
india = india[['ConvertedSalary','job']].groupby('job')['ConvertedSalary'].median().sort_values(ascending = False).head(15)

usa = stack_data[stack_data['Country'] == "United States"]
usa = usa.set_index('ConvertedSalary').DevType.str.split(';', expand=True).stack().reset_index('ConvertedSalary')
usa.columns = ['ConvertedSalary','job']
usa = usa[['ConvertedSalary','job']].groupby('job')['ConvertedSalary'].median().sort_values(ascending = False).head(15)

globl = stack_data.set_index('ConvertedSalary').DevType.str.split(';', expand=True).stack().reset_index('ConvertedSalary')
globl.columns = ['ConvertedSalary','job']
globl = globl[['ConvertedSalary','job']].groupby('job')['ConvertedSalary'].median().sort_values(ascending = False).head(15)

trace1 = go.Bar(
    y=india.index[::-1],
    x=india.values[::-1],
    orientation = 'h',
)

trace2 = go.Bar(
    y=usa.index[::-1],
    x=usa.values[::-1],
    orientation = 'h',
)

trace3 = go.Bar(
    y=globl.index[::-1],
    x=globl.values[::-1],
    orientation = 'h',
)
fig = tools.make_subplots(rows=1, cols=3, subplot_titles=('India',
                                                          'USA',
                                                          'Global'))
                                                          

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 1, 3)
                          
# fig['layout']['xaxis1'].update(title='')
# fig['layout']['xaxis2'].update(title='')
# fig['layout']['xaxis3'].update(title='')

fig['layout']['yaxis1'].update(title='DevType')
                          
fig['layout'].update(height=400, width=1500, title='Top DevType with Median Salary in ($)',margin=dict(l=300,))
iplot(fig, filename='simple-subplot')
df = stack_data.set_index(['YearsCodingProf','ConvertedSalary']).DevType.str.split(';', expand=True).stack().reset_index(['YearsCodingProf','ConvertedSalary'])
df.columns = ['YearsCodingProf','Salary','jobTitle']
df['YearsCodingProf'] = df['YearsCodingProf'].astype('category')
df['YearsCodingProf'].cat.reorder_categories(['0-2 years','3-5 years','6-8 years', '9-11 years', '12-14 years', '15-17 years', '18-20 years', 
                                            '21-23 years',  '24-26 years','27-29 years', '30 or more years'], inplace=True)
 
jobtitle = ['Full-stack developer', 'Database administrator',
       'DevOps specialist', 'System administrator', 'Engineering manager',
       'Data or business analyst',
       'Desktop or enterprise applications developer',
       'Game or graphics developer', 'QA or test developer', 'Student',
       'Back-end developer', 'Front-end developer', 'Designer',
       'C-suite executive (CEO, CTO, etc.)', 'Mobile developer',
       'Data scientist or machine learning specialist',
       'Marketing or sales professional', 'Product manager',
       'Embedded applications or devices developer',
       'Educator or academic researcher']
    
fig = {
    'data': [
        {
            'x': df[df['jobTitle']==devtypes].groupby('YearsCodingProf').agg({'Salary' : 'median'}).sort_values(by = 'YearsCodingProf').reset_index()['YearsCodingProf'],
            'y': df[df['jobTitle']==devtypes].groupby('YearsCodingProf').agg({'Salary' : 'median'}).sort_values(by = 'YearsCodingProf').reset_index()['Salary'],
            'name': devtypes, 'mode': 'line',
        } for devtypes in jobtitle
    ],
    'layout': {
        'title' : 'DevTypes V.S. years (developers coded professionally) with Median Salary ($)',
        'xaxis': {'title': 'Years experience (Developers coded professionally)'},
        'yaxis': {'title': "Median Salary ($)"}
    }
}
py.iplot(fig)
df = stack_data.set_index(['YearsCodingProf','ConvertedSalary']).Gender.str.split(';', expand=True).stack().reset_index(['YearsCodingProf','ConvertedSalary'])
df.columns = ['YearsCodingProf','Salary','gender']
df['YearsCodingProf'] = df['YearsCodingProf'].astype('category')
df['YearsCodingProf'].cat.reorder_categories(['0-2 years','3-5 years','6-8 years', '9-11 years', '12-14 years', '15-17 years', '18-20 years', 
                                            '21-23 years',  '24-26 years','27-29 years', '30 or more years'], inplace=True)
gndr = ['Male', 'Female', 'Transgender','Non-binary, genderqueer, or gender non-conforming']
fig = {
    'data': [
        {
            'x': df[df['gender']==gr].groupby('YearsCodingProf').agg({'Salary' : 'median'}).sort_values(by = 'YearsCodingProf').reset_index()['YearsCodingProf'],
            'y': df[df['gender']==gr].groupby('YearsCodingProf').agg({'Salary' : 'median'}).sort_values(by = 'YearsCodingProf').reset_index()['Salary'],
            'name': gr, 'mode': 'line',
        } for gr in gndr
    ],
    'layout': {
        'title' : 'Gender V.S. years experience (developers coded professionally) with Median Salary ($)',
        'xaxis': {'title': 'Years experience (Developers coded professionally)'},
        'yaxis': {'title': "Median Salary ($)"}
    }
}
py.iplot(fig)

df = stack_data.set_index(['YearsCodingProf','ConvertedSalary']).LanguageWorkedWith.str.split(';', expand=True).stack().reset_index(['YearsCodingProf','ConvertedSalary'])
df.columns = ['YearsCodingProf','Salary','language']
df['YearsCodingProf'] = df['YearsCodingProf'].astype('category')
df['YearsCodingProf'].cat.reorder_categories(['0-2 years','3-5 years','6-8 years', '9-11 years', '12-14 years', '15-17 years', '18-20 years', 
                                            '21-23 years',  '24-26 years','27-29 years', '30 or more years'], inplace=True)
gndr = ['Python','R','JavaScript','SQL','C++','C','Go','Scala','HTML','CSS','Bash/Shell','C#','PHP','Ruby','Swift','Matlab','TypeScript','Assembly','Objective-C','VB.NET']
fig = {
    'data': [
        {
            'x': df[df['language']==gr].groupby('YearsCodingProf').agg({'Salary' : 'median'}).sort_values(by = 'YearsCodingProf').reset_index()['YearsCodingProf'],
            'y': df[df['language']==gr].groupby('YearsCodingProf').agg({'Salary' : 'median'}).sort_values(by = 'YearsCodingProf').reset_index()['Salary'],
            'name': gr, 'mode': 'line',
        } for gr in gndr
    ],
    'layout': {
        'title' : 'Language V.S. years experience (developers coded professionally) with Median Salary ($)',
        'xaxis': {'title': 'Years experience (Developers coded professionally)'},
        'yaxis': {'title': "Median Salary ($)"}
    }
}
py.iplot(fig)
df = stack_data.set_index(['YearsCodingProf','ConvertedSalary']).FrameworkWorkedWith.str.split(';', expand=True).stack().reset_index(['YearsCodingProf','ConvertedSalary'])
df.columns = ['YearsCodingProf','Salary','Framework']
df['YearsCodingProf'] = df['YearsCodingProf'].astype('category')
df['YearsCodingProf'].cat.reorder_categories(['0-2 years','3-5 years','6-8 years', '9-11 years', '12-14 years', '15-17 years', '18-20 years', 
                                            '21-23 years',  '24-26 years','27-29 years', '30 or more years'], inplace=True)
gndr = ['Django', 'React', 'Angular', 'Node.js', 'Hadoop', 'Spark','Spring', '.NET Core', 'Cordova', 'Xamarin', 'TensorFlow','Torch/PyTorch']
fig = {
    'data': [
        {
            'x': df[df['Framework']==gr].groupby('YearsCodingProf').agg({'Salary' : 'median'}).sort_values(by = 'YearsCodingProf').reset_index()['YearsCodingProf'],
            'y': df[df['Framework']==gr].groupby('YearsCodingProf').agg({'Salary' : 'median'}).sort_values(by = 'YearsCodingProf').reset_index()['Salary'],
            'name': gr, 'mode': 'line',
        } for gr in gndr
    ],
    'layout': {
        'title' : 'Frameworks V.S. years experience (developers coded professionally) with Median Salary ($)',
        'xaxis': {'title': 'Years experience (Developers coded professionally)'},
        'yaxis': {'title': "Median Salary ($)"}
    }
}
py.iplot(fig)
df = stack_data.set_index(['YearsCodingProf','ConvertedSalary']).DatabaseWorkedWith.str.split(';', expand=True).stack().reset_index(['YearsCodingProf','ConvertedSalary'])
df.columns = ['YearsCodingProf','Salary','Database']
df['YearsCodingProf'] = df['YearsCodingProf'].astype('category')
df['YearsCodingProf'].cat.reorder_categories(['0-2 years','3-5 years','6-8 years', '9-11 years', '12-14 years', '15-17 years', '18-20 years', 
                                            '21-23 years',  '24-26 years','27-29 years', '30 or more years'], inplace=True)
gndr = ['Elasticsearch','MongoDB', 'SQL Server', 'MySQL', 'PostgreSQL', 'Amazon RDS/Aurora',
        'Amazon DynamoDB', 'Apache HBase', 'Apache Hive', 'Amazon Redshift',
       'Microsoft Azure (Tables, CosmosDB, SQL, etc)', 'Memcached',
       'Oracle', 'IBM Db2',  'Google Cloud Storage',
        'MariaDB', 'SQLite', 'Google BigQuery',
       'Cassandra', 'Neo4j','Redis']
fig = {
    'data': [
        {
            'x': df[df['Database']==gr].groupby('YearsCodingProf').agg({'Salary' : 'median'}).sort_values(by = 'YearsCodingProf').reset_index()['YearsCodingProf'],
            'y': df[df['Database']==gr].groupby('YearsCodingProf').agg({'Salary' : 'median'}).sort_values(by = 'YearsCodingProf').reset_index()['Salary'],
            'name': gr, 'mode': 'line',
        } for gr in gndr
    ],
    'layout': {
        'title' : 'Databases V.S. years experience (developers coded professionally) with Median Salary ($)',
        'xaxis': {'title': 'Years experience (Developers coded professionally)'},
        'yaxis': {'title': "Median Salary ($)"}
    }
}
py.iplot(fig)
temp = stack_data['AdBlocker'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='AdBlocker software installed or not ?', hole = 0.8, color = ['#00FFFF','#CDC0B0','#7FFFD4'])
temp = stack_data['AdBlockerDisable'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='In past month how many peoples disable theor AdBlocker ?', hole = 0.8, color = ['#FFD39B','#FF4040','#7FFF00'])
temp = pd.DataFrame(stack_data['AdBlockerReasons'].dropna().str.split(';').tolist()).stack()
temp =  temp.value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Top Reasons of disabling the AdBlocker', hole = 0.8)
temp = stack_data['AdsAgreeDisagree1'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Online advertising can be valuable when it is relevant to me ?', hole = 0.8, color = ['#228B22','#FFD39B','#FF4040','#545454'])
temp = stack_data['AdsAgreeDisagree2'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='I enjoy seeing online updates from companies that I like ?', hole = 0.8)
temp = stack_data['AdsAgreeDisagree3'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='I fundamentally dislike the concept of advertising ?', hole = 0.8, color = ['#7FFF00','#228B22','#FFD39B','#FF4040'])
print('ap1 sum =',stack_data['AdsPriorities1'].sum())
print('ap2 sum =',stack_data['AdsPriorities2'].sum())
print('ap3 sum =',stack_data['AdsPriorities3'].sum())
print('ap4 sum =',stack_data['AdsPriorities4'].sum())
print('ap5 sum =',stack_data['AdsPriorities5'].sum())
print('ap6 sum =',stack_data['AdsPriorities6'].sum())
print('ap7 sum =',stack_data['AdsPriorities7'].sum())
traces = []
newDiamond = stack_data.groupby(['WakeTime','JobSatisfaction']).size().unstack()
for c in newDiamond.columns:
    traces.append({
        'type' : 'bar',
        'x' : newDiamond.index,
        'y' : newDiamond[c],
        'name' : c
    })
fig = {
    'data' : traces,
    'layout' : {
        'title' :'Effect of Wake-up time on job satisfaction',
        #'barmode' : 'stack',
        'xaxis' : {
            'title' : 'Wake-up time'
        },        
    }
}
iplot(fig)
traces = []
newDiamond = stack_data.groupby(['WakeTime','CareerSatisfaction']).size().unstack()
for c in newDiamond.columns:
    traces.append({
        'type' : 'bar',
        'x' : newDiamond.index,
        'y' : newDiamond[c],
        'name' : c
    })
fig = {
    'data' : traces,
    'layout' : {
        'title' :'Effect of Wake-up time on career satisfaction',
#         'barmode' : 'stack',
        'xaxis' : {
            'title' : 'Wake-up time'
        },        
    }
}
iplot(fig)
traces = []
newDiamond = stack_data.groupby(['Exercise','JobSatisfaction']).size().unstack()
for c in newDiamond.columns:
    traces.append({
        'type' : 'bar',
        'x' : newDiamond.index,
        'y' : newDiamond[c],
        'name' : c
    })
fig = {
    'data' : traces,
    'layout' : {
        'title' : 'Effect of Excercise on job satisfaction',
        #'barmode' : 'stack',
        'xaxis' : {
            'title' : 'Number of time'
        },        
    }
}
iplot(fig)
traces = []
newDiamond = stack_data.groupby(['Exercise','CareerSatisfaction']).size().unstack()
for c in newDiamond.columns:
    traces.append({
        'type' : 'bar',
        'x' : newDiamond.index,
        'y' : newDiamond[c],
        'name' : c
    })
fig = {
    'data' : traces,
    'layout' : {
        'title' :'Effect of Excercise on career satisfaction',
        'xaxis' : {
            'title' : 'Number of times'
        },        
    }
}
iplot(fig)
traces = []
#print(stack_data['CompanySize'].unique())
def change_to_int(x):
    #print(x)
    x = x.split(" ")
    if x[0]=='Fewer':
        return '0 to 10 employees'
    #print(locale.atoi(x[0]))
    return str(int(x[0].replace(',', '')))+' '+' '.join(x[1:])
    
stack_data['CompanySize'] = stack_data['CompanySize'].dropna().apply(change_to_int)#map({'Fewer than 10 employees' : '0 to 10'})
newDiamond = stack_data.groupby(['CompanySize','JobSatisfaction']).size().unstack()
for c in newDiamond.columns:
    traces.append({
        'type' : 'scatter',
        'x' : newDiamond.index,
        'y' : newDiamond[c],
        'name' : c
    })
fig = {
    'data' : traces,
    'layout' : {
        'title' :'Does job satisfaction depends on  company or organization size ?',
        'xaxis' :dict(
        title='Company or organization size',
        tickfont=dict(
            size=10,
            color='rgb(107, 107, 107)'
        )
    ),     
    }
}
iplot(fig)
# locale
traces = []
#print(stack_data['CompanySize'].unique())
def change_to_int(x):
    #print(x)
    x = x.split(" ")
    if x[0]=='Fewer':
        return '0 to 10 employees'
    #print(locale.atoi(x[0]))
    return str(int(x[0].replace(',', '')))+' '+' '.join(x[1:])
    
stack_data['CompanySize'] = stack_data['CompanySize'].dropna().apply(change_to_int)#map({'Fewer than 10 employees' : '0 to 10'})
#print(temp.unique())
newDiamond = stack_data.groupby(['CompanySize','CareerSatisfaction']).size().unstack().sort_values(by ='CompanySize')
for c in newDiamond.columns:
    traces.append({
        'type' : 'scatter',
        'x' : newDiamond.index,
        'y' : newDiamond[c],
        'name' : c
    })
fig = {
    'data' : traces,
    'layout' : {
        'title' :'Does career satisfaction depends on  company or organization size ?',
        'xaxis' :dict(
        title='Company or organization size',
        tickfont=dict(
            size=10,
            color='rgb(107, 107, 107)'
        )
    ),        
    }
}
iplot(fig)
# locale
traces = []
#print(stack_data['CompanySize'].unique())
def change_to_int(x):
    #print(x)
    x = x.split(" ")
    if x[0]=='Under':
        return '0 - 18 years old'
    #print(locale.atoi(x[0]))
    return str(int(x[0].replace(',', '')))+' '+' '.join(x[1:])
    
stack_data['Age'] = stack_data['Age'].dropna().apply(change_to_int)#map({'Fewer than 10 employees' : '0 to 10'})
#print(temp.unique())
newDiamond = stack_data.groupby(['Age','CareerSatisfaction']).size().unstack().sort_values(by ='Age')
for c in newDiamond.columns:
    traces.append({
        'type' : 'scatter',
        'x' : newDiamond.index,
        'y' : newDiamond[c],
        'name' : c
    })
fig = {
    'data' : traces,
    'layout' : {
        'title' :'Does career satisfaction depends on Age ?',
        'xaxis' :dict(
        title='Age of the Developer',
        tickfont=dict(
            size=10,
            color='rgb(107, 107, 107)'
        )
    ),        
    }
}
iplot(fig)
traces = []
#print(stack_data['CompanySize'].unique())
def change_to_int(x):
    #print(x)
    x = x.split(" ")
    if x[0]=='Under':
        return '0 - 18 years old'
    #print(locale.atoi(x[0]))
    return str(int(x[0].replace(',', '')))+' '+' '.join(x[1:])
    
stack_data['Age'] = stack_data['Age'].dropna().apply(change_to_int)
newDiamond = stack_data.groupby(['Age','JobSatisfaction']).size().unstack().sort_values(by ='Age')
for c in newDiamond.columns:
    traces.append({
        'type' : 'scatter',
        'x' : newDiamond.index,
        'y' : newDiamond[c],
        'name' : c
    })
fig = {
    'data' : traces,
    'layout' : {
        'title' :'Does Job Satisfaction depends on Age ?',
        'xaxis' :dict(
        title='Age of the Developer',
        tickfont=dict(
            size=10,
            color='rgb(107, 107, 107)'
        )
    ),        
    }
}
iplot(fig)
temp1 = stack_data['AgreeDisagree1'].value_counts()
df1 = pd.DataFrame({'labels': temp1.index,'values': temp1.values})
df1.iplot(kind='pie',labels='labels',values='values', title='I feel a sense of kinship or connection to other developers', hole = 0.7, color = ['#FF4040','#FAEBD7','#7FFFD4','#7FFF00','#FFF8DC'])
temp1 = stack_data['AgreeDisagree2'].value_counts()
df1 = pd.DataFrame({'labels': temp1.index,'values': temp1.values})
df1.iplot(kind='pie',labels='labels',values='values', title='I think of myself as competing with my peers', hole = 0.7,color = ['#7FFF00','#FFF8DC','#FF4040','#FAEBD7','#7FFFD4', ])
temp1 = stack_data['AgreeDisagree3'].value_counts()
df1 = pd.DataFrame({'labels': temp1.index,'values': temp1.values})
df1.iplot(kind='pie',labels='labels',values='values', title='I\'m not as good at programming as most of my peers', hole = 0.7, )
cnt_srs = stack_data["TimeAfterBootcamp"].value_counts()
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=(cnt_srs/cnt_srs.sum() * 100)[::-1],
    orientation = 'h',
    marker=dict(
        color=['#FFF8DC','#68228B','#1E90FF','#FFC125','#8B7355','#BF3EFF','#FF6103','#FFC125'],
        colorscale = 'Blues',
        reversescale = True
    ),
)

layout = dict(
    title='Time to get a full-time job offer after doing developer training program or bootcamp (%)',
    margin=dict(
    l=500,
)
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
dm = stack_data[stack_data['DevType'] == 'Data scientist or machine learning specialist']
plt.figure(figsize = (16,6))

plt.scatter(np.sort(dm['ConvertedSalary'].values), range(dm.shape[0]))
plt.xlabel('Salary ($)', fontsize=15)
plt.title("Salary of Data Scientist / Machine Learning Specialists", fontsize=19)

plt.show()
temp1 = dm['OpenSource'].value_counts()
df1 = pd.DataFrame({'labels': temp1.index,'values': temp1.values})
df1.iplot(kind='pie',labels='labels',values='values', title='Open source contribution of Data Scientist / Machine Learning Specialists', hole = 0.7, color = ['#FF4040','#FAEBD7'])
temp = dm["Country"].value_counts().head(15)
temp.iplot(kind='bar', xTitle = 'Country name', yTitle = "Count", title = 'Top countries having highest number of respondents who are Data Scientist / Machine Learning Specialists')
temp = pd.DataFrame(dm['Gender'].dropna().str.split(';').tolist()).stack()
temp =  temp.value_counts().sort_values(ascending=False)
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Gender (Data Scientist / Machine Learning Specialists)', hole = 0.5, color = ['#FFF8DC','#8B7355','#BF3EFF','#FF6103','#FFC125','#0000FF'])

temp1 = pd.DataFrame(dm['LanguageWorkedWith'].dropna().str.split(';').tolist()).stack()
temp1 = temp1.value_counts().sort_values(ascending=False).head(15)
temp2 = pd.DataFrame(dm['LanguageDesireNextYear'].dropna().str.split(';').tolist()).stack()
temp2 = temp2.value_counts().sort_values(ascending=False).head(15)
trace1 = go.Bar(
    y=temp1.index[::-1],
    x=temp1.values[::-1],
    orientation = 'h',
    #name = ''
)
trace2 = go.Bar(
    y=temp2.index[::-1],
    x=temp2.values[::-1],
    orientation = 'h',
    #name = ''
)

fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('On which developers worked with', 'On which developers want to work in over the next year'))
                                                          

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig['layout'].update(height=500, width=1000, title='Most popular languages (Data Scientist / Machine Learning Specialists)', )
iplot(fig, filename='simple-subplot')
temp1 = pd.DataFrame(dm['DatabaseWorkedWith'].dropna().str.split(';').tolist()).stack()
temp1 = temp1.value_counts().sort_values(ascending=False).head(20)
temp2 = pd.DataFrame(dm['DatabaseDesireNextYear'].dropna().str.split(';').tolist()).stack()
temp2 = temp2.value_counts().sort_values(ascending=False).head(20)
trace1 = go.Bar(
    y=temp1.index[::-1],
    x=temp1.values[::-1],
    orientation = 'h',
    marker=dict(
        color=temp2.values[::-1],
        colorscale = 'red',
#         reversescale = True
    ),
    #name = ''
)
trace2 = go.Bar(
    y=temp2.index[::-1],
    x=temp2.values[::-1],
    orientation = 'h',
    #name = ''
)

fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('On which developers worked with', 'On which developers want to work in over the next year'))
                                                          

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
                          
fig['layout'].update(height=500, width=1100, title='Most popular Databases (Data Scientist / Machine Learning Specialists)', margin=dict(l=285,))
iplot(fig, filename='simple-subplot')
temp1 = pd.DataFrame(dm['PlatformWorkedWith'].dropna().str.split(';').tolist()).stack()
temp1 = temp1.value_counts().sort_values(ascending=False).head(20)
temp2 = pd.DataFrame(dm['PlatformDesireNextYear'].dropna().str.split(';').tolist()).stack()
temp2 = temp2.value_counts().sort_values(ascending=False).head(20)
trace1 = go.Bar(
    y=temp1.index[::-1],
    x=temp1.values[::-1],
    orientation = 'h',
    #name = ''
)
trace2 = go.Bar(
    y=temp2.index[::-1],
    x=temp2.values[::-1],
    orientation = 'h',
    #name = ''
)

fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('On which developers worked with', 'On which developers want to work in over the next year'))
                                                          

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
                          
fig['layout'].update(height=500, width=1100, title='Most popular platforms (Data Scientist / Machine Learning Specialists)', margin=dict(l=250,))
iplot(fig, filename='simple-subplot')
temp1 = pd.DataFrame(dm['FrameworkWorkedWith'].dropna().str.split(';').tolist()).stack()
temp1 = temp1.value_counts().sort_values(ascending=False).head(20)
temp2 = pd.DataFrame(dm['FrameworkDesireNextYear'].dropna().str.split(';').tolist()).stack()
temp2 = temp2.value_counts().sort_values(ascending=False).head(20)
trace1 = go.Bar(
    y=temp1.index[::-1],
    x=temp1.values[::-1],
    orientation = 'h',
    marker=dict(
        color=temp2.values[::-1],
        colorscale = 'red',
#         reversescale = True
    ),
    #name = ''
)
trace2 = go.Bar(
    y=temp2.index[::-1],
    x=temp2.values[::-1],
    orientation = 'h',
    #name = ''
)

fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('On which developers worked with', 'On which developers want to work in over the next year'))
                                                          

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
                          
fig['layout'].update(height=500, width=1100, title='Most popular Frameworks (Data Scientist / Machine Learning Specialists)', margin=dict(l=100,))
iplot(fig, filename='simple-subplot')
temp = pd.DataFrame(dm['IDE'].dropna().str.split(';').tolist()).stack()
temp =  temp.value_counts().sort_values(ascending=False)
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
    marker=dict(
        color=['#FFF8DC','#68228B','#1E90FF','#FFC125','#FF6103','#8EE5EE','#458B00','#FFF8DC','#8B7355','#BF3EFF','#FF6103','#FFC125'],
        colorscale = 'Blues',
        reversescale = True
    ),
)
data = [trace]
layout = go.Layout(
    title = "Top most used IDE  by Data Scientist / Machine Learning Specialists (%) ",
    xaxis=dict(
        title='IDE Name',
        tickfont=dict(
            size=10,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
df = dm.set_index(['YearsCodingProf','ConvertedSalary']).LanguageWorkedWith.str.split(';', expand=True).stack().reset_index(['YearsCodingProf','ConvertedSalary'])
df.columns = ['YearsCodingProf','Salary','language']
df['YearsCodingProf'] = df['YearsCodingProf'].astype('category')
df['YearsCodingProf'].cat.reorder_categories(['0-2 years','3-5 years','6-8 years', '9-11 years', '12-14 years', '15-17 years', '18-20 years', 
                                            '21-23 years',  '24-26 years','27-29 years', '30 or more years'], inplace=True)
gndr = ['Python','R','JavaScript','SQL','C++','Scala','Matlab','Julia']
fig = {
    'data': [
        {
            'x': df[df['language']==gr].groupby('YearsCodingProf').agg({'Salary' : 'median'}).sort_values(by = 'YearsCodingProf').reset_index()['YearsCodingProf'],
            'y': df[df['language']==gr].groupby('YearsCodingProf').agg({'Salary' : 'median'}).sort_values(by = 'YearsCodingProf').reset_index()['Salary'],
            'name': gr, 'mode': 'line',
        } for gr in gndr
    ],
    'layout': {
        'title' : 'Language V.S. years with Median Salary(Data Scientist/Machine Learning Specialists)',
        'xaxis': {'title': 'Years experience (Developers coded professionally)'},
        'yaxis': {'title': "Median Salary ($)"}
    }
}
py.iplot(fig)
df = dm.set_index(['YearsCodingProf','ConvertedSalary']).FrameworkWorkedWith.str.split(';', expand=True).stack().reset_index(['YearsCodingProf','ConvertedSalary'])
df.columns = ['YearsCodingProf','Salary','Framework']
df['YearsCodingProf'] = df['YearsCodingProf'].astype('category')
df['YearsCodingProf'].cat.reorder_categories(['0-2 years','3-5 years','6-8 years', '9-11 years', '12-14 years', '15-17 years', '18-20 years', 
                                            '21-23 years', '30 or more years'], inplace=True)
gndr = ['Hadoop', 'Spark', 'Spring', 'TensorFlow', 'Torch/PyTorch',
       'Django', 'Angular', 'Cordova', 'Node.js', 'React',
       'Xamarin']
fig = {
    'data': [
        {
            'x': df[df['Framework']==gr].groupby('YearsCodingProf').agg({'Salary' : 'median'}).sort_values(by = 'YearsCodingProf').reset_index()['YearsCodingProf'],
            'y': df[df['Framework']==gr].groupby('YearsCodingProf').agg({'Salary' : 'median'}).sort_values(by = 'YearsCodingProf').reset_index()['Salary'],
            'name': gr, 'mode': 'line',
        } for gr in gndr
    ],
    'layout': {
        'title' : 'Frameworks V.S. years experience with Median Salary(Data Scientist/Machine Learning Specialists)',
        'xaxis': {'title': 'Years experience (Developers coded professionally)'},
        'yaxis': {'title': "Median Salary ($)"}
    }
}
py.iplot(fig)
df = dm.set_index(['YearsCodingProf','ConvertedSalary']).DatabaseWorkedWith.str.split(';', expand=True).stack().reset_index(['YearsCodingProf','ConvertedSalary'])
df.columns = ['YearsCodingProf','Salary','Database']
df['YearsCodingProf'] = df['YearsCodingProf'].astype('category')
df['YearsCodingProf'].cat.reorder_categories(['0-2 years','3-5 years','6-8 years', '9-11 years', '12-14 years', '15-17 years', '18-20 years', 
                                            '21-23 years',  '24-26 years','27-29 years', '30 or more years'], inplace=True)
gndr = ['MySQL', 'PostgreSQL', 'Google BigQuery', 'SQL Server', 'SQLite',
       'Apache Hive', 'MongoDB', 'Oracle', 'Elasticsearch', 'Apache HBase',
       'Microsoft Azure (Tables, CosmosDB, SQL, etc)',
       'Google Cloud Storage', 'Cassandra', 'Redis', 'IBM Db2', 'Neo4j',
       'MariaDB', 'Amazon Redshift', 'Amazon DynamoDB',
       'Amazon RDS/Aurora', 'Memcached']
fig = {
    'data': [
        {
            'x': df[df['Database']==gr].groupby('YearsCodingProf').agg({'Salary' : 'median'}).sort_values(by = 'YearsCodingProf').reset_index()['YearsCodingProf'],
            'y': df[df['Database']==gr].groupby('YearsCodingProf').agg({'Salary' : 'median'}).sort_values(by = 'YearsCodingProf').reset_index()['Salary'],
            'name': gr, 'mode': 'line',
        } for gr in gndr
    ],
    'layout': {
        'title' : 'Database V.S. years experience with Median Salary(Data Scientist/Machine Learning Specialists)',
        'xaxis': {'title': 'Years experience (Developers coded professionally)'},
        'yaxis': {'title': "Median Salary ($)"}
    }
}
py.iplot(fig)
cnt_srs = stack_data["AIDangerous"].value_counts()
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=(cnt_srs/cnt_srs.sum() * 100)[::-1],
    orientation = 'h',
    marker=dict(
        color=['#8B7355','#BF3EFF','#FF6103','#FFC125'],
        colorscale = 'Blues',
        reversescale = True
    ),
)

layout = dict(
    title='Number of people thinking about dangerous aspects of increasingly advanced AI technology (%)',
    margin=dict(
    l=500,
)
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
cnt_srs = stack_data["AIInteresting"].value_counts()
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=(cnt_srs/cnt_srs.sum() * 100)[::-1],
    orientation = 'h',
    marker=dict(
        color=['#FFF8DC','#68228B','#1E90FF','#FFC125'],
        colorscale = 'Blues',
        reversescale = True
    ),
)

layout = dict(
    title='Number of people thinking about exciting aspects of increasingly advanced AI technology (%)',
    margin=dict(
    l=500,
)
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
cnt_srs = stack_data["AIResponsible"].value_counts()
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=(cnt_srs/cnt_srs.sum() * 100)[::-1],
    orientation = 'h',
    marker=dict(
        color=['#FF6103','#8EE5EE','#458B00','#FFF8DC'],
        colorscale = 'Blues',
        reversescale = True
    ),
)

layout = dict(
    title='# people responsible to consider the ramifications of increasingly advanced AI (%)',
    margin=dict(
    l=300,
)
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
cnt_srs = stack_data["AIFuture"].value_counts()
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=(cnt_srs/cnt_srs.sum() * 100)[::-1],
    orientation = 'h',
    marker=dict(
        color=['#FF6103','#8EE5EE','#458B00','#FFF8DC'],
        colorscale = 'Blues',
        reversescale = True
    ),
)

layout = dict(
    title='Number of People who think about the future of artificial intelligence (%)',
    margin=dict(
    l=500,
)
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)