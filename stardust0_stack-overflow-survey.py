# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud,STOPWORDS
import squarify


import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
from __future__ import division
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import missingno as msno
from mpl_toolkits.basemap import Basemap
from numpy import array
from matplotlib import cm
import cufflinks as cf
cf.go_offline()
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
schema = pd.read_csv("../input/survey_results_schema.csv")
public = pd.read_csv('../input/survey_results_public.csv')
pd.options.display.max_colwidth = 350
schema[:10]

public.head()
public.shape
msno.dendrogram(public)
plt.show()
null_values = public.isnull().sum().sort_values(ascending = False)
percentage = (public.isnull().sum().sort_values(ascending = False)/public.shape[0])*100
missing = pd.concat([null_values,percentage],axis = 1,keys = ['null_values','percentages'])
missing.head(10)
fig = plt.figure(figsize = (25,10))
sns.set_context("poster")
ax = sns.barplot(missing.index,missing['percentages'], palette="Blues_d")

plt.xticks(rotation = 90,fontsize=8)
plt.show()
temp = public['Country'].value_counts().head(5).sort_values(ascending=False)
values = temp.values
phases = temp.index
#values = [13873, 10553, 5443, 3703, 1708]
#phases = ['Visit', 'Sign-up', 'Selection', 'Purchase', 'Review']

# color of each funnel section
colors = ['rgb(32,155,160)', 'rgb(253,93,124)', 'rgb(28,119,139)', 'rgb(182,231,235)', 'rgb(35,154,160)']

# Shaping
n_phase = len(phases)
plot_width = 400

# height of a section and difference between sections 
section_h = 100
section_d = 10

# multiplication factor to calculate the width of other sections
unit_width = plot_width / max(values)

# width of each funnel section relative to the plot width
phase_w = [int(value * unit_width) for value in values]

# plot height based on the number of sections and the gap in between them
height = section_h * n_phase + section_d * (n_phase - 1)

# Step 3
# list containing all the plot shapes
shapes = []

# list containing the Y-axis location for each section's name and value text
label_y = []

for i in range(n_phase):
        if (i == n_phase-1):
                points = [phase_w[i] / 2, height, phase_w[i] / 2, height - section_h]
        else:
                points = [phase_w[i] / 2, height, phase_w[i+1] / 2, height - section_h]

        path = 'M {0} {1} L {2} {3} L -{2} {3} L -{0} {1} Z'.format(*points)

        shape = {
                'type': 'path',
                'path': path,
                'fillcolor': colors[i],
                'line': {
                    'width': 1,
                    'color': colors[i]
                }
        }
        shapes.append(shape)
        
        # Y-axis location for this section's details (text)
        label_y.append(height - (section_h / 2))

        height = height - (section_h + section_d)

# For phase names
label_trace = go.Scatter(
    x=[-350]*n_phase,
    y=label_y,
    mode='text',
    text=phases,
    textfont=dict(
        color='rgb(200,200,200)',
        size=15
    )
)
 
# For phase values
value_trace = go.Scatter(
    x=[350]*n_phase,
    y=label_y,
    mode='text',
    text=values,
    textfont=dict(
        color='rgb(200,200,200)',
        size=15
    )
)

data = [label_trace, value_trace]
 
layout = go.Layout(
    title="<b>Top Countries on Stack Overflow</b>",
    titlefont=dict(
        size=20,
        color='rgb(203,203,203)'
    ),
    shapes=shapes,
    height=560,
    width=800,
    showlegend=False,
    paper_bgcolor='rgba(44,58,71,1)',
    plot_bgcolor='rgba(44,58,71,1)',
    xaxis=dict(
        showticklabels=False,
        zeroline=False,
    ),
    yaxis=dict(
        showticklabels=False,
        zeroline=False
    )
)

fig = go.Figure(data=data, layout=layout)
image='png' 
from IPython.display import Image
Image('funnel_chart.png')
py.iplot(fig, filename='funnel_chart')

tree = public['Country'].value_counts().to_frame()
squarify.plot(sizes = tree['Country'].values[:50],label = tree.index[:50])
plt.rcParams.update({'font.size':6})
fig = plt.gcf()
fig.set_size_inches(45,15)
plt.savefig('area.png')
plt.show()
fig = plt.figure(figsize = (20,10))
sns.countplot(y = public['Gender'],order = public['Gender'].value_counts().index)
plt.show()
race = public['RaceEthnicity'].value_counts()
race = pd.DataFrame({'race':race.index,'percent':(race.values/sum(race.values))*100})
fig = plt.figure()
sns.barplot(race['percent'][:10],race['race'][:10])
plt.rcParams.update({'font.size':20})
cf = plt.gcf()
cf.set_size_inches(15,10)
plt.show()
student = public['Student'].value_counts()
student = pd.DataFrame({'type':student.index,'percent':(student.values)*100/sum(student.values)})
fig = plt.figure()
sns.barplot(student['percent'],student['type'])
plt.show()
edu = public['FormalEducation'].value_counts()
edu = pd.DataFrame({'type':edu.index,'percent':(edu.values)*100/sum(edu.values)})
fig = plt.figure()
sns.barplot(edu['percent'],edu['type'])
plt.show()
data = public[['Employment','FormalEducation']].groupby(['Employment'])
data.groups
full_time = data.get_group('Employed full-time')
fig = plt.figure()
sns.barplot(full_time['FormalEducation'].value_counts().values/sum(full_time['FormalEducation'].value_counts().values),
            full_time['FormalEducation'].value_counts().index)
plt.show()
full_time['FormalEducation'].value_counts()
not_emp = data.get_group('Not employed, but looking for work')
fig = plt.figure()
sns.barplot(not_emp['FormalEducation'].value_counts().values/sum(not_emp['FormalEducation'].value_counts().values),
            not_emp['FormalEducation'].value_counts().index)
plt.show()
not_emp['FormalEducation'].value_counts()
job = []
dev = public['DevType'].dropna()
for i in dev.index:
    job.extend([s for s in dev[i].split(';')]) 
from collections import Counter
a = dict(Counter(job))
job_data = pd.DataFrame(list(a.items()),columns = ['Job','count'])
job_data.sort_values(by = ['count'] , ascending= False,inplace = True)

fig = plt.figure()
sns.barplot(y = job_data['Job'],x= job_data['count'])
f = plt.gcf()
f.set_size_inches((20,15))
plt.show()
opensource = public.copy()
opensource = opensource.groupby(['OpenSource'])['Country'].value_counts()
fig = plt.figure()
ax = sns.barplot(x = opensource[opensource.index.levels[0][1]].values[:10],y = opensource[opensource.index.levels[0][1]].index[:10] )
ax.set(xlabel='Number of People who opensource')
plt.show()
stu = public.copy()
stu =  stu.groupby(['Student'])['Country'].value_counts()
fig = plt.figure()
ax = sns.barplot(x = stu[stu.index.levels[0][1]].values[:10],y = stu[stu.index.levels[0][1]].index[:10] )
ax.set(xlabel='Number of students')
plt.show()
cod = public.copy()
cod = cod.groupby(['OpenSource'])['YearsCoding'].value_counts()

fig = plt.figure()
ax = sns.barplot(x = cod[cod.index.levels[0][1]].values,y = cod[cod.index.levels[0][1]].index)
ax.set(xlabel = 'Number of people who opensource')
plt.show()
fig = plt.figure()
total = cod[cod.index.levels[0][1]].values + cod[cod.index.levels[0][0]].values
data = pd.DataFrame()
data['experience'] = cod[cod.index.levels[0][1]].index
data['ratio'] = (cod[cod.index.levels[0][1]].values/total)
data.sort_values(by = 'ratio',ascending = False,inplace = True)
ax = sns.barplot(x = data['ratio']*100,y =data['experience'] )
ax.set(xlabel = '%age of people who opensource')
plt.show()
fig,ax = plt.subplots(1,1,figsize=(9,9))
ax = public['Hobby'].value_counts().plot.pie(autopct = '%1.2f%%',shadow = True,explode = [0,0.08])
ax.set_ylabel(' ')
plt.title("Coding as a hobby")
plt.show()
country = public['Country'].value_counts().reset_index()
country_ = country['index'].str.replace(" ","")
cloud = WordCloud(scale = 6).generate(" ".join(country_))
plt.figure(figsize=(14,10))
plt.imshow(cloud,interpolation="bilinear")
plt.axis('off')
plt.savefig('cloud.png')
plt.show()
fig = plt.figure()

squarify.plot(sizes=public["CompanySize"].value_counts().values,label=public["CompanySize"].value_counts().keys(),color=sns.color_palette("muted"))
fig = plt.gcf()
plt.axis('off')
fig.set_size_inches(45,15)
plt.title("Company size of respondents",size = 30)
plt.show()

data1 = public['LanguageWorkedWith'].str.split(';',expand = True).stack().reset_index()[0].value_counts().reset_index()
data1['type'] = 'languageworkedwith'
data2 = public['LanguageDesireNextYear'].str.split(';',expand = True).stack().reset_index()[0].value_counts().reset_index()
data2['type']  = 'languagedesirenextyear'
data = pd.concat([data1,data2],axis = 0)

data.rename(columns = {0:'counts'},inplace = True)
sns.pointplot(y='index', x= 'counts', data =data,hue = 'type',join=True, markers=["o", "x"],palette="deep")
plt.grid(True,alpha=1)
g = plt.gcf()
g.set_size_inches(15,15)
exer = public.groupby(['Exercise'])['Gender'].value_counts().unstack(level = 0)
exer = exer.loc[['Female','Male']]
exer.loc['Female'] = exer.loc['Female']*100/4025
exer.loc['Male'] = exer.loc['Male']*100/59458
exer.reset_index()
exer = exer.stack().to_frame()
exer.rename(columns = {0:'percentage'},inplace = True)
exer
exer1 = pd.DataFrame()
exer1['exercise'] = list(exer.index.levels[1])*2
exer1['percent'] = exer.percentage.values
exer1['gender'] = (['male']*8)
exer1['gender'][4:] = ['female']*4
exer1.sort_values(by = 'percent', ascending = False,inplace = True)

ax = sns.barplot(x = exer1.exercise,y = exer1.percent,hue = exer1.gender)
ax.set(ylabel = 'percentage')
plt.title('How many times do you exercise ?',size= 20)
f = plt.gcf()
f.set_size_inches(20,9)
tools = public['CommunicationTools'].str.split(';',expand = True).stack().reset_index()[0].value_counts().reset_index()
tools
fig = plt.figure()
ax = sns.barplot(y = tools['index'],x = tools[0])
ax.set(xlabel = 'Number of users',ylabel = 'Communication Tools')
f = plt.gcf()
f.set_size_inches(20,15)
plt.show()
types = public['SelfTaughtTypes'].str.split(';',expand = True).stack().reset_index()[0].value_counts().reset_index()
types
fig = plt.figure()
ax = sns.barplot(x = types[0],y=types['index'])
f = plt.gcf()
f.set_size_inches(20,15)
ax.set(xlabel = 'Number of users',ylabel = 'Teaching method')
plt.show()
fig = plt.figure()
ax = public['StackOverflowHasAccount'].dropna().value_counts().plot.pie(autopct = '%1.1f%%',shadow = True,explode = [0.05,0.05,0.05])
ax.set(ylabel = ' ')
f = plt.gcf()
f.set_size_inches(10,10)
plt.title('People with stackoverflow account',size = 25)
plt.show()
fig = plt.figure()
ax = public['StackOverflowVisit'].value_counts().plot.pie(autopct = '%1.1f%%',shadow = True,explode = [0.05,0.0,0.0,0.0,0,0])
f =  plt.gcf()
f.set_size_inches(10,10)
plt.title('Stackoverflow visit',size = 25)
ax.set(ylabel = ' ')
plt.show()
fig = plt.figure()
ax = public['StackOverflowParticipate'].value_counts().plot.pie(autopct = '%1.1f%%',shadow = True)
f =  plt.gcf()
f.set_size_inches(10,10)
plt.title('Stackoverflow Participation',size = 25)
ax.set(ylabel = ' ')
plt.show()