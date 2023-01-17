import numpy as np 
import pandas as pd 
import time
import matplotlib
import matplotlib.pyplot as plt 

import seaborn as sns
color = sns.color_palette()

import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
from __future__ import division
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools

from mpl_toolkits.basemap import Basemap
from numpy import array
from matplotlib import cm
import cufflinks as cf
cf.go_offline()
from sklearn import preprocessing
import missingno as msno # to view missing values
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
from PIL import Image
import plotly.figure_factory as ff
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances
from scipy.stats import norm

import squarify
import warnings
warnings.filterwarnings('ignore')
import os

import os
print(os.listdir("../input"))
%%time
survey_results_schema = pd.read_csv('../input/survey_results_schema.csv')
survey_results_public = pd.read_csv('../input/survey_results_public.csv')
survey_results_schema.head(3)
survey_results_public.head(2)
print(survey_results_public.isnull().sum())
msno.matrix(survey_results_public)
plt.show()
# how many total missing values do we have?
missing_values_count = survey_results_public.isnull().sum()
total_cells = np.product(survey_results_public.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
print((total_missing/total_cells) * 100, '% of Missing Values in Survey Results Public')
# checking missing data in each survey results public column
total_missing = survey_results_public.isnull().sum().sort_values(ascending = False)
percentage = (survey_results_public.isnull().sum()/survey_results_public.isnull().count()*100).sort_values(ascending = False)
missing_survey_results_public = pd.concat([total_missing, percentage], axis=1, keys=['Total Missing (Column-wise)', 'Percentage (%)'])
missing_survey_results_public.head()
msno.dendrogram(survey_results_public)
plt.savefig('survey_results_public.png')
plt.show()
# Step 1
# chart stages data
temp = survey_results_public['Country'].value_counts().head(5).sort_values(ascending=False)
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
Image('a-simple-plot.png')
py.iplot(fig, filename='a-simple-plot')
temp = survey_results_public['AIDangerous'].value_counts().head(10)

df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values })
df.iplot(kind='pie',
         labels='labels',
         values='values', 
         title='Most Dangerous Aspect of Increasingly Advanced AI Technology?', 
         pull=.01,hole=.1,
         textposition='inside', 
         color = ['#B0CBE6', 'orange', 'blue', '#CCCCCC'],
         textinfo='percent')
temp = survey_results_public['AIInteresting'].value_counts().head(10)

df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values })
df.iplot(kind='pie',
         labels='labels',
         values='values', 
         title='Most Exciting Aspect of Increasingly Advanced AI Technology?', 
         pull=.02,hole=.75,
         textposition='inside', 
         color = ['#B0CBE6', 'orange', 'blue', '#CCCCCC'],
         textinfo='percent')
temp = survey_results_public['EthicsChoice'].value_counts().head(10)

df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values })
df.iplot(kind='pie',
         labels='labels',
         values='values', 
         title='Would you write unethical code?', 
         pull=.05,hole=.2,
         textposition='outside', 
         color = ['#B0CBE6', 'orange', '#CCCCCC'],
         textinfo='percent')
temp = survey_results_public['EthicsReport'].value_counts().head(10)
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values })
df.iplot(kind='pie',
         labels='labels',
         values='values', 
         title='Do you report or otherwise call out the unethical code in question?', 
         pull=.03,hole=.2,
         textposition='outside', 
         color = ['#B0CBE6', 'orange', 'blue', '#CCCCCC'],
         textinfo='percent')
temp = survey_results_public['EthicsResponsible'].value_counts().head(10)
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values })
df.iplot(kind='pie',
         labels='labels',
         values='values', 
         title='Do you report or otherwise call out the unethical code in question?', 
         pull=.03,hole=.1,
         textposition='inside', 
         color = ['#B0CBE6', 'orange', 'blue', '#CCCCCC'],
         textinfo='percent')
survey_results_public_language = survey_results_public['LanguageWorkedWith'].value_counts().head(10).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(19,9))
sns.barplot(survey_results_public_language.values, survey_results_public_language.index, ax=ax)
ax.set(xlabel= 'Count', 
       ylabel = 'Languages', 
       title = "Top Languages")
plt.show()
temp = survey_results_public['LanguageDesireNextYear'].value_counts().head(12).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(19,9))
sns.barplot(temp.values, temp.index, ax=ax)
ax.set(xlabel= 'Count', 
       ylabel = 'Language Desire Next Year', 
       title = "Top Language Desire Next Year")
plt.show()
temp = survey_results_public['DatabaseWorkedWith'].value_counts().head(10).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(19,9))
sns.barplot(temp.values, temp.index, ax=ax)
ax.set(xlabel= 'Count', 
       ylabel = 'Database Worked With', 
       title = "Top Database Worked With")
plt.show()
temp = survey_results_public['DatabaseDesireNextYear'].value_counts().head(10).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(19,9))
sns.barplot(temp.values, temp.index, ax=ax)
ax.set(xlabel= 'Count', 
       ylabel = 'Database Desire Next Year', 
       title = "Top Database Desire Next Year")
plt.show()
temp = survey_results_public['PlatformWorkedWith'].value_counts().head(10).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(19,9))
sns.barplot(temp.values, temp.index, ax=ax)
ax.set(xlabel= 'Count', 
       ylabel = 'Platform Worked With', 
       title = "Top Platform Worked With")
plt.show()
temp = survey_results_public['PlatformDesireNextYear'].value_counts().head(10).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(19,9))
sns.barplot(temp.values, temp.index, ax=ax)
ax.set(xlabel= 'Count', 
       ylabel = 'Platform Desire Next Year', 
       title = "Top Platform Desire Next Year")
plt.show()
temp = survey_results_public['FrameworkWorkedWith'].value_counts().head(10).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(19,9))
sns.barplot(temp.values, temp.index, ax=ax)
ax.set(xlabel= 'Count', 
       ylabel = 'Framework Worked With', 
       title = "Top Framework Worked With")
plt.show()
temp = survey_results_public['FrameworkDesireNextYear'].value_counts().head(10).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(19,9))
sns.barplot(temp.values, temp.index, ax=ax)
ax.set(xlabel= 'Count', 
       ylabel = 'Framework Desire Next Year', 
       title = "Top Framework Desire Next Year")
plt.show()
temp = survey_results_public['IDE'].value_counts().head(10).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(19,9))
sns.barplot(temp.values, temp.index, ax=ax)
ax.set(xlabel= 'Count', 
       ylabel = 'IDE', 
       title = "Top Top IDE Used by Developers")
plt.show()
temp = survey_results_public['OperatingSystem'].value_counts().head(10).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(19,9))
sns.barplot(temp.values, temp.index, ax=ax)
ax.set(xlabel= 'Count', 
       ylabel = 'Operating System', 
       title = "Top Operating Systems Used by Developers")
plt.show()
temp = survey_results_public['Methodology'].value_counts().head(10).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(19,9))
sns.barplot(temp.values, temp.index, ax=ax)
ax.set(xlabel= 'Count', 
       ylabel = 'Methodology', 
       title = "Top Methodologies Used by Developers")
plt.show()
temp = survey_results_public['OpenSource'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values })
df.iplot(kind='pie',
         labels='labels',
         values='values', 
         title='Open-source Project Contribution', 
         pull=.05,hole=.2,
         textposition='inside', 
         color = ['#CCCCCC', 'orange'],
         textinfo='percent+label')
temp = survey_results_public['Hobby'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values })
df.iplot(kind='pie',
         labels='labels',
         values='values', 
         title='Do you code as a hobby?', 
         pull=.001,
         hole=.7,
         textposition='outside', 
         color = ['blue', 'orange'],
         textinfo='percent+label')
temp = survey_results_public['Student'].value_counts().head(10)

df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values })
df.iplot(kind='pie',
         labels='labels',
         values='values', 
         title='Student engagement in a formal college or university?', 
         pull=.01,
         hole=.05,
         textposition='outside', 
         color = ['#CCCCCC', 'orange', 'blue'],
         textinfo='label+percent')
#survey_results_public['Salary'].info()
#temp = survey_results_public[['Country','Salary','SalaryType']]
#temp[temp['SalaryType'] == 'Yearly']
#survey_results_public['ConvertedSalary']
