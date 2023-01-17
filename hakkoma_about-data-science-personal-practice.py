#Import Pandas ,numpy and stats
import pandas as pd
import numpy as np
from scipy import stats

#Import pyplot and seaborn
import matplotlib.pyplot as plt
import seaborn as sns

#Render Matplotlib Plots Inline
%matplotlib inline

#Import Plotly and use it in the Offline Mode
import plotly
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.tools as tls
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as fig_fact
plotly.tools.set_config_file(world_readable=True, sharing='public')

#Ingnore warnings
import warnings
warnings.filterwarnings('ignore')
#Load MCQ data by Pd, encoding "ISO-8859-1"
mcq = pd.read_csv('../input/multipleChoiceResponses.csv', encoding="ISO-8859-1", low_memory=False)
mcq.shape
#Load Free Form Responses by Pd encoding "ISO-8859-1"
free = pd.read_csv('../input/freeformResponses.csv', encoding="ISO-8859-1", low_memory=False)
free.shape
#Seaborn countplot: 카운트 후 바로 렌더링을 하며 barplot 으로 나타낸다
sns.countplot(y='GenderSelect', data=mcq)
#Number of respondents by countries
con_df = pd.DataFrame(mcq['Country'].value_counts())
con_df['country'] = con_df.index
con_df.columns = ['num_resp', 'country']
con_df = con_df.reset_index().drop('index', axis=1)
con_df.head(10)
#Create a Choropleth Map: 다양한 형태의 지도 오픈소스 코드 (https://plot.ly/python/choropleth-maps/)
data = [ dict(
        type = 'choropleth',
        locations = con_df['country'],
        locationmode = 'country names',
        z = con_df['num_resp'],
        text = con_df['country'],
        colorscale = [[0,'rgb(255, 255, 255)'],[1,'rgb(201, 22, 22)']],
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Number of respondents'),
      ) ]

layout = dict(
    title = 'Number of survey respondents by countries',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='survey-world-map')
mcq['Age'].describe()
#Age distribution
fig = fig_fact.create_distplot([mcq[mcq['Age'] > 0]['Age']], ['age'], colors=['#4E40ED'])
py.iplot(fig, filename='Basic Distplot')
#sns.distplot(mcq[mcq['Age'] > 0]['Age']): 0 보다 큰값만. 
sns.countplot(y='FormalEducation', data=mcq)
plt.figure(figsize=(6,8))
sns.countplot(y='MajorSelect', data=mcq)
sns.countplot(y='EmploymentStatus', data=mcq)
#현재 전체 나라별 응답자 수
print(con_df)
sns.countplot(y='Tenure', data=mcq)
sns.countplot(y='LanguageRecommendationSelect', data=mcq)
#PR and Python users by job title
data = mcq[(mcq['CurrentJobTitleSelect'].notnull()) & ((mcq['LanguageRecommendationSelect'] == 'Python') | (mcq['LanguageRecommendationSelect'] == 'R'))]
plt.figure(figsize=(8, 10))
sns.countplot(y="CurrentJobTitleSelect", hue="LanguageRecommendationSelect", data=data)
#plot the 15 most popular ML Tools for next year
data = mcq['MLToolNextYearSelect'].value_counts().head(15)
sns.barplot(y=data.index, x=data)
data = mcq['MLMethodNextYearSelect'].value_counts().head(15)
sns.barplot(y=data.index, x=data)
#Explode the Pandas Dataframe to get the number of times each Learning Platform was mentioned
mcq['LearningPlatformSelect'] = mcq['LearningPlatformSelect'].astype('str').apply(lambda x: x.split(','))
s = mcq.apply(lambda x: pd.Series(x['LearningPlatformSelect']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'platform'
plt.figure(figsize=(6,8))
data = s[s != 'nan'].value_counts()
sns.barplot(y=data.index, x=data)
#How much is each platform useful?
use_features = [x for x in mcq.columns if x.find('LearningPlatformUsefulness') != -1]
#usefulness of each learning platform.
fdf = {}
for feature in use_features:
    a = mcq[feature].value_counts()
    a = a/a.sum()
    fdf[feature[len('LearningPlatformUsefulness'):]] = a

fdf = pd.DataFrame(fdf).transpose()#.sort_values('Very useful', ascending=False)

#Plot a Heatmap of Learning Platform Usefulness: 히트맵 사용function (https://seaborn.pydata.org/generated/seaborn.heatmap.html)
plt.figure(figsize=(6,12))
sns.heatmap(fdf.sort_values("Very useful", ascending=False), annot=True)

#Plot a grouped barplot of Learning Platform Usefulness
fdf.plot(kind='bar', figsize=(18,8), title="Usefullness of Learning Platforms")
plt.show()
plt.figure(figsize=(8,8))
sns.countplot(y='TimeSpentStudying', data=mcq, hue='EmploymentStatus').legend(loc='center left', bbox_to_anchor=(1, 0.5))
mcq['CoursePlatformSelect'] = mcq['CoursePlatformSelect'].astype('str').apply(lambda x: x.split(','))
t = mcq.apply(lambda x: pd.Series(x['CoursePlatformSelect']),axis=1).stack().reset_index(level=1, drop=True)
t.name = 'courses'
t = t[t != 'nan'].value_counts()
plt.title("Most Popular Course Platforms")
sns.barplot(y=t.index, x=t)
job_features = [x for x in mcq.columns if x.find('JobSkillImportance') != -1 and x.find('JobSkillImportanceOther') == -1]
#Skill Importance of Data Science Jobs
jdf = {}
for feature in job_features:
    a = mcq[feature].value_counts()
    a = a/a.sum()
    jdf[feature[len('JobSkillImportance'):]] = a
#fdf = pd.DataFrame(fdf)
#transpose(): 행렬화 시키기 function, 이걸로 index (a,a) 값으로 만들어서 전부다 보임.
jdf = pd.DataFrame(jdf).transpose()

jdf.plot(kind='bar', figsize=(12,6), title="Skill Importance in Data Science Jobs")
