import numpy as np

import pandas as pd

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns

import plotly

import plotly.offline as py

import plotly.tools as tls

import plotly.graph_objs as go

import plotly.tools as tls

plotly.tools.set_config_file(world_readable=True, sharing='public')

import warnings

warnings.filterwarnings('ignore')

import codecs

import base64

import wordcloud

from mpl_toolkits.basemap import Basemap

from os import path

import wordcloud

from PIL import Image

from plotly.offline import init_notebook_mode

import plotly.figure_factory as ff

init_notebook_mode(connected=True)

%matplotlib inline
data_response = pd.read_csv('../input/multipleChoiceResponses.csv', encoding="ISO-8859-1")

rates=pd.read_csv('../input/conversionRates.csv')

rates.drop('Unnamed: 0',axis=1,inplace=True)
set_dims = (12, 16)  

fig, axs = plt.subplots(ncols=2, figsize=set_dims)



df0 = {}

use_features = [x for x in data_response.columns 

                if x.find('WorkToolsFrequency') != -1]

for feature in use_features:

    a = data_response[feature].value_counts()

    a = a/a.sum()

    df0[feature[len('WorkToolsFrequency'):]] = a

df0 = pd.DataFrame(df0).transpose()

g0 = sns.heatmap(df0,linewidths=.2, cmap="YlGnBu", cbar=False, ax=axs[0])

g0.set_yticklabels(g0.get_yticklabels(), rotation = 0, fontsize = 12)

g0.set_xticklabels(g0.get_xticklabels(), rotation = 45, fontsize = 12)



df1 = {}

use_features = [x for x in data_response.columns 

                if x.find('WorkMethodsFrequency') != -1]

for feature in use_features:

    b = data_response[feature].value_counts()

    b = b/b.sum()

    df1[feature[len('WorkMethodsFrequency'):]] = b

df1 = pd.DataFrame(df1).transpose()

g1 = sns.heatmap(df1,linewidths=.2, cmap="YlGnBu", ax=axs[1])

g1.set_yticklabels(g1.get_yticklabels(), rotation = 0, fontsize = 12)

g1.set_xticklabels(g1.get_xticklabels(), rotation = 45, fontsize = 12)



plt.tight_layout()

plt.show()
df0 = {}

use_features = [x for x in data_response.columns if x.find('LearningPlatformUsefulness') != -1]

for feature in use_features:

    a = data_response[feature].value_counts()

    a = a/a.sum()

    df0[feature[len('LearningPlatformUsefulness'):]] = a



df0 = pd.DataFrame(df0).transpose()

yticks = df0.index

xticks = df0.columns



plt.figure(figsize=(15,5))

g = sns.heatmap(df0.transpose(),linewidths=.2, cmap="BuPu", 

            yticklabels=xticks, xticklabels=yticks)

g.set_yticklabels(g.get_yticklabels(), rotation = 0, fontsize = 11)

g.set_xticklabels(g.get_xticklabels(), rotation = 30, fontsize = 11)

plt.show()
set_dims = (12, 16)  

fig, axs = plt.subplots(ncols=2, figsize=set_dims)



df0 = {}

use_features = [x for x in data_response.columns 

                if x.find('WorkChallengeFrequency') != -1]

for feature in use_features:

    a = data_response[feature].value_counts()

    a = a/a.sum()

    df0[feature[len('WorkChallengeFrequency'):]] = a

df0 = pd.DataFrame(df0).transpose()

g0 = sns.heatmap(df0,linewidths=.2, cmap="YlGnBu", cbar=False, ax=axs[0])

g0.set_yticklabels(g0.get_yticklabels(), rotation = 0, fontsize = 12)

g0.set_xticklabels(g0.get_xticklabels(), rotation = 45, fontsize = 12)



df1 = {}

use_features = [x for x in data_response.columns 

                if x.find('JobFactor') != -1]

for feature in use_features:

    b = data_response[feature].value_counts()

    b = b/b.sum()

    df1[feature[len('JobFactor'):]] = b

df1 = pd.DataFrame(df1).transpose()

g1 = sns.heatmap(df1,linewidths=.2, cmap="YlGnBu", ax=axs[1])

g1.set_yticklabels(g1.get_yticklabels(), rotation = 0, fontsize = 12)

g1.set_xticklabels(g1.get_xticklabels(), rotation = 45, fontsize = 12)



plt.tight_layout()

plt.show()
df0 = {}

use_features = [x for x in data_response.columns if x.find('JobSkillImportance') != -1]

for feature in use_features:

    a = data_response[feature].value_counts()

    a = a/a.sum()

    df0[feature[len('JobSkillImportance'):]] = a



df0 = pd.DataFrame(df0).transpose()

yticks = df0.index

xticks = df0.columns



plt.figure(figsize=(15,5))

g = sns.heatmap(df0.transpose(),linewidths=.2, cmap="BuPu",

            yticklabels=xticks, xticklabels=yticks)

g.set_yticklabels(g.get_yticklabels(), rotation = 0, fontsize = 11)

g.set_xticklabels(g.get_xticklabels(), rotation = 30, fontsize = 11)

plt.show()
df = pd.read_csv('../input/multipleChoiceResponses.csv', encoding='ISO-8859-1', low_memory=False)

df = df[df['CompensationAmount'].notnull()]

# convert to str and replace the commas

df['CompensationAmountClean'] = df['CompensationAmount'].str.replace(',', '')

df = df.loc[~df['CompensationAmountClean'].str.contains('-')]

df['CompensationAmountClean'] = df['CompensationAmountClean'].astype(float)



# load the conversion rates

rates = pd.read_csv('../input/conversionRates.csv')

df_merged = pd.merge(left=df, right=rates, left_on='CompensationCurrency', 

                     right_on='originCountry')

df_merged['CompensationAmountUSD'] = df_merged['CompensationAmountClean'] * df_merged['exchangeRate']



# keep the columns we need

df_clean = df_merged[['CompensationAmountUSD', 'GenderSelect', 'MajorSelect', 

                      'Tenure', 'LanguageRecommendationSelect', 'EmployerIndustry', 

                      'EmployerSize', 'WorkToolsSelect',

                      'AlgorithmUnderstandingLevel', 'WorkMLTeamSeatSelect', 

                      'WorkInternalVsExternalTools','WorkDataVisualizations', 

                      'JobSatisfaction', 'JobHuntTime',

                     'CurrentJobTitleSelect', 'FormalEducation', 'Country']]



# keep only 9 countries with most respondents

keep_countries = ['United States', 'People \'s Republic of China', 'United Kingdom',

                  'Russia', 'India', 'Brazil', 'Germany', 'France', 'Canada']

df_clean = df_clean.loc[df_clean['Country'].isin(keep_countries)]
groups_edu = df_clean.groupby(['FormalEducation', 'Country'])

sorted_medians_edu = groups_edu['CompensationAmountUSD'].median().sort_values()

major_df_edu = pd.DataFrame(sorted_medians_edu).reset_index()



trace = go.Heatmap(z=major_df_edu['CompensationAmountUSD'],

                   x=major_df_edu['Country'],

                   y=major_df_edu['FormalEducation'],

                   colorscale=[[0.0, 'rgb(255,255,191)'], [0.2, 'rgb(254,224,139)'], 

                               [0.4, 'rgb(253,174,97)'], [0.6, 'rgb(244,109,67)'], 

                               [0.8, 'rgb(213,62,79)'], [1.0, 'rgb(158,1,66)']])



figure = dict(data=[trace], layout= dict( margin = dict(t=20,r=80,b=100,l=325)))

py.iplot(figure, filename='labelled-heatmap1')
groups_major = df_clean.groupby(['MajorSelect', 'Country'])

sorted_medians_major = groups_major['CompensationAmountUSD'].median().sort_values()

major_df_major = pd.DataFrame(sorted_medians_major).reset_index()



trace = go.Heatmap(z=major_df_major['CompensationAmountUSD'],

                   x=major_df_major['Country'],

                   y=major_df_major['MajorSelect'],

                   colorscale=[[0.0, 'rgb(255,255,191)'], [0.2, 'rgb(254,224,139)'], 

                               [0.4, 'rgb(253,174,97)'], [0.6, 'rgb(244,109,67)'], 

                               [0.8, 'rgb(213,62,79)'], [1.0, 'rgb(158,1,66)']])



figure = dict(data=[trace], layout= dict( margin = dict(t=20,r=80,b=100,l=240)))

py.iplot(figure, filename='labelled-heatmap2')
groups1 = df_clean.groupby(['EmployerIndustry', 'Country'])

sorted_medians1 = groups1['CompensationAmountUSD'].median().sort_values()

df_hm1 = pd.DataFrame(sorted_medians1).reset_index()



trace = go.Heatmap(z=df_hm1['CompensationAmountUSD'],

                   x=df_hm1['Country'],

                   y=df_hm1['EmployerIndustry'],

                   colorscale=[[0.0, 'rgb(255,255,191)'], [0.2, 'rgb(254,224,139)'], 

                               [0.4, 'rgb(253,174,97)'], [0.6, 'rgb(244,109,67)'], 

                               [0.8, 'rgb(213,62,79)'],[1.0, 'rgb(158,1,66)']])



figure = dict(data=[trace], layout= dict( margin = dict(t=20,r=80,b=100,l=240)))

py.iplot(figure, filename='labelled-heatmap3')
groups1 = df_clean.groupby(['EmployerSize', 'Country'])

sorted_medians1 = groups1['CompensationAmountUSD'].median().sort_values()

df_hm1 = pd.DataFrame(sorted_medians1).reset_index()



trace = go.Heatmap(z=df_hm1['CompensationAmountUSD'],

                   x=df_hm1['Country'],

                   y=df_hm1['EmployerSize'],

                   colorscale=[[0.0, 'rgb(255,255,191)'], [0.2, 'rgb(254,224,139)'], 

                               [0.4, 'rgb(253,174,97)'], [0.6, 'rgb(244,109,67)'], 

                               [0.8, 'rgb(213,62,79)'],[1.0, 'rgb(158,1,66)']])



figure = dict(data=[trace], layout= dict( margin = dict(t=20,r=80,b=100,l=240)))

py.iplot(figure, filename='labelled-heatmap4')
groups_tenure = df_clean.groupby(['Tenure', 'Country'])

sorted_medians_tenure = groups_tenure['CompensationAmountUSD'].median().sort_values()

major_df_tenure = pd.DataFrame(sorted_medians_tenure).reset_index()



trace = go.Heatmap(z=major_df_tenure['CompensationAmountUSD'],

                   x=major_df_tenure['Country'],

                   y=major_df_tenure['Tenure'],

                   colorscale=[[0.0, 'rgb(255,255,191)'], [0.2, 'rgb(254,224,139)'], 

                               [0.4, 'rgb(253,174,97)'], [0.6, 'rgb(244,109,67)'], 

                               [0.8, 'rgb(213,62,79)'],[1.0, 'rgb(158,1,66)']])



figure = dict(data=[trace], layout= dict( margin = dict(t=20,r=80,b=100,l=240)))

py.iplot(figure, filename='labelled-heatmap5')
groups1 = df_clean.groupby(['CurrentJobTitleSelect', 'Country'])

sorted_medians1 = groups1['CompensationAmountUSD'].median().sort_values()

df_hm1 = pd.DataFrame(sorted_medians1).reset_index()



trace = go.Heatmap(z=df_hm1['CompensationAmountUSD'],

                   x=df_hm1['Country'],

                   y=df_hm1['CurrentJobTitleSelect'],

                   colorscale=[[0.0, 'rgb(255,255,191)'], [0.2, 'rgb(254,224,139)'], 

                               [0.4, 'rgb(253,174,97)'], [0.6, 'rgb(244,109,67)'], 

                               [0.8, 'rgb(213,62,79)'],[1.0, 'rgb(158,1,66)']])



figure = dict(data=[trace], layout= dict( margin = dict(t=20,r=80,b=100,l=250)))

py.iplot(figure, filename='labelled-heatmap6')
groups1 = df_clean.groupby(['JobSatisfaction', 'Country'])

sorted_medians1 = groups1['CompensationAmountUSD'].median().sort_values()

df_hm1 = pd.DataFrame(sorted_medians1).reset_index()



trace = go.Heatmap(z=df_hm1['CompensationAmountUSD'],

                   x=df_hm1['Country'],

                   y=df_hm1['JobSatisfaction'],

                   colorscale=[[0.0, 'rgb(255,255,191)'], [0.2, 'rgb(254,224,139)'], 

                               [0.4, 'rgb(253,174,97)'], [0.6, 'rgb(244,109,67)'], 

                               [0.8, 'rgb(213,62,79)'],[1.0, 'rgb(158,1,66)']])



figure = dict(data=[trace], layout= dict( margin = dict(t=20,r=80,b=100,l=150)))

py.iplot(figure, filename='labelled-heatmap7')
groups1 = df_clean.groupby(['LanguageRecommendationSelect', 'Country'])

sorted_medians1 = groups1['CompensationAmountUSD'].median().sort_values()

df_hm1 = pd.DataFrame(sorted_medians1).reset_index()



trace = go.Heatmap(z=df_hm1['CompensationAmountUSD'],

                   x=df_hm1['Country'],

                   y=df_hm1['LanguageRecommendationSelect'][df_hm1['LanguageRecommendationSelect'] != 'Julia'],

                   colorscale=[[0.0, 'rgb(255,255,191)'], [0.2, 'rgb(254,224,139)'], 

                               [0.4, 'rgb(253,174,97)'], [0.6, 'rgb(244,109,67)'], 

                               [0.8, 'rgb(213,62,79)'],[1.0, 'rgb(158,1,66)']])



figure = dict(data=[trace], layout= dict( margin = dict(t=20,r=80,b=100,l=240)))

py.iplot(figure, filename='labelled-heatmap8')
groups1 = df_clean.groupby(['AlgorithmUnderstandingLevel', 'Country'])

sorted_medians1 = groups1['CompensationAmountUSD'].median().sort_values()

df_hm1 = pd.DataFrame(sorted_medians1).reset_index()



trace = go.Heatmap(z=df_hm1['CompensationAmountUSD'],

                   x=df_hm1['Country'],

                   y=df_hm1['AlgorithmUnderstandingLevel'],

                   colorscale=[[0.0, 'rgb(255,255,191)'], [0.2, 'rgb(254,224,139)'], 

                               [0.4, 'rgb(253,174,97)'], [0.6, 'rgb(244,109,67)'], 

                               [0.8, 'rgb(213,62,79)'],[1.0, 'rgb(158,1,66)']])



figure = dict(data=[trace], layout= dict( margin = dict(t=20,r=80,b=100,l=440)))

py.iplot(figure, filename='labelled-heatmap9')
data_freeform = pd.read_csv("../input/freeformResponses.csv", low_memory=False)

text = data_freeform[pd.notnull(data_freeform["KaggleMotivationFreeForm"]

                               )]["KaggleMotivationFreeForm"]

cloud = wordcloud.WordCloud(height=600, width=800, relative_scaling=0.2, 

                            random_state=74364).generate(" ".join(text))

plt.figure(figsize=(14,14))

plt.imshow(cloud, interpolation='bilinear')

plt.axis("off");