import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
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
survey_2018=pd.read_csv('../input/stack-overflow-2018-developer-survey/survey_results_public.csv')
survey_2017=pd.read_csv('../input/so-survey-2017/survey_results_public.csv')
survey_2018.head()
survey_2017.head()
survey_2018['Gender']=survey_2018['Gender'].fillna('Others')
survey_2017['Gender']=survey_2017['Gender'].fillna('Others')
survey_2018.loc[(survey_2018["Gender"]!= 'Male') & (survey_2018["Gender"]!='Female'), "Gender"]='Others'
survey_2017.loc[(survey_2017["Gender"]!= 'Male') & (survey_2017["Gender"]!='Female'), "Gender"]='Others'
gender2018=survey_2018['Gender'].value_counts()
gender2017=survey_2017['Gender'].value_counts()
fig = {
  "data": [
    {
      "values": gender2018.values,
      "labels": gender2018.index,
      "domain": {"x": [0, .48]},
      "name": "Gender2018",
      "hoverinfo":"label+percent+name",
      "hole": .7,
      "type": "pie"
    },
    {
      "values": gender2017.values,
      "labels": gender2017.index,
      
      "domain": {"x": [.52, 1]},
      "name": "Gender2017",
      "hoverinfo":"label+percent+name",
      "hole": .7,
      "type": "pie"
    }],
  "layout": {
        "title":"Developer Participated 2017 and 2018",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Gender2018",
                "x": 0.20,
                "y": 0.5
            },
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Gender2017",
                "x": 0.8,
                "y": 0.5
            }
        ]
    }
}
py.iplot(fig, filename='donut')
t_2017=survey_2017['Professional'].value_counts()
t_2018=survey_2018['Student'].value_counts()
fig = {
  "data": [
    {
      "values": t_2017.values,
      "labels": t_2017.index,
      "domain": {"x": [0, .48]},
      "hoverinfo":"label+percent",
      "type": "pie"
    },
    {
      "values": t_2018.values,
      "labels": t_2018.index,
      
      "domain": {"x": [.52, 1]},
      
      "hoverinfo":"label+percent",
      
      "type": "pie"
    }],
  "layout": {
        "title":"Student Developer Participated 2017 and 2018",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "2017",
                "x": 0.20,
                "y": 0.5
            },
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "2018",
                "x": 0.8,
                "y": 0.5
            }
        ]
    }
}
py.iplot(fig, filename='styled_pie_chart')
# Country participation
t_2017=survey_2017['Country'].value_counts()
t_2018=survey_2018['Country'].value_counts()
# print(t_2017[0:10])
# print(t_2018)
trace1 = go.Bar(
    x=t_2017[0:10].index,
    y=t_2017[0:10].values,
    name='2017'
)
trace2 = go.Bar(
    x=t_2018[0:10].index,
    y=t_2018[0:10].values,
    name='2018'
)
fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Country Participation 2017', 'Country Participation 2018'))
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
                          
fig['layout'].update(height=500, width=1100, title='Most popular IDE', margin=dict(l=250,))
iplot(fig, filename='simple-subplot')
t_2017=survey_2017['CompanySize'].value_counts()
t_2018=survey_2018['CompanySize'].value_counts()
# Country participation
# t_2017=survey_2017['Country'].value_counts()
# t_2018=survey_2018['Country'].value_counts()
# print(t_2017[0:10])
# print(t_2018)
fig = {
  "data": [
    {
      "values": t_2017[0:8].values,
      "labels": t_2017[0:8].index,
      "domain": {"x": [0, .48]},
      "hoverinfo":"label+percent",
      "type": "pie"
    },
    {
      "values": t_2018[0:8].values,
      "labels": t_2018[0:8].index,
      
      "domain": {"x": [.52, 1]},
      
      "hoverinfo":"label+percent",
      
      "type": "pie"
    }],
  "layout": {
        "title":"Company Size",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "2017",
                "x": 0.20,
                "y": 0.5
            },
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "2018",
                "x": 0.8,
                "y": 0.5
            }
        ]
    }
}
py.iplot(fig, filename='styled_pie_chart')
# Maping rating to Words
# survey_2018.loc[(survey_2018["Gender"]!= 'Male') & (survey_2018["Gender"]!='Female'), "Gender"]='Others'
survey_2017.loc[(survey_2017["JobSatisfaction"]== 9) , "JobSatisfaction"]='Extremely satisfied'
survey_2017.loc[(survey_2017["JobSatisfaction"]== 10) , "JobSatisfaction"]='Extremely satisfied'
survey_2017.loc[(survey_2017["JobSatisfaction"]== 8) , "JobSatisfaction"]='Moderately satisfied'
survey_2017.loc[(survey_2017["JobSatisfaction"]== 7) , "JobSatisfaction"]='Moderately satisfied'
survey_2017.loc[(survey_2017["JobSatisfaction"]== 6) , "JobSatisfaction"]='Slightly satisfied'
survey_2017.loc[(survey_2017["JobSatisfaction"]== 5) , "JobSatisfaction"]='Neither satisfied nor dissatisfied'
survey_2017.loc[(survey_2017["JobSatisfaction"]== 4) , "JobSatisfaction"]='Slightly dissatisfied'
survey_2017.loc[(survey_2017["JobSatisfaction"]== 3) , "JobSatisfaction"]='Moderately dissatisfied'
survey_2017.loc[(survey_2017["JobSatisfaction"]== 2) , "JobSatisfaction"]='Moderately dissatisfied'
survey_2017.loc[(survey_2017["JobSatisfaction"]== 1) , "JobSatisfaction"]='Extremely dissatisfied'
survey_2017.loc[(survey_2017["JobSatisfaction"]== 0) , "JobSatisfaction"]='Extremely dissatisfied'

t_2017=survey_2017['JobSatisfaction'].value_counts()
t_2018=survey_2018['JobSatisfaction'].value_counts()

fig = {
  "data": [
    {
      "values": t_2017.values,
      "labels": t_2017.index,
      "domain": {"x": [0, .48]},
      "name": "Job Satisfaction",
      "hoverinfo":"label+percent",
      "hole": .7,
      "type": "pie"
    },
    {
      "values": t_2018.values,
      "labels": t_2018.index,
      
      "domain": {"x": [.52, 1]},
      "name": "Job Satisfaction",
      "hoverinfo":"label+percent",
      "hole": .7,
      "type": "pie"
    }],
  "layout": {
        "title":"Job Satisfaction 2017 and 2018",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "2017",
                "x": 0.20,
                "y": 0.5
            },
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "2018",
                "x": 0.8,
                "y": 0.5
            }
        ]
    }
}
py.iplot(fig, filename='donut')
survey_2017.loc[(survey_2017["CareerSatisfaction"]== 9) , "CareerSatisfaction"]='Extremely satisfied'
survey_2017.loc[(survey_2017["CareerSatisfaction"]== 10) , "CareerSatisfaction"]='Extremely satisfied'
survey_2017.loc[(survey_2017["CareerSatisfaction"]== 8) , "CareerSatisfaction"]='Moderately satisfied'
survey_2017.loc[(survey_2017["CareerSatisfaction"]== 7) , "CareerSatisfaction"]='Moderately satisfied'
survey_2017.loc[(survey_2017["CareerSatisfaction"]== 6) , "CareerSatisfaction"]='Slightly satisfied'
survey_2017.loc[(survey_2017["CareerSatisfaction"]== 5) , "CareerSatisfaction"]='Neither satisfied nor dissatisfied'
survey_2017.loc[(survey_2017["CareerSatisfaction"]== 4) , "CareerSatisfaction"]='Slightly dissatisfied'
survey_2017.loc[(survey_2017["CareerSatisfaction"]== 3) , "CareerSatisfaction"]='Moderately dissatisfied'
survey_2017.loc[(survey_2017["CareerSatisfaction"]== 2) , "CareerSatisfaction"]='Moderately dissatisfied'
survey_2017.loc[(survey_2017["CareerSatisfaction"]== 1) , "CareerSatisfaction"]='Extremely dissatisfied'
survey_2017.loc[(survey_2017["CareerSatisfaction"]== 0) , "CareerSatisfaction"]='Extremely dissatisfied'
t_2017=survey_2017['CareerSatisfaction'].value_counts()
t_2018=survey_2018['CareerSatisfaction'].value_counts()
fig = {
  "data": [
    {
      "values": t_2017.values,
      "labels": t_2017.index,
      "domain": {"x": [0, .48]},
      "name": " Satisfaction",
      "hoverinfo":"label+percent",
      "hole": .7,
      "type": "pie"
    },
    {
      "values": t_2018.values,
      "labels": t_2018.index,
      
      "domain": {"x": [.52, 1]},
      "name": "Job Satisfaction",
      "hoverinfo":"label+percent",
      "hole": .7,
      "type": "pie"
    }],
  "layout": {
        "title":"Carrer Satisfaction 2017 and 2018",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "2017",
                "x": 0.20,
                "y": 0.5
            },
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "2018",
                "x": 0.8,
                "y": 0.5
            }
        ]
    }
}
py.iplot(fig, filename='donut')
#IDE
temp = pd.DataFrame(survey_2017['IDE'].dropna().str.split(';').tolist()).stack()
t_2017 =  temp.value_counts()
temp_1 = pd.DataFrame(survey_2018['IDE'].dropna().str.split(';').tolist()).stack()
t_2018=temp_1.value_counts()
trace1 = go.Bar(
    x=t_2017[0:10].index,
    y=t_2017[0:10].values,
    name='2017'
)
trace2 = go.Bar(
    x=t_2018[0:10].index,
    y=t_2018[0:10].values,
    name='2018'
)

fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('IDE 2017', 'IDE 2018'))
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
                          
fig['layout'].update(height=500, width=1100, title='Most popular IDE', margin=dict(l=250,))
iplot(fig, filename='simple-subplot')
#have Worked Framework
temp = pd.DataFrame(survey_2017['HaveWorkedFramework'].dropna().str.split(';').tolist()).stack()
t_2017 =  temp.value_counts()
temp_1 = pd.DataFrame(survey_2018['FrameworkWorkedWith'].dropna().str.split(';').tolist()).stack()
t_2018=temp_1.value_counts()
trace1 = go.Bar(
    x=t_2017[0:10].index,
    y=t_2017[0:10].values,
    name='2017'
)
trace2 = go.Bar(
    x=t_2018[0:10].index,
    y=t_2018[0:10].values,
    name='2018'
)

fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Framework 2017', 'Framework 2018'))
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
                          
fig['layout'].update(height=500, width=1100, title='Most popular FrameWork', margin=dict(l=250,))
iplot(fig, filename='simple-subplot')


#Language Worked With
temp = pd.DataFrame(survey_2017['HaveWorkedLanguage'].dropna().str.split(';').tolist()).stack()
t_2017 =  temp.value_counts()
temp_1 = pd.DataFrame(survey_2018['LanguageWorkedWith'].dropna().str.split(';').tolist()).stack()
t_2018=temp_1.value_counts()
trace1 = go.Bar(
    x=t_2017[0:10].index,
    y=t_2017[0:10].values,
    name='2017'
)
trace2 = go.Bar(
    x=t_2018[0:10].index,
    y=t_2018[0:10].values,
    name='2018'
)

fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Language 2017', 'Language 2018'))
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
                          
fig['layout'].update(height=500, width=1100, title='Most popular Language', margin=dict(l=250,))
iplot(fig, filename='simple-subplot')


