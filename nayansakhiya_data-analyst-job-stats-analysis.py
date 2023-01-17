# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import seaborn as sns

import matplotlib.pyplot as plt 

from collections import Counter



# plotly

# import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import cufflinks as cf

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)

from wordcloud import WordCloud,STOPWORDS



import warnings            

warnings.filterwarnings("ignore") 

plt.style.use('ggplot') # style of plots. ggplot is one of the most us
analyst_data = pd.read_csv('/kaggle/input/data-analyst-jobs/DataAnalyst.csv')

analyst_data.head()
analyst_data = analyst_data.drop(columns=['Unnamed: 0'])
analyst_data['Salary Estimate'].replace(['-1'],[np.nan],inplace=True)

analyst_data['Salary Estimate'].fillna('$36K-$56K (Glassdoor est.)',inplace=True)

analyst_data.isnull().sum()
# drop null values



analyst_data.dropna(inplace=True)

analyst_data.reset_index(inplace=True)
# split the company name and rating



for i in range(analyst_data.shape[0]):

    name = analyst_data.loc[i,"Company Name"]

    if "\n" in name:

        name,_ = name.split("\n")

    analyst_data.loc[i,"Company Name"] = name
company_analysis = pd.DataFrame(analyst_data['Company Name'].value_counts().sort_values(ascending=False))
# top 15 companies with highest no. of jobs





trace = go.Bar(x=company_analysis.index[:15],

               y=company_analysis['Company Name'][:15],

               marker = dict(color = 'rgba(255, 155, 128, 0.5)',

               line=dict(color='rgb(0,0,0)',width=1.5)))

layout = go.Layout(title='Top 15 companies with highest no. of jobs', xaxis=dict(title='Company Name',zeroline= False,

                                                        gridcolor='rgb(183,183,183)',

                                                        showline=True),

                                                    yaxis=dict(title='Job Counts',zeroline= False,

                                                        gridcolor='rgb(183,183,183)',

                                                        showline=True),

                  font=dict(family='Courier New, monospace', size=12, color='rgb(0,0,0)')

                  )

data = [trace]

fig = go.Figure(data = data, layout = layout)

iplot(fig)
analyst_data['Salary Estimate'],_=analyst_data['Salary Estimate'].str.split('(', 1).str
# split the salary into two columns min and max salary



analyst_data['Min_Salary'],analyst_data['Max_Salary'] = analyst_data['Salary Estimate'].str.split('-').str
analyst_data['Min_Salary'] = analyst_data['Min_Salary'].str.strip(' ').str.lstrip('$').str.rstrip('K').astype('int')

analyst_data['Max_Salary']= analyst_data['Max_Salary'].str.strip(' ').str.lstrip('$').str.rstrip('K').astype('int')
import plotly.figure_factory as ff
data_analyst = analyst_data[analyst_data['Job Title']=='Data Analyst']
hist_data = [data_analyst['Min_Salary'], data_analyst['Max_Salary']]

group_labels = ['Min_Salary', 'Max_Salary']

fig = ff.create_distplot(hist_data, group_labels)

# Add title

fig.update_layout(title_text='Distplot with Normal Distribution of salary')
# top 15 location with max and min salary

# best 15 job location with high salary



city_analysis = analyst_data.groupby('Location')[['Max_Salary','Min_Salary']].mean().sort_values(['Max_Salary','Min_Salary'],ascending=False)



trace1 = go.Bar(x=city_analysis.index[:15],

                y=city_analysis['Min_Salary'][:15],

                name='Minimum Salary',

                marker = dict(color = 'rgba(125, 215, 180, 0.5)',line=dict(color='rgb(0,0,0)',width=1.5)))



trace2 = go.Bar(x=city_analysis.index[:15],

                y=city_analysis['Max_Salary'][:15],

                name='Maximum Salary',

                marker = dict(color = 'rgba(115, 155, 214, 0.5)', line=dict(color='rgb(0,0,0)',width=1.5)))





layout = go.Layout(barmode='group', title='Top 15 cities for Data analyst', 

                   xaxis=dict(title='Name of City',zeroline= True, gridcolor='rgb(183,183,183)',showline=True),

                   yaxis=dict(title='Salary',zeroline= True, gridcolor='rgb(183,183,183)', showline=True),

                  font=dict(family='Courier New, monospace', size=12, color='rgb(0,0,0)'))





data = [trace1, trace2]

fig = go.Figure(data = data, layout = layout)

iplot(fig)
# top 20 location simply



analyst_data['Location'].value_counts()[:20].iplot(kind='bar',

                                       xTitle='Location',

                                       yTitle='Frequency of Job',

                                       title='No. of jobs at each Location',

                                       color = 'rgba(150, 200, 80, 0.5)')

# best 15 job based on role

role_analysis = analyst_data.groupby('Job Title')[['Max_Salary','Min_Salary']].mean().sort_values(['Max_Salary','Min_Salary'],ascending=False)



trace1 = go.Bar(x=role_analysis.index[:15],

                y=role_analysis['Min_Salary'][:15],

                name='Minimum Salary',

                marker = dict(color = 'rgba(255, 155, 128, 0.5)',line=dict(color='rgb(0,0,0)',width=1.5)))



trace2 = go.Bar(x=role_analysis.index[:15],

                y=role_analysis['Max_Salary'][:15],

                name='Maximum Salary',

                marker = dict(color = 'rgba(150, 60, 80, 0.5)', line=dict(color='rgb(0,0,0)',width=1.5)))





layout = go.Layout(barmode='group', title='Top 15 Roles for Data analyst based on salary', 

                   xaxis=dict(title='Name of Role',zeroline= True, gridcolor='rgb(183,183,183)',showline=True),

                   yaxis=dict(title='Salary',zeroline= True, gridcolor='rgb(183,183,183)', showline=True),

                  font=dict(family='Courier New, monospace', size=12, color='rgb(0,0,0)'))





data = [trace1, trace2]

fig = go.Figure(data = data, layout = layout)

iplot(fig)
headquarts = pd.DataFrame(analyst_data['Headquarters'].value_counts().sort_values(ascending=False))

data = {

   "values": headquarts['Headquarters'][:15],

   "labels": headquarts.index[:15],

   "domain": {"column": 0},

   "name": "Headquarters",

   "hoverinfo":"label+percent+name",

   "hole": .4,

   "type": "pie"

}

layout = go.Layout(

   {

      "title":"Headquarters Ratio",

}

)



data = [data]

fig = go.Figure(data = data, layout = layout)

iplot(fig)
# top 15 job based on rating





rating_analysis = analyst_data.groupby('Company Name')[['Rating']].mean().sort_values(['Rating'],ascending=False)

trace = go.Bar(x=rating_analysis.index[:15],

                y=rating_analysis['Rating'][:15],

                name='Rating',

                marker = dict(color = 'rgba(125, 215, 180, 0.5)',line=dict(color='rgb(0,0,0)',width=1.5)))



layout = go.Layout(title='Top 15 Company based on Rating', 

                   xaxis=dict(title='Name of Companies',zeroline= True, gridcolor='rgb(183,183,183)',showline=True),

                   yaxis=dict(title='Ratings',zeroline= True, gridcolor='rgb(183,183,183)', showline=True),

                  font=dict(family='Courier New, monospace', size=12, color='rgb(0,0,0)'))





data = [trace]

fig = go.Figure(data = data, layout = layout)

iplot(fig)
def wordcloud(string):

    wc = WordCloud(width=800,height=500,mask=None,random_state=21, max_font_size=110,stopwords=stop_words).generate(string)

    fig=plt.figure(figsize=(16,8))

    plt.axis('off')

    plt.imshow(wc)
stop_words=set(STOPWORDS)

job_string = " ".join(analyst_data['Job Title'])

company_string = " ".join(analyst_data['Company Name'])

Headquarter_string = " ".join(analyst_data['Headquarters'])

sector_string = " ".join(analyst_data['Sector'])

industry_string = " ".join(analyst_data['Industry'])
# job title wordcloud

wordcloud(job_string)
# company wordcloud

wordcloud(company_string)
wordcloud(Headquarter_string)
wordcloud(industry_string)
wordcloud(sector_string)