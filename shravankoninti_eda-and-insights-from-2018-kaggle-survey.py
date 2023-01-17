import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
%matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

import os
print(os.listdir("../input"))
# There are 3 files - create all 3 dataframes for our analysis
df_ss = pd.read_csv("../input/SurveySchema.csv", low_memory=False, header=[0,1])
df_ffr =  pd.read_csv("../input/freeFormResponses.csv",low_memory=False, header=[0,1])
df_mcr = pd.read_csv("../input/multipleChoiceResponses.csv",low_memory=False, header=[0,1])
df_mcr.head(2)
# Format Dataframes
df_ffr.columns = df_ffr.columns.map('_'.join)
df_mcr.columns = df_mcr.columns.map('_'.join)
df_ss.columns = df_ss.columns.map('_'.join)

# For getting all columns
pd.set_option('display.max_columns', None)
df_mcr.columns
#Get the number of Rows and columns of each dataframe
print("the SurveySchema file has {} total no of rows and {} total no of columns".format(df_ss.shape[0], df_ss.shape[1]))

print("the FreeFormResponse file has {} total no of rows and {} total no of columns".format(df_ffr.shape[0], df_ffr.shape[1]))


print("the MultiplChoiceResponse file has {} total no of rows and {} total no of columns".format(df_mcr.shape[0], df_mcr.shape[1]))
df_mcr.head(2)
# Rename columns
df_mcr = df_mcr.rename({'Time from Start to Finish (seconds)_Duration (in seconds)' : 'Duration', 
                 'Q1_What is your gender? - Selected Choice' : 'Gender', 
                 'Q1_OTHER_TEXT_What is your gender? - Prefer to self-describe - Text' : 'Gender_other', 
                 'Q2_What is your age (# years)?' : 'Age', 
                 'Q3_In which country do you currently reside?' : 'Country', 
                 'Q4_What is the highest level of formal education that you have attained or plan to attain within the next 2 years?' : 'Education', 
                 'Q5_Which best describes your undergraduate major? - Selected Choice' : 'UG_major', 
                 'Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice' : 'Current_title', 
                 'Q6_OTHER_TEXT_Select the title most similar to your current role (or most recent title if retired): - Other - Text' : 'Current_title_other', 
                 'Q7_In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice' : 'Industry', 
                'Q8_How many years of experience do you have in your current role?' : 'Experience', 
                'Q9_What is your current yearly compensation (approximate $USD)?' : 'Compensation', 
                'Q10_Does your current employer incorporate machine learning methods into their business?' : 'EmployerML?', 
                 }, axis='columns')
df_mcr.columns
temp_series = df_mcr['Gender'].value_counts()
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(
    title='Gender distribution'
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Gender")
df_mcr['Age'].value_counts()
Age_data =df_mcr['Age'].value_counts().to_frame()

trace = go.Bar(
    x=Age_data.index,
    y=Age_data.Age,
    marker=dict(
        color=Age_data.Age,
        colorscale = 'Reds')
)

data = [trace]
layout = go.Layout(
    title='Age distribution of the respondents', yaxis = dict(title = '# of Respondents')
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
Country_data =df_mcr['Country'].value_counts().head(10).to_frame()

trace = go.Bar(
    x=Country_data.index,
    y=Country_data.Country,
    marker=dict(
        color=Country_data.Country,
        colorscale = 'Blues')
)

data = [trace]
layout = go.Layout(
    title='Country wise number of respondents',
     xaxis=dict(tickangle= 45),
     yaxis = dict(title = '# of Respondents')
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
df_mcr['Education'].value_counts()
df_mcr['Education'] = df_mcr['Education'].replace({'Some college/university study without earning a bachelor’s degree'
                                                       :'Some college/university study',
                                                      'No (we do not use ML methods)':'No',
                                                       'No formal education past high school'
                                                       : 'No formal education'                                                 
                                                      })
temp_series = df_mcr['Education'].value_counts()
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))

trace = go.Pie(labels=labels, values=sizes, hole = 0.5)
layout = go.Layout(
    title='Education distribution'
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Education")
df_mcr['UG_major'].value_counts()
df_mcr['UG_major'].value_counts().plot.bar(figsize=(15,10), fontsize=12)
plt.xlabel('UG_major', fontsize = 15)
plt.ylabel('Count', fontsize=15)
plt.title('UG_Major - Split for Respondents',fontsize = 20)
df_mcr['Current_title'].value_counts()
title_ind = df_mcr['Current_title'].value_counts()

trace = go.Scatter(
    x=title_ind.index,
    y=title_ind.values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 20,
        color = title_ind.values,
        colorscale='Portland',
        showscale=True)
)

data = [trace]
layout = go.Layout(
    title='Current title of users', yaxis = dict(title = '# of Respondents')
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
Industry = df_mcr['Industry'].value_counts()[:10]

trace = go.Scatter(
    x=Industry.index,
    y=Industry.values,
    mode='markers',
    marker=dict(
        symbol = 'triangle-up',
        sizemode = 'diameter',
        sizeref = 1,
        size = 30,
        color = Industry.values,
        colorscale='Portland',
        showscale=True)
)

data = [trace]
layout = go.Layout(
    title='Industry of Respondents (Top 10)', yaxis = dict(title = '# of Respondents')
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
df_mcr['Experience'].value_counts()
temp_series = df_mcr['Experience'].value_counts()
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))

trace = go.Pie(labels=labels, values=sizes, hole = 0.5)
layout = go.Layout(
    title='Experience distribution in Years'
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Experience")
df_mcr['Compensation'].head(10)
df_mcr['Compensation'].head(10)
# df_mcr.groupby('Compensation').Compensation.mean().to_frame()
df_mcr['Compensation'].value_counts()
df_mcr['Compensation_mod'] = df_mcr['Compensation'].replace({'I do not wish to disclose my approximate yearly compensation': 'Not Disclosed'})

Compensation_data =df_mcr['Compensation_mod'].value_counts().head(15).to_frame()

trace = go.Bar(
    x=Compensation_data.index,
    y=Compensation_data.Compensation_mod,
    marker=dict(
        color=Compensation_data.Compensation_mod,
        colorscale = 'Greens')
)

data = [trace]
layout = go.Layout(
    title='Compensation of respondents',
     xaxis=dict(tickangle= 45),
     yaxis = dict(title = '# of Respondents')
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
df_mcr['EmployerML?'].value_counts()
df_mcr['EmployerML?'] = df_mcr['EmployerML?'].replace({'We are exploring ML methods (and may one day put a model into production)'
                                                       :'Exploring ML methods',
                                                      'No (we do not use ML methods)':'No',
                                                       'We recently started using ML methods (i.e., models in production for less than 2 years)'
                                                       : 'Recently Started',
                                                       'We have well established ML methods (i.e., models in production for more than 2 years)'
                                                       : 'Have well established ML Methods',
                                                       'We use ML methods for generating insights (but do not put working models into production)'
                                                       : 'Use ML methods for Generating Insights'
                                                       
                                                       
                                                       
                                                       
                                                      
                                                      })
df_mcr['EmployerML?'].value_counts()
temp_series = df_mcr['EmployerML?'].value_counts()
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(
    title='EmployerML distribution'
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="EmployerML?")