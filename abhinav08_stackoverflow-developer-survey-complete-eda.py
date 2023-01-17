import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns
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
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")

import cufflinks as cf
cf.go_offline()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df = pd.read_csv("../input/survey_results_public.csv")
print("Number of rows in the dataset are : {}".format(len(df)))
df.head()
print("Number of columns in the input dataset : {}".format(len(df.columns)))
country_dist = df['Country'].value_counts().head(6)
country_dist.iplot(kind='bar', xTitle='Country Name', yTitle='Num of developers', title='Most number of developers from these countries')
open_source_cnts = df['OpenSource'].dropna().value_counts()
df_new = pd.DataFrame({
    'label':open_source_cnts.index, 
    'values':open_source_cnts.values
})
df_new.iplot(kind='pie', labels='label', values='values', title='Percent of developers contributing to open source', color=['#ffff00', '#b0e0e6'])
df['FormalEducation'].isnull().sum().sum()
edu_cnts = df['FormalEducation'].dropna().value_counts()
edu_cnts / edu_cnts.sum() * 100
trace = go.Bar(
    y=edu_cnts.index[::-1],
    x=(edu_cnts/edu_cnts.sum() * 100)[::-1],
    orientation = 'h',
    marker=dict(
        color=['#00adff', '#f99372', '#fdded3', '#b0e0e6', '#ffff00', '#00fa9a', '#ffffcc', '#f2e6e9', '#fccbbb']
    ),
)

layout = dict(
    title='Level of Formal Education',
        margin=dict(
        l=500,
)
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
dev_type_sep = ";".join(df['DevType'].values.astype(str))
lst_dev = dev_type_sep.split(";")
lst_dev = pd.Series(lst_dev)
lst_dev = lst_dev.value_counts().head(7)
def get_colors(n_colors):
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for i in range(6)]) for j in range(n_colors)]
    return color
trace0 = [
    go.Bar(
        x=lst_dev.index,
        y=(lst_dev / lst_dev.sum() * 100),
        marker=dict(color=get_colors(len(lst_dev)))
    )]

layout = go.Layout(
    title='Different Job profiles of developers',
    xaxis=dict(title='Job Profiles of Developers'),
    yaxis=dict(title='Number of developers')
)

fig = go.Figure(data=trace0, layout=layout)
py.iplot(fig)
df['JobSatisfaction'].value_counts()
sat_levels = df['JobSatisfaction'].value_counts()
trace0 = [
    go.Bar(
        x=sat_levels.index,
        y=sat_levels / sat_levels.sum() * 100,
        marker=dict(color=get_colors(len(sat_levels)))
    )]

layout = go.Layout(
    title='Job satisfaction level of developers',
)

fig = go.Figure(data=trace0, layout=layout)
py.iplot(fig)
df['HopeFiveYears'].dropna().value_counts()
hope_five_yrs = df['HopeFiveYears'].value_counts()
hope_five_yrs.iplot(kind='bar', title='What developers hope to do in next 5 years', colors=get_colors(len(hope_five_yrs)))
df['Exercise'].value_counts()
exercise_num = df['Exercise'].value_counts()
df_new = pd.DataFrame({
    'label':exercise_num.index, 
    'values':exercise_num.values
})
df_new.iplot(kind='pie', labels='label', values='values', title='Times developers do exercise in a typical week', color=get_colors(len(sat_levels)))
lang_used = ";".join(df['LanguageWorkedWith'].dropna().values.astype(str))
lst_lang = lang_used.split(";")
lst_lang = pd.Series(lst_lang)
lst_lang = lst_lang.value_counts().head(10)
trace0 = [
    go.Bar(
        x=lst_lang.index,
        y=lst_lang / lst_lang.sum() * 100,
        marker=dict(color=get_colors(len(lst_lang)))
    )]

layout = go.Layout(
    title='Languages that developers have worked with',
)

fig = go.Figure(data=trace0, layout=layout)
py.iplot(fig)
lang_used = ";".join(df['LanguageDesireNextYear'].dropna().values.astype(str))
lst_lang = lang_used.split(";")
lst_lang = pd.Series(lst_lang)
lst_lang = lst_lang.value_counts().head(10)
trace0 = [
    go.Bar(
        x=lst_lang.index,
        y=lst_lang / lst_lang.sum() * 100,
        marker=dict(color=get_colors(len(lst_lang)))
    )]

layout = go.Layout(
    title='Languages that developers want to work with in the next year',
)

fig = go.Figure(data=trace0, layout=layout)
py.iplot(fig)
db_used = ";".join(df['DatabaseWorkedWith'].dropna().values.astype(str))
lst_db = db_used.split(";")
lst_db = pd.Series(lst_db)
lst_db = lst_db.value_counts().head(10)
trace0 = [
    go.Bar(
        x=lst_db.index,
        y=lst_db / lst_db.sum() * 100,
        marker=dict(color=get_colors(len(lst_db)))
    )]

layout = go.Layout(
    title='Databases that the developers have worked with',
)

fig = go.Figure(data=trace0, layout=layout)
py.iplot(fig)
db_used = ";".join(df['DatabaseDesireNextYear'].dropna().values.astype(str))
lst_db = db_used.split(";")
lst_db = pd.Series(lst_db)
lst_db = lst_db.value_counts().head(10)
trace0 = [
    go.Bar(
        x=lst_db.index,
        y=lst_db / lst_db.sum() * 100,
        marker=dict(color=get_colors(len(lst_db)))
    )]

layout = go.Layout(
    title='Databases that developers want to work with in the next year.',
)

fig = go.Figure(data=trace0, layout=layout)
py.iplot(fig)
db_used = ";".join(df['FrameworkWorkedWith'].dropna().values.astype(str))
lst_db = db_used.split(";")
lst_db = pd.Series(lst_db)
lst_db = lst_db.value_counts().head(10)
trace0 = [
    go.Bar(
        x=lst_db.index,
        y=lst_db / lst_db.sum() * 100,
        marker=dict(color=get_colors(len(lst_db)), line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),)
    )]

layout = go.Layout(
    title='Frameworks that developers have worked with.',
)

fig = go.Figure(data=trace0, layout=layout)
py.iplot(fig)
db_used = ";".join(df['FrameworkDesireNextYear'].dropna().values.astype(str))
lst_db = db_used.split(";")
lst_db = pd.Series(lst_db)
lst_db = lst_db.value_counts().head(10)
trace0 = [
    go.Bar(
        x=lst_db.index,
        y=lst_db / lst_db.sum() * 100,
        marker=dict(color=get_colors(len(lst_db)), line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),)
    )]

layout = go.Layout(
    title='Frameworks that developers want to work with in the next year.',
)

fig = go.Figure(data=trace0, layout=layout)
py.iplot(fig)
ide_used = ";".join(df['IDE'].dropna().values.astype(str))
lst_ide = ide_used.split(";")
lst_ide = pd.Series(lst_ide)
lst_ide = lst_ide.value_counts().head(10)
trace0 = [
    go.Bar(
        x=lst_ide.index,
        y=lst_ide / lst_lang.sum() * 100,
        marker=dict(color=get_colors(len(lst_ide)), line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),)
    )]

layout = go.Layout(
    title='Most Common IDEs among developers',
)

fig = go.Figure(data=trace0, layout=layout)
py.iplot(fig)
os_used = df['OperatingSystem'].value_counts()
df_new = pd.DataFrame(
{
    'label': os_used.index,
    'values': os_used.values
})
df_new.iplot(kind='pie', labels='label', values='values',title='Operating_system used', colors=get_colors(len(os_used)))
df_new = df[['Currency', 'Salary', 'SalaryType']]
# We will drop the rows which have null values
df_new = df_new.dropna()
df_new['Salary'] = [x.replace(",", "") for x in df_new['Salary']]
df_new['Salary'] = [float(x) for x in df_new['Salary'].values.astype(str)]
df_new = df_new[df_new['Salary'] != 0]
index_monthly = df_new[df_new['SalaryType'] == 'Monthly'].index
df_new.loc[index_monthly, 'Salary'] = df_new.loc[index_monthly, 'Salary'] * 12
index_weekly = df_new[df_new['SalaryType'] == 'Weekly'].index
df_new.loc[index_weekly, 'Salary'] = df_new.loc[index_weekly, 'Salary'] * 52
df_new.groupby('Currency').median()
df['Age'].dropna().value_counts()
lst_age = df['Age'].dropna().value_counts()
trace0 = [
    go.Bar(
        x=lst_age.index,
        y=lst_age / lst_age.sum() * 100,
        marker=dict(color=get_colors(len(lst_age)), line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),)
    )]

layout = go.Layout(
    title='Age Distribution of developers',
)

fig = go.Figure(data=trace0, layout=layout)
py.iplot(fig)
df_new = df[['Age', 'Hobby']]
df_new = df_new.dropna()
df_new['Hobby'] = [1 if x=='Yes' else 0 for x in df_new['Hobby']]
lst = df_new.groupby('Age').describe()
def func(row):
    return row['Hobby']['mean']*row['Hobby']['count']

lst['vals'] = lst.apply(func, axis=1)

trace1 = go.Bar(
    x=lst.index,
    y=lst['Hobby']['count'],
    name='Count of the Deveopers'
)

trace2 = go.Bar(
    x=lst.index,
    y=lst['vals'],
    name='Count of those who code as a hobby'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
df_new = df[['Age', 'OpenSource']]
df_new = df_new.dropna()
# df_new.head()
df_new['OpenSource'] = [1 if x=='Yes' else 0 for x in df_new['OpenSource']]
lst = df_new.groupby('Age').describe()
# lst.head()
def func(row):
    return row['OpenSource']['mean']*row['OpenSource']['count']

lst['vals'] = lst.apply(func, axis=1)
trace1 = go.Bar(
    x=lst.index,
    y=lst['OpenSource']['count'],
    name='Count of the Developers'
)

trace2 = go.Bar(
    x=lst.index,
    y=lst['vals'],
    name='Count of those who do open source contributions'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
df['HoursComputer'].value_counts()
lst = df['HoursComputer'].value_counts()
trace0 = [
    go.Bar(
        x=lst.index,
        y=lst / lst.sum() * 100,
        marker=dict(color=get_colors(len(lst)), line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),)
    )]

layout = go.Layout(
    title='Time developers spend in front of a computer',
)

fig = go.Figure(data=trace0, layout=layout)
py.iplot(fig)
lst = df['CompanySize'].value_counts()
trace0 = [
    go.Bar(
        x=lst.index,
        y=lst / lst.sum() * 100,
        marker=dict(color=get_colors(len(lst)), line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),)
    )]

layout = go.Layout(
    title='Company size of the developers',
)

fig = go.Figure(data=trace0, layout=layout)
py.iplot(fig)
df['AIFuture'].value_counts()
lst = df['AIFuture'].value_counts()
trace0 = [
    go.Bar(
        x=lst.index,
        y=lst / lst.sum() * 100,
        marker=dict(color=get_colors(len(lst)), line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),)
    )]

layout = go.Layout(
    title='Developers thoughts about the future of AI',
)

fig = go.Figure(data=trace0, layout=layout)
py.iplot(fig)
