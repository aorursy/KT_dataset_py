import numpy as np 

import pandas as pd 

import os

import math

import matplotlib.pyplot as plt

import seaborn as sns

import pandas_profiling

%matplotlib inline

from plotly import tools

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

from plotly.subplots import make_subplots

import warnings

warnings.filterwarnings("ignore")

init_notebook_mode(connected=True)

from ipywidgets import interact, interactive, interact_manual

import ipywidgets as widgets

import colorlover as cl
data = pd.read_csv("../input/kaggle-survey-2019/multiple_choice_responses.csv")

data = data.drop(0)

top = 5
mapping = {'Q1':'Age',

'Q2':'sex',

'Q3':'Country',

'Q4':'Degree in two years',

'Q5':'Career',

'Q6':'Company Size',

'Q7':'Team Size',

'Q8':'ML Status in Company',

'Q10':'Compensation Status',

'Q11':'Money Spent'

}



data = data.rename(columns= mapping)
data.profile_report()
data.dtypes
data.head()
dict_salary = dict({"$0-999": 500, "1,000-1,999": 1500, "2,000-2,999": 2500,

                    "3,000-3,999": 3500, "4,000-4,999": 4500, "5,000-7,499": 6250,

                    "7,500-9,999": 8750, "10,000-14,999": 12500, "15,000-19,999": 17500,

                    "20,000-24,999": 22500, "25,000-29,999": 27500, "30,000-39,999": 35000,

                    "40,000-49,999": 45000, "50,000-59,999": 55000, "60,000-69,999": 65000,

                    "70,000-79,999": 75000, "80,000-89,999": 85000, "90,000-99,999": 95000,

                    "100,000-124,999": 112500, "125,000-149,999": 137500, "150,000-199,999": 175000,

                    "200,000-249,999": 225000, "250,000-299,999": 275000, "300,000-500,000": 400000,

                    "> $500,000": 500000})

data["Compensation Status"] = data['Compensation Status'].map(dict_salary)
data["Compensation Status"].fillna(0)
salary = dict({"$0 (USD)": 0, "$1-$99": 50, "$100-$999": 500,

                    "$1,000-$9,999": 5000, "$10,000-$99,999": 50000, "> $100,000 ($USD)": 100000})

data["Money Spent"] = data['Money Spent'].map(salary)
data["Money Spent"].fillna(0)
import cufflinks

cufflinks.go_offline(connected=True)
data['Money Spent'].iplot(kind='hist', xTitle='Money Spent', yTitle='count', title='Money Spent Distribution')
data[['Money Spent', 'Compensation Status']].iplot(

    kind='hist',

    histnorm='percent',

    barmode='overlay',

    xTitle='Money Spent',

    yTitle='Compensation Status',

    title='Money Spent in 5 years to the Compensation Status')
data.pivot(columns='Career', values='Compensation Status').iplot(

        kind='box',

        yTitle='Compensation Status',

        title='Salary Distribution by Career')
from wordcloud import WordCloud, STOPWORDS

from PIL import Image



text = data.Career.values

wordcloud = WordCloud(

    width = 500,

    height = 250,

    background_color = 'black',

    stopwords = STOPWORDS).generate(str(text))

fig = plt.figure(

    figsize = (20, 20),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
male = data[data['sex']=='Male']

female = data[data['sex']=='Female']

currect_residency = data["Country"]

male_count = male['Country'].value_counts()[:top].reset_index()

female_count = female['Country'].value_counts()[:top].reset_index()
men = go.Pie(labels=male_count['index'],values=male_count['Country'],name="Men",hole=0.5,domain={'x': [0,0.50]})

women = go.Pie(labels=female_count['index'],values=female_count['Country'],name="Women",hole=0.5,domain={'x': [0.50,1]})

layout = dict(title = 'Top-5 Countries with Respondents', font=dict(size=15), legend=dict(orientation="h"),

              annotations = [dict(x=0.215, y=0.5, text='Men', showarrow=False, font=dict(size=25)),

                             dict(x=0.81, y=0.5, text='Women', showarrow=False, font=dict(size=25)) ])



top_countries_with_respondents = dict(data=[men, women], layout=layout)

iplot(top_countries_with_respondents)
age = data.groupby(['Age']).apply(lambda x: x.reset_index(drop = True))

plt.figure(figsize=(10,7))

age_bar = sns.countplot(y="Age", hue="Career", data=age)
career = data[data['Career'] == 'Data Analyst']

size_count = career['Company Size'].value_counts()[:top].reset_index()

size = go.Pie(labels=size_count['index'],values=size_count['Company Size'],name="Company Size",hole=0.5,domain={'x': [0,1]})

layout1 = dict(title = 'Top 5 company size for data analyst', font=dict(size=20), legend=dict(orientation="h"),

              annotations = [dict(x=0.5, y=0.5, text='Company Size', showarrow=False, font=dict(size=20))])

comp_size = dict(data=size, layout=layout1)

iplot(comp_size)
("There are ",career.shape[0], "data analyst respondents ",career.shape[0]/data.shape[0],"%")
degree_count = career['Degree in two years'].value_counts()[:top].reset_index()

degree_size = go.Pie(labels=degree_count['index'],values=degree_count['Degree in two years'],name="Degree",hole=0.5,domain={'x': [0,1]})

layout2 = dict(title = 'Top 5 degree for data analyst', font=dict(size=15), legend=dict(orientation="h"),

              annotations = [dict(x=0.5, y=0.5, text='Degree', showarrow=False, font=dict(size=20))])

deg_size = dict(data=degree_size, layout=layout2)

iplot(deg_size)
data.Age.value_counts()
career.Age.value_counts()
country_count = career['Country'].value_counts()[:top].reset_index()

country_size = go.Pie(labels=country_count['index'],values=country_count['Country'],name="Country",hole=0.5,domain={'x': [0,1]})

layout3 = dict(title = 'Top 5 country with data analyst as a career', font=dict(size=15), legend=dict(orientation="h"),

              annotations = [dict(x=0.5, y=0.5, text='Country', showarrow=False, font=dict(size=20))])

coun_size = dict(data=country_size, layout=layout3)

iplot(coun_size)
male_total = career[career['sex'] == 'Male']

female_total = career[career['sex'] == 'Female']

male_total = male_total['Country'].value_counts()[:top].reset_index()

female_total = female_total['Country'].value_counts()[:top].reset_index()

men_coun = go.Pie(labels=male_total['index'],values=male_total['Country'],name="Men",hole=0.5,domain={'x': [0,0.50]})

women_coun = go.Pie(labels=female_total['index'],values=female_total['Country'],name="Women",hole=0.5,domain={'x': [0.50,1]})

layout4 = dict(title = 'Top-5 Countries with Data Analyst', font=dict(size=15), legend=dict(orientation="h"),

              annotations = [dict(x=0.215, y=0.5, text='Men', showarrow=False, font=dict(size=25)),

                             dict(x=0.81, y=0.5, text='Women', showarrow=False, font=dict(size=25)) ])



top_countries = dict(data=[men_coun, women_coun], layout=layout4)

iplot(top_countries)
def charts(col1,col2,x_title,y_title, data):

    index = data[col1].dropna().unique()

    vals = data[col2].unique()

    layout = go.Layout()

    trace = []

    for j,y_axis in enumerate(vals):

        trace.append(go.Bar(x = data[data[col2] == y_axis][col1].value_counts().index,

                            y = data[data[col2] == y_axis][col1].sort_values().value_counts().values,

                opacity = 0.6, name = vals[j]))

    fig = go.Figure(data = trace, layout = layout)

    fig.update_layout(

        title = x_title,

        yaxis = dict(title = y_title),

        legend = dict( bgcolor = 'rgba(255, 255, 255, 0)', bordercolor = 'rgba(255, 255, 255, 0)'),

        bargap = 0.15, bargroupgap = 0.1,legend_orientation="h")

    fig.show()

charts("Company Size","Age","Company size VS age group","Count",career)
data['Q13'] = data.Q13_Part_1.combine_first(data.Q13_Part_2).combine_first(data.Q13_Part_3).combine_first(data.Q13_Part_4).combine_first(data.Q13_Part_5).combine_first(data.Q13_Part_6).combine_first(data.Q13_Part_7).combine_first(data.Q13_Part_2).combine_first(data.Q13_Part_8).combine_first(data.Q13_Part_9).combine_first(data.Q13_Part_10).combine_first(data.Q13_Part_11)

mapp = {'Q13':'Learning'}

data = data.rename(columns= mapp)
career = data[data['Career'] == 'Data Analyst']

male_total = career[career['sex'] == 'Male']

female_total = career[career['sex'] == 'Female']

male_learn = male_total['Learning'].value_counts()[:top].reset_index()

female_learn = female_total['Learning'].value_counts()[:top].reset_index()

men_learn = go.Pie(labels=male_learn['index'],values=male_learn['Learning'],name="Men",hole=0.5,domain={'x': [0,0.50]})

women_learn = go.Pie(labels=female_learn['index'],values=female_learn['Learning'],name="Women",hole=0.5,domain={'x': [0.50,1]})

layout5 = dict(title = 'Top-5 learning sites for Data Analyst', font=dict(size=15), legend=dict(orientation="h"),

              annotations = [dict(x=0.215, y=0.5, text='Men', showarrow=False, font=dict(size=25)),

                             dict(x=0.81, y=0.5, text='Women', showarrow=False, font=dict(size=25)) ])



learn = dict(data=[men_learn, women_learn], layout=layout5)

iplot(learn)
def charts(col1,col2,x_title,y_title, data):

    index = data[col1].dropna().unique()

    vals = data[col2].unique()

    layout = go.Layout()

    trace = []

    for j,y_axis in enumerate(vals):

        trace.append(go.Bar(x = data[data[col2] == y_axis][col1].value_counts().index,

                            y = data[data[col2] == y_axis][col1].sort_values().value_counts().values,

                opacity = 0.6, name = vals[j]))

    fig = go.Figure(data = trace, layout = layout)

    fig.update_layout(

        title = x_title,

        yaxis = dict(title = y_title),

        legend = dict( bgcolor = 'rgba(255, 255, 255, 0)', bordercolor = 'rgba(255, 255, 255, 0)'),

        bargap = 0.15, bargroupgap = 0.1,legend_orientation="h")

    fig.show()

charts("Team Size","Company Size","Company size VS Team Size","Count",career)
data['Q16'] = data.Q16_Part_1.combine_first(data.Q16_Part_2).combine_first(data.Q16_Part_3).combine_first(data.Q16_Part_4).combine_first(data.Q16_Part_5).combine_first(data.Q16_Part_6).combine_first(data.Q16_Part_7).combine_first(data.Q16_Part_2).combine_first(data.Q16_Part_8).combine_first(data.Q16_Part_9).combine_first(data.Q16_Part_10).combine_first(data.Q16_Part_11)

col_map = {'Q16':'Developmental Envorinments'}

data = data.rename(columns= col_map)
career = data[data['Career'] == 'Data Analyst']

male_total = career[career['sex'] == 'Male']

female_total = career[career['sex'] == 'Female']

male_envi = male_total['Developmental Envorinments'].value_counts()[:top].reset_index()

female_envi = female_total['Developmental Envorinments'].value_counts()[:top].reset_index()

men_envi = go.Pie(labels=male_envi['index'],values=male_envi['Developmental Envorinments'],name="Men",hole=0.5,domain={'x': [0,0.50]})

women_envi = go.Pie(labels=female_envi['index'],values=female_envi['Developmental Envorinments'],name="Women",hole=0.5,domain={'x': [0.50,1]})

layout6 = dict(title = 'Top-5 developmental environments(IDE) for Data Analyst', font=dict(size=15), legend=dict(orientation="h"),

              annotations = [dict(x=0.215, y=0.5, text='Men', showarrow=False, font=dict(size=25)),

                             dict(x=0.81, y=0.5, text='Women', showarrow=False, font=dict(size=25)) ])



environment = dict(data=[men_envi, women_envi], layout=layout6)

iplot(environment)
def charts(col1,col2,x_title,y_title, data):

    index = data[col1].dropna().unique()

    vals = data[col2].unique()

    layout = go.Layout()

    trace = []

    for j,y_axis in enumerate(vals):

        trace.append(go.Bar(x = data[data[col2] == y_axis][col1].value_counts().index,

                            y = data[data[col2] == y_axis][col1].sort_values().value_counts().values,

                opacity = 0.6, name = vals[j]))

    fig = go.Figure(data = trace, layout = layout)

    fig.update_layout(

        title = x_title,

        yaxis = dict(title = y_title),

        legend = dict( bgcolor = 'rgba(255, 255, 255, 0)', bordercolor = 'rgba(255, 255, 255, 0)'),

        bargap = 0.15, bargroupgap = 0.1,legend_orientation="h")

    fig.show()

charts("Age","Compensation Status","Salary VS age group","Count",career)
data['Q20'] = data.Q20_Part_1.combine_first(data.Q20_Part_2).combine_first(data.Q20_Part_3).combine_first(data.Q20_Part_4).combine_first(data.Q20_Part_5).combine_first(data.Q20_Part_6).combine_first(data.Q20_Part_7).combine_first(data.Q20_Part_2).combine_first(data.Q20_Part_8).combine_first(data.Q20_Part_9).combine_first(data.Q20_Part_10).combine_first(data.Q20_Part_11)

column_map = {'Q20':'visualization Libraries'}

data = data.rename(columns= column_map)
career = data[data['Career'] == 'Data Analyst']

visualization = career['visualization Libraries'].value_counts()[:top].reset_index()

visualization = go.Pie(labels=visualization['index'],values=visualization['visualization Libraries'],name="Libraries",hole=0.5,domain={'x': [0,0.50]})

layout7 = dict(title = 'Top-5 visualization libraries for Data Analyst', font=dict(size=15), legend=dict(orientation="h"),

              annotations = [dict(x=0.2, y=0.5, text='Libraries', showarrow=False, font=dict(size=20))])

environment = dict(data=[visualization], layout=layout7)

iplot(environment)
data['Q32'] = data.Q32_Part_1.combine_first(data.Q32_Part_2).combine_first(data.Q32_Part_3).combine_first(data.Q32_Part_4).combine_first(data.Q32_Part_5).combine_first(data.Q32_Part_6).combine_first(data.Q32_Part_7).combine_first(data.Q32_Part_8).combine_first(data.Q32_Part_9).combine_first(data.Q32_Part_10).combine_first(data.Q32_Part_11)

colm_map = {'Q32':'ML products'}

data = data.rename(columns= colm_map)
career = data[data['Career'] == 'Data Analyst']

products = career['ML products'].value_counts()[:top].reset_index()

products = go.Pie(labels=products['index'],values=products['ML products'],name="Products",hole=0.5,domain={'x': [0,0.50]})

layout8 = dict(title = 'Top-5 ML products for Data Analyst', font=dict(size=15), legend=dict(orientation="h"),

              annotations = [dict(x=0.2, y=0.5, text='Products', showarrow=False, font=dict(size=20))])

prod = dict(data=[products], layout=layout8)

iplot(prod)
data['Q18'] = data.Q18_Part_1.combine_first(data.Q18_Part_2).combine_first(data.Q18_Part_3).combine_first(data.Q18_Part_4).combine_first(data.Q18_Part_5).combine_first(data.Q18_Part_6).combine_first(data.Q18_Part_7).combine_first(data.Q18_Part_8).combine_first(data.Q18_Part_9).combine_first(data.Q18_Part_10).combine_first(data.Q18_Part_11)

co_map = {'Q18':'Programming languages'}

data = data.rename(columns= co_map)
career = data[data['Career'] == 'Data Analyst']

lang = career['Programming languages'].value_counts()[:top].reset_index()

lang = go.Pie(labels=lang['index'],values=lang['Programming languages'],name="Language",hole=0.5,domain={'x': [0,0.50]})

layout9 = dict(title = 'Top-5 Programming languages of Data Analyst', font=dict(size=15), legend=dict(orientation="h"),

              annotations = [dict(x=0.2, y=0.5, text='Language', showarrow=False, font=dict(size=20))])

prod_lang = dict(data=[lang], layout=layout9)

iplot(prod_lang)
data['Q24'] = data.Q24_Part_1.combine_first(data.Q24_Part_2).combine_first(data.Q24_Part_3).combine_first(data.Q24_Part_4).combine_first(data.Q24_Part_5).combine_first(data.Q24_Part_6).combine_first(data.Q24_Part_7).combine_first(data.Q24_Part_8).combine_first(data.Q24_Part_9).combine_first(data.Q24_Part_10).combine_first(data.Q24_Part_11)

co_mp = {'Q24':'ML algorithm'}

data = data.rename(columns= co_mp)
career = data[data['Career'] == 'Data Analyst']

algorithm = career['ML algorithm'].value_counts()[:top].reset_index()

algorithm = go.Pie(labels=algorithm['index'],values=algorithm['ML algorithm'],name="Algorithms",hole=0.5,domain={'x': [0,0.50]})

layout10 = dict(title = 'Top-5 commonly used ML algorithm by Data Analyst', font=dict(size=15), legend=dict(orientation="h"),

              annotations = [dict(x=0.18, y=0.5, text='Algorithms', showarrow=False, font=dict(size=20))])

prod_alg = dict(data=[algorithm], layout=layout10)

iplot(prod_alg)
soft = data[data['Career'] == 'Software Engineer']

country_size = career['Country'].value_counts()[:top].reset_index()

country_size = go.Pie(labels=country_size['index'],values=country_size['Country'],name="Country",hole=0.5,domain={'x': [0,1]})

layout3 = dict(title = 'Top 5 country with software engineer as a career', font=dict(size=15), legend=dict(orientation="h"),

              annotations = [dict(x=0.5, y=0.5, text='Country', showarrow=False, font=dict(size=20))])

coun_size = dict(data=country_size, layout=layout3)

iplot(coun_size)
soft = data[data['Career'] == 'Software Engineer']

size_count = soft['Company Size'].value_counts()[:top].reset_index()

size = go.Pie(labels=size_count['index'],values=size_count['Company Size'],name="Company Size",hole=0.5,domain={'x': [0,1]})

layout1 = dict(title = 'Top 5 company size for software engineers', font=dict(size=20), legend=dict(orientation="h"),

              annotations = [dict(x=0.5, y=0.5, text='Company Size', showarrow=False, font=dict(size=20))])

comp_size = dict(data=size, layout=layout1)

iplot(comp_size)
soft = data[data['Career'] == 'Software Engineer']

degree_count = soft['Degree in two years'].value_counts()[:top].reset_index()

degree_size = go.Pie(labels=degree_count['index'],values=degree_count['Degree in two years'],name="Degree",hole=0.5,domain={'x': [0,1]})

layout2 = dict(title = 'Top 5 degree for software engineers', font=dict(size=15), legend=dict(orientation="h"),

              annotations = [dict(x=0.5, y=0.5, text='Degree', showarrow=False, font=dict(size=20))])

deg_size = dict(data=degree_size, layout=layout2)

iplot(deg_size)
soft.Age.value_counts()
soft = data[data['Career'] == 'Software Engineer']

male_total = soft[soft['sex'] == 'Male']

female_total = soft[soft['sex'] == 'Female']

male_total = male_total['Country'].value_counts()[:top].reset_index()

female_total = female_total['Country'].value_counts()[:top].reset_index()

men_coun = go.Pie(labels=male_total['index'],values=male_total['Country'],name="Men",hole=0.5,domain={'x': [0,0.50]})

women_coun = go.Pie(labels=female_total['index'],values=female_total['Country'],name="Women",hole=0.5,domain={'x': [0.50,1]})

layout4 = dict(title = 'Top-5 Countries with software engineers', font=dict(size=15), legend=dict(orientation="h"),

              annotations = [dict(x=0.215, y=0.5, text='Men', showarrow=False, font=dict(size=25)),

                             dict(x=0.81, y=0.5, text='Women', showarrow=False, font=dict(size=25)) ])



top_countries = dict(data=[men_coun, women_coun], layout=layout4)

iplot(top_countries)
def charts(col1,col2,x_title,y_title, data):

    index = data[col1].dropna().unique()

    vals = data[col2].unique()

    layout = go.Layout()

    trace = []

    for j,y_axis in enumerate(vals):

        trace.append(go.Bar(x = data[data[col2] == y_axis][col1].value_counts().index,

                            y = data[data[col2] == y_axis][col1].sort_values().value_counts().values,

                opacity = 0.6, name = vals[j]))

    fig = go.Figure(data = trace, layout = layout)

    fig.update_layout(

        title = x_title,

        yaxis = dict(title = y_title),

        legend = dict( bgcolor = 'rgba(255, 255, 255, 0)', bordercolor = 'rgba(255, 255, 255, 0)'),

        bargap = 0.15, bargroupgap = 0.1,legend_orientation="h")

    fig.show()

charts("Company Size","Age","Company size VS age group","Count",soft)
soft = data[data['Career'] == 'Software Engineer']

male_total = soft[soft['sex'] == 'Male']

female_total = soft[soft['sex'] == 'Female']

male_total = male_total['Learning'].value_counts()[:top].reset_index()

female_total = female_total['Learning'].value_counts()[:top].reset_index()

men_learn = go.Pie(labels=male_learn['index'],values=male_learn['Learning'],name="Men",hole=0.5,domain={'x': [0,0.50]})

women_learn = go.Pie(labels=female_learn['index'],values=female_learn['Learning'],name="Women",hole=0.5,domain={'x': [0.50,1]})

layout5 = dict(title = 'Top-5 learning sites for Software Engineers', font=dict(size=15), legend=dict(orientation="h"),

              annotations = [dict(x=0.215, y=0.5, text='Men', showarrow=False, font=dict(size=25)),

                             dict(x=0.81, y=0.5, text='Women', showarrow=False, font=dict(size=25)) ])



learn = dict(data=[men_learn, women_learn], layout=layout5)

iplot(learn)
def charts(col1,col2,x_title,y_title, data):

    index = data[col1].dropna().unique()

    vals = data[col2].unique()

    layout = go.Layout()

    trace = []

    for j,y_axis in enumerate(vals):

        trace.append(go.Bar(x = data[data[col2] == y_axis][col1].value_counts().index,

                            y = data[data[col2] == y_axis][col1].sort_values().value_counts().values,

                opacity = 0.6, name = vals[j]))

    fig = go.Figure(data = trace, layout = layout)

    fig.update_layout(

        title = x_title,

        yaxis = dict(title = y_title),

        legend = dict( bgcolor = 'rgba(255, 255, 255, 0)', bordercolor = 'rgba(255, 255, 255, 0)'),

        bargap = 0.15, bargroupgap = 0.1,legend_orientation="h")

    fig.show()

charts("Team Size","Company Size","Company size VS Team Size","Count",soft)
soft = data[data['Career'] == 'Software Engineer']

male_total = soft[soft['sex'] == 'Male']

female_total = soft[soft['sex'] == 'Female']

male_envi = male_total['Developmental Envorinments'].value_counts()[:top].reset_index()

female_envi = female_total['Developmental Envorinments'].value_counts()[:top].reset_index()

men_envi = go.Pie(labels=male_envi['index'],values=male_envi['Developmental Envorinments'],name="Men",hole=0.5,domain={'x': [0,0.50]})

women_envi = go.Pie(labels=female_envi['index'],values=female_envi['Developmental Envorinments'],name="Women",hole=0.5,domain={'x': [0.50,1]})

layout6 = dict(title = 'Top-5 developmental environments(IDE) for Software Engineers', font=dict(size=15), legend=dict(orientation="h"),

              annotations = [dict(x=0.215, y=0.5, text='Men', showarrow=False, font=dict(size=25)),

                             dict(x=0.81, y=0.5, text='Women', showarrow=False, font=dict(size=25)) ])



environment = dict(data=[men_envi, women_envi], layout=layout6)

iplot(environment)
def charts(col1,col2,x_title,y_title, data):

    index = data[col1].dropna().unique()

    vals = data[col2].unique()

    layout = go.Layout()

    trace = []

    for j,y_axis in enumerate(vals):

        trace.append(go.Bar(x = data[data[col2] == y_axis][col1].value_counts().index,

                            y = data[data[col2] == y_axis][col1].sort_values().value_counts().values,

                opacity = 0.6, name = vals[j]))

    fig = go.Figure(data = trace, layout = layout)

    fig.update_layout(

        title = x_title,

        yaxis = dict(title = y_title),

        legend = dict( bgcolor = 'rgba(255, 255, 255, 0)', bordercolor = 'rgba(255, 255, 255, 0)'),

        bargap = 0.15, bargroupgap = 0.1,legend_orientation="h")

    fig.show()

charts("Age","Compensation Status","Salary VS age group","Count",soft)
soft = data[data['Career'] == 'Software Engineer']

visualization = soft['visualization Libraries'].value_counts()[:top].reset_index()

visualization = go.Pie(labels=visualization['index'],values=visualization['visualization Libraries'],name="Libraries",hole=0.5,domain={'x': [0,0.50]})

layout7 = dict(title = 'Top-5 visualization libraries for Software Engineers', font=dict(size=15), legend=dict(orientation="h"),

              annotations = [dict(x=0.2, y=0.5, text='Libraries', showarrow=False, font=dict(size=20))])

environment = dict(data=[visualization], layout=layout7)

iplot(environment)
soft = data[data['Career'] == 'Software Engineer']

lang = soft['Programming languages'].value_counts()[:top].reset_index()

lang = go.Pie(labels=lang['index'],values=lang['Programming languages'],name="Language",hole=0.5,domain={'x': [0,0.50]})

layout9 = dict(title = 'Top-5 Programming languages of Software Engineers', font=dict(size=15), legend=dict(orientation="h"),

              annotations = [dict(x=0.2, y=0.5, text='Language', showarrow=False, font=dict(size=20))])

prod_lang = dict(data=[lang], layout=layout9)

iplot(prod_lang)
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler

from collections import Counter
df = pd.DataFrame(data,columns = ['Age','sex','Country','Degree in two years','Career','Company Size','Team Size','Learning','Developmental Envoriments','visualization Libraries','ML products', 'Programming languages', 'ML algorithm'])
df = df.astype('category')

df["Age"] = df["Age"].cat.codes

df['sex'] = df["sex"].cat.codes

df['Country'] = df["Country"].cat.codes

df['Degree in two years'] = df["Degree in two years"].cat.codes

df['Career'] = df["Career"].cat.codes

df['Company Size'] = df["Company Size"].cat.codes

df['Team Size'] = df["Team Size"].cat.codes

df['Learning'] = df["Learning"].cat.codes

df['Developmental Envoriments'] = df["Developmental Envoriments"].cat.codes

df['visualization Libraries'] = df["visualization Libraries"].cat.codes

df['ML products'] = df["ML products"].cat.codes

df['Programming languages'] = df["Programming languages"].cat.codes

df['ML algorithm'] = df["ML algorithm"].cat.codes
df.head()
def find_k(X):

    stat=[]

    diff=[]

    no_of_clusters = []

    K = list(range(2, krange+1))

    for k in K:

        if k==2:

            model = KMeans(n_clusters=k).fit(X)

            stat.append(model.inertia_)

            no_of_clusters.append(k)

        else:

            model = KMeans(n_clusters=k, random_state=random_state ).fit(X)

            diff.append((stat[len(stat)-1]-model.inertia_))

            stat.append(model.inertia_)

            no_of_clusters.append(k)



    ratio = diff/ (stat[1] - stat[len(stat)-1])

    minpos = ((pd.DataFrame(ratio).values<0.1).argmax())

    optimal_k = minpos+1

    return optimal_k

def create_kmodel(dataset):

    X  = scaler.fit_transform(dataset.values)

    k_val = find_k(X) 

    kmodel = KMeans(n_clusters=k_val, random_state=random_state, copy_x=True).fit(X)

    return kmodel,k_val

krange = 3

random_state = 2222

scaler = StandardScaler()

kmodel,optimal_k = create_kmodel(df)
from scipy.spatial.distance import cdist

def create_kmodel_metrics(dataset, kmodel):

    X  = scaler.fit_transform(df.values)

    dist = np.min(cdist(X, kmodel.cluster_centers_, 'euclidean'),axis=1)

    dist=pd.DataFrame(dist.tolist(),columns=['dist_from_cc'])

    labels = pd.DataFrame((kmodel.labels_.astype(str)).tolist(),columns=['cluster_label'])

    concatdf = pd.concat([labels,dist],axis=1)

    wss = (concatdf.groupby('cluster_label')['dist_from_cc'].mean().reset_index()[['dist_from_cc']])

    wss_sd=concatdf.groupby('cluster_label')['dist_from_cc'].std().reset_index()[['dist_from_cc']]

    wss_sd=wss_sd.fillna(0)

    df_with_labels=pd.concat([labels,dataset],axis=1)

    cluster_means=df_with_labels.groupby('cluster_label').mean().reset_index().set_index('cluster_label')

    members=Counter(kmodel.labels_)

    return labels,concatdf, wss, wss_sd, cluster_means, members, df_with_labels
def create_meta(dataset, kmodel):

    labels,concatdf, wss, wss_sd, cluster_means, members, df_with_labels = create_kmodel_metrics(dataset, kmodel)

    centers_dict=pd.DataFrame((kmodel.cluster_centers_.astype(str)).tolist(), columns=dataset.columns, index=sorted(labels['cluster_label'].unique().tolist())).to_dict(orient="index")

    ms_df = pd.DataFrame(index=['feature_means','feature_stddev'], columns=dataset.columns)

    ms_df.loc['feature_means'] = dataset.mean()

    ms_df.loc['feature_stddev'] = dataset.std()

    wss_dict = wss.to_dict(orient='dict')[wss.columns.tolist()[0]]

    wss_sd_dict = wss_sd.to_dict(orient='dict')[wss_sd.columns.tolist()[0]]

    kmeta_dict = ms_df.to_dict(orient='index')

    kmeta_dict['cluster_centers'] = centers_dict

    kmeta_dict['cluster_means'] = cluster_means.to_dict(orient='index')

    kmeta_dict['wss'] = wss_dict

    kmeta_dict['wss_sd']=wss_sd_dict

    kmeta_dict['cluster_labels'] = sorted(labels['cluster_label'].unique())

    kmeta_dict['optimal_k']=str(optimal_k)

    return kmeta_dict

meta_data = create_meta(df,kmodel)
career = df.groupby(['Career']).apply(lambda x: x.reset_index(drop = True))

sns.catplot(x="Career", y="Programming languages", hue="sex", kind="point", data=career);

data_f = data['Money Spent'].astype('category')

df = pd.concat([data_f,df],axis=1)
sns.catplot(x="Career", y="Money Spent", hue="sex", kind="point", data=data);
sns.catplot(x="Money Spent", y="Compensation Status", kind="box", data=data)
sns.catplot(x="Programming languages", y="Money Spent", hue="Compensation Status", kind="point", data=data)