# default imports

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



# visualization imports

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import missingno as msno

import networkx as nx

import plotly.graph_objects as go



sns.set_palette(sns.color_palette(['#20639B', '#ED553B', '#3CAEA3', '#F5D55C']))





# read input files

question = pd.read_csv('../input/kaggle-survey-2019/questions_only.csv')

schema = pd.read_csv('../input/kaggle-survey-2019/survey_schema.csv')

multiple_choice = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv')

other_text =  pd.read_csv('../input/kaggle-survey-2019/other_text_responses.csv')
def q_list(s):

    lst = []

    for i in multiple_choice.columns:

        if i[:3]==s:

            lst.append(i)



    df = multiple_choice[lst]



    df_sub = df.iloc[0].apply(lambda x: ''.join(x.split('-')[2:]))

    q = ''.join([f'<li>{i}</li>' for i in df_sub.values])

    display(HTML(f'<h2 style="color:#20639B">{s} : {question.T[0][int(s[1:])]}</h2><ol>{q}</ol>'))

    return df, df_sub



from IPython.core.display import display, HTML

q = ''.join([f'<li>{i}</li>' for i in question.T[0][1:]])

display(HTML(f'<h2 style="color:#20639B">Question List</h2><ol>{q}</ol>'))
dist = multiple_choice[['Q1', 'Q2', 'Q3', 'Q4']]

dist = dist.rename(columns={"Q1": "Age", "Q2": "Gender", "Q3":"Country", "Q4":"Education"})

dist.drop(0, axis=0, inplace=True)
!pip install pywaffle
from pywaffle import Waffle



gender = dist['Gender'].value_counts()



fig = plt.figure(

    FigureClass=Waffle, 

    rows=10,

    columns=10,

    values=gender,

    colors = ('#20639B', '#ED553B', '#3CAEA3', '#F5D55C'),

    title={'label': 'Gender Distribution', 'loc': 'center'},

    labels=["{}({})".format(a, b) for a, b in zip(gender.index, gender) ],

    legend={'loc': 'lower left', 'bbox_to_anchor': (1.3, .5)},

    font_size=25,

    icons='user',

    figsize=(10, 10),  

    icon_legend=True

)

fig.set_facecolor('#EEEEEE')
dist['Country']=dist['Country'].replace({'United States of America':'USA',

                                     'United Kingdom of Great Britain and Northern Ireland':'UK'})

plt.figure(figsize=(10,8))

sns.countplot(x='Country',

              data=dist,

              order=dist['Country'].value_counts().index)

plt.xticks(rotation=90)

plt.ylabel('Number of participants')

plt.title('Country wise distribution in Survey')



plt.show()
fig, ax = plt.subplots(1, 1, figsize=(20, 5))



sns.set_palette(sns.color_palette(['#20639B', '#ED553B', '#3CAEA3', '#F5D55C']))



sns.countplot(x='Age', hue='Gender', data=dist, 

              order = dist['Age'].value_counts().sort_index().index, 

              ax=ax )



plt.title('Age & Gender Distribution', size=15)

plt.show()
fig, ax = plt.subplots(1, 1, figsize=(20, 5))



sns.set_palette(sns.color_palette(['#20639B', '#ED553B', '#3CAEA3', '#F5D55C']))



ageRatio = dist.groupby(['Age', 'Gender']).size()

sns.countplot(x='Age', hue='Gender', data=ageRatio, 

              ax=ax )



plt.title('Age & Gender Ratio', size=15)

plt.show()
ageRatio = dist.groupby(['Age', 'Gender']).size()

ageRatio



# dist.groupby(['Age', 'Gender']).size().plot.barh()



# import matplotlib.pyplot as plt



# for title, group in ageRatio:

#     group.plot(x='Age', y='Gender', title=title)
#Importing the 2017 Dataset

df_2017=pd.read_csv('../input/kaggle-survey-2017/multipleChoiceResponses.csv',encoding='ISO-8859-1')

df_2017



gender_count_2017 = df_2017['GenderSelect'].value_counts(sort=True)



fig = plt.figure(

    FigureClass=Waffle, 

    rows=10,

    columns=10,

    values=gender_count_2017,

    colors = ('#20639B', '#ED553B', '#3CAEA3', '#F5D55C'),

    title={'label': 'Gender Distribution 2017', 'loc': 'center'},

    labels=["{}({})".format(a, b) for a, b in zip(gender_count_2017.index, gender_count_2017) ],

    legend={'loc': 'lower left', 'bbox_to_anchor': (1.3, .5)},

    font_size=25,

    icons='user',

    figsize=(10, 10),  

    icon_legend=True

)

fig.set_facecolor('#EEEEEE')
#Importing the 2018 Dataset

df_2018 = pd.read_csv('../input/kaggle-survey-2018/multipleChoiceResponses.csv')

df_2018.columns = df_2018.iloc[0]

df_2018=df_2018.drop([0])



gender_count_2018 = df_2018['What is your gender? - Selected Choice'].value_counts(sort=True)



fig = plt.figure(

    FigureClass=Waffle, 

    rows=10,

    columns=10,

    values=gender_count_2018,

    colors = ('#20639B', '#ED553B', '#3CAEA3', '#F5D55C'),

    title={'label': 'Gender Distribution 2018', 'loc': 'center'},

    labels=["{}({})".format(a, b) for a, b in zip(gender_count_2018.index, gender_count_2018) ],

    legend={'loc': 'lower left', 'bbox_to_anchor': (1.3, .5)},

    font_size=25,

    icons='user',

    figsize=(10, 10),  

    icon_legend=True

)

fig.set_facecolor('#EEEEEE')
role_2019 = multiple_choice[['Q5']]

role_2019.drop(0, axis=0, inplace=True)



language_2019 = multiple_choice[['Q19']]

language_2019.drop(0, axis=0, inplace=True)



experience_2019 = multiple_choice[['Q23']]

experience_2019.drop(0, axis=0, inplace=True)
plt.figure(figsize=(10,8))



sns.countplot(x='Q5',

              data=role_2019)

plt.xticks(rotation=90)

plt.ylabel('Number of participants')

plt.title('Country wise distribution in Survey')

plt.show()
plt.figure(figsize=(10,8))

sns.countplot(x='Q19',

              data=language_2019)

plt.xticks(rotation=90)

plt.ylabel('Number of participants')

plt.title('Country wise distribution in Survey')



plt.show()
plt.figure(figsize=(10,8))

sns.countplot(x='Q23',

              data=experience_2019)

plt.xticks(rotation=90)

plt.ylabel('Number of participants')

plt.title('Country wise distribution in Survey')



plt.show()