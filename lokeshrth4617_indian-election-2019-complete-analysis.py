import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from plotly.offline import init_notebook_mode, iplot 

import plotly.graph_objects as go

import plotly.offline as py

import plotly.express as px

import pycountry

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

print('Successfully loaded')

data = pd.read_csv('/kaggle/input/indian-candidates-for-general-election-2019/LS_2.0.csv')

data.head()
data = data.rename(columns={"CRIMINAL\nCASES": "Criminal", "GENERAL\nVOTES": "Genral_votes","POSTAL\nVOTES":"Postal_votes","TOTAL\nVOTES":"Total_votes"})

data.head()
round(data.isnull().sum()/len(data.index),2)

data = data.dropna() ## deleting all the NaN values for Analysis purposes
Num_cons = data.groupby('STATE')['CONSTITUENCY'].nunique().sort_values(ascending = False).reset_index()



ax = px.bar(Num_cons,y='CONSTITUENCY',x='STATE',color = 'CONSTITUENCY', title='The Number of Constituencies from each State')

ax.show()
# Data Cleaning

data['Criminal'] = data['Criminal'].replace('Not Available','0').astype('int')
data['EDUCATION'] = data['EDUCATION'].replace('Post Graduate\n','Post Graduate')

data['EDUCATION'] = data['EDUCATION'].replace('Not Available','Others')

education = data['EDUCATION'].value_counts().reset_index()

education.columns = ['EDUCATION','COUNT']

ax = px.bar(education,x = 'EDUCATION', y = 'COUNT',color = 'EDUCATION', title= 'Education Level of the Candidates')

ax.show()
winner = data[data['WINNER']==1]

ax = px.bar(winner,x = 'EDUCATION',y = 'WINNER', title='Winning Candidates Educational Degree').update_xaxes(categoryorder = "total descending")

ax.show()
young_winner = data[data['WINNER']==1]

young_winner = young_winner.sort_values('AGE').head(10)

ax = px.bar(young_winner,x = 'NAME',y = 'AGE',color = 'AGE',hover_data = ['PARTY','STATE','CONSTITUENCY'], title='Youngest Winners')

ax.show()
old_winner = data[data['WINNER']==1]

old_winner = old_winner.sort_values('AGE',ascending = False).head(10)

ax = px.bar(old_winner,x = 'NAME',y = 'AGE',color = 'AGE',hover_data = ['PARTY','STATE','CONSTITUENCY'], title = 'Oldest Winners and their Details:')

ax.show()
sns.distplot(data['AGE'],

             kde=False,

             hist_kws=dict(edgecolor="black", linewidth=2),

             color='#00BFC4')
criminal_cases = data.groupby('PARTY')['Criminal'].sum().reset_index().sort_values('Criminal',ascending=False).head(30)

ax = px.bar(criminal_cases, x = 'PARTY',y = 'Criminal',color = 'PARTY', title='Total Criminal Cases by respective parties')

ax.show()
crime = data[data['WINNER']==1]

criminal_cases = crime.groupby('PARTY')['Criminal'].sum().reset_index().sort_values('Criminal',ascending=False).head(30)

ax = px.bar(criminal_cases, x = 'PARTY',y = 'Criminal',color = 'PARTY', title='Winning Parties with Number of Criminal Cases')

ax.show()
## changing the datatype

data['GENDER'] = data['GENDER'].astype('category') 

data['WINNER'] = data['WINNER'].astype('category') 
Female_winners = data[(data['WINNER']==1) & (data['GENDER']=='FEMALE')]

ax = px.histogram(Female_winners, 'STATE', title = 'Female Winners from different States')

ax.show()
male_winners = data[(data['WINNER']==1) & (data['GENDER']=='MALE')]

ax = px.histogram(male_winners, 'STATE', title='Male Winners from different States')

ax.show()
votes = data.groupby('STATE')['Total_votes'].sum().sort_values(ascending = False).reset_index()

ax = px.bar(votes,x = 'STATE',y = 'Total_votes',color='STATE', title='Total Votes cast State Wise')

ax.show()
category = data['CATEGORY'].value_counts().reset_index()

category.columns= ['CATEGORY','COUNT']

ax = px.bar(category,x = 'CATEGORY', y = 'COUNT', color = 'CATEGORY')

ax.show()
df = data[data['WINNER']==1]

category = df['CATEGORY'].value_counts().reset_index()

category.columns= ['CATEGORY','COUNT']

ax = px.bar(category,x = 'CATEGORY', y = 'COUNT', color = 'CATEGORY', title='Winners from Various Categories')

ax.show()