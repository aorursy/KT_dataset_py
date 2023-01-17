import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

from plotly.offline import init_notebook_mode, iplot 

import plotly.graph_objects as go

import plotly.offline as py

import plotly.express as px

import pycountry

py.init_notebook_mode(connected=True)

import folium 

from folium import plugins

%config InlineBackend.figure_format = 'retina' 

plt.rcParams['figure.figsize'] = 8, 5

pd.options.mode.chained_assignment = None 

pd.set_option('display.max_columns',None)

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/indian-candidates-for-general-election-2019/LS_2.0.csv')

df.head()
print('Number of rows and columns :',df.shape) # Number of rows and columns
df.columns = df.columns.str.replace('\n',' ') # To rename columns
df.describe()
df1 = df[df['PARTY']!= 'NOTA']

percentage_missing_values = round(df1.isnull().sum()*100/len(df1),2).reset_index()

percentage_missing_values.columns = ['column_name','percentage_missing_values']

percentage_missing_values = percentage_missing_values.sort_values('percentage_missing_values',ascending = False)

percentage_missing_values
constituencies_per_state = df.groupby('STATE')['CONSTITUENCY'].nunique().reset_index().sort_values('CONSTITUENCY',ascending = False)

fig = px.bar(constituencies_per_state, x='STATE', y='CONSTITUENCY', color='CONSTITUENCY', height=600)

fig.show()
candidates_per_party = df.PARTY.value_counts().reset_index().rename(columns = {'index':'Party','PARTY':'Total Candidates'}).head(50)

candidates_per_party = candidates_per_party[candidates_per_party['Party'] != 'NOTA']

fig = px.bar(candidates_per_party, x='Party', y='Total Candidates', color='Total Candidates', height=500)

fig.show()
vote_share_top5 = df.groupby('PARTY')['TOTAL VOTES'].sum().nlargest(5).index.tolist()

def vote_share(row):

    if row['PARTY'] not in vote_share_top5:

        return 'Other'

    else:

        return row['PARTY']

df['Party New'] = df.apply(vote_share,axis =1)

counts = df.groupby('Party New')['TOTAL VOTES'].sum(sort=True)

labels = counts.index

values = counts.values

pie = go.Pie(labels=labels, values=values, marker=dict(line=dict(color='#000000', width=1)))

layout = go.Layout(title='Partywise Vote Share')

fig = go.Figure(data=[pie], layout=layout)

py.iplot(fig)
winning_candidates_per_party = df.groupby(['PARTY','SYMBOL'])['WINNER'].sum().reset_index().sort_values('WINNER',ascending = False)

winning_candidates_per_party = winning_candidates_per_party[winning_candidates_per_party['WINNER'] > 0]

fig = px.bar(winning_candidates_per_party, x='PARTY', y='WINNER',hover_data =['SYMBOL'], color='WINNER', height=500)

fig.show()
fig = px.histogram(df, x="AGE")

fig.show()
fig = px.histogram(df.dropna(), x="AGE", y="WINNER", color="GENDER", marginal="violin",hover_data=df.columns)

fig.show()
df_winners = df[df['WINNER']==1]

df_winners = df_winners.sort_values('AGE').head(10)

fig = px.bar(df_winners, x='NAME', y='AGE', color='AGE', height=500, hover_data=['PARTY','SYMBOL','CONSTITUENCY','STATE'])

fig.show()
df_winners = df[df['WINNER']==1]

df_winners = df_winners.sort_values('AGE',ascending=False).head(10)

fig = px.bar(df_winners, x='NAME', y='AGE', color='AGE', height=500, hover_data=['PARTY','SYMBOL','CONSTITUENCY','STATE'])

fig.show()
df['CRIMINAL CASES'] = df['CRIMINAL CASES'].str.replace('Not Available','0')

df['CRIMINAL CASES'] = df['CRIMINAL CASES'].fillna(0)

df['CRIMINAL CASES'] = df['CRIMINAL CASES'].astype(int)

criminal_cases = df[(df['CRIMINAL CASES'] != 'Not Available') & (df['CRIMINAL CASES'].notnull())]

criminal_cases = criminal_cases.groupby('PARTY')['CRIMINAL CASES'].sum().reset_index().sort_values('CRIMINAL CASES',ascending=False).head(30)

fig = px.bar(criminal_cases, x='PARTY', y='CRIMINAL CASES', color='CRIMINAL CASES', height=500)

fig.show()
df['EDUCATION'] = df['EDUCATION'].str.replace('Post Graduate\n','Post Graduate')

df['EDUCATION'] = df['EDUCATION'].fillna('Others') 

education = df[df['EDUCATION'] != 'Not Available']

education = education['EDUCATION'].value_counts().reset_index()

education.columns = ['EDUCATION','COUNT']

fig = px.bar(education, x='EDUCATION', y='COUNT', color='COUNT', height=500)

fig.show()
won_educated_candidates = df[df['WINNER']==1]

fig = px.bar(won_educated_candidates, x='EDUCATION', y='WINNER', color='EDUCATION', height=500).update_xaxes(categoryorder = "total descending")

fig.show()
fig = px.histogram(df.dropna(), x="AGE", y="PARTY", color="WINNER", marginal="violin",hover_data=df.columns)

fig.show()
df_criminal_cases = df.loc[(df['CRIMINAL CASES'].notnull()) & (df['CRIMINAL CASES'] != 'Not Available')]

def criminal_cases(row):

    if row['CRIMINAL CASES'] == 0:

        return 'No'

    else:

        return 'Yes'

df_criminal_cases['HAS CRIMINAL CASE'] = df_criminal_cases.apply(criminal_cases,axis = 1)

df_criminal_cases_count = df_criminal_cases.groupby(['HAS CRIMINAL CASE','WINNER']).size().reset_index()

df_criminal_cases_count.columns = ['HAS CRIMINAL CASE','WINNER','COUNT']

# pivot_df_criminal_cases_count = df_criminal_cases_count.pivot(index='HAS CRIMINAL CASE', columns='WINNER', values='COUNT')

# pivot_df_criminal_cases_count.plot.bar(stacked=True,figsize=(10,7))

# plt.xticks(rotation=0)

# plt.show()

fig = px.bar(df_criminal_cases_count, x="HAS CRIMINAL CASE", y="COUNT", color='WINNER')

fig.show()
df_votes_perct_constituency = df.groupby(['STATE','CONSTITUENCY','TOTAL ELECTORS'])['TOTAL VOTES'].sum().reset_index()

df_votes_perct_constituency['% VOTED IN CONSTITUENCY'] = round(df_votes_perct_constituency['TOTAL VOTES']*100/df_votes_perct_constituency['TOTAL ELECTORS'],2)

df_voters_state = df[['STATE','CONSTITUENCY','TOTAL ELECTORS']].drop_duplicates()

df_voters_state = df_voters_state.groupby('STATE')['TOTAL ELECTORS'].sum().reset_index()

df_votes_state = df.groupby('STATE')['TOTAL VOTES'].sum().reset_index().sort_values('TOTAL VOTES',ascending = False)

df_votes_perct_state = pd.merge(df_votes_state,df_voters_state, on ='STATE',how = 'left')

df_votes_perct_state['% VOTED IN STATE'] = round(df_votes_perct_state['TOTAL VOTES']*100/df_votes_perct_state['TOTAL ELECTORS'],2)

df_votes_perct_state = df_votes_perct_state.sort_values('% VOTED IN STATE',ascending = False)

fig = px.bar(df_votes_perct_state, x='STATE', y='% VOTED IN STATE', color='% VOTED IN STATE', height=500)

fig.show()
fig = px.bar(df_votes_state, x='STATE', y='TOTAL VOTES', color='TOTAL VOTES', height=500)

fig.show()
df_assets = df.copy()

df_assets[['ASSETS2','ASSETS_VALUE']] = df_assets['ASSETS'].str.split('~',expand=True)

df_assets.drop(['ASSETS2'],axis =1,inplace=True)

df_assets = df_assets[df_assets['ASSETS_VALUE'].notnull()]

def asset_range(row):

    if row['ASSETS_VALUE'].endswith('Crore+'):

        return 'Crore+'

    elif row['ASSETS_VALUE'].endswith('Lacs+'):

        return 'Lakh+'

    elif row['ASSETS_VALUE'].endswith('Thou+'):

        return 'Thousand+'

    else:

        return 'NAN'



df_assets['ASSETS_RANGE'] = df_assets.apply(asset_range,axis =1)

df_assets['COUNT'] = 1

df_assets = df_assets[df_assets['ASSETS_RANGE'] != 'NAN']

counts = df_assets.groupby('ASSETS_RANGE')['COUNT'].sum(sort=True)

labels = counts.index

values = counts.values

pie = go.Pie(labels=labels, values=values, marker=dict(line=dict(color='#000000', width=1)))

layout = go.Layout(title='Assests of Candidates')

fig = go.Figure(data=[pie], layout=layout)

py.iplot(fig)
df_category = df['CATEGORY'].value_counts().reset_index()

df_category.columns = ['CATEGORY','COUNT']

fig = px.bar(df_category, x='CATEGORY', y='COUNT', color='CATEGORY', height=500)

fig.show()
df_gender = df['GENDER'].value_counts().reset_index()

df_gender.columns = ['GENDER','COUNT']

pie = go.Pie(labels=df_gender['GENDER'], values=df_gender['COUNT'], marker=dict(line=dict(color='black', width=1)))

layout = go.Layout(title='Male vs Female Ratio - All Candidates')

fig = go.Figure(data=[pie], layout=layout)

py.iplot(fig)
df_gender_won =df[df['WINNER'] == 1]

df_gender_won = df_gender_won['GENDER'].value_counts().reset_index()

df_gender_won.columns = ['GENDER','COUNT']

pie = go.Pie(labels=df_gender_won['GENDER'], values=df_gender_won['COUNT'], marker=dict(line=dict(color='black', width=1)))

layout = go.Layout(title='Male vs Female Ratio - Winners')

fig = go.Figure(data=[pie], layout=layout)

py.iplot(fig)
df = df[df['PARTY']!= 'NOTA']

df[['ASSETS2','ASSETS_VALUE']] = df['ASSETS'].str.split('~',expand=True)

df.drop(['ASSETS2'],axis =1,inplace=True)

df = df[df['ASSETS_VALUE'].notnull()]

df['ASSETS_RANGE'] = df.apply(asset_range,axis =1)



df[['LIABILITY2','LIABILITY_VALUE']] = df['LIABILITIES'].str.split('~',expand=True)

df.drop(['LIABILITY2'],axis =1,inplace=True)

df = df[df['LIABILITY_VALUE'].notnull()]



def liability_range(row):

    if row['LIABILITY_VALUE'].endswith('Crore+'):

        return 'Crore+'

    elif row['LIABILITY_VALUE'].endswith('Lacs+'):

        return 'Lakh+'

    elif row['LIABILITY_VALUE'].endswith('Thou+'):

        return 'Thousand+'

    else:

        return 'NAN'

df['LIABILITY_RANGE'] = df.apply(liability_range,axis =1)
df1 = df[['STATE','CONSTITUENCY','WINNER','Party New','SYMBOL','GENDER','CRIMINAL CASES','AGE','CATEGORY','EDUCATION','TOTAL VOTES','TOTAL ELECTORS','ASSETS_RANGE','LIABILITY_RANGE']]

cat_cols = ['STATE','CONSTITUENCY','Party New','SYMBOL','GENDER','CATEGORY','EDUCATION','ASSETS_RANGE','LIABILITY_RANGE']

num_cols = ['CRIMINAL CASES','AGE','TOTAL VOTES','TOTAL ELECTORS']
df_winner = df1['WINNER'].value_counts().reset_index()

df_winner.columns = ['RESULT','COUNT']

pie = go.Pie(labels=df_winner['RESULT'], values=df_winner['COUNT'], marker=dict(line=dict(color='black', width=1)))

layout = go.Layout(title='Total Candidates vs Winners')

fig = go.Figure(data=[pie], layout=layout)

py.iplot(fig)
dataset = pd.get_dummies(df1, columns = cat_cols)

from sklearn.preprocessing import StandardScaler

standardScaler = StandardScaler()

columns_to_scale = num_cols

dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])

dataset.head()
from sklearn.utils import resample

df_majority = dataset[dataset.WINNER == 0]

df_minority = dataset[dataset.WINNER == 1]

df_minority_upsampled = resample(df_minority, replace = True,n_samples = 1452, random_state = 0) 

df_upsampled = pd.concat([df_majority, df_minority_upsampled])

df_upsampled.WINNER.value_counts()

y = df_upsampled['WINNER']

X = df_upsampled.drop(['WINNER'], axis = 1)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

rfc_scores = []

for k in range(1,21):

    randomforest_classifier= RandomForestClassifier(n_estimators=k,random_state=0)

    score=cross_val_score(randomforest_classifier,X,y,cv=10)

    rfc_scores.append(score.mean())

plt.figure(figsize =(20,7))

plt.plot([k for k in range(1, 21)], rfc_scores, color = 'red')

for i in range(1,21):

    plt.text(i, rfc_scores[i-1], (i, round(rfc_scores[i-1],3)))

plt.xticks([i for i in range(1, 21)])

plt.xlabel('Number of Estimators (K)')

plt.ylabel('Scores')

plt.title('Random Forest Classifier scores for different K values')
randomforest_classifier= RandomForestClassifier(n_estimators=14,random_state=0)

score=cross_val_score(randomforest_classifier,X,y,cv=10)

print('% Accuracy :', round(score.mean()*100,4))