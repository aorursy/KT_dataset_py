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
# Renaming of COLUMNS

df.rename(columns={'OVER TOTAL VOTES POLLED \nIN CONSTITUENCY':'VOTE PERCENTAGE','TOTAL\nVOTES':'TOTAL VOTES','CRIMINAL\nCASES':'CRIMINAL CASES'},inplace=True)
# Checking for NULL values in DataFrame



df.isnull().sum()
# Creating copy of DataFrame excluding NOTA records



df_exclude_NOTA = df.copy()

df_exclude_NOTA.dropna(inplace=True)
df_exclude_NOTA.shape
df_exclude_NOTA.head()
df_exclude_NOTA.EDUCATION.unique()
df_exclude_NOTA['EDUCATION'][df_exclude_NOTA['EDUCATION']=='Post Graduate\n'] = 'Post Graduate'
# Education Qualification of Candidates for LOK SABHA - 2019



ax = df_exclude_NOTA.EDUCATION.value_counts().plot.bar(

figsize=(12,4),

color = 'green',

fontsize =14    

)



ax.set_title('Education Qualification of Candidates for LOK SABHA - 2019',fontsize=18)

ax.set_ylabel('Number of candidates',fontsize=16)



sns.despine()
df[(df['EDUCATION']=='Illiterate')&(df['WINNER']==1)]
df[df['EDUCATION']=='Illiterate']
df[(df['EDUCATION']=='Illiterate')&(df['WINNER']==1)]
# Number of Seats Contested by PARTIES (TOP 20)



ax=df_exclude_NOTA.PARTY.value_counts().head(20).plot.bar(

figsize=(18,5),

color = '#2A89A1',

fontsize=12)



ax.set_title('Number of Seats Contested by PARTIES (TOP 20)',fontsize=20)

ax.set_ylabel('Number of Seats',fontsize=16)

ax.set_xlabel('Political Parties',fontsize=16)



sns.despine(left=True,bottom=True)
def win_percent_convertor(party):

    total_contested_seats = df[df['PARTY']==party].shape[0]

    total_seats_won = df[(df['PARTY']==party)&(df['WINNER']==1)].shape[0]

    win_percent = (total_seats_won/total_contested_seats)*100

    return win_percent
# Creating a SERIES containing information of Win Percentage PARTYWISE



party_win_percent = {}



for party in df['PARTY'].unique():

    party_win_percent[party] = win_percent_convertor(party)

    

party_win_percent_series = pd.Series(party_win_percent)  



party_win_percent_series
# Let's find out Seat Conversion Rate PARTYWISE



ax=party_win_percent_series.sort_values(ascending=False).head(36).plot.bar(

figsize=(17,5),

color='lawngreen'    

)



ax.set_title('Seat Conversion Rate',fontsize=20)

ax.set_xlabel('Political Parties',fontsize=14)

ax.set_ylabel('Win Percentage',fontsize=14)



sns.despine(bottom=True,left=True)
winning_candidates_per_party = df.groupby(['PARTY','SYMBOL'])['WINNER'].sum().reset_index().sort_values('WINNER',ascending = False)

winning_candidates_per_party = winning_candidates_per_party[winning_candidates_per_party['WINNER'] > 0]

fig = px.bar(winning_candidates_per_party, x='PARTY', y='WINNER',hover_data =['SYMBOL'], color='WINNER', height=500)

fig.show()
# Top 20 Parties on the basis of Seats Contesting 



top_20_parties = pd.Series(df_exclude_NOTA['PARTY'].value_counts().head(21))

top_20_parties = top_20_parties.index.drop(['IND'])



top_20_parties
# Creating DataFrame which consists of Top 20 Parties on the basis of Seats Contested



df_partiwise_seats_comparison = pd.DataFrame(columns=df_exclude_NOTA.columns)



for count,party in enumerate(df['PARTY']):

    if party in top_20_parties:

        df_partiwise_seats_comparison = df_partiwise_seats_comparison.append(df.loc[count],ignore_index=True)
# Comparison of Seats Won and Lost by Parties (TOP 20 PARTIES)



plt.figure(figsize=(17,6))

ax = sns.countplot(x='PARTY',hue='WINNER',data=df_partiwise_seats_comparison,palette='Set1')

ax.set_title('Comparison of Seats Won and Lost by Parties (TOP 20 PARTIES)',fontsize=20)

ax.legend(['Seats Lost','Seats Won'],loc='upper right',frameon=False),

ax.set_xlabel('Political Parties',fontsize=16)

ax.set_ylabel('Number of Seats',fontsize=16)



sns.despine(bottom=True,left=True)
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
# Number of Tickets given by PARTIES to People with CRIMINAL BACKGROUND



ax=df_exclude_NOTA['PARTY'][df_exclude_NOTA['CRIMINAL CASES']!='0'].value_counts().head(20).plot.bar(

figsize=(18,6),

color='red'    

)



ax.set_title('Number of Tickets given by PARTIES to People with CRIMINAL BACKGROUND',fontsize=20)

ax.set_ylabel('Number of Tickets',fontsize=16)



sns.despine(bottom=True,left=True)
def criminal_or_not(value):

    if value !='0':

        criminal_value = 1

    else:

        criminal_value = 0

    return criminal_value
# Creating 1 column in DataFrame named 'CRIMINAL BACKGROUND'



df_exclude_NOTA['CRIMINAL BACKGROUND'] = df_exclude_NOTA['CRIMINAL CASES'].apply(criminal_or_not)

df_exclude_NOTA.head()
# Creating a Series consisting of Names of Top 20 Political Parties having most number of CRIMINAL CANDIDATES



top_20_crim_cand_parties = df_exclude_NOTA['PARTY'][df_exclude_NOTA['CRIMINAL CASES']!='0'].sort_index().value_counts().head(20)

top_20_crim_cand_parties = top_20_crim_cand_parties.index



top_20_crim_cand_parties
# Creating DataFrame consisting of Top 20 Political Parties having most number of CRIMINAL CANDIDATES



df_top_20_criminal_parties = df_exclude_NOTA.copy()



for party,index in zip(df_top_20_criminal_parties['PARTY'],df_top_20_criminal_parties['PARTY'].index):

    if party not in top_20_crim_cand_parties:

        df_top_20_criminal_parties.drop(index=index, inplace=True)
df_exclude_NOTA.head()
# Political Party Candidates CRIMINAL BACKGROUND check (TOP 20)



plt.figure(figsize=(16,5))

ax = sns.countplot(data=df_top_20_criminal_parties,x='PARTY',hue='CRIMINAL BACKGROUND')



ax.legend(['CLEAN IMAGE','CRIMINAL BACKGROUND'],loc='upper right',frameon=False)

ax.set_title('Political Party Candidates CRIMINAL BACKGROUND check (TOP 20)',fontsize=20)

ax.set_ylabel('Number of Candidates',fontsize=16)

ax.set_xlabel('Political Parties',fontsize=16)



sns.despine(bottom=True,left=True)
# Creating DataFrame consisting of only Candidates having CRIMINAL BACKGROUND



df_criminal = pd.DataFrame(columns=df_exclude_NOTA.columns)

df_criminal = df_exclude_NOTA.copy()

df_criminal = df_criminal[df_criminal['CRIMINAL BACKGROUND']==1]



for party,index in zip(df_criminal['PARTY'],df_criminal['PARTY'].index):

    if party not in top_20_crim_cand_parties:

        df_criminal.drop(index=index, inplace=True)
df_criminal.shape
# Comparison of Seats Won and Lost by CRIMINAL CANDIDATES of Parties (TOP 20 PARTIES)



plt.figure(figsize=(17,5))

ax = sns.countplot(x='PARTY',hue='WINNER',data=df_criminal,palette='husl')

ax.set_title('Comparison of Seats Won and Lost by CRIMINAL CANDIDATES of Parties (TOP 20 PARTIES)',fontsize=20)

ax.legend(['Seats Lost','Seats Won'],loc='upper right',frameon=False),

ax.set_xlabel('CRIMINAL CANDIDATES of Political Parties',fontsize=16)

ax.set_ylabel('Number of Seats',fontsize=16)



sns.despine(bottom=True,left=True)
# Total Number of Candidates in BIHAR



ax = sns.countplot(data=df_exclude_NOTA[(df_exclude_NOTA['STATE']=='Bihar')],x='CRIMINAL BACKGROUND',palette='dark')



ax.set_title('Total Number of Candidates in BIHAR',fontsize=20)

ax.set_ylabel('Number of Candidates',fontsize=16)

ax.set_xlabel('        Clean Image         Criminal Background',fontsize=16)



sns.despine(bottom=True,left=True)
# Total Number of Candidates in BIHAR



plt.figure(figsize=(16,5))

ax = sns.countplot(data=df_exclude_NOTA[(df_exclude_NOTA['STATE']=='Bihar')],x='WINNER',hue='CRIMINAL BACKGROUND',palette='colorblind')



ax.set_title('Total Number of Candidates in BIHAR',fontsize=20)

ax.set_ylabel('Number of Candidates',fontsize=16)

ax.legend(['CLEAN IMAGE','CRIMINAL BACKGROUND'],frameon=False)

ax.set_xlabel('LOSING CANDIDATES                                                               WINNING CANDIDATES',fontsize=16)



sns.despine(bottom=True,left=True)
# Total Number of Candidates in Uttar Pradesh



ax = sns.countplot(data=df_exclude_NOTA[(df_exclude_NOTA['STATE']=='Uttar Pradesh')],x='CRIMINAL BACKGROUND',palette='dark')



ax.set_title('Total Number of Candidates in Uttar Pradesh',fontsize=20)

ax.set_ylabel('Number of Candidates',fontsize=16)

ax.set_xlabel('        Clean Image         Criminal Background',fontsize=16)



sns.despine(bottom=True,left=True)
# Total Number of Candidates in Uttar Pradesh



plt.figure(figsize=(16,5))

ax = sns.countplot(data=df_exclude_NOTA[(df_exclude_NOTA['STATE']=='Uttar Pradesh')],x='WINNER',hue='CRIMINAL BACKGROUND',palette='colorblind')



ax.set_title('Total Number of Candidates in Uttar Pradesh',fontsize=20)

ax.set_ylabel('Number of Candidates',fontsize=16)

ax.legend(['CLEAN IMAGE','CRIMINAL BACKGROUND'],frameon=False)

ax.set_xlabel('LOSING CANDIDATES                                                               WINNING CANDIDATES',fontsize=16)



sns.despine(bottom=True,left=True)
# Total Number of Candidates in West Bengal



ax = sns.countplot(data=df_exclude_NOTA[(df_exclude_NOTA['STATE']=='West Bengal')],x='CRIMINAL BACKGROUND',palette='dark')



ax.set_title('Total Number of Candidates in West Bengal',fontsize=20)

ax.set_ylabel('Number of Candidates',fontsize=16)

ax.set_xlabel('        Clean Image         Criminal Background',fontsize=16)



sns.despine(bottom=True,left=True)
# Total Number of Candidates in West Bengal



plt.figure(figsize=(16,5))

ax = sns.countplot(data=df_exclude_NOTA[(df_exclude_NOTA['STATE']=='West Bengal')],x='WINNER',hue='CRIMINAL BACKGROUND',palette='colorblind')



ax.set_title('Total Number of Candidates in West Bengal',fontsize=20)

ax.set_ylabel('Number of Candidates',fontsize=16)

ax.legend(['CLEAN IMAGE','CRIMINAL BACKGROUND'],frameon=False)

ax.set_xlabel('LOSING CANDIDATES                                                               WINNING CANDIDATES',fontsize=16)



sns.despine(bottom=True,left=True)
# Total Number of Candidates in Punjab



plt.figure(figsize=(16,5))

ax = sns.countplot(data=df_exclude_NOTA[(df_exclude_NOTA['STATE']=='Punjab')],x='WINNER',hue='CRIMINAL BACKGROUND',palette='colorblind')



ax.set_title('Total Number of Candidates in Punjab',fontsize=20)

ax.set_ylabel('Number of Candidates',fontsize=16)

ax.legend(['CLEAN IMAGE','CRIMINAL BACKGROUND'],frameon=False)

ax.set_xlabel('LOSING CANDIDATES                                                               WINNING CANDIDATES',fontsize=16)



sns.despine(bottom=True,left=True)
# Political Parties having CANDIDATES ABOVE 70



ax = df_exclude_NOTA['PARTY'][df_exclude_NOTA['AGE']>70].value_counts().plot.bar(

figsize=(17,6),   

color='#EB984E'

)



ax.set_title('Political Parties having CANDIDATES ABOVE 70',fontsize=20)

ax.set_ylabel('Number of Candidates',fontsize=16)

ax.set_xlabel('Political Parties',fontsize=16)



sns.despine(bottom=True,left=True)
# Political Parties having CANDIDATES BELOW 35



ax = df_exclude_NOTA['PARTY'][df_exclude_NOTA['AGE']<35].value_counts().plot.bar(

figsize=(17,6),   

color='#1ABC9C'

)



ax.set_title('Political Parties having CANDIDATES BELOW 35',fontsize=20)

ax.set_ylabel('Number of Candidates',fontsize=16)

ax.set_xlabel('Political Parties',fontsize=16)



sns.despine(bottom=True,left=True)
df_exclude_NOTA.sort_values(by='AGE',ascending=False).head()
df_exclude_NOTA[df_exclude_NOTA['WINNER']==1].sort_values(by='AGE',ascending=False).head()
df_exclude_NOTA[df_exclude_NOTA['WINNER']==1].sort_values(by='AGE').head()
# Number of MALES and FEMALES contesting Election



ax = sns.countplot(data=df_exclude_NOTA,x='GENDER',palette='cubehelix')



ax.set_title('Number of MALES and FEMALES contesting Election',fontsize=20)

sns.despine(bottom=True,left=True)
# Comparison of Seats WON and LOST by Candidates



ax = sns.countplot(data=df_exclude_NOTA,x='GENDER',hue='WINNER',palette='hls')



ax.legend(['seats lost','seats won'],frameon=False)

ax.set_title('Comparison of Seats WON and LOST by Candidates',fontsize=20)

sns.despine(bottom=True,left=True)
# Elected FEMALE PARLIAMENTARIANS - State Wise



ax = df_exclude_NOTA['STATE'][(df_exclude_NOTA['GENDER']=='FEMALE')&(df_exclude_NOTA['WINNER']==1)].value_counts().plot.bar(

figsize=(16,5),

color='#AF7AC5'

)



ax.set_title('Elected FEMALE PARLIAMENTARIANS - State Wise',fontsize=20)

ax.set_ylabel('Number of Elected Parliamentarians',fontsize=16)

ax.set_xlabel('States',fontsize=16)



sns.despine(bottom=True,left=True)
# Seats won by BJP - Statewise



ax = df_exclude_NOTA['STATE'][(df_exclude_NOTA['PARTY']=='BJP')&(df_exclude_NOTA['WINNER']==1)].value_counts().plot.bar(

figsize=(16,5),

color ='#FBA21C'

)



ax.set_title('Seats won by BJP - Statewise',fontsize=20)

ax.set_ylabel('Number of Seats',fontsize=16)

ax.set_xlabel('States',fontsize=16)



sns.despine(bottom=True,left=True)
# Seats won by Congress - Statewise



ax = df_exclude_NOTA['STATE'][(df_exclude_NOTA['PARTY']=='INC')&(df_exclude_NOTA['WINNER']==1)].value_counts().plot.bar(

figsize=(16,5),

color ='#29AA2C'

)



ax.set_title('Seats won by Congress - Statewise',fontsize=20)

ax.set_ylabel('Number of Seats',fontsize=16)

ax.set_xlabel('States',fontsize=16)



sns.despine(bottom=True,left=True)
# Seats in Uttar Pradesh



plt.figure(figsize=(8,5))

ax = sns.countplot(data=df_exclude_NOTA[(df_exclude_NOTA['STATE']=='Uttar Pradesh')&(df_exclude_NOTA['WINNER']==1)],x='PARTY')



ax.set_title('Seats in Uttar Pradesh',fontsize=20)

ax.set_ylabel('Number of Seats',fontsize=16)

ax.set_xlabel('Political Parties',fontsize=16)



sns.despine(bottom=True,left=True)
# Seats in West Bengal



plt.figure(figsize=(8,5))

ax = sns.countplot(data=df_exclude_NOTA[(df_exclude_NOTA['STATE']=='West Bengal')&(df_exclude_NOTA['WINNER']==1)],x='PARTY')



ax.set_title('Seats in West Bengal',fontsize=20)

ax.set_ylabel('Number of Seats',fontsize=16)

ax.set_xlabel('Political Parties',fontsize=16)



sns.despine(bottom=True,left=True)
# Seats in Maharashtra



plt.figure(figsize=(8,5))

ax = sns.countplot(data=df_exclude_NOTA[(df_exclude_NOTA['STATE']=='Maharashtra')&(df_exclude_NOTA['WINNER']==1)].sort_values(by='WINNER'),x='PARTY')



ax.set_title('Seats in Maharashtra',fontsize=20)

ax.set_ylabel('Number of Seats',fontsize=16)

ax.set_xlabel('Political Parties',fontsize=16)



sns.despine(bottom=True,left=True)
# Seats in Bihar



plt.figure(figsize=(8,5))

ax = sns.countplot(data=df_exclude_NOTA[(df_exclude_NOTA['STATE']=='Bihar')&(df_exclude_NOTA['WINNER']==1)].sort_values(by='WINNER'),x='PARTY')



ax.set_title('Seats in Bihar',fontsize=20)

ax.set_ylabel('Number of Seats',fontsize=16)

ax.set_xlabel('Political Parties',fontsize=16)



sns.despine(bottom=True,left=True)
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
df1 = df[['STATE','CONSTITUENCY','WINNER','PARTY','SYMBOL','GENDER','CRIMINAL CASES','AGE','CATEGORY','EDUCATION','TOTAL VOTES','TOTAL ELECTORS','ASSETS_RANGE','LIABILITY_RANGE']]

cat_cols = ['STATE','CONSTITUENCY','PARTY','SYMBOL','GENDER','CATEGORY','EDUCATION','ASSETS_RANGE','LIABILITY_RANGE']

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