# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import os, sys

from collections import defaultdict

from urllib.request import urlopen

import json

import plotly.graph_objects as go

from plotly.subplots import make_subplots

from ipywidgets import widgets

import geopandas as gpd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import random

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.colors import n_colors

from plotly.subplots import make_subplots

init_notebook_mode(connected=True)

import cufflinks as cf

cf.go_offline()

from wordcloud import WordCloud , ImageColorGenerator

from PIL import Image

from sklearn.utils import resample

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
vote=pd.read_csv('/kaggle/input/indian-candidates-for-general-election-2019/LS_2.0.csv')

vote.head()
vote.isnull().sum()
vote[vote.SYMBOL.isnull()==True]['NAME'].unique()
def value_cleaner(x):

    try:

        str_temp = (x.split('Rs')[1].split('\n')[0].strip())

        str_temp_2 = ''

        for i in str_temp.split(","):

            str_temp_2 = str_temp_2+i

        return str_temp_2

    except:

        x = 0

        return x

vote['ASSETS'] = vote['ASSETS'].apply((value_cleaner))

vote['LIABILITIES'] = vote['LIABILITIES'].apply((value_cleaner))

vote.head()
vote.rename(columns={"CRIMINAL\nCASES": "CRIMINAL CASES", "GENERAL\nVOTES": "GENERAL VOTES", "POSTAL\nVOTES": "POSTAL VOTES","TOTAL\nVOTES": "TOTAL VOTES","OVER TOTAL ELECTORS \nIN CONSTITUENCY": "OVER TOTAL ELECTORS IN CONSTITUENCY","OVER TOTAL VOTES POLLED \nIN CONSTITUENCY": "OVER TOTAL VOTES POLLED IN CONSTITUENCY"}, inplace=True)

vote.head()
vote.EDUCATION.unique()
vote.EDUCATION.replace({'Post Graduate\n':'Post Graduate'},inplace=True)

vote.EDUCATION.unique()
vote.dtypes
vote[vote['CRIMINAL CASES']=='Not Available'].head()
vote['ASSETS']=pd.to_numeric(vote['ASSETS'])

vote['LIABILITIES']=pd.to_numeric(vote['LIABILITIES'])

vote['CRIMINAL CASES'].replace({np.NaN:0})

vote['CRIMINAL CASES'] = pd.to_numeric(vote['CRIMINAL CASES'], errors='coerce').fillna(0).astype(np.int64)
st_con=vote.groupby('STATE').apply(lambda x:x['CONSTITUENCY'].nunique()).reset_index(name='# Constituency')

shp_gdf = gpd.read_file('/kaggle/input/india-states/Igismap/Indian_States.shp')

merged = shp_gdf.set_index('st_nm').join(st_con.set_index('STATE'))

fig, ax = plt.subplots(1, figsize=(10, 10))

ax.axis('off')

ax.set_title('State-wise Distribution of Indian Constituencies',

             fontdict={'fontsize': '15', 'fontweight' : '3'})

fig = merged.plot(column='# Constituency', cmap='inferno_r',linewidth=0.5, ax=ax, edgecolor='0.2',legend=True)

st_con.sort_values(by='# Constituency',ascending=False,inplace=True)

fig2 = px.bar(st_con, x='STATE', y='# Constituency',

                     color='# Constituency',

             labels={'pop':'Constituencies of India'})

fig2.update_layout(title_text='Statewise distribution of the Constituencies all over India',template='plotly_dark')

fig2.show()
st_con_vt=vote[['STATE','CONSTITUENCY','TOTAL ELECTORS']]

fig = px.sunburst(st_con_vt, path=['STATE','CONSTITUENCY'], values='TOTAL ELECTORS',

                  color='TOTAL ELECTORS',

                  color_continuous_scale='viridis_r')

fig.update_layout(title_text='Sunburst Image of State and Constituency by Voters',template='plotly_dark')

fig.show()
vote_prty=vote[vote['PARTY']!='NOTA']

prty_cnt=vote_prty.groupby('PARTY').apply(lambda x:x['CONSTITUENCY'].count()).reset_index(name='# Constituency')

prty_st=vote_prty.groupby('PARTY').apply(lambda x:x['STATE'].nunique()).reset_index(name='# State')

prty_cnt.sort_values(by='# Constituency',ascending=False,inplace=True)

prty_top_cn=prty_cnt[:25]

prty_top_all=pd.merge(prty_top_cn,prty_st,how='inner',left_on='PARTY',right_on='PARTY')

fig = px.scatter(prty_top_all, x='# Constituency', y='# State', color='# State',

                 size='# Constituency', hover_data=['PARTY'])

fig.update_layout(title_text='Constituency vs Statewise participation for the most contesting Political Parties',template='plotly_dark')

fig.show()
st_prty=vote_prty.groupby(['PARTY','STATE']).apply(lambda x:x['WINNER'].sum()).reset_index(name='Wins')

pvt_st_prty=pd.pivot(st_prty,index='PARTY',columns='STATE',values='Wins')

plt.figure(figsize=(15,35))

sns.heatmap(pvt_st_prty,annot=True,fmt='g',cmap='terrain')

plt.xlabel('States',size=20)

plt.ylabel('Party',size=20)

plt.title('Statewise report card for the Political Parties in India',size=25)
part_win=vote.groupby('PARTY').apply(lambda x:x['WINNER'].sum()).reset_index(name='# Wins')

part_win.sort_values(by='# Wins',ascending=False,inplace=True)

top_part_win=part_win[0:15]

fig = px.bar(top_part_win, x='PARTY', y='# Wins',

                     color='# Wins',title='Win Counts by a Political Party in 2019')

fig.update_layout(title_text='Win Counts by a Political Party in 2019',template='plotly_dark')

fig.show()
prty_cnt_win=pd.merge(prty_cnt,part_win,how='inner',left_on='PARTY',right_on='PARTY')

prty_cnt_win['Lost']=prty_cnt_win['# Constituency']-prty_cnt_win['# Wins']

prty_wins_cnt=prty_cnt_win[['PARTY','# Wins']]

prty_wins_cnt['Verdict']='Constituency Won'

prty_loss_cnt=prty_cnt_win[['PARTY','Lost']]

prty_loss_cnt['Verdict']='Constituency Lost'

prty_wins_cnt.columns=['Party','Counts','Verdict']

prty_loss_cnt.columns=['Party','Counts','Verdict']

top_prty_wins_cnt=prty_wins_cnt[:15]

prty_loss_cnt_cnt=prty_loss_cnt[:15]

prt_win_loss=pd.concat([top_prty_wins_cnt,prty_loss_cnt_cnt])

fig = px.bar(prt_win_loss, x='Party', y='Counts', color='Verdict')

fig.update_layout(title_text='Win vs Loss Analysis for the Top Parties',template='plotly_dark')

fig.show()
vote_gndr=vote[vote['PARTY']!='NOTA']

gndr_overall=vote_gndr.groupby('GENDER').apply(lambda x:x['NAME'].count()).reset_index(name='Counts')

gndr_overall['Category']='Overall Gender Ratio'

winners=vote_gndr[vote_gndr['WINNER']==1]

gndr_winner=winners.groupby('GENDER').apply(lambda x:x['NAME'].count()).reset_index(name='Counts')

gndr_winner['Category']='Winning Gender Ratio'

gndr_overl_win=pd.concat([gndr_winner,gndr_overall])

fig = px.bar(gndr_overl_win, x='GENDER', y='Counts',

             color='Category', barmode='group')

fig.update_layout(title_text='Participation vs Win Counts analysis for the Genders',template='plotly_dark')

fig.show()
ed_valid=vote[vote['PARTY']!="NOTA"]

ed_cnt=ed_valid.groupby('EDUCATION').apply(lambda x:x['PARTY'].count()).reset_index(name='Counts')

fig = go.Figure(data=[go.Pie(labels=ed_cnt['EDUCATION'], values=ed_cnt['Counts'], pull=[0.1, 0.2, 0, 0.1, 0.2, 0,0.1, 0.2, 0,0.1, 0.2, 0.1])])

fig.update_layout(title_text='Overall Education Qualification of all the Nominees',template='plotly_dark')

fig.show()

ed_won=ed_valid[ed_valid['WINNER']==1]

ed_win_cnt=ed_won.groupby('EDUCATION').apply(lambda x:x['PARTY'].count()).reset_index(name='Counts')

fig2 = go.Figure(data=[go.Pie(labels=ed_win_cnt['EDUCATION'], values=ed_win_cnt['Counts'], pull=[0.1, 0.2, 0, 0.1, 0.2, 0,0.1, 0.1, 0.2,0, 0.1, 0.2],title='Education Qualification of the Winners')])

fig2.update_layout(title_text='Education Qualification of the Winners',template='plotly_dark')

fig2.show()
age_cnt=ed_valid.groupby(['AGE','GENDER']).apply(lambda x:x['NAME'].count()).reset_index(name='Counts')

fig = px.histogram(age_cnt, x="AGE",y='Counts',color='GENDER',marginal='violin',title='Age Counts Distribution among the politicians')

fig.update_layout(title_text='Age Counts Distribution among the politicians',template='plotly_dark')

fig.show()
vote_cat=vote[vote['PARTY']!='NOTA']

cat_overall=vote_cat.groupby('CATEGORY').apply(lambda x:x['NAME'].count()).reset_index(name='Counts')

cat_overall['Category']='Overall Category Counts'

winners_cat=vote_gndr[vote_gndr['WINNER']==1]

cat_winner=winners_cat.groupby('CATEGORY').apply(lambda x:x['NAME'].count()).reset_index(name='Counts')

cat_winner['Category']='Winning Category Ratio'

cat_overl_win=pd.concat([cat_winner,cat_overall])

fig = px.bar(cat_overl_win, x='CATEGORY', y='Counts',

             color='Category', barmode='group')

fig.update_layout(title_text='Participation vs Win Counts for the Category in Politics',template='plotly_dark')

fig.show()
crim_cnt=ed_valid.groupby('CRIMINAL CASES').apply(lambda x:x['NAME'].count()).reset_index(name='Counts')

fig = px.histogram(crim_cnt, x='CRIMINAL CASES',y='Counts',marginal='violin')

fig.update_layout(title_text='Criminal Cases Counts Distribution among the politicians',template='plotly_dark')

fig.show()
as_liab_name=ed_valid[['NAME','PARTY','ASSETS','LIABILITIES','STATE','CONSTITUENCY','WINNER']]

as_liab_name.WINNER.replace({1:'Yes',0:'No'},inplace=True)

win_as_liab_name=as_liab_name[as_liab_name['WINNER']=='Yes']

win_as_liab_name.sort_values(by='ASSETS',ascending=False,inplace=True)

fig = px.scatter(win_as_liab_name, x='ASSETS', y='LIABILITIES', 

                 color='STATE',size='ASSETS', 

                 hover_data=(['NAME','PARTY','CONSTITUENCY','STATE','WINNER']),

                 title='Assets vs Liabilities for the Winning Politicians')

fig.update_layout(title_text='Assets vs Liabilities for the Winning Politicians',template='plotly_dark')

fig.show()
vote_df=vote[vote['PARTY']!='NOTA']

vote_df['GENDER'].replace({'MALE':1,'FEMALE':2},inplace=True)

vote_df['CATEGORY'].replace({'GENERAL':1,'SC':2,'ST':3},inplace=True)

i=1

parties_dict={}

for j in vote_df['PARTY']:

    if j in parties_dict:

        continue

    else:

        parties_dict[j]=i

        i+=1

vote_df['PARTY'].replace(parties_dict,inplace=True)

a=1

edu_dict={}

for b in vote_df['EDUCATION']:

    if b in edu_dict:

        continue

    else:

        edu_dict[b]=a

        a+=1

vote_df['EDUCATION'].replace(edu_dict,inplace=True)

df1 = vote_df[['STATE','CONSTITUENCY','WINNER','PARTY','SYMBOL','GENDER','CRIMINAL CASES','AGE','CATEGORY','EDUCATION','TOTAL VOTES','TOTAL ELECTORS','ASSETS','LIABILITIES']]

num_cols = ['PARTY','EDUCATION','CRIMINAL CASES','AGE','TOTAL VOTES','TOTAL ELECTORS','ASSETS','CATEGORY','LIABILITIES','GENDER']

dataset = pd.get_dummies(df1)

from sklearn.preprocessing import StandardScaler

standardScaler = StandardScaler()

scaling_columns = num_cols

dataset[scaling_columns] = standardScaler.fit_transform(dataset[scaling_columns])

dataset.head()
df_not_winner = dataset[dataset.WINNER == 0]

df_winner = dataset[dataset.WINNER == 1]

df_winner_upsampled = resample(df_winner, replace = True,n_samples = 1452, random_state = 0) 

df_total_upsampled = pd.concat([df_not_winner, df_winner_upsampled])

df_total_upsampled.WINNER.value_counts()

y = df_total_upsampled['WINNER']

X = df_total_upsampled.drop(['WINNER'], axis = 1)

rf_scores = []

for k in range(1,60):

    randomforest_classifier= RandomForestClassifier(n_estimators=k,random_state=0)

    score=cross_val_score(randomforest_classifier,X,y,cv=10)

    rf_scores.append(score.mean())

fig=px.scatter(x=[k for k in range(1, 60)],y= rf_scores,color=rf_scores,size=rf_scores)

fig.update_layout(title_text='Assets vs Liabilities for the Winning Politicians',template='plotly_dark')

fig.show()
randomforest_classifier= RandomForestClassifier(n_estimators=38,random_state=0)

score=cross_val_score(randomforest_classifier,X,y,cv=10)

print('% Accuracy :', round(score.mean()*100,4))