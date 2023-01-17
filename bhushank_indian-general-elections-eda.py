import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# import plotly.graph_objects as go

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
files = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        files.append(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv(files[0])
df.head()
df.describe(include = 'all')
df.isnull().sum()
df.groupby('PARTY').count()['NAME'].sort_values(ascending=False)
df_NOTA = df[df['PARTY']!='NOTA']
constituencies = df.groupby('STATE')['CONSTITUENCY'].nunique().to_frame().sort_values('CONSTITUENCY',ascending = False).reset_index()
# plt.style.use('ggplot')
# constituencies.plot(kind = 'bar', 
#                     figsize=(10,6),
#                     width = 0.8)
constituencies
px.bar(constituencies, x = 'STATE', y = 'CONSTITUENCY')
party_stats = df_NOTA['PARTY'].value_counts().to_frame().reset_index().rename(columns = {'index':'PARTY','PARTY':'Total Candidates'})#.sort_values('PARTY',ascending = False)
#party_stats[party_stats>100]

party_stats
# plt.style.use('seaborn')
# party_stats.head(60).plot(kind = 'bar', 
#                     figsize=(15,15))

px.bar(party_stats, x = 'PARTY', y = 'Total Candidates')
states_with_highest_candidates_per_const = df_NOTA[['STATE', 'CONSTITUENCY', 'NAME']].groupby(['STATE','CONSTITUENCY']).count().groupby('STATE').mean().sort_values(by = 'NAME', ascending = 0)
states_with_highest_candidates_per_const = states_with_highest_candidates_per_const.reset_index()
states_with_highest_candidates_per_const
px.bar(states_with_highest_candidates_per_const, x = 'STATE', y = 'NAME')

# states_with_highest_candidates_per_const.plot(kind = 'bar',x = 'STATE', y = 'NAME')
# df_NOTA
voteshare = df_NOTA[['PARTY','TOTAL\nVOTES']].groupby('PARTY').sum()
# voteshare['share']
voteshare = voteshare.sort_values('TOTAL\nVOTES', ascending = 0).reset_index()
# voteshare

voteshare['ConsolidatedParty'] = [voteshare.iloc[x]['PARTY'] if x <= 10 else 'OTHER' for x in voteshare.index.values] 
  
voteshare = voteshare[['ConsolidatedParty','TOTAL\nVOTES']].groupby('ConsolidatedParty').sum().reset_index().sort_values('TOTAL\nVOTES', ascending = 0).reset_index()
voteshare['Voter Percentage Share'] = voteshare['TOTAL\nVOTES'] * 100 / voteshare['TOTAL\nVOTES'].sum(axis = 0)
# for index, row in voteshare.iterrows():
#     if index <= 4:
#         row['ConsolidatedParty'] = row['PARTY'] 
#     else:
#         row['ConsolidatedParty'] = 'OTHER'

sns.set_context('notebook')

sns.set_palette('pastel')
sns.set_style('whitegrid')
sns.set_style('ticks',
              {'x_ticks.major.size':8,
               'y_ticks.major.size':8})
voteshare.plot(kind = 'bar', 
               x = 'ConsolidatedParty', 
               y = 'Voter Percentage Share',
              figsize = (10,10))
tmp = df_NOTA[df_NOTA['WINNER'] == 1].groupby('PARTY').count().reset_index()[['PARTY','WINNER']].sort_values(by = 'WINNER', ascending = False)
px.pie(tmp, values='WINNER', names='PARTY', title='Elected Candidates by Party')

df_NOTA['CRIMINAL\nCASES'] = df_NOTA['CRIMINAL\nCASES'].replace({'Not Available': 0}).astype(int)

def update_education(row):
    if row['EDUCATION'] in ['Illiterate','Not Available']:
        return 'Iliterate/NA'
    if row['EDUCATION'] in ['5th Pass','8th Pass','10th Pass','12th Pass']:
        return 'School/College Level'
    elif row['EDUCATION'] in ['Graduate Professional','Graduate']: 
        return 'Graduate'
    elif row['EDUCATION'] in ['Post Graduate\n','Post Graduate', 'Doctorate']:
        return 'Post Graduate/Doctorate'
    else:
        'Literate'
        
df_NOTA['EDUCATION_NEW'] = df_NOTA.apply(update_education,axis =1)
tmp = df_NOTA.groupby('EDUCATION_NEW').count()['WINNER'].reset_index().sort_values(by = 'WINNER', ascending = False)
# tmp.plot(kind = 'bar', x = 'EDUCATION_NEW')
sns.set_context('notebook')

sns.set_style('whitegrid')
sns.set_style('ticks',
              {'x_ticks.major.size':8,
               'y_ticks.major.size':8})
sns.set(rc={'figure.figsize':(10,8)})
sns.set_palette('pastel')

ax = sns.barplot(x = 'EDUCATION_NEW', 
            y = 'WINNER',
            data = tmp)


ax.set(xlabel = 'Education')
temp = df_NOTA.groupby(['AGE','EDUCATION_NEW']).mean()['CRIMINAL\nCASES'].to_frame().reset_index()

education_ranking = ['Illiterate',
                     'School/College Level',
                     'Graduate',
                     'Post Graduate/Doctorate',
                     'Literate']


f, ax = plt.subplots(figsize=(10,10))
# ax.set_ylim(0,50)
sns.lineplot(x="AGE", y="CRIMINAL\nCASES",
                hue="EDUCATION_NEW", #size="depth",
#                 palette="pastel",
                hue_order=education_ranking,
                data=temp,
                ax = ax
#                 marker="+"
               )
sns.catplot(x="GENDER",
            y="AGE", 
            hue="WINNER", 
            kind="box", data=df_NOTA);

# sns.distplot(df_NOTA['AGE'], 
#              kde=False, 
#              color=df_NOTA['GENDER'])
# plt.figure(figsize=(15,50))
g = sns.catplot(x="CATEGORY", 
            kind="count", 
            palette="ch:.25",
            hue="WINNER",
            data=df_NOTA,
            legend_out=True
           )
# plt.show()
g.fig.set_size_inches(6, 4)
g.set_axis_labels("Category", "Number of Candidates")