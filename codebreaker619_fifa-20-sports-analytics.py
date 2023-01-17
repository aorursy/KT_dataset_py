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



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

import plotly.express as px

from plotly.offline import init_notebook_mode, iplot, plot

from plotly.subplots import make_subplots

from collections import Counter

from math import sqrt

from random import randrange

import os

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC



init_notebook_mode(connected=True)

pd.set_option('display.max_columns', None)

pd.set_option('precision', 0)

pd.options.mode.chained_assignment = None
df_20 = pd.read_csv("/kaggle/input/fifa-20-complete-player-dataset/players_20.csv")

df_19 = pd.read_csv("/kaggle/input/fifa-20-complete-player-dataset/players_19.csv")

df_18 = pd.read_csv("/kaggle/input/fifa-20-complete-player-dataset/players_18.csv")

df_17 = pd.read_csv("/kaggle/input/fifa-20-complete-player-dataset/players_17.csv")

df_16 = pd.read_csv("/kaggle/input/fifa-20-complete-player-dataset/players_16.csv")

df_15 = pd.read_csv("/kaggle/input/fifa-20-complete-player-dataset/players_15.csv")



df_20.head()
df_20.shape
for i in df_20.columns:

    print(i)
df_20.describe()
df_20.isnull().sum()
missing_data = df_20.isnull().sum().sort_values(ascending=False)

missing_data = missing_data.reset_index(drop = False)

missing_data = missing_data.rename(columns={"index": "Columns", 0:"Value"})

missing_data['proportion'] = (missing_data['Value']/len(df_20))*100

missing_data.head()
sample = missing_data[missing_data['proportion']>10]

fig = px.pie(sample, names='Columns', values='proportion',

             color_discrete_sequence=px.colors.sequential.Viridis_r,

             title='Percentage of Missing values in Columns')

fig.update_traces(textposition='inside', textinfo='label')

fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',

                  plot_bgcolor='rgba(0,0,0,0)',

                  font=dict(family='Cambria, monospace', size=12, color='#000000'))

fig.show()
df_20.info()
df_20['release_clause_eur'].fillna(0,inplace=True)

df_20['player_tags'].fillna('#Team Player',inplace=True)

df_20['team_position'].fillna('Not Decided',inplace=True)                                  

df_20['team_jersey_number'].fillna(0,inplace=True)

df_20['loaned_from'].fillna('Disclosed',inplace=True)

df_20['joined'].fillna('Disclosed',inplace=True)

df_20['contract_valid_until'].fillna('Disclosed',inplace=True)

df_20['pace'].fillna(df_20['pace'].mean(),inplace=True)

df_20['shooting'].fillna(df_20['shooting'].mean(),inplace=True)

df_20['passing'].fillna(df_20['passing'].mean(),inplace=True)

df_20['dribbling'].fillna(df_20['dribbling'].mean(),inplace=True)

df_20['defending'].fillna(df_20['defending'].mean(),inplace=True)

df_20['physic'].fillna(df_20['physic'].mean(),inplace=True)

df_20['gk_diving'].fillna(df_20['gk_diving'].mean(),inplace=True)

df_20['gk_handling'].fillna(df_20['gk_handling'].mean(),inplace=True)

df_20['gk_kicking'].fillna(df_20['gk_kicking'].mean(),inplace=True)

df_20['gk_reflexes'].fillna(df_20['gk_reflexes'].mean(),inplace=True)

df_20['gk_speed'].fillna(df_20['gk_speed'].mean(),inplace=True)

df_20['gk_positioning'].fillna(df_20['gk_positioning'].mean(),inplace=True)

df_20['player_traits'].fillna('NA',inplace=True)

df_20['ls'].fillna('NA',inplace=True)

df_20['st'].fillna('NA',inplace=True)

df_20['rs'].fillna('NA',inplace=True)

df_20['lw'].fillna('NA',inplace=True)

df_20['lf'].fillna('NA',inplace=True)

df_20['cf'].fillna('NA',inplace=True)

df_20['rf'].fillna('NA',inplace=True)

df_20['rw'].fillna('NA',inplace=True)

df_20['lam'].fillna('NA',inplace=True)

df_20['cam'].fillna('NA',inplace=True)

df_20['ram'].fillna('NA',inplace=True)

df_20['lm'].fillna('NA',inplace=True)

df_20['lcm'].fillna('NA',inplace=True)

df_20['cm'].fillna('NA',inplace=True)

df_20['rcm'].fillna('NA',inplace=True)

df_20['rm'].fillna('NA',inplace=True)

df_20['lwb'].fillna('NA',inplace=True)

df_20['ldm'].fillna('NA',inplace=True)

df_20['cdm'].fillna('NA',inplace=True)

df_20['rdm'].fillna('NA',inplace=True)

df_20['rwb'].fillna('NA',inplace=True)

df_20['lb'].fillna('NA',inplace=True)

df_20['lcb'].fillna('NA',inplace=True)

df_20['cb'].fillna('NA',inplace=True)

df_20['rcb'].fillna('NA',inplace=True)

df_20['rb'].fillna('NA',inplace=True)
df_20.drop(['sofifa_id','player_url','real_face','nation_position','nation_jersey_number','long_name'], axis=1, inplace=True)

df_20.head()
df_20.shape
df_20.club.unique()
len(df_20.club.unique())
plt.figure(figsize=(18,8))

plt.title('Age Distribution FIFA 20', fontsize=20)

sns.distplot(a=df_20['age'], kde=True, bins=20)
plt.figure(dpi=125)

sns.distplot(a=df_20['age'],kde=False,bins=20)

plt.axvline(x=np.mean(df_20['age']),c='green',label='Mean Age of All Players')

plt.legend()

plt.xlabel('Age')

plt.ylabel('Count')

plt.title('Distribution of Age')

plt.show()
plt.figure(figsize = (15, 7))

df_20['age'].plot(kind="hist", color="orangered")

plt.title("Age Distribution(Histogram)", fontsize=20)

plt.xlabel("Age")

plt.show()
plt.figure(figsize= (15,7))



ax = sns.countplot(x='age', data=df_20, palette='bright')

ax.set_title(label='Count of Players on Basis of Age in FIFA 20', fontsize=20)



ax.set_xlabel(xlabel='Age', fontsize=16)

ax.set_ylabel(ylabel='Count', fontsize=16)



plt.show()
print(df_20.nationality.unique())
print(len(df_20.nationality.unique()))
plt.figure(figsize= (15, 7))



ax = sns.countplot(x='nationality', data=df_20, palette='bright', order=df_20.nationality.value_counts().iloc[:10].index)

ax.set_title(label='Count of Players on Basis of Nationality in FIFA 20', fontsize=20)



ax.set_xlabel(xlabel='Nationality', fontsize=16)

ax.set_ylabel(ylabel='Count', fontsize=16)

plt.xticks(rotation=30, fontsize=12)



plt.show()
fig = go.Figure()



fig = go.Figure(data=go.Scatter(

    x = df_20['nationality'],

    y = df_20['overall'],

    mode='markers',

    marker=dict(

        size=16,

        color=df_20['overall'], #set color equal to a variable

        colorscale='Plasma', # one of plotly colorscales

        showscale=True

    ),

    text= df_20['short_name'],

))



fig.update_layout(title='Nationality vs Overall Rating',

                  xaxis_title='Nationality',

                  yaxis_title='Overall Rating',

                  paper_bgcolor='rgba(0,0,0,0)',

                  plot_bgcolor='rgba(0,0,0,0)',

                  font=dict(size=12, color='#000000'))

fig.show()
plt.figure(figsize=(18,8))

plt.title('Height(in centimeters) Distribution FIFA 20', fontsize=20)

sns.distplot(a=df_20['height_cm'], kde=True, bins=20)
plt.figure(dpi=125)

sns.distplot(a=df_20['height_cm'],kde=False,bins=20)

plt.axvline(x=np.mean(df_20['height_cm']),c='green',label='Mean Height of All Players')

plt.legend()

plt.xlabel('Height(in cm)')

plt.ylabel('Count')

plt.title('Distribution of Height')

plt.show()
plt.figure(figsize = (15, 7))

df_20['height_cm'].plot(kind="hist", color="red")

plt.title("Height(Histogram)", fontsize=20)

plt.xlabel("Height(in cm)")

plt.show()
plt.figure(figsize= (15, 7))



ax = sns.countplot(x='height_cm', data=df_20, palette='bright', order=df_20.height_cm.value_counts().iloc[:20].index)

ax.set_title(label='Count of Players on Basis of Height(in cm) in FIFA 20', fontsize=20)



ax.set_xlabel(xlabel='Height(in centimeters)', fontsize=16)

ax.set_ylabel(ylabel='Count', fontsize=16)

plt.xticks(rotation=30, fontsize=12)



plt.show()
plt.figure(dpi=125)

x=df_20.head(20)['height_cm']

y=df_20.head(20)['pace']



sns.regplot(x,y)

plt.title('Height v Pace')

plt.xlabel('Height')

plt.ylabel('Pace')

plt.show()
plt.figure(figsize=(18,8))

plt.title('Weight(in kg) Distribution FIFA 20', fontsize=20)

sns.distplot(a=df_20['weight_kg'], kde=True, bins=20)
plt.figure(dpi=125)

sns.distplot(a=df_20['weight_kg'],kde=False,bins=20)

plt.axvline(x=np.mean(df_20['weight_kg']),c='green',label='Mean Weight of All Players')

plt.legend()

plt.xlabel('Weight(in kg)')

plt.ylabel('Count')

plt.title('Distribution of Weight')

plt.show()
plt.figure(figsize = (15, 7))

df_20['weight_kg'].plot(kind="hist", color="green")

plt.title("Weights(Histogram)", fontsize=20)

plt.xlabel("Weight")

plt.show()
plt.figure(figsize= (15, 7))



ax = sns.countplot(x='weight_kg', data=df_20, palette='bright', order=df_20.weight_kg.value_counts().iloc[:20].index)

ax.set_title(label='Count of Players on Basis of Weight(in kg) in FIFA 20', fontsize=20)



ax.set_xlabel(xlabel='Weight(in kg)', fontsize=16)

ax.set_ylabel(ylabel='Count', fontsize=16)

plt.xticks(rotation=30, fontsize=12)



plt.show()
plt.figure(dpi=125)

x=df_20.head(20)['weight_kg']

y=df_20.head(20)['pace']



sns.regplot(x,y)

plt.title('Weight v Pace')

plt.xlabel('Weight')

plt.ylabel('Pace')

plt.show()
plt.figure(dpi=125)

x=df_20.head(20)['height_cm']

y=df_20.head(20)['weight_kg']



sns.regplot(x,y)

plt.title('Height v Weight')

plt.xlabel('Height')

plt.ylabel('Weight')

plt.show()
df_20['bmi'] = df_20['weight_kg'] // (df_20['height_cm']/100)**2

df_20['bmi'].head(20)
plt.figure(figsize= (15, 7))



ax = sns.countplot(x='bmi', data=df_20, palette='bright', order=df_20.bmi.value_counts().iloc[:20].index)

ax.set_title(label='Count of Players on Basis of BMI(Body Mass Index) in FIFA 20', fontsize=20)



ax.set_xlabel(xlabel='BMI(Body Mass Index)', fontsize=16)

ax.set_ylabel(ylabel='Count', fontsize=16)

plt.xticks(rotation=30, fontsize=12)



plt.show()
plt.figure(dpi=125)

x=df_20.head(20)['bmi']

y=df_20.head(20)['height_cm']



sns.regplot(x,y)

plt.title('BMI v Height')

plt.xlabel('BMI')

plt.ylabel('Height')

plt.show()
plt.figure(dpi=125)

x=df_20.head(20)['bmi']

y=df_20.head(20)['weight_kg']



sns.regplot(x,y)

plt.title('BMI v Weight')

plt.xlabel('BMI')

plt.ylabel('Weight')

plt.show()
plt.figure(figsize=(40, 20))

plt.scatter(df_20['age'], df_20['overall'], marker="x")

plt.ylabel('Overall Rating', fontsize=40)

plt.xlabel("Age", fontsize=40)

plt.title("Relationship between Overall Rating and Age", fontsize=40)

plt.show()
plt.figure(figsize=(18,8))

plt.title('Overall Rating Distribution FIFA 20', fontsize=20)

sns.distplot(a=df_20['overall'], kde=True, bins=20)
plt.figure(dpi=125)

sns.distplot(a=df_20['overall'],kde=False,bins=20)

plt.axvline(x=np.mean(df_20['overall']),c='green',label='Mean Overall Rating of All Players')

plt.legend()

plt.xlabel('Overall Rating')

plt.ylabel('Count')

plt.title('Distribution of Overall Rating')

plt.show()
plt.figure(figsize=(15, 7))

sns.countplot(df_20['overall'])

plt.title("Overall Rating", fontsize=20)

plt.show()
plt.figure(figsize = (15, 7))

df_20['overall'].plot(kind="hist", color="purple")

plt.title("Overall Rating(Histogram)", fontsize=20)

plt.xlabel("Overall")

plt.show()
plt.figure(figsize= (15, 7))



ax = sns.countplot(x='overall', data=df_20, palette='bright', order=df_20.overall.value_counts().iloc[:20].index)

ax.set_title(label='Count of Players on Basis of Overall Rating in FIFA 20(Top 20)', fontsize=20)



ax.set_xlabel(xlabel='Overall Rating', fontsize=16)

ax.set_ylabel(ylabel='Count', fontsize=16)

plt.xticks(rotation=30, fontsize=12)



plt.show()
plt.figure(figsize= (15, 7))



ax = sns.countplot(x='overall', data=df_20, palette='bright', order=df_20.overall.value_counts().iloc[-20:].index)

ax.set_title(label='Count of Players on Basis of Overall Rating in FIFA 20(Bottom 20)', fontsize=20)



ax.set_xlabel(xlabel='Overall Rating', fontsize=16)

ax.set_ylabel(ylabel='Count', fontsize=16)

plt.xticks(rotation=30, fontsize=12)



plt.show()
fig = go.Figure()

data = df_20.sort_values(by='nationality')



fig.add_trace(go.Box(

    x = data['nationality'],

    y = data['bmi'],

    name = "Suspected Outliers",

    boxpoints = "suspectedoutliers",

    marker = dict(

        size = 12,

        color = 'rgb(180, 222, 43)',

        outliercolor = 'rgba(31, 158, 137, 0.6)',

        line = dict(

            outliercolor = 'rgba(31, 158, 137, 0.6)',

            outlierwidth = 2

        )),

    line_color = 'rgba(72, 40, 120)',

    text = data['short_name']

))



fig.update_layout(title='Styled Box Plot (with Suspected Outliers) - Nationality vs BMI',

                  xaxis_title='Nationality',

                  yaxis_title='BMI',

                  paper_bgcolor='rgba(0,0,0,0)',

                  plot_bgcolor='rgba(0,0,0,0)',

                  font=dict(family='Cambria, monospace', size=12, color='#000000'),

                  xaxis_rangeslider_visible=True)

fig.show()
plt.figure(figsize= (15, 7))



ax = sns.countplot(x='team_position', data=df_20, palette='bright', order=df_20.team_position.value_counts().index)

ax.set_title(label='Count of Players on Basis of Positions in FIFA 20', fontsize=20)



ax.set_xlabel(xlabel='Positions', fontsize=16)

ax.set_ylabel(ylabel='Count', fontsize=16)

plt.xticks(rotation=60, fontsize=12)



plt.show()
attack = ['LW', 'RW', 'ST', 'LF', 'RF', 'CF', 'LS', 'RS']

data = df_20.query('team_position in @attack')    

fig = px.pie(data, names='team_position',

             color_discrete_sequence=px.colors.sequential.Bluered,

             title='Percentages of Player Attacking Positions')

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',

                  plot_bgcolor='rgba(0,0,0,0)',

                  font=dict(size=12, color='#000000'))

fig.show()
defence = ['LWB', 'RWB', 'CB', 'LB', 'RB', 'LCB', 'RCB']

data = df_20.query('team_position in @defence')    

fig = px.pie(data, names='team_position',

             color_discrete_sequence=px.colors.sequential.Sunsetdark,

             title='Percentages of Player Defending Positions')

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',

                  plot_bgcolor='rgba(0,0,0,0)',

                  font=dict(size=12, color='#000000'))

fig.show()
mid = ['CM', 'RCM', 'LCM', 'RM', 'LM', 'CAM', 'RDM', 'LDM', 'CDM', 'RAM', 'LAM']

data = df_20.query('team_position in @mid')    

fig = px.pie(data, names='team_position',

             color_discrete_sequence=px.colors.sequential.Agsunset,

             title='Percentages of Player Midfield Positions')

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',

                  plot_bgcolor='rgba(0,0,0,0)',

                  font=dict(size=12, color='#000000'))

fig.show()
df_20['short_name'][df_20['overall']==max(df_20['overall'])]
df_20['overall'].mean()
df_20_top_rated = df_20.sort_values(by=["overall"], ascending=False)

df_20_top_rated.head(10)
fig = px.pie(df_20_top_rated.head(25), names='club',

             title='Percentage of Clubs in top 25 players')

fig.show()
fig = px.pie(df_20_top_rated.head(25), names='nationality',

             title='Percentage of Nations in top 25 players')

fig.show()
df_20_high_value = df_20.sort_values(by=['value_eur'], ascending=False)

df_20_high_value.head(10)
plt.figure(figsize = (50, 5))

plt.subplot(131)

plt.bar(df_20_high_value["short_name"].head(10), df_20_high_value["value_eur"].head(10), color='navy')

plt.tick_params(axis="x", rotation=30)

plt.ylabel("Value")

plt.xlabel("Player Name")



plt.title("Most Valued players in FIFA 20", fontsize=20)

plt.show()
plt.rcParams['figure.figsize'] = (15, 5)

df_20['value_eur'].plot(kind="line")

plt.title("Players Value distribution in Fifa 20", fontsize=20)

plt.show()
df_20_high_release = df_20.sort_values(by=['release_clause_eur'], ascending=False)

df_20_high_release.head(10)
plt.figure(figsize = (50, 5))

plt.subplot(131)

plt.bar(df_20_high_release["short_name"].head(10), df_20_high_release["release_clause_eur"].head(10), color="indigo")

plt.tick_params(axis="x", rotation=30)

plt.ylabel("Release Clause Value")

plt.xlabel("Player Name")



plt.title("Players with highest release clauses in FIFA 20", fontsize=20)

plt.show()
plt.rcParams['figure.figsize'] = (15, 5)

df_20['release_clause_eur'].plot(kind="line")

plt.title("Players Release Clause distribution in Fifa 20", fontsize=20)

plt.show()
print(df_20_high_release[:10]['age'].mean())

print(df_20_high_value[:10]['age'].mean())
sns.barplot(df_20_top_rated['short_name'].head(10), df_20_top_rated['value_eur'].head(10), palette="OrRd").set_title("Value of top 10 players", fontsize=20)
sns.boxplot(df_20['value_eur'])
plt.rcParams['figure.figsize'] = (15, 5)

df_20['wage_eur'].plot(kind="line")

plt.title("Players Wage distribution in Fifa 20", fontsize=20)

plt.show()
df_20_high_wage = df_20.sort_values(by=['wage_eur'], ascending=False)

df_20_high_wage.head(10)
sns.barplot(df_20_high_wage['short_name'].head(10), df_20_high_wage['wage_eur'].head(10), palette="PuBu").set_title("Top 10 Player wages", fontsize=20)
sns.barplot(df_20_top_rated['short_name'].head(10), df_20_top_rated['wage_eur'].head(10), palette="rocket").set_title("Wage of top 10 players", fontsize=20)
sns.boxplot(df_20['wage_eur'])
fig = px.scatter_3d(df_20_top_rated.head(20), x='potential', y='overall', z='wage_eur',

              color='short_name')

fig.update_layout(title='3D Plot of Potential, Overall and Wage in Euros of Top 20 FIFA Players')

fig.show()
fig = px.scatter_3d(df_20_top_rated.head(20), x='potential', y='overall', z='value_eur',

              color='short_name')

fig.update_layout(title='3D Plot of Potential, Overall and Value in Euros of Top 20 FIFA Players')

fig.show()
df_20_top_attackers = df_20[df_20['team_position'].str.contains('ST|RW|LW|CF|LS|RS')].sort_values(by="overall", ascending=False)

df_20_top_attackers.head(10)
sns.barplot(df_20_top_attackers['short_name'].head(10), df_20_top_attackers['overall'].head(10), palette="colorblind").set_title("Overall Ratings of Top 10 Attackers", fontsize=20)
sns.barplot(df_20_top_attackers['short_name'].head(10), df_20_top_attackers['value_eur'].head(10), palette="GnBu").set_title("Values of Top 10 Attackers", fontsize=20)
sns.barplot(df_20_top_attackers['short_name'].head(10), df_20_top_attackers['wage_eur'].head(10), palette="PiYG").set_title("Wages of Top 10 Attackers", fontsize=20)
df_20_top_midfield = df_20[df_20['team_position'].str.contains('CM|CAM|CDM|LM|RM')].sort_values(by="overall", ascending=False)

df_20_top_midfield.head(10)
sns.barplot(df_20_top_midfield['short_name'].head(10), df_20_top_midfield['overall'].head(10), palette="Spectral").set_title("Overall Ratings of Top 10 Midfielders", fontsize=20)
sns.barplot(df_20_top_midfield['short_name'].head(10), df_20_top_midfield['value_eur'].head(10), palette="RdYlBu").set_title("Values of Top 10 Midfielders", fontsize=20)
sns.barplot(df_20_top_midfield['short_name'].head(10), df_20_top_midfield['wage_eur'].head(10), palette="RdGy").set_title("Wages of Top 10 Midfielders", fontsize=20)
df_20_top_defend = df_20[df_20['team_position'].str.contains('CB|LB|RB|LWB|RWB')].sort_values(by="overall", ascending=False)

df_20_top_defend.head(10)
sns.barplot(df_20_top_defend['short_name'].head(10), df_20_top_defend['overall'].head(10), palette="PRGn").set_title("Overall Ratings of Top 10 Defenders", fontsize=20)
sns.barplot(df_20_top_defend['short_name'].head(10), df_20_top_defend['value_eur'].head(10), palette="muted").set_title("Values of Top 10 Defenders", fontsize=20)
sns.barplot(df_20_top_defend['short_name'].head(10), df_20_top_defend['wage_eur'].head(10), palette="dark").set_title("Wages of Top 10 Defenders", fontsize=20)
df_20.preferred_foot.value_counts()
plt.rcParams['figure.figsize']=(13, 7)

sns.countplot(df_20['preferred_foot'])

plt.title("Most Preferred Foot", fontsize=30)

plt.show()
foot = ['Left', 'Right']

data = df_20.query('preferred_foot in @foot')    

fig = px.pie(data, names='preferred_foot',

             color_discrete_sequence=px.colors.sequential.Bluered,

             title='Overall Preferred foot percentages')

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',

                  plot_bgcolor='rgba(0,0,0,0)',

                  font=dict(size=12, color='#000000'))

fig.show()
df_20_high_value[df_20_high_value['preferred_foot']=='Right'].head(10)
df_20_high_value[df_20_high_value['preferred_foot']=='Left'].head(10)
sns.lmplot(x='skill_ball_control', y='skill_dribbling', data=df_20[:1000], col='preferred_foot')
df_20['potential'].mean()
young_talent = df_20[df_20['age']<=22]

young_talent.sort_values(by=["potential"], ascending=False).head(10)
most_valuable_young = young_talent.sort_values(by=["overall"], ascending=False)

most_valuable_young.head()
print(most_valuable_young['age'][:10].mean())

print(most_valuable_young['value_eur'][:10].mean())
young_talent.sort_values(by=["release_clause_eur"], ascending=False).head()
for p in range(80, 95):

    overall_req = df_20[df_20['overall']==p]

    potential_req = df_20[df_20['potential']==p]

    mean = round(overall_req.loc[:, 'value_eur'].mean())

    df_20.at[potential_req.index, 'potential_value'] = mean

    

df_copy = df_20

df_copy['value_change'] = df_copy['potential_value'] - df_copy['value_eur']

df_copy.loc[:, ['short_name', 'age', 'overall', 'potential', 'value_eur', 'potential_value', 'value_change']].sort_values(by="value_change", ascending=False).head(20)
corr_mat = df_20.corr()

corr_mat
corr_mat.loc['age':'wage_eur', 'age':'wage_eur']
plt.figure(figsize=(15,10))

sns.heatmap(corr_mat.loc['age':'wage_eur', 'age':'wage_eur'], annot=True, cbar=True)
plt.figure(figsize=(15,10))

sns.heatmap(corr_mat.loc['pace':'physic', 'pace':'physic'], annot=True, cbar=True)
plt.figure(figsize=(20, 20))

mask = np.zeros_like(corr_mat, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr_mat, cmap="GnBu", vmax=.3, center=0, square=True, linewidths=.7, cbar_kws={"shrink": .7})
plt.figure(figsize=(20, 20))

mask = np.zeros_like(corr_mat.loc['pace':'physic', 'pace':'physic'], dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr_mat.loc['pace':'physic', 'pace':'physic'], cmap="GnBu", vmax=.3, center=0, square=True, linewidths=.7, cbar_kws={"shrink": .7})
fig, ax = plt.subplots(ncols=2, figsize=(18, 10))

sns.regplot(x = df_20['age'], y = df_20['overall'], ax=ax[0])

sns.regplot(x = df_20['age'], y = df_20['potential'], ax=ax[1])
fig, ax = plt.subplots(ncols=3, figsize=(18, 10))

sns.regplot(x = df_20['age'], y = df_20['value_eur'], ax=ax[0])

sns.regplot(x = df_20['age'], y = df_20['wage_eur'], ax=ax[1])

sns.regplot(x = df_20['age'], y = df_20['release_clause_eur'], ax=ax[2])
fig, ax = plt.subplots(figsize=(16, 8))

ax.scatter(df_20['age'], df_20['value_eur'])

ax.set_xlabel("Age")

ax.set_ylabel("Value")

plt.show()
sns.pairplot(df_20[["international_reputation","skill_moves", "attacking_finishing", "skill_fk_accuracy", "movement_sprint_speed"]], palette="deep")
df_20.sort_values(by="pace", ascending=False).head(10)
df_20[df_20["international_reputation"]==5].head(10)
top_speed = df_20.sort_values(by=['movement_sprint_speed'], ascending=False)

free_kick = df_20.sort_values(by=['skill_fk_accuracy'], ascending=False)

skill_moves = df_20.sort_values(by=['skill_moves'], ascending=False)

finishing = df_20.sort_values(by=['attacking_finishing'], ascending=False)
top_speed.head()
free_kick.head()
skill_moves.head()
finishing.head()
top_speed = top_speed.head(10)

free_kick = free_kick.head(10)

skill_moves = skill_moves.head(10)

finishing = finishing.head(10)
sns.lineplot(top_speed['short_name'], top_speed['movement_sprint_speed']).set_title("Fastest Players", fontsize=20)
sns.lineplot(free_kick['short_name'], free_kick['skill_fk_accuracy']).set_title("Best Free Kick Takers", fontsize=20)
sns.lineplot(skill_moves['short_name'], skill_moves['skill_moves']).set_title("Best Skill Moves", fontsize=20)
sns.lineplot(finishing['short_name'], finishing['attacking_finishing']).set_title("Best Finishing", fontsize=20)
plt.rcParams['figure.figsize']=(15, 5)

sns.countplot(df_20['work_rate'], palette="Blues_d")

plt.title("Work Rate Frequency", fontsize=20)

plt.show()
sns.barplot(df_20['work_rate'], df_20['age'], palette="Paired").set_title("Relationship between Age and Work Rate of Players")
rate = ['Medium/Low', 'High/Low', 'High/Medium', 'Medium/Medium', 'High/High', 'Medium/High', 'Low/High', 'Low/Medium', 'Low/Low']

data = df_20.query('work_rate in @rate')    

fig = px.pie(data, names='work_rate',

             color_discrete_sequence=px.colors.sequential.Plasma_r,

             title='Percentage of players in Various Work Rates')

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',

                  plot_bgcolor='rgba(0,0,0,0)',

                  font=dict(size=12, color='#000000'))

fig.show()
features = df_20[['age', 'overall', 'skill_moves', 'attacking_finishing', 'skill_fk_accuracy', 'movement_sprint_speed']]

ax = sns.boxplot(data=features)

plt.tick_params(axis='x', rotation=30)
fig = go.Figure()



fig = go.Figure(data=go.Scatter(

    x = df_20['overall'],

    y = df_20['value_eur'],

    mode='markers',

    marker=dict(

        size=16,

        color=df_20['age'], #set color equal to a variable

        colorscale='Plasma', # one of plotly colorscales

        showscale=True

    ),

    text= df_20['short_name'],

))



fig.update_layout(title='Overall Rating vs Value in Euros',

                  xaxis_title='Overall Rating',

                  yaxis_title='Value in Euros',

                  paper_bgcolor='rgba(0,0,0,0)',

                  plot_bgcolor='rgba(0,0,0,0)',

                  font=dict(size=12, color='#000000'))

fig.show()
sns.relplot(x='overall',y='value_eur',hue='age',palette = 'viridis',size="bmi", sizes=(15, 200),aspect=2.5,data=df_20)

plt.title('Overall Rating v  Value in Euros',fontsize = 20)

plt.xlabel('Overall Rating')

plt.ylabel('Value in Euros')

plt.show()
fig = go.Figure()



fig = go.Figure(data=go.Scatter(

    x = df_20['overall'],

    y = df_20['wage_eur'],

    mode='markers',

    marker=dict(

        size=16,

        color=df_20['age'], #set color equal to a variable

        colorscale='Plasma', # one of plotly colorscales

        showscale=True

    ),

    text= df_20['short_name'],

))



fig.update_layout(title='Overall Rating vs Wage in Euros',

                  xaxis_title='Overall Rating',

                  yaxis_title='Value in Euros',

                  paper_bgcolor='rgba(0,0,0,0)',

                  plot_bgcolor='rgba(0,0,0,0)',

                  font=dict(size=12, color='#000000'))

fig.show()
sns.relplot(x='overall',y='wage_eur',hue='age',palette = 'viridis',size="bmi", sizes=(15, 200),aspect=2.5,data=df_20)

plt.title('Overall Rating  v  Wage in Euros',fontsize = 20)

plt.xlabel('Overall')

plt.ylabel('Wage in Euros')

plt.show()
fig = go.Figure()



fig = go.Figure(data=go.Scatter(

    x = df_20['overall'],

    y = df_20['bmi'],

    mode='markers',

    marker=dict(

        size=16,

        color=df_20['age'], #set color equal to a variable

        colorscale='Plasma', # one of plotly colorscales

        showscale=True

    ),

    text= df_20['short_name'],

))



fig.update_layout(title='Overall Rating vs BMI',

                  xaxis_title='Overall Rating',

                  yaxis_title='Value in Euros',

                  paper_bgcolor='rgba(0,0,0,0)',

                  plot_bgcolor='rgba(0,0,0,0)',

                  font=dict(size=12, color='#000000'))

fig.show()
df_20_potential = df_20[(df_20.age.astype(int)>=18) & (df_20.age.astype(int)<=40)].groupby(['age'])['potential'].mean()

df_20_overall = df_20[(df_20.age.astype(int)>=18) & (df_20.age.astype(int)<=40)].groupby(['age'])['overall'].mean()

df_20_summary = pd.concat([df_20_potential, df_20_overall], axis=1)



fig, ax = plt.subplots(figsize=(16, 8))

ax.plot(df_20_summary)

ax.set_xlabel("Age", fontsize=20)

ax.set_ylabel("Rating", fontsize=20)

ax.set_title("FIFA 20 - Average Rating by Age", fontsize=20)

plt.show()
def club(x):

    return df_20[df_20['club']==x]



def teamValue(club):

    value = 0

    for i in range(len(club)):

        value += club.iloc[i]['value_eur']

    value = value/1000000

    return value



def teamWeeklyWage(club):

    wage = 0

    for i in range(len(club)):

        wage += club.iloc[i]['wage_eur']

    wage = wage/1000000

    return wage



def avgSquadAge(club):

    return round(club['age'].mean(), 1)



def avgOverallRating(club):

    return round(club['overall'].mean(), 1)



def avgPotential(club):

    return round(club['potential'].mean(), 1)
MU = club('Manchester United')

MU.head()
MU.shape
MU = club("Manchester United")

print("Team Value of Manchester United is",teamValue(MU),"million Euros")

print("Manchester United spends", teamWeeklyWage(MU),"million Euros as per Week Wages")

print("Manchester United has average squad age of", avgSquadAge(MU), "years")

print("Manchester United has average overall rating of", avgOverallRating(MU))

print("Manchester United has average potential of", avgPotential(MU))
plt.figure(figsize=(18,8))

plt.title('Age Distribution of Manchester United in FIFA 20', fontsize=20)

sns.distplot(a=MU['age'], kde=True, bins=20)

plt.show()
plt.figure(figsize= (15,7))



ax = sns.countplot(x='nationality', data=MU, palette='bright')

ax.set_title(label='Count of Manchester United Players on Basis of Nationality in FIFA 20', fontsize=20)



ax.set_xlabel(xlabel='Nationality', fontsize=16)

ax.set_ylabel(ylabel='Count', fontsize=16)

plt.xticks(rotation=30, fontsize=12)



plt.show()
fig = go.Figure()



fig = go.Figure(data=go.Scatter(

    x = MU['nationality'],

    y = MU['overall'],

    mode='markers',

    marker=dict(

        size=16,

        color=MU['overall'], #set color equal to a variable

        colorscale='Plasma', # one of plotly colorscales

        showscale=True

    ),

    text= MU['short_name'],

))



fig.update_layout(title='Nationality vs Overall Rating in Manchester United',

                  xaxis_title='Nationality',

                  yaxis_title='Overall Rating',

                  paper_bgcolor='rgba(0,0,0,0)',

                  plot_bgcolor='rgba(0,0,0,0)',

                  font=dict(size=12, color='#000000'))

fig.show()
plt.figure(figsize=(18,8))

plt.title('Rating Distribution of Manchester United in FIFA 20', fontsize=20)

sns.distplot(a=MU['overall'], kde=True, bins=20)

plt.show()
MU_most_valuable = MU.sort_values(by="value_eur", ascending=False)

MU_most_valuable.head(10)
plt.figure(figsize=(20, 15))

plt.bar(MU_most_valuable['short_name'], MU_most_valuable['value_eur'], color="k")

plt.tick_params(axis='x', rotation=60)

plt.ylabel("Value", fontsize=40)

plt.xlabel("Name", fontsize=40)

plt.show()
plt.figure(figsize=(15, 7))

ax = sns.countplot(x="team_position", data=MU, palette="deep")

ax.set_title("Count of Manchester United's Players based on Positions", fontsize=20)

ax.set_xlabel(xlabel="Positons", fontsize=16)

ax.set_ylabel(ylabel="Count", fontsize=16)

plt.xticks(rotation=30, fontsize=10)

plt.show()
fig = go.Figure()



fig = go.Figure(data=go.Scatter(

    x = MU['overall'],

    y = MU['value_eur'],

    mode='markers',

    marker=dict(

        size=16,

        color=MU['age'], #set color equal to a variable

        colorscale='Plasma', # one of plotly colorscales

        showscale=True

    ),

    text= MU['short_name'],

))



fig.update_layout(title='Overall Rating vs Value in Euros for Manchester United',

                  xaxis_title='Overall Rating',

                  yaxis_title='Value in Euros',

                  paper_bgcolor='rgba(0,0,0,0)',

                  plot_bgcolor='rgba(0,0,0,0)',

                  font=dict(size=12, color='#000000'))

fig.show()
CH = club("Chelsea")

print("Team Value of Chelsea is",teamValue(CH),"million Euros")

print("Chelsea spends", teamWeeklyWage(CH),"million Euros as per Week Wages")

print("Chelsea has average squad age of", avgSquadAge(CH), "years")

print("Chelsea has average overall rating of", avgOverallRating(CH))

print("Chelsea has average potential of", avgPotential(CH))
plt.figure(figsize=(18,8))

plt.title('Age Distribution of Chelsea in FIFA 20', fontsize=20)

sns.distplot(a=CH['age'], kde=True, bins=20)

plt.show()
plt.figure(figsize= (15,7))



ax = sns.countplot(x='nationality', data=CH, palette='bright')

ax.set_title(label='Count of Chelsea Players on Basis of Nationality in FIFA 20', fontsize=20)



ax.set_xlabel(xlabel='Nationality', fontsize=16)

ax.set_ylabel(ylabel='Count', fontsize=16)

plt.xticks(rotation=30, fontsize=12)



plt.show()
fig = go.Figure()



fig = go.Figure(data=go.Scatter(

    x = CH['nationality'],

    y = CH['overall'],

    mode='markers',

    marker=dict(

        size=16,

        color=CH['overall'], #set color equal to a variable

        colorscale='Plasma', # one of plotly colorscales

        showscale=True

    ),

    text= CH['short_name'],

))



fig.update_layout(title='Nationality vs Overall Rating in Chelsea',

                  xaxis_title='Nationality',

                  yaxis_title='Overall Rating',

                  paper_bgcolor='rgba(0,0,0,0)',

                  plot_bgcolor='rgba(0,0,0,0)',

                  font=dict(size=12, color='#000000'))

fig.show()
plt.figure(figsize=(18,8))

plt.title('Rating Distribution of Chelsea in FIFA 20', fontsize=20)

sns.distplot(a=CH['overall'], kde=True, bins=20)

plt.show()
CH_most_valuable = CH.sort_values(by="value_eur", ascending=False)

CH_most_valuable.head(10)
plt.figure(figsize=(20, 15))

plt.bar(CH_most_valuable['short_name'], CH_most_valuable['value_eur'], color="k")

plt.tick_params(axis='x', rotation=60)

plt.ylabel("Value", fontsize=40)

plt.xlabel("Name", fontsize=40)

plt.show()
plt.figure(figsize=(15, 7))

ax = sns.countplot(x="team_position", data=CH, palette="deep")

ax.set_title("Count of Chelsea's Players based on Positions", fontsize=20)

ax.set_xlabel(xlabel="Positons", fontsize=16)

ax.set_ylabel(ylabel="Count", fontsize=16)

plt.xticks(rotation=30, fontsize=10)

plt.show()
fig = go.Figure()



fig = go.Figure(data=go.Scatter(

    x = CH['overall'],

    y = CH['value_eur'],

    mode='markers',

    marker=dict(

        size=16,

        color=CH['age'], #set color equal to a variable

        colorscale='Plasma', # one of plotly colorscales

        showscale=True

    ),

    text= CH['short_name'],

))



fig.update_layout(title='Overall Rating vs Value in Euros in Chelsea',

                  xaxis_title='Overall Rating',

                  yaxis_title='Value in Euros',

                  paper_bgcolor='rgba(0,0,0,0)',

                  plot_bgcolor='rgba(0,0,0,0)',

                  font=dict(size=12, color='#000000'))

fig.show()
LI = club('Liverpool')

print("Team Value of Liverpool is",teamValue(LI),"million Euros")

print("Liverpool spends", teamWeeklyWage(LI),"million Euros as per Week Wages")

print("Liverpool has average squad age of", avgSquadAge(LI), "years")

print("Liverpool has average overall rating of", avgOverallRating(LI))

print("Liverpool has average potential of", avgPotential(LI))
plt.figure(figsize=(18,8))

plt.title('Age Distribution of Liverpool in FIFA 20', fontsize=20)

sns.distplot(a=LI['age'], kde=True, bins=20)

plt.show()
plt.figure(figsize= (15,7))



ax = sns.countplot(x='nationality', data=LI, palette='bright')

ax.set_title(label='Count of Liverpool Players on Basis of Nationality in FIFA 20', fontsize=20)



ax.set_xlabel(xlabel='Nationality', fontsize=16)

ax.set_ylabel(ylabel='Count', fontsize=16)

plt.xticks(rotation=30, fontsize=12)



plt.show()
fig = go.Figure()



fig = go.Figure(data=go.Scatter(

    x = LI['nationality'],

    y = LI['overall'],

    mode='markers',

    marker=dict(

        size=16,

        color=LI['overall'], #set color equal to a variable

        colorscale='Plasma', # one of plotly colorscales

        showscale=True

    ),

    text= LI['short_name'],

))



fig.update_layout(title='Nationality vs Overall Rating in Liverpool',

                  xaxis_title='Nationality',

                  yaxis_title='Overall Rating',

                  paper_bgcolor='rgba(0,0,0,0)',

                  plot_bgcolor='rgba(0,0,0,0)',

                  font=dict(size=12, color='#000000'))

fig.show()
plt.figure(figsize=(18,8))

plt.title('Rating Distribution of Liverpool in FIFA 20', fontsize=20)

sns.distplot(a=LI['overall'], kde=True, bins=20)

plt.show()
LI_most_valuable = LI.sort_values(by="value_eur", ascending=False)

LI_most_valuable.head(10)
plt.figure(figsize=(20, 15))

plt.bar(LI_most_valuable['short_name'], LI_most_valuable['value_eur'], color="k")

plt.tick_params(axis='x', rotation=60)

plt.ylabel("Value", fontsize=40)

plt.xlabel("Name", fontsize=40)

plt.show()
plt.figure(figsize=(15, 7))

ax = sns.countplot(x="team_position", data=LI, palette="deep")

ax.set_title("Count of Liverpool's Players based on Positions", fontsize=20)

ax.set_xlabel(xlabel="Positons", fontsize=16)

ax.set_ylabel(ylabel="Count", fontsize=16)

plt.xticks(rotation=30, fontsize=10)

plt.show()
fig = go.Figure()



fig = go.Figure(data=go.Scatter(

    x = LI['overall'],

    y = LI['value_eur'],

    mode='markers',

    marker=dict(

        size=16,

        color=LI['age'], #set color equal to a variable

        colorscale='Plasma', # one of plotly colorscales

        showscale=True

    ),

    text= LI['short_name'],

))



fig.update_layout(title='Overall Rating vs Value in Euros in Liverpool',

                  xaxis_title='Overall Rating',

                  yaxis_title='Value in Euros',

                  paper_bgcolor='rgba(0,0,0,0)',

                  plot_bgcolor='rgba(0,0,0,0)',

                  font=dict(size=12, color='#000000'))

fig.show()
AS = club('Arsenal')

print("Team Value of Arsenal is",teamValue(AS),"million Euros")

print("Arsenal spends", teamWeeklyWage(AS),"million Euros as per Week Wages")

print("Arsenal has average squad age of", avgSquadAge(AS), "years")

print("Arsenal has average overall rating of", avgOverallRating(AS))

print("Arsenal has average potential of", avgPotential(AS))
plt.figure(figsize=(18,8))

plt.title('Age Distribution of Arsenal in FIFA 20', fontsize=20)

sns.distplot(a=AS['age'], kde=True, bins=20)

plt.show()
plt.figure(figsize= (15,7))



ax = sns.countplot(x='nationality', data=AS, palette='bright')

ax.set_title(label='Count of Arsenal Players on Basis of Nationality in FIFA 20', fontsize=20)



ax.set_xlabel(xlabel='Nationality', fontsize=16)

ax.set_ylabel(ylabel='Count', fontsize=16)

plt.xticks(rotation=30, fontsize=12)



plt.show()
fig = go.Figure()



fig = go.Figure(data=go.Scatter(

    x = AS['nationality'],

    y = AS['overall'],

    mode='markers',

    marker=dict(

        size=16,

        color=AS['overall'], #set color equal to a variable

        colorscale='Plasma', # one of plotly colorscales

        showscale=True

    ),

    text= AS['short_name'],

))



fig.update_layout(title='Nationality vs Overall Rating in Arsenal',

                  xaxis_title='Nationality',

                  yaxis_title='Overall Rating',

                  paper_bgcolor='rgba(0,0,0,0)',

                  plot_bgcolor='rgba(0,0,0,0)',

                  font=dict(size=12, color='#000000'))

fig.show()
plt.figure(figsize=(18,8))

plt.title('Rating Distribution of Arsenal in FIFA 20', fontsize=20)

sns.distplot(a=AS['overall'], kde=True, bins=20)

plt.show()
AS_most_valuable = AS.sort_values(by="value_eur", ascending=False)

AS_most_valuable.head(10)
plt.figure(figsize=(20, 15))

plt.bar(AS_most_valuable['short_name'], AS_most_valuable['value_eur'], color="k")

plt.tick_params(axis='x', rotation=60)

plt.ylabel("Value", fontsize=40)

plt.xlabel("Name", fontsize=40)

plt.show()
plt.figure(figsize=(15, 7))

ax = sns.countplot(x="team_position", data=AS, palette="deep")

ax.set_title("Count of Arsenal's Players based on Positions", fontsize=20)

ax.set_xlabel(xlabel="Positons", fontsize=16)

ax.set_ylabel(ylabel="Count", fontsize=16)

plt.xticks(rotation=30, fontsize=10)

plt.show()
fig = go.Figure()



fig = go.Figure(data=go.Scatter(

    x = AS['overall'],

    y = AS['value_eur'],

    mode='markers',

    marker=dict(

        size=16,

        color=df_20['age'], #set color equal to a variable

        colorscale='Plasma', # one of plotly colorscales

        showscale=True

    ),

    text= AS['short_name'],

))



fig.update_layout(title='Overall Rating vs Value in Euros in Arsenal',

                  xaxis_title='Overall Rating',

                  yaxis_title='Value in Euros',

                  paper_bgcolor='rgba(0,0,0,0)',

                  plot_bgcolor='rgba(0,0,0,0)',

                  font=dict(size=12, color='#000000'))

fig.show()
MC = club('Manchester City')

print("Team Value of Manchester City is",teamValue(MC),"million Euros")

print("Manchester City spends", teamWeeklyWage(MC),"million Euros as per Week Wages")

print("Manchester City has average squad age of", avgSquadAge(MC), "years")

print("Manchester City has average overall rating of", avgOverallRating(MC))

print("Manchester City has average potential of", avgPotential(MC))
plt.figure(figsize=(18,8))

plt.title('Age Distribution of Manchester City in FIFA 20', fontsize=20)

sns.distplot(a=MC['age'], kde=True, bins=20)

plt.show()
plt.figure(figsize= (15,7))



ax = sns.countplot(x='nationality', data=MC, palette='bright')

ax.set_title(label='Count of Manchester City Players on Basis of Nationality in FIFA 20', fontsize=20)



ax.set_xlabel(xlabel='Nationality', fontsize=16)

ax.set_ylabel(ylabel='Count', fontsize=16)

plt.xticks(rotation=30, fontsize=12)



plt.show()
fig = go.Figure()



fig = go.Figure(data=go.Scatter(

    x = MC['nationality'],

    y = MC['overall'],

    mode='markers',

    marker=dict(

        size=16,

        color=MC['overall'], #set color equal to a variable

        colorscale='Plasma', # one of plotly colorscales

        showscale=True

    ),

    text= MC['short_name'],

))



fig.update_layout(title='Nationality vs Overall Rating in Manchester City',

                  xaxis_title='Nationality',

                  yaxis_title='Overall Rating',

                  paper_bgcolor='rgba(0,0,0,0)',

                  plot_bgcolor='rgba(0,0,0,0)',

                  font=dict(size=12, color='#000000'))

fig.show()
plt.figure(figsize=(18,8))

plt.title('Rating Distribution of Manchester City in FIFA 20', fontsize=20)

sns.distplot(a=MC['overall'], kde=True, bins=20)

plt.show()
MC_most_valuable = MC.sort_values(by="value_eur", ascending=False)

MC_most_valuable.head(10)
plt.figure(figsize=(20, 15))

plt.bar(MC_most_valuable['short_name'], MC_most_valuable['value_eur'], color="k")

plt.tick_params(axis='x', rotation=60)

plt.ylabel("Value", fontsize=40)

plt.xlabel("Name", fontsize=40)

plt.show()
plt.figure(figsize=(15, 7))

ax = sns.countplot(x="team_position", data=MC, palette="deep")

ax.set_title("Count of Manchester City's Players based on Positions", fontsize=20)

ax.set_xlabel(xlabel="Positons", fontsize=16)

ax.set_ylabel(ylabel="Count", fontsize=16)

plt.xticks(rotation=30, fontsize=10)

plt.show()
fig = go.Figure()



fig = go.Figure(data=go.Scatter(

    x = MC['overall'],

    y = MC['value_eur'],

    mode='markers',

    marker=dict(

        size=16,

        color=MC['age'], #set color equal to a variable

        colorscale='Plasma', # one of plotly colorscales

        showscale=True

    ),

    text= MC['short_name'],

))



fig.update_layout(title='Overall Rating vs Value in Euros in Manchester City',

                  xaxis_title='Overall Rating',

                  yaxis_title='Value in Euros',

                  paper_bgcolor='rgba(0,0,0,0)',

                  plot_bgcolor='rgba(0,0,0,0)',

                  font=dict(size=12, color='#000000'))

fig.show()
BAR = club('FC Barcelona')

print("Team Value of Barcelona is",teamValue(BAR),"million Euros")

print("Barcelona spends", teamWeeklyWage(BAR),"million Euros as per Week Wages")

print("Barcelona has average squad age of", avgSquadAge(BAR), "years")

print("Barcelona has average overall rating of", avgOverallRating(BAR))

print("Barcelona has average potential of", avgPotential(BAR))
plt.figure(figsize=(18,8))

plt.title('Age Distribution of FC Barcelona in FIFA 20', fontsize=20)

sns.distplot(a=BAR['age'], kde=True, bins=20)

plt.show()
plt.figure(figsize= (15,7))



ax = sns.countplot(x='nationality', data=BAR, palette='bright')

ax.set_title(label='Count of Barcelona Players on Basis of Nationality in FIFA 20', fontsize=20)



ax.set_xlabel(xlabel='Nationality', fontsize=16)

ax.set_ylabel(ylabel='Count', fontsize=16)

plt.xticks(rotation=30, fontsize=12)



plt.show()
fig = go.Figure()



fig = go.Figure(data=go.Scatter(

    x = BAR['nationality'],

    y = BAR['overall'],

    mode='markers',

    marker=dict(

        size=16,

        color=BAR['overall'], #set color equal to a variable

        colorscale='Plasma', # one of plotly colorscales

        showscale=True

    ),

    text= BAR['short_name'],

))



fig.update_layout(title='Nationality vs Overall Rating in Barcelona',

                  xaxis_title='Nationality',

                  yaxis_title='Overall Rating',

                  paper_bgcolor='rgba(0,0,0,0)',

                  plot_bgcolor='rgba(0,0,0,0)',

                  font=dict(size=12, color='#000000'))

fig.show()
plt.figure(figsize=(18,8))

plt.title('Rating Distribution of Barcelona in FIFA 20', fontsize=20)

sns.distplot(a=BAR['overall'], kde=True, bins=20)

plt.show()
BAR_most_valuable = BAR.sort_values(by="value_eur", ascending=False)

BAR_most_valuable.head(10)
plt.figure(figsize=(20, 15))

plt.bar(BAR_most_valuable['short_name'], BAR_most_valuable['value_eur'], color="k")

plt.tick_params(axis='x', rotation=60)

plt.ylabel("Value", fontsize=40)

plt.xlabel("Name", fontsize=40)

plt.show()
plt.figure(figsize=(15, 7))

ax = sns.countplot(x="team_position", data=BAR, palette="deep")

ax.set_title("Count of Barcelona's Players based on Positions", fontsize=20)

ax.set_xlabel(xlabel="Positons", fontsize=16)

ax.set_ylabel(ylabel="Count", fontsize=16)

plt.xticks(rotation=30, fontsize=10)

plt.show()
fig = go.Figure()



fig = go.Figure(data=go.Scatter(

    x = BAR['overall'],

    y = BAR['value_eur'],

    mode='markers',

    marker=dict(

        size=16,

        color=BAR['age'], #set color equal to a variable

        colorscale='Plasma', # one of plotly colorscales

        showscale=True

    ),

    text= BAR['short_name'],

))



fig.update_layout(title='Overall Rating vs Value in Euros in Barcelona',

                  xaxis_title='Overall Rating',

                  yaxis_title='Value in Euros',

                  paper_bgcolor='rgba(0,0,0,0)',

                  plot_bgcolor='rgba(0,0,0,0)',

                  font=dict(size=12, color='#000000'))

fig.show()
RM = club('Real Madrid')

print("Team Value of Real Madrid is",teamValue(RM),"million Euros")

print("Real Madrid spends", teamWeeklyWage(RM),"million Euros as per Week Wages")

print("Real Madrid has average squad age of", avgSquadAge(RM), "years")

print("Real Madrid has average overall rating of", avgOverallRating(RM))

print("Real Madrid has average potential of", avgPotential(RM))
plt.figure(figsize=(18,8))

plt.title('Age Distribution of Real Madrid in FIFA 20', fontsize=20)

sns.distplot(a=RM['age'], kde=True, bins=20)

plt.show()
plt.figure(figsize= (15,7))



ax = sns.countplot(x='nationality', data=RM, palette='bright')

ax.set_title(label='Count of Real Madrid Players on Basis of Nationality in FIFA 20', fontsize=20)



ax.set_xlabel(xlabel='Nationality', fontsize=16)

ax.set_ylabel(ylabel='Count', fontsize=16)

plt.xticks(rotation=30, fontsize=12)



plt.show()
fig = go.Figure()



fig = go.Figure(data=go.Scatter(

    x = RM['nationality'],

    y = RM['overall'],

    mode='markers',

    marker=dict(

        size=16,

        color=RM['overall'], #set color equal to a variable

        colorscale='Plasma', # one of plotly colorscales

        showscale=True

    ),

    text= RM['short_name'],

))



fig.update_layout(title='Nationality vs Overall Rating in Real Madrid',

                  xaxis_title='Nationality',

                  yaxis_title='Overall Rating',

                  paper_bgcolor='rgba(0,0,0,0)',

                  plot_bgcolor='rgba(0,0,0,0)',

                  font=dict(size=12, color='#000000'))

fig.show()
plt.figure(figsize=(18,8))

plt.title('Rating Distribution of Real Madrid in FIFA 20', fontsize=20)

sns.distplot(a=RM['overall'], kde=True, bins=20)

plt.show()
RM_most_valuable = RM.sort_values(by="value_eur", ascending=False)

RM_most_valuable.head(10)
plt.figure(figsize=(20, 15))

plt.bar(RM_most_valuable['short_name'], RM_most_valuable['value_eur'], color="k")

plt.tick_params(axis='x', rotation=60)

plt.ylabel("Value", fontsize=40)

plt.xlabel("Name", fontsize=40)

plt.show()
plt.figure(figsize=(15, 7))

ax = sns.countplot(x="team_position", data=RM, palette="deep")

ax.set_title("Count of Real Madrid's Players based on Positions", fontsize=20)

ax.set_xlabel(xlabel="Positons", fontsize=16)

ax.set_ylabel(ylabel="Count", fontsize=16)

plt.xticks(rotation=30, fontsize=10)

plt.show()
fig = go.Figure()



fig = go.Figure(data=go.Scatter(

    x = RM['overall'],

    y = RM['value_eur'],

    mode='markers',

    marker=dict(

        size=16,

        color=RM['age'], #set color equal to a variable

        colorscale='Plasma', # one of plotly colorscales

        showscale=True

    ),

    text= RM['short_name'],

))



fig.update_layout(title='Overall Rating vs Value in Euros in Real Madrid',

                  xaxis_title='Overall Rating',

                  yaxis_title='Value in Euros',

                  paper_bgcolor='rgba(0,0,0,0)',

                  plot_bgcolor='rgba(0,0,0,0)',

                  font=dict(size=12, color='#000000'))

fig.show()
JUV = club('Juventus')

print("Team Value of Juventus is",teamValue(JUV),"million Euros")

print("Juventus spends", teamWeeklyWage(JUV),"million Euros as per Week Wages")

print("Juventus has average squad age of", avgSquadAge(JUV), "years")

print("Juventus has average overall rating of", avgOverallRating(JUV))

print("Juventus has average potential of", avgPotential(JUV))
plt.figure(figsize=(18,8))

plt.title('Age Distribution of Juventus in FIFA 20', fontsize=20)

sns.distplot(a=JUV['age'], kde=True, bins=20)

plt.show()
plt.figure(figsize= (15,7))



ax = sns.countplot(x='nationality', data=JUV, palette='bright')

ax.set_title(label='Count of Juventus Players on Basis of Nationality in FIFA 20', fontsize=20)



ax.set_xlabel(xlabel='Nationality', fontsize=16)

ax.set_ylabel(ylabel='Count', fontsize=16)

plt.xticks(rotation=30, fontsize=12)



plt.show()
fig = go.Figure()



fig = go.Figure(data=go.Scatter(

    x = JUV['nationality'],

    y = JUV['overall'],

    mode='markers',

    marker=dict(

        size=16,

        color=JUV['overall'], #set color equal to a variable

        colorscale='Plasma', # one of plotly colorscales

        showscale=True

    ),

    text= JUV['short_name'],

))



fig.update_layout(title='Nationality vs Overall Rating in Juventus',

                  xaxis_title='Nationality',

                  yaxis_title='Overall Rating',

                  paper_bgcolor='rgba(0,0,0,0)',

                  plot_bgcolor='rgba(0,0,0,0)',

                  font=dict(size=12, color='#000000'))

fig.show()
plt.figure(figsize=(18,8))

plt.title('Rating Distribution of Juventus in FIFA 20', fontsize=20)

sns.distplot(a=JUV['overall'], kde=True, bins=20)

plt.show()
JUV_most_valuable = JUV.sort_values(by="value_eur", ascending=False)

JUV_most_valuable.head(10)
plt.figure(figsize=(20, 15))

plt.bar(JUV_most_valuable['short_name'], JUV_most_valuable['value_eur'], color="k")

plt.tick_params(axis='x', rotation=60)

plt.ylabel("Value", fontsize=40)

plt.xlabel("Name", fontsize=40)

plt.show()
plt.figure(figsize=(15, 7))

ax = sns.countplot(x="team_position", data=JUV, palette="deep")

ax.set_title("Count of Juventus's Players based on Positions", fontsize=20)

ax.set_xlabel(xlabel="Positons", fontsize=16)

ax.set_ylabel(ylabel="Count", fontsize=16)

plt.xticks(rotation=30, fontsize=10)

plt.show()
fig = go.Figure()



fig = go.Figure(data=go.Scatter(

    x = JUV['overall'],

    y = JUV['value_eur'],

    mode='markers',

    marker=dict(

        size=16,

        color=JUV['age'], #set color equal to a variable

        colorscale='Plasma', # one of plotly colorscales

        showscale=True

    ),

    text= JUV['short_name'],

))



fig.update_layout(title='Overall Rating vs Value in Euros in Juventus',

                  xaxis_title='Overall Rating',

                  yaxis_title='Value in Euros',

                  paper_bgcolor='rgba(0,0,0,0)',

                  plot_bgcolor='rgba(0,0,0,0)',

                  font=dict(size=12, color='#000000'))

fig.show()
games = [df_15, df_16, df_17, df_18, df_19, df_20]

versions = ["fifa 15", "fifa 16", "fifa 17", "fifa 18", "fifa 19", "fifa 20"]

for data, name in zip(games, versions):

    data['FIFA'] = name

df_all = pd.concat(games)

df_all.head()
comparison = df_all[(df_all.short_name=="L. Messi")|(df_all.short_name=="Cristiano Ronaldo")]

fig = px.bar(comparison, x="short_name", y="overall", animation_frame="FIFA", color="club", hover_name="value_eur", range_y=[90, 96])

fig.update_layout(title={

    'text': 'Overall Rating Comparison over the last 6 years',

    'y': 0.95,

    'x': 0.5,

    'xanchor': 'center',

    'yanchor': 'top',

    },

    showlegend=True,

    xaxis_title="Player Name",

    yaxis_title="Overall Rating")



fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 800

fig.show()
fig = px.bar(comparison, x="short_name", y="wage_eur", animation_frame="FIFA", color="club", hover_name="wage_eur", range_y=[300000, 700000])

fig.update_layout(title={

    'text': 'Wage Comparison over the last 6 years',

    'y': 0.95,

    'x': 0.5,

    'xanchor': 'center',

    'yanchor': 'top',

    },

    showlegend=True,

    xaxis_title="Player Name",

    yaxis_title="Wages")



fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 800

fig.show()