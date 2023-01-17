import numpy as np

import scipy as sp

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot



init_notebook_mode(connected = True)

import plotly.graph_objs as go
import os

print(os.listdir("../input/fifa19"))
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df =pd.read_csv('/kaggle/input/fifa19/data.csv')
df.head()
df.columns
df.drop(['Unnamed: 0', 'ID', 'Photo', 'Flag', 'Club Logo', 'Special', 'Real Face','Loaned From' ,'Release Clause',

                   'Joined', 'Contract Valid Until'],axis=1,inplace=True)
df.isnull().sum()
df['Club'].fillna(value='No Club', inplace=True)
df[df['Position'].isna()][['Name', 'Nationality', 'LS', 'ST','RS', 'LW', 'LF', 'CF', 'RF', 'RW',

                              'LAM', 'CAM', 'RAM', 'LM', 'LCM','CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 

                              'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']].head()
df.drop(index=df[df['Position'].isna()].index, inplace=True)
df.isnull().sum()[df.isnull().sum() > 0]
len(df[df['Position'] == 'GK'])
df.fillna(value=0, inplace=True)
df['Nationality'].nunique()
df['Club'].nunique()
df.sort_values(by='Age')[['Name','Age','Club','Nationality','Position']].head(1)
df.sort_values(by='Age',ascending = False)[['Name','Age','Club','Nationality','Position']].head(1)


# defining a function for cleaning the value and wage column



def extract_value_from(Value):

    out = Value.replace('€', '')

    if 'M' in out:

        out = float(out.replace('M', ''))*1000000

    elif 'K' in Value:

        out = float(out.replace('K', ''))*1000

    return float(out)





df['Value'] = df['Value'].apply(lambda x: extract_value_from(x))

df['Wage'] = df['Wage'].apply(lambda x: extract_value_from(x))

df.sort_values(by='Value',ascending = False)[['Name','Age','Club','Nationality','Value','Position']].head()
df.sort_values(by='Wage',ascending = False)[['Name','Age','Club','Nationality','Wage','Position']].head()
def top(x):

    return df[df['Overall'] > x][['Name','Nationality','Club','Overall','Position']]



top(90)
def Inttop(x):

    return df[df['International Reputation'] == x][['Name','Nationality','Club','Overall','International Reputation','Position']]



Inttop(5)                                                   
df[(df['Skill Moves'] == 5)][['Name','Nationality','Club','Overall','Skill Moves','Position']].head(5)
pr_cols=['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',

       'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',

       'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',

       'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',

       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',

       'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving',

       'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']

i=0

while i < len(pr_cols):

    print('Best {0} : {1}'.format(pr_cols[i],df.loc[df[pr_cols[i]].idxmax()][0]))

    i += 1
df[df['Preferred Foot'] == 'Left'][['Name', 'Age', 'Club', 'Nationality']].head(5)
df[df['Preferred Foot'] == 'Right'][['Name', 'Age', 'Club', 'Nationality']].head(5)
df.groupby(df['Club'])['Nationality'].nunique().sort_values(ascending = False).head(5)
def club(x):

    return df[(df['Club'] == x)][['Name','Age','Nationality','Overall','Potential','Position','Value','Wage']]



club('Manchester City').head(15)
def team(x):

    return df[(df['Nationality'] == x)][['Name','Age','Club','Overall','Potential','Position','Value','Wage']]



team('Belgium').head(15)
df[['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM',

       'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM',

       'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']].head()
def skillConverter(val):

    if type(val) == str:

        s1 = val[0:2]

        s2 = val[-1]

        val = int(s1) + int(s2)

        return val

    

    else:

        return val
skill_columns = ['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM',

       'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM',

       'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']

                      

for col in skill_columns:

    df[col] = df[col].apply(skillConverter)
df[['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM',

       'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM',

       'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']].head()
df[['Height', 'Weight']].head()
def height_converter(val):

    f = val.split("'")[0]

    i = val.split("'")[1]

    h = (int(f) * 30.48) + (int(i)*2.54)

    return h



def weight_converter(val):

    w = int(val.split('lbs')[0])

    return w
df['Height in Cms'] = df['Height'].apply(height_converter)

df['Weight in Pounds'] = df['Weight'].apply(weight_converter)



df.drop(columns=['Height', 'Weight'], inplace=True)

df[['Height in Cms', 'Weight in Pounds']].head()
#Below is the reason for splitting the players into the above mentioned categories

print(df['Position'].unique())

print(df['Position'].nunique())
def position_simplifier(val):

    

    if val == 'RF' or val == 'ST' or val == 'LF' or val == 'RS' or val == 'LS' or val == 'CF' or val == 'RW' or val == 'LW':

        val = 'F'

        return val

        

    elif val == 'RCM' or val == 'LCM' or val == 'LDM' or val == 'CAM' or val == 'CDM' or val == 'RM' or val == 'LAM' or val == 'LM' or val == 'RDM' or val == 'CM' or val == 'RAM':

        val = 'M'

        return val



    

    elif val == 'RCB' or val == 'CB' or val == 'LCB' or val == 'LB' or val == 'RB' or val == 'RWB' or val == 'LWB':

        val = 'D'

        return val

    

    else:

        return val
df['Position'] = df['Position'].apply(position_simplifier)

df['Position'].value_counts()
df[(df['Position'] == 'F') & (df['Overall'] > 90)]
df[(df['Position'] == 'M') & (df['Overall'] > 90)]
df[(df['Position'] == 'D') & (df['Overall'] > 90)]
df[(df['Position'] == 'GK') & (df['Overall'] > 90)]
df[df['Age']<20].sort_values(by = 'Potential' , ascending = False).head()
sns.distplot(df['Age'],kde=False,bins=30)

plt.title('Distribution of Age among players',fontsize = 15)

plt.show()
sns.countplot(x='Preferred Foot' , hue = 'Position' , hue_order = ('GK','D','M','F') , data = df)

plt.title('Most Preferred Foot of the Players', fontsize = 15)

plt.show()
labels = df[['Name','Finishing']].head(10)

values = df['Finishing'].head(10)

colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']



fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.update_traces(hoverinfo='label', textinfo='percent', textfont_size=10,

                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))

fig.update_layout(title_text="Distribution of Finishing rating of Top 10 Players")

fig.show()
fig = plt.figure(figsize = (12,8))

sns.heatmap(df.corr(),cmap='coolwarm')

plt.title('Fifa Correlation of variables',fontsize = 15)
fig = plt.figure(figsize = (14,8))

sns.boxplot(x='Age',y='Overall',data=df , palette = 'rainbow')

plt.title('Age Vs Overall Rating distribution',fontsize = 15)
rating = pd.DataFrame(df.groupby(['Nationality'])['Overall'].sum().reset_index())

count = pd.DataFrame(rating.groupby('Nationality')['Overall'].sum().reset_index())



trace = [go.Choropleth(

            colorscale = 'YlOrRd',

            locationmode = 'country names',

            locations = count['Nationality'],

            text = count['Nationality'],

            z = count['Overall'],

)]



layout = go.Layout(title = '<b>Country vs Ratings</b>')



fig = go.Figure(data = trace, layout = layout)

py.iplot(fig)
df_nations = df.groupby(by='Nationality').size().reset_index()

df_nations.columns = ['Nation', 'Count']
df_nations[(df_nations['Nation'] == 'England') | (df_nations['Nation'] == 'Wales') 

           | (df_nations['Nation'] == 'Scotland') | (df_nations['Nation'] == 'Northern Ireland') ]
df_temp = pd.DataFrame(data= [['United Kingdom', 2148]], columns=['Nation', 'Count'])

df_nations = df_nations.append(df_temp, ignore_index=True)

df_nations.tail()
trace2 = dict(type='choropleth',

              locations=df_nations['Nation'],

              z=df_nations['Count'],

              locationmode='country names',

              colorscale='Portland'

             )



layout = go.Layout(title='<b>Number of Players in each Country</b>',

                   geo=dict(showocean=True,

                            oceancolor='#AEDFDF',

                            projection=dict(type='natural earth'),

                        )

                  )



fig = go.Figure(data=[trace2], layout=layout)

py.iplot(fig)
palette = sns.cubehelix_palette(light=.8, n_colors=6)

sns.relplot(x = 'Weight in Pounds' , y = 'Height in Cms' , hue = 'Position',style = 'Preferred Foot', kind= 'line' , data = df ,ci = "sd" ,dashes = False , markers = True )

plt.title('Height vs Weight based on Player position and Preferred Foot', fontsize = 15)

plt.show()
ndf = df[['Position' , 'Crossing' , 'Finishing' , 'SlidingTackle' , 'GKDiving' , 'HeadingAccuracy']]

g = sns.PairGrid(ndf , hue = 'Position')

g.map_diag(plt.hist,edgecolor = 'w')

g.map_offdiag(plt.scatter,edgecolor = 'w')

g.add_legend()

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Special Attributes distribution of players based on their Position')
g = sns.FacetGrid(df, col="Position", row="Preferred Foot",hue = 'International Reputation' ,margin_titles=True)

g = (g.map(plt.scatter, "Wage", "Value").add_legend())

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Player Value and Wage distribution according to International Reputation based on players position and preferred foot')
g = sns.FacetGrid(df, col="Position",  row="Preferred Foot",margin_titles=True , hue = 'Skill Moves')

g = (g.map(plt.hist, "Overall").add_legend())

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Overall distribution according to skill moves based on players position and preferred foot')