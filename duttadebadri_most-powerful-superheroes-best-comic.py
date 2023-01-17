import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
color = sns.color_palette()
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
import plotly.tools as tls
df1=pd.read_csv('../input/heroes_information.csv')
df1.head()
df1.isnull().sum()
df1.replace(to_replace='-',value='Other',inplace=True)
df1['Publisher'].fillna('Other',inplace=True)
df1.drop('Unnamed: 0',axis=1,inplace=True)
temp_series = df1['Publisher'].value_counts()
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(
    title='Comic-wise Superheroes distribution',
    width=900,
    height=900,
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Comic")
df_powerful_hero=pd.read_csv('../input/super_hero_powers.csv')
df_powerful_hero.head(1)
df_powerful_hero=df_powerful_hero*1 # converting True and False to "0" & "1", so that we can calculate maximum power
df_powerful_hero.loc[:, 'no_of_powers'] = df_powerful_hero.iloc[:, 1:].sum(axis=1)
df_powerful_hero.head(1)
df_all_power_hero=df_powerful_hero[['hero_names','no_of_powers']]
df_all_power_hero=df_all_power_hero.sort_values('no_of_powers',ascending=False)
df_all_power_hero.head(1)
type_colors = ['#78C850',  # Grass
                    '#F08030',  # Fire
                    '#6890F0',  # Water
                    '#A8B820',  # Bug
                    '#A8A878',  # Normal
                    '#A040A0',  # Poison
                    '#F8D030',  # Electric
                    '#E0C068',  # Ground
                    '#EE99AC',  # Fairy
                    '#C03028',  # Fighting
                    '#F85888',  # Psychic
                    '#B8A038',  # Rock
                    '#705898',  # Ghost
                    '#98D8D8',  # Ice
                    '#7038F8',  # Dragon
                   ]


fig, ax = plt.subplots()

fig.set_size_inches(13.7, 10.27)

sns.set_context("paper", font_scale=1.5)
f=sns.barplot(x=df_all_power_hero["hero_names"].head(20), y=df_all_power_hero['no_of_powers'].head(20), data=df_all_power_hero,palette=type_colors)
f.set_xlabel("Name of Superhero",fontsize=18)
f.set_ylabel("No. of Superpowers",fontsize=18)
f.set_title('Top 20 Superheroes having highest no. powers')
for item in f.get_xticklabels():
    item.set_rotation(90)
cnt_srs = df1['Gender'].value_counts().head()
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=cnt_srs.values[::-1],
    orientation = 'h',
    marker=dict(
        color=cnt_srs.values[::-1]
    ),
)

layout = dict(
    title='Gender distribution of Superheroes',
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Superheroes")
plt.figure(figsize=(12,8))
sns.boxplot(x="Gender", y="Height", data=df1)
plt.ylabel('Height Distribution (cm.)', fontsize=12)
plt.xlabel('Gender', fontsize=12)
plt.title("Height Distribution by Gender", fontsize=14)
plt.xticks(rotation='vertical')
plt.show()


plt.figure(figsize=(12,8))
sns.boxplot(x="Gender", y="Weight", data=df1)
plt.ylabel('Weight Distribution (kg.)', fontsize=12)
plt.xlabel('Gender', fontsize=12)
plt.title("Weight Distribution by Gender", fontsize=14)
plt.xticks(rotation='vertical')
plt.show()
df_alignment1=df1.ix[df1['Gender']=='Male']
cnt_srs = df_alignment1['Alignment'].value_counts()

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values,
        colorscale = 'Picnic',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Alignment of Male Superheroes'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Alignment1")


df_alignment2=df1.ix[df1['Gender']=='Female']
cnt_srs = df_alignment2['Alignment'].value_counts()

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values,
        colorscale = 'Picnic',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Alignment of Female Superheroes'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Alignment2")
cnt_srs = df1['Race'].value_counts()

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color="blue",
        #colorscale = 'Blues',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Race Type of Superheroes'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Race")  
type_colors = ['#78C850',  # Grass
                    '#F08030',  # Fire
                    '#6890F0',  # Water
                    '#A8B820',  # Bug
                    '#A8A878',  # Normal
                    '#A040A0',  # Poison
                    '#F8D030',  # Electric
                    '#E0C068',  # Ground
                    '#EE99AC',  # Fairy
                    '#C03028',  # Fighting
                    '#F85888',  # Psychic
                    '#B8A038',  # Rock
                    '#705898',  # Ghost
                    '#98D8D8',  # Ice
                    '#7038F8',  # Dragon
                   ]


temp_series = df1.ix[df1['Skin color']!='Other']['Skin color'].value_counts()
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))

trace = go.Pie(labels=labels, values=sizes,
               hoverinfo='label+percent', textinfo='value',
               textfont=dict(size=20),
               marker=dict(colors=type_colors,
                           line=dict(color='#CAF8BA', width=2)))
layout = go.Layout(
    title='Skin Color distribution of Superheroes',
    width=900,
    height=900,
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="skin")



temp_series = df1.ix[df1['Eye color']!='Other']['Eye color'].value_counts()
labels = (np.array(temp_series.index))
colors = ['#DFC1BA', '#FFA545', '#9AA9FC', '#F8BAEE']
sizes = (np.array((temp_series / temp_series.sum())*100))

trace = go.Pie(labels=labels, values=sizes,
               hoverinfo='label+percent', textinfo='value',
               textfont=dict(size=20),
               marker=dict(colors=colors,
                           line=dict(color='#CAF8BA', width=2)))
layout = go.Layout(
    title='Eye Color distribution of Superheroes',
    width=900,
    height=900,
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="skin")


temp_series = df1.ix[df1['Hair color']!='No Hair']['Hair color'].value_counts() #Not considering bald heroes :P
labels = (np.array(temp_series.index))
colors = ['#BACEF8', '#F8E6BA']
sizes = (np.array((temp_series / temp_series.sum())*100))

trace = go.Pie(labels=labels, values=sizes,
               hoverinfo='label+percent', textinfo='value',
               textfont=dict(size=20),
               marker=dict(colors=colors,
                           line=dict(color='#CAF8BA', width=2)))
layout = go.Layout(
    title='Hair Color distribution of Superheroes',
    width=900,
    height=900,
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="skin")
df1['hair'] = np.where(df1['Hair color']=="No Hair", 'Bald', 'Non-Balded')
cnt_srs = df1['hair'].value_counts()
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=cnt_srs.values[::-1],
    orientation = 'h',
    marker=dict(
        color=cnt_srs.values[::-1]
    ),
)

layout = dict(
    title='How many superheroes are bald?',
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Hair")
df2=pd.read_csv('../input/super_hero_powers.csv')
df2.head()
df2.isnull().any()
df_superhero=df2['hero_names']

df2.drop('hero_names',axis=1,inplace=True)
df3=pd.DataFrame()

for i in df2.columns:
    df3[i]=df2[i].value_counts()
df3.drop(df3.index=='False', inplace=True)
df3
df3.shape
df3=df3.T
df3=df3.reset_index()
df3.head()
df3.columns
df3['No_of_Superheroes']=df3[True]
df3.drop(True,axis=1,inplace=True)
df3.rename(columns={'index': 'Super_Power_Name'}, inplace=True)
df3 = df3.sort_values('No_of_Superheroes', ascending=False)
df3=df3.ix[df3['No_of_Superheroes']>50] #We take only when more than 50 ssuperheroes have the superpower
fig, ax = plt.subplots()

fig.set_size_inches(11.7, 8.27)

sns.set_context("paper", font_scale=1.5)
f=sns.barplot(x=df3["Super_Power_Name"], y=df3['No_of_Superheroes'], data=df3)
f.set_xlabel("Name of Superpower",fontsize=15)
f.set_ylabel("No. of Superheroes",fontsize=15)
f.set_title('Top Common Superpowers')
for item in f.get_xticklabels():
    item.set_rotation(90)