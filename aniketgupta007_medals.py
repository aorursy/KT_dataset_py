import numpy as np

import pandas as pd

%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt
data = pd.read_csv('../input/olympic-medals/medal.csv')
data.describe()
data.head()
data.dtypes
data.info()
sns.heatmap(data.isnull(),cbar=False, cmap='viridis')
group_city_gender =data.groupby(['Edition','Gender']).agg({'Event_gender':['count']})
group_city_gender.columns = ['participants']

group_city_gender = group_city_gender.reset_index()
group_city_gender.head(2)
sns.lineplot(x = 'Edition',y='participants',hue='Gender', data= group_city_gender)
#Best athelete of all time

group_athelete_medals = data.groupby(['Athlete','NOC','Discipline']).agg({'Medal':['count']})

group_athelete_medals.columns = ['medals_count']

group_athelete_medals = group_athelete_medals.reset_index()

group_athelete_medals.nlargest(10,'medals_count')
#Historical analysis of Olympics: Long periods gaps

group_participants = data.groupby('Edition').agg({'Athlete':'count'})

group_participants = group_participants.reset_index()

group_participants



fig,ax = plt.subplots(figsize = (20,5))

ax=sns.lineplot(x = 'Edition',y='Athlete',data = group_participants,marker = 'o')

ax.set(xticks=group_participants['Edition'])

ax.text(1913,1200, s='WW1',fontsize=16)

ax.text(1937,1200,s='WW2',fontsize=16)

plt.show()

#Sports which is having the maximum medals : Artistic Gymnastic is all time favorite

group_discipline = data.groupby(['Edition','Discipline','NOC']).agg({'Medal' : 'count'})

group_discipline = group_discipline.reset_index()



group_discipline



df = group_discipline.groupby(['Edition']).agg({'Medal':'max','Discipline':'first','NOC':'first'}).reset_index()



df.groupby('Discipline').agg({'Discipline':'count','Discipline':'first','NOC':'first'}).max()
#Yearly participation of countries

group_countries = data.groupby(['Edition','NOC']).agg({'Athlete':'count'})

group_countries = group_countries.reset_index()

group_countries_df = group_countries.groupby(['Edition']).agg({'NOC':'count'})



group_countries_df = group_countries_df.reset_index()

group_countries_df

fig,ax = plt.subplots(figsize = (20,5))

ax=sns.lineplot(x='Edition',y='NOC',data=group_countries_df)

ax.set(xticks=group_countries_df['Edition'])

ax.text(1968,60,'Montreal Olympics Boycott')

ax.text(1976,25,'Moscow Olympics Boycott')

plt.show()
import matplotlib.ticker as ticker

import matplotlib.animation as animation

from IPython.display import HTML
df = group_countries

df['NOC'].unique
#import random



#umber_of_colors =70000



#colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])

#             for i in range(number_of_colors)]

#countries1 = df['NOC'].unique

#countries2 = df['NOC'].values

#countries2

#color_dict = dict(zip(countries2,colors))

#color_dict

df.head()

#group_lk = df.set_index('NOC')['Athlete'].to_dict()

#group_lk



colors = ['b', 'g', 'r', 'c', 'm', 'y', 'g']

#colors
fig, ax = plt.subplots(figsize=(15, 8))



def draw_barchart(current_year):

    dff = df[df['Edition'].eq(current_year)].sort_values(by='Athlete', ascending=True).tail(10)

    ax.clear()

    ax.barh(dff['NOC'], dff['Athlete'],color = colors)

    dx = dff['Athlete'].max() / 200

    for i, (value, name) in enumerate(zip(dff['Athlete'], dff['NOC'])):

        ax.text(value-dx, i,     name,           size=14, weight=600, ha='right', va='bottom')

        ax.text(value-dx, i-.25, name, size=10, color='#444444', ha='right', va='baseline')

        ax.text(value+dx, i,     f'{value:,.0f}',  size=14, ha='left',  va='center')

    ax.text(1, 0.4, current_year, transform=ax.transAxes, color='#777777', size=46, ha='right', weight=800)

    ax.text(0, 1.06, 'Number of Athletes Participating', transform=ax.transAxes, size=12, color='#777777')

    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    ax.xaxis.set_ticks_position('top')

    ax.tick_params(axis='x', colors='#777777', labelsize=12)

    ax.set_yticks([])

    ax.margins(0, 0.01)

    ax.grid(which='major', axis='x', linestyle='-')

    ax.set_axisbelow(True)

    ax.text(0, 1.15, 'Countries with maximum medals',

            transform=ax.transAxes, size=24, weight=600, ha='left', va='top')

    plt.box(False)

    

draw_barchart(2008)
fig, ax = plt.subplots(figsize=(15, 8))

animator = animation.FuncAnimation(fig, draw_barchart, frames=range(1896, 2008,4))

HTML(animator.to_jshtml())
group_india =data[data['NOC'] == 'IND'].groupby(['Edition','Gender']).agg({'Event_gender':['count']})

group_india.columns = ['participants']

group_india = group_india.reset_index()
group_india
sns.lineplot(x = 'Edition',y='participants',hue='Gender', data= group_india)
#All time favorite comparisons of Indian media

famous =  ['IND', 'PAK','CHN']

group_india_medals_tally =data[data.NOC.isin(famous)].groupby(['NOC','Medal']).agg({'Medal':['count']})

group_india_medals_tally.columns = ['medal_count']

group_india_medals_tally = group_india_medals_tally.reset_index()

group_india_medals_tally

sns.barplot(x='NOC',y='medal_count',data=group_india_medals_tally, hue='Medal')