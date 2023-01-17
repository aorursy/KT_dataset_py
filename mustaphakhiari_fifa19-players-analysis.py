import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import scipy as sp
data = pd.read_csv("../input/fifa19/data.csv")
data.columns 
data.info()
data.isnull().sum()
data.head(10)
data.describe()
#Line Plot

data.Overall.plot(kind = 'line', color = 'r' , label = 'OVERALL' , linewidth = 1 , alpha = .8 , grid = True ,linestyle = '-' )

data.Potential.plot(color = 'g' , label = 'POTENTIAL' , linewidth = .8 , alpha = .5 , linestyle = ':' ) 

plt.legend(loc = 'upper right')

plt.xlabel('Number of Players')

plt.ylabel('Overall Values')

plt.title('Line Plot Example')

plt.show()
some_countries = ('England', 'Germany', 'Spain', 'Argentina', 'France', 'Brazil', 'Italy')

data_countries = data.loc[data['Nationality'].isin(some_countries) & data['Overall']]



plt.rcParams['figure.figsize'] = (15, 7)

ax = sns.barplot(x = data_countries['Nationality'], y = data_countries['Overall'], palette = 'spring')

ax.set_xlabel(xlabel = 'Countries', fontsize = 9)

ax.set_ylabel(ylabel = 'Overall Scores', fontsize = 9)

ax.set_title(label = 'Distribution of overall scores of players from different countries', fontsize = 20)

plt.show()


data.SprintSpeed.plot(kind = 'hist' , bins = 100 , figsize = (20,20))

plt.xlabel('Sprint Speed')

plt.show()
data.plot(kind = 'scatter' , x='Stamina' , y = 'SprintSpeed' , alpha = .5 , color = 'b')

plt.xlabel('Stamina')

plt.ylabel('Sprint Speed')

plt.title('Stamina-Sprint Speed Scatter Plot')

plt.show()

data[data['Preferred Foot'] == 'Left'][['Name', 'Age', 'Nationality','Age','Overall','SprintSpeed','Stamina']].head(10)
data[data['Preferred Foot'] == 'Right'][['Name', 'Age', 'Nationality','Age','Overall','SprintSpeed','Stamina']].head(10)
sns.lineplot(data['Age'], data['Stamina'])

plt.title('Age vs Stamina', fontsize = 20)



plt.show()
def extract_value_from(Value):

    out = Value.replace('€', '')

    if 'M' in out:

        out = float(out.replace('M', ''))*1000000

    elif 'K' in Value:

        out = float(out.replace('K', ''))*1000

    return float(out)





data['Value'] = data['Value'].apply(lambda x: extract_value_from(x))

data['Wage'] = data['Wage'].apply(lambda x: extract_value_from(x))
data.sort_values(by='Value',ascending = False)[['Name','Age','Club','Nationality','Value','Position']].head()
def top(x):

    return data[data['Overall'] > x][['Name','Age','Nationality','Club','Overall','Position']]



top(90)
data[(data['Skill Moves'] == 5)][['Name','Age','Nationality','Club','Overall','Skill Moves','Position']].head(5)