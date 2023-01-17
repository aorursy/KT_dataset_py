import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import os

print(os.listdir("../input"))

import seaborn as sns

from nltk.corpus import stopwords

from wordcloud import WordCloud, STOPWORDS

fifa = pd.read_csv('../input/data.csv')
fifa.columns
fifa.shape
fifa.Finishing.plot(kind = 'line', color = 'g',label = 'Finishing',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

fifa.Positioning.plot(color = 'r',label = 'Positioning',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right') 

plt.xlabel('x axis') 

plt.ylabel('y axis')

plt.title('Line Plot') 

plt.show()
fifa.SprintSpeed.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
data1 = fifa.loc[:,["Age","Potential","Overall"]]

data1.plot()
data1.plot(subplots = True)

plt.show()
data1 = fifa.set_index(["Name","Club","Value"]) 

data1.head(100)
fifa.isnull().sum()


fifa.plot(kind = 'scatter',x ='Age',y = 'Jumping',Color = 'red')

fifa.Strength.plot(kind = 'hist',bins = 40 ,figsize = (5,5))
chosen_columns = ['Name','Age','Nationality','Overall','Potential','Special','Acceleration','Aggression','Agility','Balance',

'BallControl','Body Type','Composure','Crossing','Curve','Club','Dribbling','FKAccuracy','Finishing','GKDiving','GKHandling','GKKicking',

'GKPositioning','GKReflexes','HeadingAccuracy','Interceptions','International Reputation','Jersey Number','Jumping','Joined','LongPassing',

'LongShots','Marking','Penalties','Position','Positioning','Preferred Foot','Reactions','ShortPassing','ShotPower','Skill Moves','SlidingTackle',

'SprintSpeed','Stamina','StandingTackle','Strength','Value','Vision','Volleys','Wage','Weak Foot','Work Rate'

]
df = pd.DataFrame(fifa, columns = chosen_columns)

df.head()
df.set_index('Name', inplace=True)
df.describe()
df['Nationality'].value_counts().head()
df['Club'].value_counts().head()
en_genc = df.sort_values('Age', ascending = True)[[ 'Age', 'Club', 'Nationality']]

en_yasli = df.sort_values('Age', ascending = False)[['Age', 'Club', 'Nationality']]



en_genc.head()
en_yasli.head()
sns.set(style="darkgrid") 

ax = sns.countplot(x = 'Position' ,data = df) 

ax.set_title(label='Pozisyondaki Oyuncu Sayısı', fontsize=30);

plt.xticks(Rotation = 90)

plt.show()
filter_age = df.Age >= 35

df[filter_age].head()
plt.figure(figsize = (30,30))

stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='White',

                          stopwords=stopwords,

                          max_words=1200,

                          max_font_size=120, 

                          random_state=42

                         ).generate(str(df['Club']))

print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.title("WORD CLOUD - TAGS")

plt.axis('off')

plt.show()