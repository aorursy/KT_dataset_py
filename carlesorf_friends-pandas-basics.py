import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/friends-series-dataset/friends_episodes_v2.csv')

df.head(5)
df.query('Season==1 and Director=="James Burrows"')
df.query('(Season==1 or Director=="James Burrows") & (Stars > 8.5)')
df[((df['Season']==1) | (df['Director']=='James Burrows')) & (df['Stars']>8.5)] #el mateix que abans
df2 = df.groupby('Director')[['Stars','Votes']].sum()

df2.sort_values(by='Stars',ascending=False)
df.groupby('Director').count()
#dues columnes noves amb numero de episodis per director i stars / num de programes fets

df2['Ep_totals'] = df['Director'].value_counts()

df2['Ratting_stars'] = df2['Stars'] / df2['Ep_totals'] #busquem la mitjana per cada episodi

df2['Ratting_stars'] = df2['Ratting_stars'].round(2)

df2['Max_puntuacio'] = df.groupby(['Director'], sort=False)['Stars'].max() #agafem la maxima puntacio dels seus episodis

df2.sort_values(by='Ratting_stars',ascending=False)

%config InlineBackend.figure_format='retina'

p = df2.plot.bar(y ='Ratting_stars',figsize=(14,4), alpha=0.5)
#escalar la columa de 0-1

from sklearn.preprocessing import MinMaxScaler

df2[['Ratting_stars','Votes']] = MinMaxScaler().fit_transform(df2[['Ratting_stars','Votes']])



%config InlineBackend.figure_format='retina'

p = df2.plot.bar(y ='Ratting_stars',figsize=(14,4), alpha=0.5, grid=True, title="Average Stars for each Director")
#set limits for a graph contrary to normalize

g = df.groupby('Season')['Stars'].mean()

p = g.plot.bar(y ='Stars',figsize=(14,4), alpha=0.5, grid=True, fontsize= 10, title="Average Stars for each season")

p = p.set_ylim(8.2,8.8)
#top ten episodes

df3 = df[['Episode_Title','Stars']].sort_values('Stars', ascending=False).head(10).reset_index(drop=True)

df3.Episode_Title[2] = 'The Last One: Part 2'



p = df3.plot.barh(y ='Stars',x='Episode_Title',figsize=(9,4), alpha=0.5, grid=True, title="Top ten episodes")

p = p.set_xlim(9,10)
for key, value in df['Summary'].iteritems(): 

    pass
#iterate and append to pandas new dataframe, busquem on surt la paraula "Janice" i creem una nova taula amb el que ens interesi

t = pd.DataFrame(columns=['Key', 'Season', 'Stars'])

columns = list(t)



for key, value in df['Summary'].iteritems(): 

    if "Janice" in value:

        t.loc[key] = [key ,df['Season'][key], df['Stars'][key]]



t.reset_index(drop = True)
#iterate and append to pandas new dataframe, busquem on surt la paraula "Janice" i creem una nova taula amb el que ens interesi

t = pd.DataFrame(columns=['Key', 'Season', 'Stars', 'Prota'])

columns = list(t)



for key, value in df['Summary'].iteritems(): 

    if "Janice" in value:

        t.loc[key] = [key ,df['Season'][key], df['Stars'][key], 'Janice']

    if "pregnant" in value:

        t.loc[key] = [key ,df['Season'][key], df['Stars'][key], 'pregnant']

    if "Richard" in value:

        t.loc[key] = [key ,df['Season'][key], df['Stars'][key], 'Richard']



t.reset_index(drop = True)
t = t.groupby('Prota')['Stars'].mean()

p = t.plot.bar(y ='Stars',figsize=(7,3), alpha=0.5, grid=True)
#llista de totes les paraules en summary segons la ocurrencia

list_word = df.Summary.str.split(expand=True).stack().value_counts()

list_word = list_word.drop(labels = ['to', 'and', 'a', 'the', 'is', 'for', 'her', 'his', 'with', 'an', 'their', 'when', 'of','in','get','at','he','up',])  

list_word[0:20]
#buscar les paraules de mes de 4 lletres

words4 = df['Summary'].str.findall('\w{4,}').str.join(' ')

words4[0:10]
words4b = words4.str.split(expand=True).stack().value_counts()

words4b[0:20]
type(words4)
#convert a list to whole text

words4_list = words4.tolist()

words4 = ["%s" % item for item in words4_list]

wholetext = ' '.join(words4)

str1[0:300]
from wordcloud import WordCloud, STOPWORDS



wordcloud_spam = WordCloud(background_color="white", stopwords = set(STOPWORDS)).generate(wholetext)



# Lines 2 - 5

plt.figure(figsize = (20,20))

plt.imshow(wordcloud_spam, interpolation='bilinear')

plt.axis("off")

plt.show()