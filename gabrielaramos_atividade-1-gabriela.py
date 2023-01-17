import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

import re



video_games = pd.read_csv('https://raw.githubusercontent.com/wtamu-cisresearch/scraper/master/gamedata-20140215-11_53_10.csv',skiprows=[0])



#variaveis númericas importantes para responder as perguntas:

#country sales

#critic score

#release year



#variaveis categoricas importantes para responder as perguntas:

#genre

#rating

#publisher

#user score



video_games.head()
#describe: irá retornar os dados estatisticos das variaveis numéricas

video_games.describe()

#Os valores de vendas estão em milhões. O que é impactante no histograma, pois há concentração de valores.
#analisando a distribuição de todas as variaveis numericas, para focar nas variaveis importantes às perguntas.

video_games.hist(figsize=(15, 15))

plt.show()
year = video_games[~video_games['release date'].isna()]

year = year[year['release date'].apply(lambda x: re.search(r"[A-Z][a-z]{2}\s+\d+,\s+\d{4}", x) is not None)]

plt.plot([min(year['release year']), max(year['release year'])], [min(year['release year']), max(year['release year'])])

plt.scatter(x=year['release year'], y = year['release date'].apply(lambda y : int(y[-4:])))

plt.show()
#analisando  a distribuição do ano de lançamento

video_games['release year'].plot.hist()

release = video_games[video_games['release year'] != 2020]
video_games.groupby('rating')['release year'].describe()
#Para a análise do ano de lançamento foi retirado o valor de 'release year' 2020, pois notou-se é um outlier.

sns.boxplot(x= 'rating', y= 'release year', data = release)
#analisar as notas dos criticos e usuarios por plataforma:

video_games['user score'].unique()



#há valores 'tbd': o que significa 'to be determined': https://www.metacritic.com/faq#item13

score = video_games[video_games['user score'] != 'tbd']

score['user score'] = score['user score'].astype(float)

score.info()
#comparando a nota dada pelos criticos e usuários: 

#obs.: >0 a nota do crítico foi maior que a dos usuários.

#      <0 a nota do crítico foi menor que a dos usuários.

score['diff score'] = score['critic score']/10 - score['user score']

score = score[~score['diff score'].isna()]
sns.boxplot(y ='diff score', data=score)
#Exclui valores que não possuem diferença entre notas consideradas outliers.

critic_score = score[score['diff score'] > 3]

user_score = score[score['diff score'] <-3]



#Analisando a proporção de jogos que tiveram as notas dos críticos maiores que as dos usuários

prop_critic = (critic_score.groupby('publisher')['name'].count()/video_games.groupby('publisher')['name'].count()).dropna().to_frame()

prop_critic.columns = ['prop_games']



#Analisando a proporção de jogos que tiveram as notas dos criticos maiores que as dos usuários

prop_user = (user_score.groupby('publisher')['name'].count()/video_games.groupby('publisher')['name'].count()).dropna().to_frame()

prop_user.columns = ['prop_games']
plt.figure(figsize=(20, 6))

sns.barplot(x=prop_critic.index, y = 'prop_games', data=prop_critic).set_xticklabels(prop_critic.index, rotation=30)

plt.title("Proporção de jogos com notas dos críticos maiores que as dos usuários por editora")

plt.show()
plt.figure(figsize=(20, 6))

sns.barplot(x=prop_user.index, y = 'prop_games', data=prop_user).set_xticklabels(prop_user.index, rotation=30)

plt.title("Proporção de jogos com notas dos usuários maiores que as dos críticos por editora")

plt.show()
sns.boxplot(x='genre', y='global sales', data=video_games).set_xticklabels(video_games['genre'],rotation=30)

plt.show()
NA_sales = video_games.groupby('genre')['north america sales'].sum()/video_games['north america sales'].sum()

EU_sales = video_games.groupby('genre')['europe sales'].sum()/video_games['europe sales'].sum()

JP_sales = video_games.groupby('genre')['japan sales'].sum()/video_games['japan sales'].sum()

Rest_sales = video_games.groupby('genre')['rest of world sales'].sum()/video_games['rest of world sales'].sum()
fig, ax = plt.subplots(6,2,figsize=(20, 20)) 

color = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf','#cd715e','#8588f5']



for i,genre in enumerate(video_games['genre'].unique()):

    

    y = [NA_sales[genre], EU_sales[genre], JP_sales[genre], Rest_sales[genre]]

    x = ['North America', 'Europe', 'Japan', 'Rest of world']

    ax[i//2,i%2].bar(x,y,color=color[i])

    ax[i//2,i%2].title.set_text(genre)



plt.show()