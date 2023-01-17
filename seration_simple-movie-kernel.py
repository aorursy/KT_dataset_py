import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))



data = pd.read_csv('../input/tmdb_5000_movies.csv')



del data['id']
data.columns
data.head()
filter_of_vote_count = data.vote_count > 1000

filtered_data_vote_count = data[filter_of_vote_count] 

filtered_data_vote_count[['title', 'popularity','vote_count', 'vote_average']].nlargest(15, 'vote_average')
def to_date(series):

    return str(series)[:4]



data['release_date'] = data['release_date'].apply(to_date)

xx = data.groupby('release_date').vote_average.mean().reset_index()



xx.tail(30)
filtered_data =  data.nlargest(15, 'revenue')



plt.rcdefaults()

fig, ax = plt.subplots()



# Example data

title = filtered_data.title

y_pos = np.arange(len(title))

revenue = filtered_data.revenue

error = np.random.rand(len(title))



ax.barh(y_pos, revenue, xerr=error, align='center',

        color='blue', ecolor='black')

ax.set_yticks(y_pos)

ax.set_yticklabels(title)

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Revenue (Billion $)')

ax.set_title('Who earned more?')



plt.show()

# film surelerine gore puan ortalamasi

data_new = data.groupby('vote_average').runtime.mean().reset_index()

data_new = data_new.drop([69,70],axis=0)

data_new.tail(15).plot(kind='scatter', x='runtime', y='vote_average',color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')

plt.xlabel('Duration(minute)')

plt.ylabel('Raking')

plt.title('Movies Duration By Ranking')

plt.show()



most_budget_data = data.nlargest(15, 'budget')



fig, axs = plt.subplots()



budget = most_budget_data.title

y_pos = np.arange(len(budget))

budget = most_budget_data.budget

error = np.random.rand(len(budget))



axs.barh(y_pos, budget, xerr=error, align='center',

        color='red', ecolor='black')

axs.set_yticks(y_pos)

axs.set_yticklabels(most_budget_data.title)

axs.invert_yaxis()  # labels read top-to-bottom

axs.set_xlabel('Revenue (Billion $)')

axs.set_title('Who spent more?')



plt.show()





data[['title', 'popularity','vote_count', 'vote_average','budget','revenue']].nlargest(15, 'popularity')
f,ax = plt.subplots(figsize=(18, 12))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
