import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('../input/tweets.csv')

summum_popularity = pd.DataFrame(data.groupby('username').followers.max())
summum_popularity.hist(bins=50)


small=500
mid=2000
big=10000
summum_popularity['starhood']='none'
summum_popularity.loc[summum_popularity.followers<small,'starhood']='small'
summum_popularity.loc[(summum_popularity.followers>small) & (summum_popularity.followers<mid),'starhood']='mid'
summum_popularity.loc[(summum_popularity.followers>mid) & (summum_popularity.followers<big),'starhood']='large'
summum_popularity.loc[(summum_popularity.followers>big),'starhood']='very large'
data.set_index('username', inplace=True)
data['starhood']=summum_popularity.starhood
data.reset_index(inplace=True)

print(data.groupby('starhood').username.unique().apply(lambda x: x.size))


data[data.starhood=='large'].groupby('name').username.last()

data[data.starhood=='large'].groupby('name').username.last()

grid = sns.FacetGrid(data=data, col='starhood',col_wrap=2, hue='username', sharex=False, sharey=False)
grid=grid.map(plt.scatter,'numberstatuses','followers')
grid.axes[2].legend(loc='upper left')
grid.axes[3].legend(loc='upper left')
grid.fig.set_size_inches((14,7))
rami_data = data[data.username=='RamiAlLolah']
pd.to_datetime(rami_data.time)
rami_data.time=pd.to_datetime(rami_data.time)
rami_data.set_index('time', inplace=True)
impact = ((rami_data.followers-rami_data.followers.shift(-1))/(rami_data.numberstatuses-rami_data.numberstatuses.shift(-1))).dropna()
impact.plot(kind='bar')

for i in range(0, 6):
    print(rami_data[rami_data.index == impact.sort_values().index[-i]].tweets.get_values())