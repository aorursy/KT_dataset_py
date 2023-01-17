import numpy as np

import pandas as pd

import seaborn as sns

import random

import scipy.stats as stt

import warnings

warnings.filterwarnings('ignore')

%pylab inline
data = pd.read_csv('../input/athlete_events.csv')
data.head()
data.shape[0]
categ = pd.cut(data.Year,[1896,1916,1936,1956,1976,1996,2016])

pd.value_counts(categ).plot(kind='barh');
dat_new = data[['ID','Sex','Age','Height','Weight','NOC','Season','Sport','Medal']]

dat_new['Years'] = categ

plt.figure(figsize=(7,7))

#sns.boxplot(x='Years',y='Age',data=dat_new, width=0.7);

sns.violinplot(x='Years',y='Age',data=dat_new, width=0.7);
dat = dat_new.dropna()

sample = []

for j in range(6):

        sample.append(dat[dat.Years==dat.Years.unique()[j]].Age.values)

stt.f_oneway(*sample)
dat_new = data[['ID','Sex','Age','Height','Weight','NOC','Season','Sport','Medal']]

dat_new['Years'] = categ

plt.figure(figsize=(7,7))

sns.boxplot(x='Years',y='Weight',data=dat_new, width=0.7);

#sns.swarmplot(x='Years',y='Age',data=dat_new);

sample = []

for j in range(6):

        sample.append(dat[dat.Years==dat.Years.unique()[j]].Weight.values)

print(stt.f_oneway(*sample))

fig = plt.figure(figsize=(15,15))

ax1 = fig.add_subplot(3,3,1)

ax1.set_title('1996 - 2016')

pd.value_counts(data[data.Year>1996].Sport)[:10].plot(kind='barh');

ax2 = fig.add_subplot(3,3,3)

ax2.set_title('1976 - 1996')

pd.value_counts(data[(data.Year<1996) & (data.Year>1976)].Sport)[:10].plot(kind='barh');

ax3 = fig.add_subplot(3,3,4)

ax3.set_title('1956 - 1976')

pd.value_counts(data[(data.Year<1976) & (data.Year>1956)].Sport)[:10].plot(kind='barh');

ax4 = fig.add_subplot(3,3,6)

ax4.set_title('1936 - 1956')

pd.value_counts(data[(data.Year<1956) & (data.Year>1936)].Sport)[:10].plot(kind='barh');

ax5 = fig.add_subplot(3,3,7)

ax5.set_title('1916 - 1936')

pd.value_counts(data[(data.Year<1936) & (data.Year>1916)].Sport)[:10].plot(kind='barh');

ax6 = fig.add_subplot(3,3,9)

ax6.set_title('1896 - 1916')

pd.value_counts(data[(data.Year<1916)].Sport)[:10].plot(kind='barh');
grop_sport = data['Year'].groupby(data['Sport']).min()

pd.value_counts(grop_sport).plot.barh();
#fig = plt.figure(figsize=(15,15))

pd.value_counts(data[data.Medal.notnull()]['Team']).head(10).plot.bar();
fig = plt.figure(figsize=(15,15))

ax1 = fig.add_subplot(3,3,1)

ax1.set_title('Gold')

pd.value_counts(data[data.Medal=='Gold']['Team']).head(10).plot.bar();

ax2 = fig.add_subplot(3,3,2)

ax2.set_title('Silver')

pd.value_counts(data[data.Medal=='Silver']['Team']).head(10).plot.bar();

ax3 = fig.add_subplot(3,3,3)

ax3.set_title('Bronze')

pd.value_counts(data[data.Medal=='Bronze']['Team']).head(10).plot.bar();
fig = plt.figure(figsize=(17,17))

ax1 = fig.add_subplot(3,3,1)

ax1.set_title('Swimming')

pd.value_counts(data[(data.Medal.notnull()) & (data['Sport']=='Swimming')]['Team'])[:10].plot.barh();

ax2 = fig.add_subplot(3,3,2)

ax2.set_title('Gymnastics')

pd.value_counts(data[(data.Medal.notnull()) & (data['Sport']=='Gymnastics')]['Team'])[:10].plot.barh();

ax3 = fig.add_subplot(3,3,3)

ax3.set_title('Athletics')

pd.value_counts(data[(data.Medal.notnull()) & (data['Sport']=='Athletics')]['Team'])[:10].plot.barh();
ss = pd.value_counts(dat_new.Sport)[:10]

qq = list(ss.index)

for i in qq:

    print(i,data[data.Sport==i].Year.min())
sns.heatmap(data[['Age','Height','Weight']].corr());
sns.boxplot(x='Season',y='Age',hue='Sex',data=data);
sns.boxplot(x='Season',y='Weight',hue='Sex',data=data);

mal_weig = data.Weight[data.Sex=='M'].dropna()

fem_weig = data.Weight[data.Sex=='F'].dropna()

#mal_weig.head()
mal_weig = data.Weight[data.Sex=='M'].dropna()

fem_weig = data.Weight[data.Sex=='F'].dropna()

fig = plt.figure(figsize=(10,10))

fig.add_subplot(3,3,1)

mal_weig.hist(bins=30)

fem_weig.hist(bins=30)

fig.add_subplot(3,3,2)

stt.probplot(mal_weig, plot=pylab);

fig.add_subplot(3,3,3)

stt.probplot(fem_weig, plot=pylab);
mal_weig = np.log(data.Weight[data.Sex=='M'].dropna())

fem_weig = np.log(data.Weight[data.Sex=='F'].dropna())

fig = plt.figure(figsize=(10,10))

fig.add_subplot(3,3,1)

mal_weig.hist(bins=30)

fem_weig.hist(bins=30)

fig.add_subplot(3,3,2)

stt.probplot(mal_weig, plot=pylab);

fig.add_subplot(3,3,3)

stt.probplot(fem_weig, plot=pylab);
male_sample = mal_weig.iloc[[random.randint(0,mal_weig.shape[0]) for i in range(int(mal_weig.shape[0]*0.01))]]

female_sample = fem_weig.iloc[[random.randint(0,fem_weig.shape[0]) for i in range(int(fem_weig.shape[0]*0.01))]]



stt.ttest_ind_from_stats(male_sample.mean(),male_sample.std(),male_sample.shape[0],\

                         female_sample.mean(),female_sample.std(),female_sample.shape[0],equal_var=True)
mal_age = data.Age[data.Sex=='M'].dropna()

fem_age = data.Age[data.Sex=='F'].dropna()



male_sample = mal_age.iloc[[random.randint(0,mal_age.shape[0]) for i in range(int(mal_age.shape[0]*0.01))]]

female_sample = fem_age.iloc[[random.randint(0,fem_age.shape[0]) for i in range(int(fem_age.shape[0]*0.01))]]



stt.ttest_ind_from_stats(male_sample.mean(),male_sample.std(),male_sample.shape[0],\

                         female_sample.mean(),female_sample.std(),female_sample.shape[0],equal_var=True)
mal_height = data.Height[data.Sex=='M'].dropna()

fem_height = data.Height[data.Sex=='F'].dropna()



male_sample = mal_height.iloc[[random.randint(0,mal_height.shape[0]) for i in range(int(mal_height.shape[0]*0.01))]]

female_sample = fem_height.iloc[[random.randint(0,fem_height.shape[0]) for i in range(int(fem_height.shape[0]*0.01))]]



stt.ttest_ind_from_stats(male_sample.mean(),male_sample.std(),male_sample.shape[0],\

                         female_sample.mean(),female_sample.std(),female_sample.shape[0],equal_var=True)