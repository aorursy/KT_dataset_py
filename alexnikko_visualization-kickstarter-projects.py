%matplotlib inline



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

sns.set(style="ticks")
data = pd.read_csv('../input/kickstarter-projects/ks-projects-201801.csv')
data.head()
data.columns
data.goal = data.usd_goal_real

data.pledged = data.usd_pledged_real
data.drop(columns=['ID', 'usd pledged', 'usd_pledged_real', 'usd_goal_real', 'currency'], inplace=True)
data.head()
state = round(data["state"].value_counts() / len(data["state"]) * 100, 2)

ax = sns.barplot(state.index, state.values)

ax.set(ylabel='%')

plt.show()
data = data.loc[data.state.isin(['successful', 'failed'])]
failed = np.log1p(data[data.state == 'failed'].goal)

success = np.log1p(data[data.state == 'successful'].goal)

sns.distplot(failed, bins=30, norm_hist=True)

sns.distplot(success, bins=30, norm_hist=True)

plt.legend(title='State', loc='upper left', labels=['failed', 'successful'])

plt.show()
goal = np.log1p(data.goal)

pledged = np.log1p(data.pledged)

fig, axs = plt.subplots(figsize=[20, 10], ncols=2)

sns.distplot(goal, bins=30, norm_hist=True, ax=axs[0])

sns.distplot(pledged, bins=30, norm_hist=True, ax=axs[1])

plt.show()
sns.boxplot(x=data.state, y=goal)

plt.ylabel('goal_log')

plt.title('Goal')

plt.show()
sns.boxplot(x=data.state, y=pledged)

plt.ylabel('pledged_log')

plt.title('Pledged')

plt.show()
sns.scatterplot(x=goal, y=pledged)

plt.title('goal x pledged')

plt.xlabel('goal_log')

plt.ylabel('pledged_log')

plt.show()
main_cats = data.main_category.value_counts()

main_cats_fail = data[data.state == 'failed'].main_category.value_counts()

main_cats_suc = data[data.state == 'successful'].main_category.value_counts()
ax = sns.barplot(x=main_cats_suc.index, y=main_cats_suc.values)

ax.set_xticklabels(main_cats_suc.index, rotation=90)

plt.title('Successful')

plt.show()
ax = sns.barplot(x=main_cats_fail.index, y=main_cats_fail.values)

ax.set_xticklabels(main_cats_fail.index, rotation=90)

plt.title('Failed')

plt.show()
ax = sns.barplot(x=main_cats.index, y=main_cats.values)

ax.set_xticklabels(main_cats.index, rotation=90)

plt.title('General')

plt.show()
suc_music = data[(data.main_category == 'Music') & (data.state == 'successful')]

suc_film = data[(data.main_category == 'Film & Video') & (data.state == 'successful')]

plt.figure(figsize=(14,16))

plt.subplot(211)

ax0 = sns.countplot(x='category', data=suc_music, color='cyan')

ax0.set_xticklabels(ax0.get_xticklabels(), rotation=45)

ax0.set_title("Успешные подкатегории Music", fontsize=22)

ax0.set_xlabel("Подкатегории Music", fontsize=15)

ax0.set_ylabel("Количество", fontsize=15)

sizes=[]

for p in ax0.patches:

    height = p.get_height()

    sizes.append(height)

    ax0.text(p.get_x()+p.get_width()/2, height + 3, f'{height/len(suc_music)*100:.2f}%', ha="center", fontsize=12) 

ax0.set_ylim(0, max(sizes) * 1.15)

plt.subplot(212)

ax1 = sns.countplot(x='category', data=suc_film, color='cyan')

ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

ax1.set_title('Успешные подкатегории Film & Video', fontsize=22)

ax1.set_xlabel('Подкатегории Film & Video', fontsize=15)

ax1.set_ylabel('Количество', fontsize=15)

sizes=[]

for p in ax1.patches:

    height = p.get_height()

    sizes.append(height)

    ax1.text(p.get_x()+p.get_width()/2, height + 3, f'{height/len(suc_film)*100:.2f}%', ha='center', fontsize = 12)

ax1.set_ylim(0, max(sizes) * 1.15)



plt.subplots_adjust(wspace = 0.3, hspace = 0.6,top = 0.9)

plt.show()
fail_film = data[(data.main_category == 'Film & Video') & (data.state == 'failed')]

fail_pub = data[(data.main_category == 'Publishing') & (data.state == 'failed')]

plt.figure(figsize=(14,16))

plt.subplot(211)

ax0 = sns.countplot(x='category', data=fail_film, color='cyan')

ax0.set_xticklabels(ax0.get_xticklabels(), rotation=45)

ax0.set_title('Неуспешные подкатегории Film & Video', fontsize=22)

ax0.set_xlabel('Подкатегории Film & Video', fontsize=15)

ax0.set_ylabel('Количество', fontsize=15)

sizes=[]

for p in ax0.patches:

    height = p.get_height()

    sizes.append(height)

    ax0.text(p.get_x()+p.get_width()/2, height + 3, f'{height/len(fail_film)*100:.2f}%', ha='center', fontsize = 12)

ax0.set_ylim(0, max(sizes) * 1.15)



plt.subplot(212)

ax1 = sns.countplot(x='category', data=fail_pub, color='cyan')

ax1.set_xticklabels(ax0.get_xticklabels(), rotation=45)

ax1.set_title('Неуспешные подкатегории Publishing', fontsize=22)

ax1.set_xlabel('Подкатегории Publishing', fontsize=15)

ax1.set_ylabel('Количество', fontsize=15)

sizes=[]

for p in ax1.patches:

    height = p.get_height()

    sizes.append(height)

    ax1.text(p.get_x()+p.get_width()/2, height + 3, f'{height/len(fail_pub)*100:.2f}%', ha='center', fontsize = 12)

ax1.set_ylim(0, max(sizes) * 1.15)



plt.subplots_adjust(wspace = 0.3, hspace = 0.6,top = 0.9)

plt.show()
data.columns
data.launched = pd.to_datetime(data.launched)

data.deadline = pd.to_datetime(data.deadline)



data['lon_year'] = data.launched.dt.year

data['lon_month'] = data.launched.dt.month

year = data.lon_year.value_counts()

month = data.lon_month.value_counts()
plt.figure(figsize=(12,10))

sns.countplot(x="lon_year", hue='state', data=data)

plt.title('Количество проектов в год')

plt.show()
plt.figure(figsize=(12,10))

sns.countplot(x='lon_month', hue='state', data=data)

plt.title('Количество проектов по месяцам')

plt.show()
# посчитаем сколько длился сбор средств (длительность кампании)

data['time'] = (data['deadline'] - data['launched']).dt.days

data['time'] = data['time'].astype(int)



data = data[data['time'] != 14867]

data['time'] = round(data['time'] / 30 )
plt.figure(figsize = (12,5))

sns.countplot(x='time', hue='state', data=data)

plt.title('Распределение проектов по длительности кампании')

plt.show()
plt.figure(figsize = (14,17))



plt.subplot(211)

g =sns.boxplot(x='state', y=np.log1p(data.goal), data=data, hue='time')

g.set_title("Распределение целей успешных и неуспешных проектов в зависимости от времени кампании", fontsize=22)

g.set_xlabel("", fontsize=17)

g.set_ylabel("goal_log", fontsize=17)

g.legend(loc='upper right')



plt.subplot(212, sharex=g)

g1 = sns.boxplot(x='state', y=np.log1p(data.pledged), data=data, hue='time')

g1.set_title("Распределение собранных средств успешных и неуспешных проектов в зависимости от времени кампании", fontsize=22)

g1.set_xlabel("", fontsize=17)

g1.set_ylabel("pledged_log", fontsize=17)





plt.subplots_adjust(hspace = 0.50, top = 0.8)

plt.show()
plt.figure(figsize = (12,6))

ax = sns.distplot(np.log1p(data.backers))

ax.set_xlabel("Распределение", fontsize=17)

ax.set_ylabel("Частота", fontsize=17)

ax.set_title("Распределение покровителей (логарифм)", fontsize=22)

plt.show()
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
text = suc_music.category.to_string()

wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()