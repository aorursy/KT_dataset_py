# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(rc={'figure.figsize':(10, 8)}); # you can change this if needed
df = pd.read_csv('../input/OnlineNewsPopularityReduced.csv')

df.head().T
df.info()
sns.boxplot(df['shares']);
sns.boxplot(df['n_tokens_title']);
sns.boxplot(df['num_videos']);
sns.boxplot(df['num_imgs']);
sns.boxplot(df['n_tokens_content']);
def outliers_indices(feature):



    mid = df[feature].mean()

    sigma = df[feature].std()

    return df[(df[feature] < mid - 3*sigma) | (df[feature] > mid + 3*sigma)].index
wrong_share = outliers_indices('shares')

wrong_vid = outliers_indices('num_videos')

wrong_img = outliers_indices('num_imgs')

wrong_content = outliers_indices('n_tokens_content')

wrong_title = outliers_indices('n_tokens_title')

out = set(wrong_share) | set(wrong_vid) | set(wrong_img) | set(wrong_content) | set(wrong_title)



print(len(out))
df.drop(out, inplace=True)
# Your code

cols = ['weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday', 'weekday_is_thursday',

               'weekday_is_friday', 'weekday_is_saturday', 'weekday_is_sunday'

               ]

df[cols].sum().plot(kind='bar')

plt.ylabel('count')

# Больше всего статей было опубликованно во вторник. Меньше всего статей было опубликонванно в субботу.
# Your code

df['n_tokens_title'].hist()
# sns.jointplot(x='n_tokens_title', y='shares', data=df);

# sns.set(rc={'figure.figsize':(15, 8)})

sns.regplot(x="n_tokens_title", y="shares", data=df)
cols = ['shares', 'n_tokens_title']

df[cols].corr(method='spearman')

# Длина заголовка плохо коррелирует с популярностью
cols = ['shares', 'num_videos', 'num_imgs']

df[cols].corr(method='spearman')

# Видим, что корреляция между кол-вом видео и популярностью больше, чем корреляция между кол-вом картинок и популярностью.
# Your code

a = df.groupby('num_videos')['shares'].mean()

b = df.groupby('num_imgs')['shares'].mean()

b = pd.DataFrame(b)

b.index.names = ['count']

b.columns = ['mean_shares_img']

a = pd.DataFrame(a)

a.index.names = ['count']

a.columns = ['mean_shares_videos']

frames = [a, b]

sns.set(rc={'figure.figsize':(25, 8)});

c = pd.merge(a, b, on='count', how='outer')

c.plot(kind='bar')

# При 1-2 картинке или видео. Статьи с видеороликами в среднем популярнее. При большем кол-во картинок или видео в статье - в среднем статьи с картинками популярнее, кроме пары случаев.
# sns.heatmap(df.groupby('is_weekend')['shares'].mean(), cmap="YlGnBu", annot=True, cbar=False);

# df.groupby('is_weekend')['shares'].mean()



sns.heatmap(df.pivot_table('shares', 'is_weekend', aggfunc='mean'), cmap="YlGnBu", annot=True, cbar=False)

sns.set(rc={'figure.figsize':(4, 4)});
# Your code

df.groupby('is_weekend')['shares'].mean().plot(kind='bar') 

plt.ylabel('shares')

sns.set(rc={'figure.figsize':(16, 10)});

# Да, в среднем статьи опубликованные в выходные имеют большую популярность.
cols = ['n_tokens_content', 'shares']

df[cols].corr(method='spearman')

# нет, прямой зависимости у кол-ва и популярности нет.
# Your code



sns.jointplot(x='n_tokens_content', y='shares', data=df);

# Пик популряности статей находится на промежутке до 1000 символов с статье.
# Your code

cols = ['data_channel_is_lifestyle', 'data_channel_is_entertainment', 'data_channel_is_bus', 'data_channel_is_socmed',

               'data_channel_is_tech']

df[cols].sum().plot(kind='pie', autopct='%1.1f%%')

sns.set(rc={'figure.figsize':(16, 10)});

# Больше всего статей было опубликованно по теме "развлечения" 36.4%, а меньше всего - соц. медиа 5.6%

cols = ['data_channel_is_lifestyle', 'data_channel_is_entertainment', 'data_channel_is_bus', 'data_channel_is_socmed',

               'data_channel_is_tech']

results = [df[df[x] == 1]['shares'].mean() for x in cols]

sns.barplot(x=cols, y=results)

sns.set(rc={'figure.figsize':(16, 10)});

# И среднем самые  статиь раздела соц медиа популярнее остальных. 




