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
df = pd.read_csv("../input/OnlineNewsPopularityReduced.csv")
df.head()
weekday = ['weekday_is_monday',	

           'weekday_is_tuesday',	

           'weekday_is_wednesday',	

           'weekday_is_thursday',	

           'weekday_is_friday',	

           'weekday_is_saturday',	

           'weekday_is_sunday',	

           'is_weekend',

          ]

plt.bar(x=weekday, height=df[weekday].sum())

plt.xticks(rotation=45)

plt.show()
df['n_tokens_title'].hist(bins=20)
plt.scatter(df['n_tokens_title'], df['shares'])
have_img = df[df['num_imgs']!=0]['shares'].sum()

non_have_img = df[df['num_imgs']==0]['shares'].sum()



have_vid = df[df['num_videos']!=0]['shares'].sum()

non_have_vid = df[df['num_videos']==0]['shares'].sum()



plt.bar(x = ['have_img', 'non_have_img', 'have_vid', 'non_have_vid'], height= [have_img, non_have_img, have_vid, non_have_vid], color=['green', 'Fuchsia', 'red', 'yellow'])



plt.show()







weekday = ['weekday_is_monday',	

           'weekday_is_tuesday',	

           'weekday_is_wednesday',	

           'weekday_is_thursday',	

           'weekday_is_friday',	

          ]

weekend  = 'is_weekend'

sum_share_on_weekday = 0

for w in weekday:

    sum_share_on_weekday += df[df[w] == 1]['shares'].sum()

    

sum_share_on_weekend  = df[df[weekend] == 1]['shares'].sum()

print(sum_share_on_weekday, sum_share_on_weekend)

print(sum_share_on_weekday + sum_share_on_weekend)

print(df['shares'].sum())



plt.bar(x=['weekday', 'weekend'], height=[sum_share_on_weekday/5, sum_share_on_weekend/2], color=['blue', 'green'])

plt.xticks(rotation=45)

plt.show()
plt.hist(df['shares'], bins=2000, normed=True)

plt.xlim(0, 10000)

df['shares'].describe()

plt.hist(df['n_tokens_content'], bins=50)

plt.xlim(0, 2000)

df['shares'].describe()

from scipy.stats import pearsonr, spearmanr, kendalltau

r = pearsonr(df['n_tokens_content'], df['shares'])

print('Pearson correlation:', r[0], 'p-value:', r[1])

r = spearmanr(df['n_tokens_content'], df['shares'])

print('Spearmanr correlation:', r[0], 'p-value:', r[1])

r = kendalltau(df['n_tokens_content'], df['shares'])

print('Kendalltau correlation:', r[0], 'p-value:', r[1])
fig = plt.figure(figsize=(11,11))





# # Отфильтруем строки в который нет ни одного коэф. корр. меньше koef

koef = 0.3



corr = df.corr()

corr = corr[corr.abs() > koef]

corr = corr[(corr.sum(axis=1)!=0)]



mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    f, ax = plt.subplots(figsize=(11, 11))

    ax = sns.heatmap(corr, mask=mask, square=True)
df.columns
cat = df[['data_channel_is_entertainment', 

          'data_channel_is_bus',

          'data_channel_is_socmed', 

          'data_channel_is_tech',

          'data_channel_is_world']].sum(axis=1)



cat.unique()