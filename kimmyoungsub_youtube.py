

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import json

plt.style.use('seaborn')

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
US_youtube_data=pd.read_csv('../input/youtube-new/USvideos.csv')
US_youtube_data.info()
US_youtube_data.head()
US_youtube_data.isnull().sum()
##로그

US_youtube_data['likes_log'] = np.log(US_youtube_data['likes'] + 1)

US_youtube_data['views_log'] = np.log(US_youtube_data['views'] + 1)

US_youtube_data['dislikes_log'] = np.log(US_youtube_data['dislikes'] + 1)

US_youtube_data['comment_log'] = np.log(US_youtube_data['comment_count'] + 1)



## 그래프 사이즈

plt.figure(figsize = (12,6))



## 로그 정규분포 - views

plt.subplot(221)

g1 = sns.distplot(US_youtube_data['views_log'])

g1.set_title("VIEWS LOG DISTRIBUITION", fontsize=16)



## 로그 정규분포 - likes

plt.subplot(224)

g2 = sns.distplot(US_youtube_data['likes_log'],color='green')

g2.set_title('LIKES LOG DISTRIBUITION', fontsize=16)



## 로그 정규분포 - dislikes

plt.subplot(223)

g3 = sns.distplot(US_youtube_data['dislikes_log'], color='r')

g3.set_title("DISLIKES LOG DISTRIBUITION", fontsize=16)



## 로그 정규분포 - comment_counts

plt.subplot(222)

g4 = sns.distplot(US_youtube_data['comment_log'])

g4.set_title("COMMENTS LOG DISTRIBUITION", fontsize=16)



plt.subplots_adjust(wspace = 0.2, hspace = 0.4,top = 0.9)



plt.show()





US_youtube_data['category_name'] = np.nan



US_youtube_data.loc[(US_youtube_data["category_id"] == 1),"category_name"] = 'Film and Animation'

US_youtube_data.loc[(US_youtube_data["category_id"] == 2),"category_name"] = 'Cars and Vehicles'

US_youtube_data.loc[(US_youtube_data["category_id"] == 10),"category_name"] = 'Music'

US_youtube_data.loc[(US_youtube_data["category_id"] == 15),"category_name"] = 'Pets and Animals'

US_youtube_data.loc[(US_youtube_data["category_id"] == 17),"category_name"] = 'Sport'

US_youtube_data.loc[(US_youtube_data["category_id"] == 19),"category_name"] = 'Travel and Events'

US_youtube_data.loc[(US_youtube_data["category_id"] == 20),"category_name"] = 'Gaming'

US_youtube_data.loc[(US_youtube_data["category_id"] == 22),"category_name"] = 'People and Blogs'

US_youtube_data.loc[(US_youtube_data["category_id"] == 23),"category_name"] = 'Comedy'

US_youtube_data.loc[(US_youtube_data["category_id"] == 24),"category_name"] = 'Entertainment'

US_youtube_data.loc[(US_youtube_data["category_id"] == 25),"category_name"] = 'News and Politics'

US_youtube_data.loc[(US_youtube_data["category_id"] == 26),"category_name"] = 'How to and Style'

US_youtube_data.loc[(US_youtube_data["category_id"] == 27),"category_name"] = 'Education'

US_youtube_data.loc[(US_youtube_data["category_id"] == 28),"category_name"] = 'Science and Technology'

US_youtube_data.loc[(US_youtube_data["category_id"] == 29),"category_name"] = 'Non Profits and Activism'

US_youtube_data.loc[(US_youtube_data["category_id"] == 25),"category_name"] = 'News & Politics'
US_youtube_data['category_log'] = np.log(US_youtube_data['likes'] + 1)
print("Category Name count")

print(US_youtube_data.category_name.value_counts()[:5])



plt.figure(figsize = (14,9))



plt.subplot(211)

g = sns.countplot('category_name', data=US_youtube_data, palette="Set1")

g.set_xticklabels(g.get_xticklabels(),rotation=45)

g.set_title("Counting the Video Category's ", fontsize=15)

g.set_xlabel("", fontsize=12)

g.set_ylabel("Count", fontsize=12)





plt.show()
plt.figure(figsize = (14,9))

plt.subplot(212)

g1 = sns.boxplot(x='category_name', y='views_log', data=US_youtube_data, palette="Set1")

g1.set_xticklabels(g.get_xticklabels(),rotation=45)

g1.set_title("Views Distribuition by Category Names", fontsize=20)

g1.set_xlabel("", fontsize=15)

g1.set_ylabel("Views(log)", fontsize=15)

plt.show()
plt.figure(figsize = (14,6))



g = sns.boxplot(x='category_name', y='likes_log', data=US_youtube_data, palette="Set1")

g.set_xticklabels(g.get_xticklabels(),rotation=45)

g.set_title("Likes Distribuition by Category Names ", fontsize=15)

g.set_xlabel("", fontsize=12)

g.set_ylabel("Likes(log)", fontsize=12)

plt.show()
plt.figure(figsize = (14,6))



g = sns.boxplot(x='category_name', y='dislikes_log', data=US_youtube_data, palette="Set1")

g.set_xticklabels(g.get_xticklabels(),rotation=45)

g.set_title("Dislikes distribuition by Category's", fontsize=15)

g.set_xlabel("", fontsize=12)

g.set_ylabel("Dislikes(log)", fontsize=12)

plt.show()
plt.figure(figsize = (14,6))



g = sns.boxplot(x='category_name', y='comment_log', data=US_youtube_data, palette="Set1")

g.set_xticklabels(g.get_xticklabels(),rotation=45)

g.set_title("Comments Distribuition by Category Names", fontsize=15)

g.set_xlabel("", fontsize=12)

g.set_ylabel("Comments Count(log)", fontsize=12)



plt.show()
plt.figure(figsize = (10,8))



#Let's verify the correlation of each value

sns.heatmap(US_youtube_data[[ 'comment_log',

         'views_log','likes_log','dislikes_log', "category_name"]].corr(), annot=True)

plt.show()
plt.figure(figsize = (12,6))

plt.subplot(221)

sns.regplot(x=US_youtube_data['views_log'], 

           y=US_youtube_data['likes_log'], 

           fit_reg=True) # no regression line

plt.title('views_log between likes_log', fontsize=15)

plt.xlabel('views_log', fontsize=8)

plt.ylabel('likes_log', fontsize=8)





plt.subplot(222)

sns.regplot(x=US_youtube_data['views_log'], 

           y=US_youtube_data['dislikes_log'], 

           fit_reg=True) # no regression line

plt.title('views_log between dislikes_log', fontsize=15)

plt.xlabel('views_log', fontsize=8)

plt.ylabel('dislikes_log', fontsize=8)





plt.subplots_adjust(wspace = 0.2, hspace = 0.4,top = 0.9)

plt.show()





plt.figure(figsize = (12,6))

plt.subplot(221)

sns.regplot(x=US_youtube_data['comment_log'], 

           y=US_youtube_data['likes_log'], 

           fit_reg=True) # no regression line

plt.title('likes_log between comment_log', fontsize=15)

plt.xlabel('comment_log', fontsize=8)

plt.ylabel('likes_log', fontsize=8)





plt.subplot(222)

sns.regplot(x=US_youtube_data['comment_log'], 

           y=US_youtube_data['dislikes_log'], 

           fit_reg=True) # no regression line

plt.title('dislikes_log between comment_log', fontsize=15)

plt.xlabel('comment_log', fontsize=8)

plt.ylabel('dislikes_log', fontsize=8)





plt.subplot(223)

sns.regplot(x=US_youtube_data['comment_log'], 

           y=US_youtube_data['views_log'], 

           fit_reg=True) # no regression line

plt.title('views_log between comment_log', fontsize=15)

plt.xlabel('comment_log', fontsize=8)

plt.ylabel('views_log', fontsize=8)



plt.subplots_adjust(wspace = 0.2, hspace = 0.4,top = 0.9)

plt.show()





plt.figure(figsize = (12,6))

plt.subplot(221)

sns.regplot(x=US_youtube_data['likes_log'], 

           y=US_youtube_data['dislikes_log'], 

           fit_reg=True) # no regression line

plt.title('likes_log between dislikes_log', fontsize=15)

plt.xlabel('likes_log', fontsize=8)

plt.ylabel('dislikes_log', fontsize=8)

plt.show()