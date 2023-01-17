# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib as mpl

import os

from subprocess import check_output

from wordcloud import WordCloud, STOPWORDS
with open("../input/youtube-new/KRvideos.csv", "rt", encoding="utf8",errors='ignore') as f:

    df = pd.read_csv(f)



print(df.shape)

df.head(10)

df_yout = df
df_yout['likes_log'] = np.log(df_yout['likes'] + 1)

df_yout['views_log'] = np.log(df_yout['views'] + 1)

df_yout['dislikes_log'] = np.log(df_yout['dislikes'] + 1)

df_yout['comment_log'] = np.log(df_yout['comment_count'] + 1)



plt.figure(figsize = (12,6))



plt.subplot(221)

g1 = sns.distplot(df_yout['views_log'])

g1.set_title("VIEWS LOG DISTRIBUITION", fontsize=16)



plt.subplot(224)

g2 = sns.distplot(df_yout['likes_log'],color='green')

g2.set_title('LIKES LOG DISTRIBUITION', fontsize=16)



plt.subplot(223)

g3 = sns.distplot(df_yout['dislikes_log'], color='r')

g3.set_title("DISLIKES LOG DISTRIBUITION", fontsize=16)



plt.subplot(222)

g4 = sns.distplot(df_yout['comment_log'])

g4.set_title("COMMENTS LOG DISTRIBUITION", fontsize=16)



plt.subplots_adjust(wspace = 0.2, hspace = 0.4,top = 0.9)



plt.show()
df_yout['category_name'] = np.nan



df_yout.loc[(df_yout["category_id"] == 1),"category_name"] = 'Film and Animation'

df_yout.loc[(df_yout["category_id"] == 2),"category_name"] = 'Cars and Vehicles'

df_yout.loc[(df_yout["category_id"] == 10),"category_name"] = 'Music'

df_yout.loc[(df_yout["category_id"] == 15),"category_name"] = 'Pets and Animals'

df_yout.loc[(df_yout["category_id"] == 17),"category_name"] = 'Sport'

df_yout.loc[(df_yout["category_id"] == 19),"category_name"] = 'Travel and Events'

df_yout.loc[(df_yout["category_id"] == 20),"category_name"] = 'Gaming'

df_yout.loc[(df_yout["category_id"] == 22),"category_name"] = 'People and Blogs'

df_yout.loc[(df_yout["category_id"] == 23),"category_name"] = 'Comedy'

df_yout.loc[(df_yout["category_id"] == 24),"category_name"] = 'Entertainment'

df_yout.loc[(df_yout["category_id"] == 25),"category_name"] = 'News and Politics'

df_yout.loc[(df_yout["category_id"] == 26),"category_name"] = 'How to and Style'

df_yout.loc[(df_yout["category_id"] == 27),"category_name"] = 'Education'

df_yout.loc[(df_yout["category_id"] == 28),"category_name"] = 'Science and Technology'

df_yout.loc[(df_yout["category_id"] == 29),"category_name"] = 'Non Profits and Activism'

df_yout.loc[(df_yout["category_id"] == 25),"category_name"] = 'News & Politics'
print("Category Name count")

print(df_yout.category_name.value_counts()[:5])



plt.figure(figsize = (14,9))



plt.subplot(211)

g = sns.countplot('category_name', data=df_yout, palette="Set1")

g.set_xticklabels(g.get_xticklabels(),rotation=45)

g.set_title("Counting the Video Category's ", fontsize=15)

g.set_xlabel("", fontsize=12)

g.set_ylabel("Count", fontsize=12)



plt.subplot(212)

g1 = sns.boxplot(x='category_name', y='views_log', data=df_yout, palette="Set1")

g1.set_xticklabels(g.get_xticklabels(),rotation=45)

g1.set_title("Views Distribuition by Category Names", fontsize=20)

g1.set_xlabel("", fontsize=15)

g1.set_ylabel("Views(log)", fontsize=15)



plt.subplots_adjust(hspace = 0.9, top = 0.9)



plt.show()
plt.figure(figsize = (14,6))



g = sns.boxplot(x='category_name', y='likes_log', data=df_yout, palette="Set1")

g.set_xticklabels(g.get_xticklabels(),rotation=45)

g.set_title("Likes Distribuition by Category Names ", fontsize=15)

g.set_xlabel("", fontsize=12)

g.set_ylabel("Likes(log)", fontsize=12)

plt.show()
plt.figure(figsize = (14,6))



g = sns.boxplot(x='category_name', y='dislikes_log', data=df_yout, palette="Set1")

g.set_xticklabels(g.get_xticklabels(),rotation=45)

g.set_title("Dislikes distribuition by Category's", fontsize=15)

g.set_xlabel("", fontsize=12)

g.set_ylabel("Dislikes(log)", fontsize=12)

plt.show()
mpl.rcParams['font.size']= 15              

mpl.rcParams['savefig.dpi']= 100         

mpl.rcParams['figure.subplot.bottom']= .1 
yout_string = str(df_yout['title'])

print(yout_string)
plt.figure(figsize = (15,15))



stopwords = set(STOPWORDS)



#font settings

wordcloud = WordCloud(

                          background_color='black',font_path='../input/nanum34/NanumSquareR.ttf',

                          stopwords=stopwords,

                          max_words=1000,

                          max_font_size=120, 

                          random_state=42

                         ).generate(str(df_yout['title']))



print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.title("WORD CLOUD - TITLES")

plt.axis('off')

plt.show()
plt.figure(figsize = (15,15))



stopwords = set(STOPWORDS)



newStopWords= ['https', 'youtube', 'VIDEO','youtu','CHANNEL', 'WATCH']



stopwords.update(newStopWords)



wordcloud = WordCloud(

                          background_color='black',font_path='../input/nanum34/NanumSquareR.ttf',

                          stopwords=stopwords,

                          max_words=1200,

                          max_font_size=80, 

                          random_state=42

                         ).generate(str(df_yout['description']))



print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.title("WORD CLOUD - DESCRIPTION")

plt.axis('off')

plt.show()