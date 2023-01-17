# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

from wordcloud import WordCloud

from nltk.corpus import stopwords



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

file_ext = []

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         # print(os.path.join(dirname, filename))

#         file_nm, ext = os.path.splitext(filename)

#         file_ext.append(ext)

#         if ext == ".csv":

#             df = pd.read_csv(os.path.join(dirname, filename))

#             df.head()

#             break;

#print(set(file_ext))



#dir_name = "/kaggle/input/CORD-19-research-challenge/cord_19_embeddings/"

dir_name = "/kaggle/input/CORD-19-research-challenge/"

# file_name = "cord_19_embeddings_2020-05-19.csv"

file_name = "metadata.csv"

df = pd.read_csv(os.path.join(dir_name, file_name))

print(list(df.columns))

# print(df.describe(include='all'))

series_title = df.loc[:, ["title"]].dropna()

series_abstract = df.loc[:, ["abstract"]].dropna()



# print(series_title.info())



titles = ' '.join(series_title["title"])

abstracts = ' '.join(series_abstract["abstract"])





stopword = stopwords.words('english') 



wordcloud_title = WordCloud(max_font_size=None, background_color='white', 

                      collocations=False, stopwords=stopword,

                      width=1000, height=1000).generate(titles)





wordcloud_abstract = WordCloud(max_font_size=None, background_color='white', 

                      collocations=False, stopwords=stopword,

                      width=1000, height=1000).generate(abstracts)





plt.figure(figsize=(15,15))

plt.subplot(1,2,1)

plt.axis("off")

plt.imshow(wordcloud_title)

plt.title('Common Words in Title')

plt.subplot(1,2,2)

plt.axis("off")

plt.imshow(wordcloud_abstract)

plt.title('Common Words in Abstract')

plt.show()





# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print(df['publish_time'].describe())

def get_year(x):

    # print(x)

    y = str(x).split('-')

    return y[0]

df_publish_year = df['publish_time'].apply(lambda x: get_year(x))

df_publish_year = df_publish_year.value_counts()



df_publish_year.plot(title="Year Wise Article Count", figsize=(15,10))

df_author = df['authors'].dropna().apply(lambda a: str(a).strip().lower()).value_counts()

print(df_author.head(10).sort_values().plot(kind='barh', title='TOP 10 AUTHOR', figsize=(15,8)))
df_topic = df['title'].dropna().apply(lambda a: str(a).strip().lower()).value_counts()

print(df_topic.head(10).sort_values().plot(kind='barh', title='Most Frequent title', figsize=(15,8)))