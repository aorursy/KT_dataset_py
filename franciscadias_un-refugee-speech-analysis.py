%matplotlib inline

import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

import seaborn as sns

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import *

from sklearn.cluster import KMeans

stop = set(stopwords.words("english"))

from wordcloud import WordCloud, STOPWORDS

from nltk import FreqDist

from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
df = pd.read_csv("../input/speeches.csv",encoding="UTF-8")

df["content"].head()
df.groupby(['author']).size()
with sns.color_palette("GnBu_d", 10):

    ax= sns.countplot(y="author",data=df)

    ax.set(xlabel='Number of Speaches', ylabel='High Commissioner´s Name')

plt.title("Number of Speeches given by each High Commissioner")
df["date"] = df['by'].str[-4:]



number_speechs_year = df.groupby(['date']).size()



with sns.color_palette("GnBu_d", 10):

    ax= sns.countplot(y="date",data=df)

    ax.set(xlabel='Number of Speaches', ylabel='High Commissioner´s Name')

    ax.figure.set_size_inches(20,12)

plt.title("Number of Speeches given each year")
def cleaning(s):

    s = str(s)

    s = s.rstrip('\n')

    s = s.lower()

    s = re.sub('\s\W',' ',s)

    s = re.sub('\W,\s',' ',s)

    s = re.sub(r'[^\w]', ' ', s)

    s = re.sub("\d+", "", s)

    s = re.sub('\s+',' ',s)

    s = re.sub('[!@#$_]', '', s)

    s = s.replace("co","")

    s = s.replace("https","")

    s = s.replace(",","")

    s = s.replace("[\w*"," ")

    return s

df['content'] = [cleaning(s) for s in df['content']]



df_date_1993 = df[(df["date"]=='1993')]

df_date_1993.head()
count_vect = CountVectorizer(min_df = 1, max_features = 500)

X_counts = count_vect.fit_transform(df_date_1993['content'])

X_counts = X_counts.toarray()

X_counts.shape

vocab = count_vect.get_feature_names()
dist = np.sum(X_counts, axis=0)



for tag, count in zip(vocab, dist):

    print(count, tag)
stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(df['content']))

plt.imshow(wordcloud)

plt.axis('off')

plt.title("Content of all speeches")
def cleaning(s):

    s = str(s)

    s = s.lower()

    s = re.sub('\s\W',' ',s)

    s = re.sub('\W,\s',' ',s)

    s = re.sub(r'[^\w]', ' ', s)

    s = re.sub("\d+", "", s)

    s = re.sub('\s+',' ',s)

    s = re.sub('[!@#$_]', '', s)

    s = s.replace("co","")

    s = s.replace("https","")

    s = s.replace(",","")

    s = s.replace("[\w*"," ")

    return s

df['content'] = [cleaning(s) for s in df['content']]

df['content'].head()
# Tokenization segments text into basic units (or tokens ) such as words and punctuation.

df['content'] = df.apply(lambda row: nltk.word_tokenize(row['content']),axis=1)



#  Removes stop words from column "content" and iterates over each row and item.

df['content'] = df['content'].apply(lambda x : [item for item in x if item not in stop])
df["Total_Words"] = df["content"].apply(lambda x : len(x))

df.loc[df['Total_Words'].idxmax()]
index_speech = df.loc[df['Total_Words'] == 11001]

content_index_speech = index_speech["content"]



allWords = []

for wordList in content_index_speech:

    allWords += wordList

fdist = FreqDist(allWords)



mostcommon = fdist.most_common(50)

mostcommon
wordcloud = WordCloud(

                          background_color='white',

                            max_words=50,

                         ).generate(str(mostcommon))

plt.figure(figsize=(15,7))

plt.imshow(wordcloud)

plt.axis('off')

plt.title('Most Common Words from the longest Speech ')

def grey_color_func(word, font_size, position,orientation,random_state=None, **kwargs):

    return("hsl(300,27,8)")

wordcloud.recolor(color_func = grey_color_func)