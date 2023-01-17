# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import pandas as pd

import numpy as np



from nltk.corpus import stopwords

from nltk.tokenize import RegexpTokenizer

from nltk.stem.wordnet import WordNetLemmatizer



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.cluster import KMeans

from sklearn import metrics



from wordcloud import WordCloud

import matplotlib.pyplot as plt

import seaborn as sns





def word_freq_plot(top_w):

    top_df = pd.DataFrame(top_w)

    top_df.columns = ["Word", "Freq"]

    sns.set(rc={'figure.figsize': (13, 8)})

    g = sns.barplot(x="Word", y="Freq", data=top_df)

    g.set_xticklabels(g.get_xticklabels(), rotation=30)





def get_top_n_words(bow, vec, n=None):

    sum_words = bow.sum(axis=0)

    words_freq = [(word, sum_words[0, idx]) for word, idx in

                   vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1],

                       reverse=True)

    return words_freq[:n]





def word_cloud_gen(token_list):

    wordcloud = WordCloud(background_color='white',

                          stopwords=stop_words,

                          max_words=100,

                          max_font_size=50,

                          random_state=42

                          ).generate(' '.join(token_list))



    print(wordcloud)

    fig = plt.figure(1)

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()

    fig.savefig("word1.png", dpi=900)
df = pd.read_csv('../input/listings.csv')

host_desc = df[['host_id', 'host_about']]   # Host's about-descriptions and IDs

host_desc.drop_duplicates(inplace=True)     # drop duplicate entries of host IDs and abouts

host_desc.dropna(inplace=True)              # drop NaNs

"""Fetch wordcount for each review. Add it to the DataFrame"""

host_desc['word_count'] = host_desc['host_about'].apply(lambda x: len(str(x).split(" ")))

print(host_desc['word_count'].describe())

"""Data Cleaning and word/token frequency calculation"""

whole_text = ' '.join(host_desc['host_about'])

tokenizer2 = RegexpTokenizer(r'\w+')

tokens = tokenizer2.tokenize(whole_text)

freq = pd.Series(tokens).value_counts()

print(freq)
# Stop words

# Creating a list of stop words and adding custom stopwords

stop_words = set(stopwords.words("english"))

# Creating a list of custom stopwords obtained through careful selection after many trials

new_words = ['love', 'seattle', 'home', 'year', 'enjoy', 'new', 'place', 'time', 'city',

             'like', 'make', 'live', 'airbnb', 'great', 'also', 'world', 'guest', 'host',

             'lived', 'around', 'thing', 'one', 'two', 'many']

stop_words = stop_words.union(new_words)
desc_list = []

for text in host_desc['host_about'].values:

    token_list = tokenizer2.tokenize(text)  # Removes punctuations and special characters

    token_list_lowered = [s.lower() for s in token_list]

    lem = WordNetLemmatizer()  # Lemmatization

    # Removal of stop words

    tokens_final = [lem.lemmatize(word) for word in token_list_lowered if not word in stop_words]

    stringed = ' '.join(tokens_final)

    desc_list.append(stringed)

# print(desc_list)
all_tokens = ' '.join(desc_list).split()

frequency = pd.Series(all_tokens).value_counts()

print(frequency)



"""Visualization of most frequent words in word-cloud form"""

word_cloud_gen(all_tokens)
"""Tokenization"""

cv = CountVectorizer(max_df=1.0,stop_words=stop_words, max_features=10000, ngram_range=(1,3))

bag_of_words = cv.fit_transform(desc_list)

print(list(cv.vocabulary_.keys())[:10])



"""Convert most freq words to dataframe for plotting bar plot"""

top_words = get_top_n_words(bag_of_words, cv, n=20)

word_freq_plot(top_words)
# TF-IDF

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)

tfidf_transformer.fit(bag_of_words)

# get feature names

feature_names=cv.get_feature_names()

# generate tf-idf for the given list of about descriptions

tf_idf_vector=tfidf_transformer.transform(bag_of_words)
"""KMeans Clustering"""

cal_score = []

for k in np.arange(11)+2:

    cluster_list = KMeans(init='k-means++', n_clusters=k).fit_predict(tf_idf_vector.A)

    assert len(cluster_list) == len(desc_list)

    s_score = metrics.silhouette_score(tf_idf_vector.A, cluster_list)

    cal_score.append(s_score)

    print(s_score)

"""Elbow method for determining number of clusters - k"""

idxs = np.arange(1, len(cal_score)+1)

plt.plot(idxs, cal_score)

acceleration = np.diff(cal_score, 2)  # 2nd derivative of the distances

plt.plot(idxs[:-2] + 1, acceleration, marker='*')

plt.show()

k = acceleration.argmin() + 2  # if idx 0 is the max of this we want 2 clusters

print("clusters:"+str(k))
"""Write the resulted host groups in csv"""

cluster_list = KMeans(n_clusters=k, init='k-means++').fit_predict(tf_idf_vector.A)

host_desc['labels'] = list(cluster_list)

host_desc.to_csv('clustered_hosts.csv', ',')