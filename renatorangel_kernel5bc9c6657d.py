import numpy as np 

import pandas as pd 

import seaborn as sns

from nltk.corpus import stopwords

stop = stopwords.words('english')



import scipy.cluster.hierarchy as shc

from sklearn.cluster import AgglomerativeClustering



import matplotlib.pyplot as plt



plt.rcParams["figure.figsize"] = (20,10)



from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer



pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# got from here https://medium.com/@cristhianboujon/how-to-list-the-most-common-words-from-text-corpus-using-scikit-learn-dad4d0cab41d

def get_top_n_words(corpus, n=None):

    """

    List the top n words in a vocabulary according to occurrence in a text corpus.

    

    get_top_n_words(["I love Python", "Python is a language programming", "Hello world", "I love the world"]) -> 

    [('python', 2),

     ('world', 2),

     ('love', 2),

     ('hello', 1),

     ('is', 1),

     ('programming', 1),

     ('the', 1),

     ('language', 1)]

    """

    

    regex1 = '[a-zA-Z]{1,50}'

#     vectorizer = CountVectorizer(analyzer='word', stop_words = 'english', token_pattern  = regex1)

#     vectorizer1 = vectorizer.fit_transform(doc1)

    

    vec = CountVectorizer(stop_words=stop,

                          min_df=0,

                          token_pattern = regex1,

                          ngram_range=(1,2)).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]
df = pd.read_excel("/kaggle/input/job_skills.xlsx")

df["Location"] = df["Location"].str.replace('United States', 'USA', regex=False)

df["Countries"] = df["Location"].str.split(",").str[-1].str.strip()
minimum_qualifications_splited = df["Minimum Qualifications"].str.split("\n").apply(pd.Series).stack().reset_index()[0]

minimum_qualifications_unique = pd.Series(minimum_qualifications_splited.unique())

print(minimum_qualifications_splited.shape)

print(minimum_qualifications_unique.shape)
most_common_words = pd.DataFrame(get_top_n_words(df["Minimum Qualifications"][df["Minimum Qualifications"].notnull()]))

for i in range(10):

    print(most_common_words.iloc[i].values)
df.shape
df.head()
print("Total de valores não preenchidos por coluna:")

len(df) - df.count()
df["Company"].value_counts(dropna=False)
df["Title"].value_counts(dropna=False)
df["Category"].value_counts(dropna=False)
top_countries = df["Countries"].value_counts().head()



ax = sns.countplot(y="Countries", data=df[df["Countries"].isin(top_countries.index)], order = top_countries.index)
print("Existem {} vagas no Brasil".format(df["Countries"].value_counts()["Brazil"]))
vectorizer = TfidfVectorizer(#min_df=1,

                             stop_words=stop,

#                              max_df=.9

                            )

X = vectorizer.fit_transform(minimum_qualifications_unique)
dend = shc.dendrogram(shc.linkage(X.toarray(), metric='euclidean', method='ward'))
cluster = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')

cluster_qualifications = pd.Series(cluster.fit_predict(X.toarray()))

minimum_qualifications_cluster = pd.DataFrame(cluster_qualifications.values, index=minimum_qualifications_unique)

minimum_qualifications_splited_cluster = pd.DataFrame({'minimum_qualifications_splited': minimum_qualifications_splited.values, 

                                                       'cluster': minimum_qualifications_splited.apply(lambda x: minimum_qualifications_cluster.loc[x])[0]})

minimum_qualifications_splited_cluster["cluster"].value_counts()
np.random.seed(1)

for c in cluster_qualifications.value_counts().sort_index().index:

    print("-----Cluster {}-----".format(c))

    for m in minimum_qualifications_unique[cluster_qualifications == c].sample(5):

        print(m)
programming_words = ["cloud computing", "python", "java", "c++", "c", "scala", 

                     "sql", "web development", "back end" "javascript", "mobile", "r", "api", "go", "matlab", 

                     "rest", "soap"]
most_common_words[most_common_words[0].isin(programming_words)].head()
language_words = ["english", "portuguese", "spanish", "mandarin", "chinese", "russian", "french", "german", "hindi", "arabic", "danish"]

most_common_words[most_common_words[0].isin(language_words)].head()