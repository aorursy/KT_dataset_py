# Importing the used dependency libraries



import math

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotig graphs library

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.cluster import MiniBatchKMeans # Efficient clustering algorithm

from sklearn.decomposition import TruncatedSVD # SVD decomposition for performing LSA

from sklearn.preprocessing import Normalizer # Normalizer with instance, useful as preprocessing before clustering



from sklearn.pipeline import make_pipeline

from sklearn import metrics # used to output siuette score



from scipy import stats # using Kruskal Wallis unparametric test

import nltk.stem # Stemming tool for processing the transcript tokens

import datetime

import ast # Parser for python objects



# Our deterministic seed

SEED = 12



################# Inherit TfidfVectorizer to adapt it to stemming ###################

english_stemmer = nltk.stem.SnowballStemmer('english')



class StemmedTfidfVectorizer(TfidfVectorizer):

    """Tf-idf Vectorizer which is applying also stemming i.e. it boils down

    each word to its root. In the semantic analysis we are doing where we don't

    take into account the parts of speach stemming is a good choice to generalize

    better over the general meaning the text of interest has."""

    # Overriding the build_analyzer

    def build_analyzer(self):

        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()

        return lambda doc: ([english_stemmer.stem(w) for w in analyzer(doc)])

    

####################################################################################

class RatingsVectorizer():

    """Used to parse and convert to vector the ratings for each TED talk. It has

    to 'book keep' all of the rating terms and their corresponding indices."""



    def __init__(self):

        # It stores as key the rating name and as value

        # the corresponding index.

        self.vocabulary_ = dict()

        

    def get_feature_names(self):

        return list(self.vocabulary_.keys())

    

    def parse_ratings(self, json_collection):

        """It returns a frequency table where the rows are talks and columns - ratings"""



        # First pass.

        for json_string in json_collection:

            ratings = ast.literal_eval(json_string)

            

            for rating in ratings:

                if rating['name'] not in self.vocabulary_:

                    self.vocabulary_[rating['name']] = len(self.vocabulary_)

                    

        # Second pass.

        freq_table = np.zeros(shape = (len(json_collection), len(self.vocabulary_)))

        for i in range(len(json_collection)):

            ratings = ast.literal_eval(json_collection[i])

            

            for rating in ratings:

                freq_table[i,self.vocabulary_[rating['name']]] = rating['count']



        return freq_table

            

########################## Diagnostic Tools ######################



def pearson_r(x, y):

    """Compute Pearson correlation coefficient between two arrays."""



    # Compute correlation matrix

    corr_mat = np.corrcoef(x, y)



    # Return entry [0,1]

    return corr_mat[0,1]



def k_means_diagnostics(x,ted_df, max_k):

    

    def k_wss(km, k, target):

        wss = 0

        for i in range(k):

            y = target[i == km.labels_]

            mean = y.mean()

            delta = y - mean

            wss += np.sum(delta * delta)

        return wss



    def k_bss(km, k, target):

        bss = 0

        mean = target.mean()

        for i in range(k):

            y = target[i == km.labels_]

            delta = y.mean() - mean

            bss += y.shape[0] * delta * delta



        return bss



    sum_of_squared_distances = []

    comments_wss_bss = []

    views_wss_bss = []

    

    K = range(2,max_k)

    for k in K:

        km = MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=1,

                             init_size=1000, batch_size=1000, random_state = SEED)

        km = km.fit(X)

        sum_of_squared_distances.append(km.inertia_)



        bss_c = k_bss(km, k, ted_df["comments"].values)

        wss_c = k_wss(km, k, ted_df["comments"].values)



        bss_v = k_bss(km, k, ted_df["views"].values)

        wss_v = k_wss(km, k, ted_df["views"].values)



        comments_wss_bss.append(wss_c/bss_c)

        views_wss_bss.append(wss_v/bss_v)

        

    plt.plot(K, sum_of_squared_distances, 'bx-')

    plt.xlabel('k')

    plt.ylabel('Sum of Squared Distances')

    plt.title('Elbow Method For Optimal k')

    plt.show()



    plt.plot(K, comments_wss_bss, 'bx-')

    plt.xlabel('k')

    plt.ylabel('WSS/BSS of the variable \'comments\'')

    plt.title('Elbow Method For Optimal k')

    plt.show()



    plt.plot(K, views_wss_bss, 'bx-')

    plt.xlabel('k')

    plt.ylabel('WSS/BSS of the variable \'views\'')

    plt.title('Elbow Method For Optimal k')

    plt.show()

    

def cluster_stratification(k, ted_df, order_centroids, df_clust_name, terms):



    for i in range(k):

        print("Cluster %d:" % i, end='')

        for ind in order_centroids[i, :10]:

            print(' %s;' % terms[ind], end='')

        print()



    plt.figure(1, figsize=(21, 7))

    tran_count = ted_df.groupby([df_clust_name]).size()

    tran_count.plot.bar()

    plt.ticklabel_format(style='plain', axis='y')

    plt.xticks(rotation=0)

    plt.xlabel('Cluster indices')

    plt.ylabel('Count of Ted talks')

    plt.title('Frequency of the talks within the clusters')

    plt.show()



    plt.figure(1, figsize=(21, 7))

    tran_comments = ted_df.groupby([df_clust_name])['comments'].agg('mean')

    tran_comments.plot.bar()

    plt.ticklabel_format(style='plain', axis ='y')

    plt.xticks(rotation=0)

    plt.xlabel('Cluster indices')

    plt.ylabel('Average comments count')

    plt.title('Average comments count within the clusters')

    plt.show()



    plt.figure(1, figsize = (21, 7))

    tran_views = ted_df.groupby([df_clust_name])['views'].agg('mean')

    tran_views.plot.bar()

    plt.ticklabel_format(style ='plain', axis ='y')

    plt.xticks(rotation=0)

    plt.xlabel('Cluster indices')

    plt.ylabel('Average reviews count')

    plt.title('Average reviews count within the clusters')

    plt.show()
ted_meta = pd.read_csv('../input/ted_main.csv')



print("Total count of TED Talk views: {} and comments: {}"

      .format(ted_meta['views'].sum(), ted_meta['comments'].sum()))

print() 

with pd.option_context('display.max_rows', None, 'display.max_columns', None):

    print(ted_meta.describe())
print(ted_meta.isnull().any())

print(ted_meta.isnull().sum())

ted_meta.fillna('Not specified', inplace = True)
plt.figure(1)

plt.title('Box of the talks views')

boxplot = ted_meta.boxplot(column=['views'])

print(boxplot)

plt.ylabel('View counts')

axes = plt.gca()

# We prune the long tail up to 6 000 000 for the sake of visibility

axes.set_ylim([0,6e6])



plt.show()



plt.figure(2)

ted_meta['views'].plot.hist(grid=True, bins=9, rwidth=0.9, color='#607c8e')

plt.title('Histogramm of the talks views')

plt.xlabel('View counts')

plt.ylabel('Freqency')

plt.grid(axis='y', alpha=0.75)

plt.show()
tfidf_vectorizer = StemmedTfidfVectorizer(stop_words = 'english', max_df=0.3)



# Inefficient stroring of the word frequencies within the talk description

# in the form of a dense matrix with row for each talk and a column for each observed word.

talks_word_tf_idf = tfidf_vectorizer.fit_transform(ted_meta['description'].values).toarray()



id_to_word = dict((v, k) for k, v in tfidf_vectorizer.vocabulary_.items())



total_word_tf_idf = talks_word_tf_idf.sum(axis=0)



word_views_corr = np.empty(talks_word_tf_idf.shape[1])

for i in range(talks_word_tf_idf.shape[1]):

    word_views_corr[i] = pearson_r(talks_word_tf_idf[:,i], ted_meta['views'].values)



word_views_ordered_corr = np.argsort(-np.abs(word_views_corr))



fig = plt.figure(1, figsize=(7, 7))

p = fig.add_subplot(111)



k = 0

i = 0

while(k < 30 and i < total_word_tf_idf.shape[0]):

    # We filter just the words with bigger tf_idf than 10

    if total_word_tf_idf[word_views_ordered_corr[i]] > 10:

        x  = total_word_tf_idf[word_views_ordered_corr[i]]

        y = word_views_corr[word_views_ordered_corr[i]]

        p.scatter(x, y, alpha=0.5)

        p.text(x, y,id_to_word[word_views_ordered_corr[i]])



        k += 1

    i += 1

    

plt.title('Word TF-IDF vs Talks Views count Correlation')

plt.xlabel('Overall TF-IDF word mass')

plt.ylabel('TF-IDF vd Views counts Correlation')

plt.show()
ted_meta.plot.scatter(x='views',y='duration', alpha=0.4)

print("The correlation between the 'duration' and 'views': {}".format(ted_meta['views'].corr(ted_meta['duration'])))
ted_meta.plot.scatter(x='views',y='languages', alpha = 0.4)

print("The correlation between the 'languages' and 'views': {}".format(ted_meta['views'].corr(ted_meta['languages'])))
plt.figure(1, figsize=(21, 7))

speaker_views = ted_meta.groupby(['main_speaker'])['views'].agg('sum')

speaker_views_top_60 = speaker_views.sort_values(ascending=False).head(60)

speaker_views_top_60.plot.bar()

plt.ticklabel_format(style='plain', axis='y')

plt.title('Top 60 speakers with most views of their talks')

plt.xlabel('Speakers')

plt.ylabel('Total views count')
plt.figure(1, figsize=(21, 7))

speaker_occupation_views = ted_meta.groupby(['speaker_occupation'])['views'].agg('sum')

speaker_occupation_views_top_60 = speaker_occupation_views.sort_values(ascending=False).head(60)

speaker_occupation_views_top_60.plot.bar()

plt.ticklabel_format(style='plain', axis='y')

plt.title('Top 60 speaker occupations with most talk views')

plt.xlabel('Occupations')

plt.ylabel('Total views count')
speaker_views_avg = ted_meta.groupby(['main_speaker'])['views'].agg('mean')

speaker_occurrenes = ted_meta.groupby('main_speaker')['views'].nunique()

speaker_views_avg = speaker_views_avg.rename('avg_views')

speaker_occurrenes = speaker_occurrenes.rename('speaker_occurrenes')



speaker_views = pd.concat([speaker_occurrenes, speaker_views_avg], axis=1).reset_index()



speaker_views_top_60_occ = speaker_views.sort_values(ascending=False, by = 'speaker_occurrenes').head(60)



fig = plt.figure(1, figsize=(10, 10))

p = fig.add_subplot(111)

plt.ticklabel_format(style='plain', axis='y')

for i in range(speaker_views.shape[0]):

    x  = speaker_views.at[i,'speaker_occurrenes']

    y = speaker_views.at[i,'avg_views']

    p.scatter(x, y, alpha=0.5)

    p.text(x, y, speaker_views.at[i,'main_speaker'])



plt.title('Speakers with a lot of views and a lot of occurrances')

plt.xlabel('Number of Occurrances')

plt.ylabel('Average views count per speaker')

plt.show()
ted_meta[ted_meta['main_speaker'] == 'Ken Robinson']['views']
plt.figure(1, figsize=(21, 7))

events_views = ted_meta.groupby(['event'])['views'].agg('sum')

events_views_top_60 = events_views.sort_values(ascending=False).head(60)

events_views_top_60.plot.bar()

plt.ticklabel_format(style='plain', axis='y')

plt.title('Top 60 events with most talk views')

plt.xlabel('Events')

plt.ylabel('Total views count')

plt.show()

ted_events = ted_meta.groupby('event').size()

ted_events.columns = ['event', 'count']

ted_events = ted_events.sort_values(ascending=False)



plt.figure(figsize=(15,5))

ted_events.head(10).plot.bar()

plt.title('Top 10 events with most talks')

plt.xlabel('Events')

plt.ylabel('Talks count within an event')

plt.show()
plt.figure(figsize=(30,5))

ted_meta.sort_values(by='film_date', ascending=False)[ted_meta['event'].isin(ted_events.head(20).index.values)][['views','event']].boxplot(by='event', grid=False, rot=45, fontsize=15)

axes = plt.gca()

axes.set_ylim([0,6e6])

plt.ylabel('Total views count')

plt.show()



stats.kruskal(*[group["views"].values for name, group in ted_meta.groupby("event")])
plt.figure(1)

ted_meta['num_speaker'].plot.hist(grid=True, bins=5, rwidth=0.9, color='#607c8e')

plt.title('Histogramm of the talks speaker count')

plt.xlabel('Speaker counts')

plt.xticks(range(1,5))

plt.ylabel('Freqency')

plt.grid(axis='y', alpha=0.75)



plt.figure(2)

ted_meta['num_speaker'].plot.hist(grid=True, bins=5, rwidth=0.9, color='#607c8e')

plt.title('Log Histogramm of the talks speaker count')

plt.xlabel('Speaker counts')

plt.xticks(range(1,5))

plt.ylabel('Log Freqency')

plt.yscale('log')

plt.grid(axis='y', alpha=0.75)
ted_meta.plot.scatter(x='views',y='num_speaker', alpha = 0.4)

print("The correlation between the 'num_speaker' and 'views': {}".format(ted_meta['views'].corr(ted_meta['num_speaker'])))
# Preparing the tags to be parsed by the CountVectorizer(We separate the tags by space and unite

# the words within a tag by '_'), expecting to be parsed as sentence and treated as sentence tokens.

ted_meta['tags_text'] = ted_meta['tags'].str.replace('(?<=[A-Za-z])(\s)(?=[A-Za-z])','_')

ted_meta['tags_text'] = ted_meta['tags_text'].str.replace('\[|\'|\]', ' ')

count_vectorizer = CountVectorizer()



# Very inefficient stroring of the word frequencies within the talk description,

# in the form of a dense matrix with row for each talk and a column for each word ever observed.

# Effectivelly we are using count vectorizer for quick and dirty transforming of the tags into

# One-hot Encoding, since we have max one tag occurrence in a talk

tags_frequency = count_vectorizer.fit_transform(ted_meta['tags_text'].values).toarray()



plt.figure(1, figsize=(21, 7))

tags_df = pd.Series(tags_frequency.sum(axis=0))

tags_df.name = 'tags_frequency'

tags_df.index = count_vectorizer.get_feature_names()

tags_df_top60 = tags_df.sort_values(ascending=False).head(60)

plt.title('Top 60 most frequent tags')

plt.xlabel('Tags')

plt.ylabel('Tags frequency')

tags_df_top60.plot.bar()
ted_meta['published_month'] = ted_meta['published_date'].apply(lambda x: datetime.datetime.fromtimestamp( int(x)).strftime('%m'))



plt.figure(1, figsize=(21, 7))

month_views = ted_meta.groupby(['published_month'])['views'].agg('sum')

month_views.plot.bar()

plt.ticklabel_format(style='plain', axis='y')

plt.title('Views across Months')

plt.xlabel('Published Month')

plt.ylabel('Views Counts')

plt.show()
# The dataset has been published in 2017-09-09 ~ 1504915200(unix)



# Caluclate the approximate age in days

ted_meta['talk_age'] = ted_meta['published_date'].apply(lambda x: int((1504915200 - x)/(3600 * 24)))



ted_meta.plot.scatter(x='views',y='talk_age', alpha = 0.4)

print("The correlation between the 'talk_age' and 'views': {}".format(ted_meta['views'].corr(ted_meta['talk_age'])))
plt.figure(1)

plt.title('Box of the talks comments counts')

boxplot = ted_meta[ted_meta['comments'] < 2e3].boxplot(column=['comments'])

plt.ylabel('View counts')

axes = plt.gca()

# We prune the long tail up to 2 000 for the sake of visibility

axes.set_ylim([0,2e3])



plt.show()



plt.figure(2)

ted_meta[ted_meta['comments'] < 2e3]['comments'].plot.hist(grid=True, bins=9, rwidth=0.9, color='#607c8e')

plt.title('Histogramm of the talks comments')

plt.xlabel('View comments')

plt.ylabel('Freqency')

plt.grid(axis='y', alpha=0.75)



plt.show()
ted_meta.plot.scatter(x='comments',y='views', alpha=0.4)

print("The correlation between the 'views' and 'comments': {}".format(ted_meta['comments'].corr(ted_meta['views'])))
ted_meta.plot.scatter(x='comments',y='duration', alpha=0.4)

print("The correlation between the 'duration' and 'comments': {}".format(ted_meta['comments'].corr(ted_meta['duration'])))
plt.figure(1, figsize=(21, 7))

month_comments = ted_meta.groupby(['published_month'])['comments'].agg('sum')

month_comments.plot.bar()

plt.ticklabel_format(style='plain', axis='y')

plt.title('Comments across Months')

plt.xlabel('Published Month')

plt.ylabel('Comments Counts')

plt.show()
ted_meta.plot.scatter(x='comments',y='talk_age', alpha = 0.4)

print("The correlation between the 'talk_age' and 'comments': {}".format(ted_meta['comments'].corr(ted_meta['talk_age'])))
word_comments_corr = np.empty(talks_word_tf_idf.shape[1])

for i in range(talks_word_tf_idf.shape[1]):

    word_comments_corr[i] = pearson_r(talks_word_tf_idf[:,i], ted_meta['comments'].values)



word_comments_ordered_corr = np.argsort(-np.abs(word_views_corr))



fig = plt.figure(1, figsize=(7, 7))

p = fig.add_subplot(111)



k = 0

i = 0

while(k < 30 and i < total_word_tf_idf.shape[0]):

    # We filter just the words with bigger frequency than 10

    if total_word_tf_idf[word_comments_ordered_corr[i]] > 10:

        x  = total_word_tf_idf[word_comments_ordered_corr[i]]

        y = word_comments_corr[word_comments_ordered_corr[i]]

        p.scatter(x, y, alpha=0.5)

        p.text(x, y,id_to_word[word_comments_ordered_corr[i]])



        k += 1

    i += 1

plt.title('Word TF-IDF vs Talks Comments count Correlation')

plt.xlabel('Overall TF-IDF word mass')

plt.ylabel('TF-IDF vs Comments counts Correlation')

plt.show()
ratings_vectorizer = RatingsVectorizer()

rating_freq = ratings_vectorizer.parse_ratings(ted_meta['ratings'])

all_talks_ratings_count = rating_freq.sum(axis=0)



# Standardizing the ratings

print(rating_freq.std(axis=0, keepdims=True).shape)

rating_freq = rating_freq / rating_freq.std(axis=0, keepdims=True)



id_to_word = dict((v, k) for k, v in ratings_vectorizer.vocabulary_.items())



print("The distinct ratings types count is {}".format(len(id_to_word)))



ratings_comments_corr = np.empty(rating_freq.shape[1])

for i in range(rating_freq.shape[1]):

    ratings_comments_corr[i] = pearson_r(rating_freq[:,i], ted_meta['comments'].values)



ratings_comments_ordered_corr = np.argsort(-np.abs(ratings_comments_corr))



fig = plt.figure(1, figsize=(7, 7))

p = fig.add_subplot(111)



i = 0

while(i < all_talks_ratings_count.shape[0]):

    x  = all_talks_ratings_count[ratings_comments_ordered_corr[i]]

    y = ratings_comments_corr[ratings_comments_ordered_corr[i]]

    p.scatter(x, y, alpha=0.5)

    p.text(x, y,id_to_word[ratings_comments_ordered_corr[i]])

    i += 1

    

plt.ylabel('Correlation')

plt.xlabel('Ratings Total Count')

plt.title('Ratings Counts vs Comments Counts Correlation')

plt.show()



N_CLUSTERS_R = 6

normalizer = Normalizer(copy=False)



# Truncated frequency table

X = normalizer.transform(rating_freq)



k_means_diagnostics(X, ted_meta, 40)

km0 = MiniBatchKMeans(n_clusters = N_CLUSTERS_R, init ='k-means++', n_init=1, tol = 0.01,

                         init_size = 1000, batch_size = 1000, random_state = SEED)



km0.fit(X)



order_centroids = km0.cluster_centers_.argsort()[:, ::-1]

terms = ratings_vectorizer.get_feature_names()



for i in range(N_CLUSTERS_R):

    print("Cluster %d:" % i, end='')

    current_mass = 0

    mass_limit = 0.5 * km0.cluster_centers_[i,].sum()

    for ind in order_centroids[i,:]:

        current_mass += km0.cluster_centers_[i,ind]

        # We filter the most present words up to 50% of thw whole centroid mass

        if(current_mass > mass_limit):

            break

        print(' %s' % terms[ind], end='')

    print()

    

ted_meta['rating_clust'] = km0.labels_



plt.figure(1, figsize=(21, 7))

rating_counts = ted_meta.groupby(['rating_clust']).size()

rating_counts.plot.bar()

plt.ticklabel_format(style='plain', axis='y')

plt.ylabel('Talks Count within Clusters')

plt.xlabel('Clusters')

plt.title('Talks Count representation within Raiting Clusters')

plt.show()



plt.figure(1, figsize=(21, 7))

desc_views = ted_meta.groupby(['rating_clust'])['views'].agg('mean')

desc_views.plot.bar()

plt.ticklabel_format(style='plain', axis='y')

plt.ylabel('Average Views Count within Clusters')

plt.xlabel('Clusters')

plt.title('Average Views Count representation within Raiting Clusters')

plt.show()



plt.figure(1, figsize=(21, 7))

desc_comments = ted_meta.groupby(['rating_clust'])['comments'].agg('mean')

desc_comments.plot.bar()

plt.ticklabel_format(style='plain', axis='y')

plt.ylabel('Average Comments Count within Clusters')

plt.xlabel('Clusters')

plt.title('Average Comments Count representation within Raiting Clusters')

plt.show()
plt.figure(1, figsize=(21, 7))

ratings_df = pd.Series(all_talks_ratings_count)

ratings_df.name = 'rating_frequency'

ratings_df.index = terms

ratings_df = ratings_df.sort_values(ascending=False)

ratings_df.plot.bar()

plt.ylabel('Total count of ratings across the talks')

plt.xlabel('Ratings')

plt.title('Total ratings count')

plt.show()
ted_transcripts = pd.read_csv('../input/transcripts.csv')

ted_transcripts = pd.merge(ted_transcripts, ted_meta[['comments','url']], on='url')
tfidf_vectorizer = StemmedTfidfVectorizer(stop_words = 'english', max_df=0.3, min_df=0.01)



# Very inefficient stroring of the word frequencies within the talk description,

# in the form of a dense matrix with row for each talk and a column for each word ever observed.

talks_word_tf_idf = tfidf_vectorizer.fit_transform(ted_transcripts['transcript'].values).toarray()



id_to_word = dict((v, k) for k, v in tfidf_vectorizer.vocabulary_.items())



total_word_tf_idf = talks_word_tf_idf.sum(axis=0)



word_comments_corr = np.empty(talks_word_tf_idf.shape[1])

for i in range(talks_word_tf_idf.shape[1]):

    word_comments_corr[i] = pearson_r(talks_word_tf_idf[:,i], ted_transcripts['comments'].values)



word_comments_ordered_corr = np.argsort(-np.abs(word_comments_corr))



fig = plt.figure(1, figsize=(7, 7))

p = fig.add_subplot(111)





i = 0

while(i < 10):

    

    x  = total_word_tf_idf[word_comments_ordered_corr[i]]

    y = word_comments_corr[word_comments_ordered_corr[i]]

    p.scatter(x, y, alpha=0.5)

    p.text(x, y,id_to_word[word_comments_ordered_corr[i]])

    i += 1

    

plt.title('Word TF-IDF vs Talks Comments count Correlation')

plt.xlabel('Overall TF-IDF word mass')

plt.ylabel('TF-IDF vd Views counts Correlation')

plt.show()
plt.figure(1, figsize=(21, 7))

tr_words_df = pd.Series(total_word_tf_idf)

tr_words_df.name = 'tr_words_tf_idf'

tr_words_df.index = list(id_to_word.values())

tr_words_df_top60 = tr_words_df.sort_values(ascending=False).head(30)

plt.title('Total TF-IDF word importance of the top 60 words')

plt.xlabel('Words')

plt.ylabel('Overall TF-IDF word mass over all the talks')

tr_words_df_top60.plot.bar()
# Inner join between the bot tables. We rop some rows, but it is ok.

ted_df = pd.merge(pd.read_csv('../input/transcripts.csv'), pd.read_csv('../input/ted_main.csv'), on='url')



# Preparing the tags to be parsed by the CountVectorizer(We separate the tags by space and unite

# the words within a tag by '_'), expecting to be parsed as sentence and treated as sentence tokens.

ted_df['tags_text'] = ted_df['tags'].str.replace('(?<=[A-Za-z])(\s)(?=[A-Za-z])','_')

ted_df['tags_text'] = ted_df['tags_text'].str.replace('\[|\'|\]', ' ')



count_vectorizer = CountVectorizer()



# Very inefficient stroring of the word frequencies within the talk description,

# in the form of a dense matrix with row for each talk and a column for each word ever observed.

# Effectivelly we are using count vectorizer for quick and dirty transforming of the tags into

# One-hot Encoding, since we have max one tag occurrence in a talk

tags_frequency = count_vectorizer.fit_transform(ted_df['tags_text'].values).toarray()
N_COMPONENTS = 40

N_CLUSTERS_T = 20



tfidf_vectorizer = StemmedTfidfVectorizer(stop_words = 'english', max_df=0.3, min_df=0.01)

tfidf = tfidf_vectorizer.fit_transform(ted_df['transcript'].values)



normalizer = Normalizer(copy=False)

svd = TruncatedSVD(n_components=N_COMPONENTS, n_iter=10, random_state=SEED)

lsa = make_pipeline(svd, normalizer)



X = lsa.fit_transform(tfidf)



explained_variance = svd.explained_variance_ratio_.sum()

print("Explained variance of the SVD step for {} kept components: {}%".format(N_COMPONENTS,

    int(explained_variance * 100)))



print("Bag of words count {}".format(

    tfidf.shape[1]))



index = np.arange(N_COMPONENTS)

plt.bar(index,svd.explained_variance_ratio_)

plt.title('Explained Variance Chart')

plt.xlabel('Sungular value index')

plt.ylabel('Explained variance ratio')

plt.show()



k_means_diagnostics(X,ted_df, max_k = 40)



km0 = MiniBatchKMeans(n_clusters=N_CLUSTERS_T, init='k-means++', n_init=1, tol = 0.01,

                         init_size=1000, batch_size=1000, random_state = SEED)



km0.fit(X)



original_space_centroids = svd.inverse_transform(km0.cluster_centers_)

order_centroids = original_space_centroids.argsort()[:, ::-1]

terms = tfidf_vectorizer.get_feature_names()



ted_df['transcript_clust'] = km0.labels_

cluster_stratification(N_CLUSTERS_T, ted_df, order_centroids, 'transcript_clust', terms)

N_COMPONENTS = 40

N_CLUSTERS_D = 17



tfidf_vectorizer = StemmedTfidfVectorizer(stop_words = 'english', max_df=0.35, min_df=2)

tfidf = tfidf_vectorizer.fit_transform(ted_df['description'].values)



normalizer = Normalizer(copy=False)

svd = TruncatedSVD(n_components=N_COMPONENTS, n_iter=10, random_state=SEED)

lsa = make_pipeline(svd, normalizer)



X = lsa.fit_transform(tfidf)



explained_variance = svd.explained_variance_ratio_.sum()

print("Explained variance of the SVD step: {}%".format(

    int(explained_variance * 100)))



print("Bag of words count {}".format(

    tfidf.shape[1]))



index = np.arange(N_COMPONENTS)

plt.bar(index,svd.explained_variance_ratio_)

plt.title('Explained Variance Chart')

plt.xlabel('Sungular value index')

plt.ylabel('Explained variance ratio')

plt.show()



k_means_diagnostics(X, ted_df, max_k = 40)



km0 = MiniBatchKMeans(n_clusters = N_CLUSTERS_D, init ='k-means++', n_init=1, tol = 0.01,

                         init_size = 1000, batch_size = 1000, random_state = SEED)



km0.fit(X)



original_space_centroids = svd.inverse_transform(km0.cluster_centers_)

order_centroids = original_space_centroids.argsort()[:, ::-1]

terms = tfidf_vectorizer.get_feature_names()

    

ted_df['description_clust'] = km0.labels_

cluster_stratification(N_CLUSTERS_D, ted_df, order_centroids, 'description_clust', terms)

N_CLUSTERS_TAGS = 6



k_means_diagnostics(tags_frequency, ted_df, max_k = 40)



# We are using MiniBatchKMeans which is well suited in case of sparse examples as in our case with one-hot

# encoded tags. See the very good paper of the algorithm  http://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf

km = MiniBatchKMeans(n_clusters=N_CLUSTERS_TAGS, init='k-means++', n_init=1, tol=0.01,

                         init_size=1000, batch_size=1000, random_state = SEED)



km.fit(tags_frequency)



terms = count_vectorizer.get_feature_names()

cluster_tags_freq = np.empty((N_CLUSTERS_TAGS,tags_frequency.shape[1]))

for i, c in enumerate(km.labels_):

    cluster_tags_freq[c,:] += tags_frequency[i,:]

    

# Normalized by the total tag frequency

cluster_tags_freq /= cluster_tags_freq.sum(axis = 0, keepdims = True)

order_centroids = cluster_tags_freq.argsort()[:, ::-1]



ted_df['tag_clust'] = km.labels_



cluster_stratification(N_CLUSTERS_TAGS, ted_df, order_centroids, 'tag_clust', terms)
clust_occ_freq = pd.crosstab(ted_df.tag_clust,ted_df.speaker_occupation)

terms = clust_occ_freq.columns.values.tolist()

clust_occ_freq = clust_occ_freq.values



# Normalized by the total tag frequency

clust_occ_freq = clust_occ_freq / clust_occ_freq.sum(axis= 0, keepdims = True)

order_centroids = clust_occ_freq.argsort()[:, ::-1]



for i in range(N_CLUSTERS_TAGS):

    print("Cluster %d:" % i, end='')

    for ind in order_centroids[i, :10]:

        print(' %s;' % terms[ind], end='')

    print()
