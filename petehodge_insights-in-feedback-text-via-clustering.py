# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
raw_data = pd.read_csv('../input/Womens Clothing E-Commerce Reviews.csv', encoding='utf-8')

raw_data.head(10)
raw_data.describe()
raw_data.Age[raw_data['Review Text'].isnull()].count()
raw_data['Division Name'].unique()
raw_data['Department Name'].unique()
raw_data['Class Name'].unique()
print('Blank reviews', raw_data.Age[raw_data['Review Text'].isnull()].count())

print('Blank Titles', raw_data.Age[raw_data['Title'].isnull()].count())
raw_data[raw_data['Review Text'].isnull()].count()
rev_data = raw_data[raw_data['Review Text'].notnull()]

rev_data['Title'].fillna('', inplace=True)

rev_data.head()
rev_data.groupby('Class Name').agg({'Rating':['count', 'mean']}).sort_values(by=[('Rating', 'count')], ascending=False)
rev_data.groupby('Class Name').agg({'Recommended IND':['count', 'sum']}).sort_values(by=[('Recommended IND', 'count')], ascending=False)
rev_data.groupby('Class Name').agg({'Positive Feedback Count':['count', 'sum']}).sort_values(by=[('Positive Feedback Count', 'count')], ascending=False)
rev_data['Positive Feedback Count'].unique()
rev_data['Recommended IND'].unique()
len(rev_data['Clothing ID'].unique())
# Pull out individual products and the class they are in

cloth_id = pd.DataFrame(rev_data.groupby(['Clothing ID', 'Class Name']).agg({'Clothing ID':['count']}).

                        sort_values(by=[('Clothing ID', 'count')], ascending=False))



cloth_id[cloth_id['Clothing ID']['count'] > 100]
cloth_id[cloth_id['Clothing ID']['count'] > 100].plot(title='Individual Product Review Count', legend=False, figsize=(10,6), rot=90)
cloth_id['Clothing ID'][cloth_id['Clothing ID']['count'] ==1].count()
# See if the classes of the number of products with reviews > 100 reflects overall numbers

cloth_id[cloth_id['Clothing ID']['count'] >100].groupby('Class Name').agg({('Clothing ID', 'count'):['count', 'sum']})
rev_data[('Age')].hist()
print ('Number below average of 43', rev_data.Age[rev_data['Age'] < 43].count())

print ('Number 43 & above', rev_data.Age[rev_data['Age'] >= 43].count())
rev_data[('Rating')].hist()
rev_data['Title Word Count'] = rev_data['Title'].map(lambda x: len(x.split()))

rev_data['Review Word Count'] = rev_data['Review Text'].map(lambda x: len(x.split()))
len(rev_data)
# rev_data[('Review Word Count')].hist()

plot = rev_data[('Review Word Count')].plot(kind="hist")

plot.set_xlabel("Word Count")

plot.set_ylabel("Number of Reviews")
rev_data['Review Word Count'].describe()
rev_data.boxplot(column='Review Word Count', by='Class Name', figsize=(12,12), rot=45)
rev_data.boxplot(column='Review Word Count', by='Rating')
rev_data['Title Word Count'].hist()
pd.set_option('display.max_colwidth', -1)

rev_data['Review Text'].sample(n=100)
def percentage(part, whole):

  return 100 * float(part)/float(whole)
total_rec_cnt = rev_data.Rating.count()

rate_4_above = rev_data.Rating[rev_data['Rating'] > 3].count()

rate_3_below = rev_data.Rating[rev_data['Rating'] < 4].count()

rate_2_below = rev_data.Rating[rev_data['Rating'] < 3].count()

rate_4_above_perc = percentage(rate_4_above, total_rec_cnt)

rate_3_below_perc = percentage(rate_3_below, total_rec_cnt)

rate_2_below_perc = percentage(rate_2_below, total_rec_cnt)



print ('Total number of ratings {}. '.format(total_rec_cnt))



print ('Number of ratings 4 and above {}. Percentage of Total: {:.2f}'.format(rate_4_above,

                                                                           rate_4_above_perc))

print ('Number of ratings 3 and below {}. Percentage of Total: {:.2f}'.format(rate_3_below,

                                                                           rate_3_below_perc))

print ('Number of ratings 2 and below {}. Percentage of Total: {:.2f}'.format(rate_2_below,

                                                                       rate_2_below_perc))
# Pull out individual products that are low rated and the class they are in

cloth_id_low_rate = pd.DataFrame(rev_data[rev_data['Rating'] < 3].groupby(['Clothing ID', 'Class Name']).agg({'Clothing ID':['count']}).sort_values(by=[('Clothing ID', 'count')], ascending=False))



cloth_id_low_rate
# See if the classes of products rated 1 have just 1 review as well

# To see if negative reviews are on their own with no positives to balance them

cloth_id_low_rate[cloth_id_low_rate['Clothing ID']['count'] == 1].groupby('Class Name').agg({('Clothing ID', 'count'):['count', 'sum']})
import re

def re_check_slash(text):

    return re.findall(r'(\n|\r|\t|\f)', text)
# Check for control characters

all_checks = []

for rev in rev_data['Review Text']:

    all_checks += re_check_slash(rev)



# Convert list to set of unique values

set(all_checks)
rev_data[rev_data.duplicated(subset='Review Text')]
undata = rev_data.drop_duplicates(subset='Review Text')
import string

import nltk

import nltk.data

nltk.download()   



from nltk import data, bigrams

# from nltk.stem import PorterStemmer

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem.snowball import SnowballStemmer
# A few standard operations on the text to get to words that add value



def scrub_text(text,

               html=False,

               hyphen=False,

               stemmer=False,

               stopwords=False,

               punctuation=False,

               lem=False,

               w2v=False):

    """Return cleaned up text."""

    if html:

        text = BeautifulSoup(text).get_text()  # Deal with HTML



    if hyphen:

        text = ' '.join(re.split('[-]+', text)) # Split out hyphenated words 

    

    text = re.sub('\.(?!\s)(?!$)', '. ', text) # Add space after "."

    

    word_tokens = word_tokenize(text.lower())

    if punctuation:

        word_tokens = [word for word in word_tokens if word not in punctuation]

    word_tokens = [word for word in word_tokens if word.isalnum()]

    word_tokens = [word for word in word_tokens if not word.isdigit()]

    

    if stopwords:

        word_tokens = [word for word in word_tokens if word not in stopwords]

    

    if lem:

        word_tokens = [lem.lemmatize(word) for word in word_tokens]

    

    if stemmer:

        word_tokens=[stemmer.stem(word) for word in word_tokens]

    

    if w2v:

        return word_tokens

    else:

        return ' '.join(word_tokens)
# clean text - tokenize, remove stopwords & punctuation but don't stem or lem



#lem = nltk.WordNetLemmatizer()

#stemmer = SnowballStemmer("english")

sw = stopwords.words('english')

punctuation=set(list(string.punctuation))



undata['cleantxt'] = undata['Review Text'].map(lambda x: scrub_text(x,

                                                                    hyphen=True,

                                                                    stopwords=sw,

                                                                    punctuation=punctuation))
undata.head()
# Build a list of all words from reviews in order to create a corpus



def create_corpus(data, text_field):

    """Return NLTK text corpus."""

    cnt = 0 

    build_text = []

    for row in data[text_field]:

        token = word_tokenize(row)

        build_text.extend(token)

        cnt += 1

        if cnt % 5000 == 0:

            print("Docs processed:", cnt)



    corp = nltk.Text(build_text)

    return corp, build_text
corp_data, tokens = create_corpus(undata, 'cleantxt')
len(tokens)
freq_dist = nltk.FreqDist(corp_data)

freq_dist.most_common(50)
top_words = pd.DataFrame(freq_dist.most_common(10), columns=('Word', 'Count'))

top_words
import matplotlib.pyplot as plt



plt.figure(figsize=(10, 6))



freq_dist.plot(20,cumulative=False)
corp_data.collocations()
# calculate bigrams

terms_bigram = bigrams(tokens)

bi_fdist = nltk.FreqDist(terms_bigram)

top_bigrams = pd.DataFrame(bi_fdist.most_common(50), columns=('Word Pair', 'Count'))

top_bigrams
corp_data.similar('dress')
corp_data.concordance('back')
corp_data.common_contexts(['going', 'back'])
undata['Rating'][undata['Review Text'].str.contains('going', 'back')].hist()
print(undata['Rating'][undata['Review Text'].str.contains('going', 'back')].count())

print(undata['Rating'][undata['Review Text'].str.contains('going', 'back')].value_counts())
print(undata['Rating'][undata['Review Text'].str.contains('ordered')].count())

print(undata['Rating'][undata['Review Text'].str.contains('ordered')].value_counts())
corp_data.similar('ordered')
# Can use the Unnamed column as a unique review ID

undata.rename(columns={'Unnamed: 0': 'Review ID'}, inplace=True)

undata.head(2)
# Download the punkt tokenizer for sentence splitting

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
def review_to_sentences(review, tokenizer, sw, punctuation):

    """Split a review into parsed sentences.

    

    Returns 2 lists of sentences:

    1. Raw sentences - just the review split by sentence

    2. Scrubbed sentences.

    

    Each sentence is a list of words."""

    

    raw_sentences = tokenizer.tokenize(review.strip())

    

    scrub_sentences = [scrub_text(raw_sentence, stopwords=sw, punctuation=punctuation)

                       for raw_sentence in raw_sentences if len(raw_sentence) > 0]



    return raw_sentences, scrub_sentences
# Build a review only list

review_text = undata[['Review ID', 'Review Text']].values.tolist()

review_text[0:3]
# Create a list of sentences with both raw and scrubbed sentences

# Scrubbed to be used in models, raw to refer back to for human undertanding and display

# [[review id <1203>, raw_sent, scrbd_sent], [...]]

# Review id is not unique. May need to add unique sentence ID at some point



review_sentences = []

for rev in review_text:

    sent, scrbd_sent = review_to_sentences(rev[1], tokenizer, sw, punctuation)

    for item in zip(sent, scrbd_sent):

        review_sentences.append([rev[0], item[0], item[1]])

review_sentences[0:10]
# Convert to a dataframe for convenience

rev_sent = pd.DataFrame(review_sentences,

                        columns=('Review ID', 'Review Sent', 'Review Scrub'))

rev_sent.head()
# Scrubbing may have removed all words - drop sentences that are blank after scrubbing

rev_sent = rev_sent[rev_sent['Review Scrub'] != '']

rev_sent_un = rev_sent.drop_duplicates(subset='Review Scrub')

rev_sent_un.reset_index(inplace=True)
rev_sent_un['Word Count'] = rev_sent_un['Review Scrub'].map(lambda x: len(x.split()))

rev_sent_large = rev_sent_un[rev_sent_un['Word Count'] > 5]

rev_sent_large.reset_index(inplace=True)

rev_sent_large.count()
rev_sent_lrg_sample = rev_sent_large[['Review ID',

                                      'Review Sent',

                                      'Review Scrub',

                                      'Word Count']].sample(n=30000)

rev_sent_lrg_sample.reset_index(inplace=True)

rev_sent_lrg_sample.rename(columns={'index': 'Sentence ID'}, inplace=True)

rev_sent_lrg_sample.head()
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.cluster import KMeans
# Calculate TF and tfidf scores

# Creates a bag of words with a sparse matrix for each sentence



vectorizer = TfidfVectorizer(min_df=2)

vz = vectorizer.fit_transform([rev for rev in rev_sent_lrg_sample['Review Scrub']])

tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

checkword = 'dress'

print("{}: {}".format(checkword, str(tfidf[checkword])))
vz.shape
num_clusters = 50

kmeans_model = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10,verbose=False)



kmeans = kmeans_model.fit(vz)

kmeans_clusters = kmeans.predict(vz)

kmeans_distances = kmeans.transform(vz)
# print the cluster center

sorted_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

terms = vectorizer.get_feature_names()

for i in range(num_clusters):

    print('Cluster {}:'.format(i))

    for j in sorted_centroids[i, :20]:

        print(' {}'.format(terms[j]))

    print()
# Add cluster number to the sentences and create df



rev_sent_lrg_sample['K Cluster'] = kmeans_clusters

rev_sent_lrg_sample.head()
def create_stats(clusters):

    """Return stats in a dataframe."""



    unique, counts = np.unique(clusters, return_counts=True)

    stats = pd.DataFrame(list(zip(unique, counts)), columns=['cluster','count'])



    stats.sort_values('count', ascending=False, inplace=True)

    return stats
def cluster_wordrank(docs, min_df=1):

    """Returns a dataframe of word and count."""

    cvec = CountVectorizer(min_df=min_df)

    cts = cvec.fit_transform(docs)

    vocab = list(cvec.get_feature_names())

    counts = cts.sum(axis=0).A1



    d = pd.DataFrame(list(zip(vocab, counts)))

    d.columns = ['word', 'raw count']

    d.sort_values('raw count', inplace=True, ascending=False)

    # d.reset_index('word',inplace=True)

    return d
from wordcloud import WordCloud



def make_wordcloud(word_list, top_n):

    """Returns a wordcloud of the top words provided."""

    counts = {}

    for r in cluster_wordrank(word_list)[['word','raw count']].head(top_n).iterrows():

        counts[r[1]['word']] = int(r[1]['raw count'])



    wordcloud = WordCloud(scale=10)

    wordcloud.fit_words(counts)



    # Display the generated image the matplotlib way

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis("off")
create_stats(kmeans_clusters)
inquiry_cluster = 1

rev_sent_lrg_sample['Review Sent'][rev_sent_lrg_sample['K Cluster'] == inquiry_cluster].head(10)
cluster_wordrank(rev_sent_lrg_sample['Review Scrub'][rev_sent_lrg_sample['K Cluster'] == inquiry_cluster])
def show_cloud_for_cluster(cluster_no):

    make_wordcloud(rev_sent_lrg_sample['Review Scrub'][rev_sent_lrg_sample['K Cluster'] == cluster_no], 20)
show_cloud_for_cluster(inquiry_cluster)
inquiry_cluster = 8

rev_sent_lrg_sample['Review Sent'][rev_sent_lrg_sample['K Cluster'] == inquiry_cluster].head(10)
show_cloud_for_cluster(inquiry_cluster)
# Convert string of words to a list

rev_sent_lrg_sample['sent w2v'] = rev_sent_lrg_sample['Review Scrub'].map(lambda x: word_tokenize(x))

rev_sent_lrg_sample.head()
# Run W2V on all text to create a larger vectorspace



import gensim

w2v_full = gensim.models.Word2Vec(rev_sent_lrg_sample['sent w2v'], workers=8, iter=10, size=100, min_count=2)
w2v_full.most_similar('dress', topn=20)
len(w2v_full.wv.vocab)
len(w2v_full.wv.index2word)
def create_doc_vector (doc, model, vector_size):

    # Average all of the word vectors in a given text document

    

    doc_vector = np.zeros((vector_size,), dtype='float32')

    num_words = 0.    

    index2word_set = set(model.wv.index2word)

    

    for word in doc:

        if word in index2word_set:

            num_words += 1.

            doc_vector = np.add(doc_vector, model[word])

            

    # Divide result by number of words to get average

    doc_vector = np.divide(doc_vector, num_words)

    return doc_vector
def get_doc_vecs(docs, model, vector_size):

    # Calculate the average vector for each text document

    

    cnt = 0

    doc_vecs = np.zeros((len(docs), vector_size), dtype='float32')

    

    for doc in docs:

        if cnt % 100000 == 0:

            print('Doc {} of {}'.format(cnt, len(docs)))

            

        doc_vecs[cnt] = create_doc_vector(doc, model, vector_size)

        cnt += 1

    return doc_vecs
vector_size = 100



vector_space = get_doc_vecs(rev_sent_lrg_sample['sent w2v'], w2v_full, vector_size)

vector_space
vector_space.shape
# Checks if nulls found in vector space

chck = pd.DataFrame(vector_space)

chck[chck.isnull().T.any().T]
# Drop relevant nulls and re-run (save them first)

unused_sent = rev_sent_clusters[chck.isnull().T.any().T]

unused_sent
rev_sent_clusters = rev_sent_clusters[chck.notnull().T.any().T]
from sklearn.preprocessing import normalize
vector_space_norm = normalize(vector_space)

vector_space_norm
def get_sims(vecs, raw=False):

    """Calculate cosine similarity - matrix dot product."""

    # expects np.matrix

    if not raw:

        vecs_adj = (vecs > 0).astype(int)

    else:

        vecs_adj = vecs

    

    extras = [vecs_adj]

    sims = np.dot(vecs_adj, vecs_adj.T)

    

    np.fill_diagonal(sims,0)

    return sims, extras
sims, etc = get_sims(vector_space_norm, raw=True)

len(sims[0])
sims
import networkx as nx

import random



def prune_graph(G, threshold):

    """Remove any nodes that have no edges."""

    for n in G.nodes(data=True):

        if G.degree(n[0]) <= threshold:

            G.remove_node(n[0])
def cw(G, iterations=3, class_label='class', weight_label='weight'):

    """Run the Chinese Whispers algorithm on the graph."""

    for c,n in enumerate(G.nodes()):

        G.node[n]['class'] = c



    for z in range(0, iterations):

        gn = G.nodes()

        # Randomize the nodes to give an arbitrary start point

        # random.shuffle(gn)

        random.sample(gn, k=len(gn))

        for node in gn:

            neighs = G[node]

            classes = {}

            # do an inventory of the given nodes neighbours and edge weights

            for ne in neighs:

                if G.node[ne][class_label] in classes:

                    classes[G.node[ne][class_label]] += G[node][ne][weight_label]

                else:

                    classes[G.node[ne][class_label]] = G[node][ne][weight_label]

            # find the class with the highest edge weight sum

            maxi = 0

            maxclass = 0

            for c in classes:

                if classes[c] > maxi:

                    maxi = classes[c]

                    maxclass = c

            # set the class of target node to the winning local class

            G.node[node][class_label] = maxclass

    return G
def cw_all(sims, lookup=False, lookup_id=False, threshold=0.5, iterations=3, 

           write_df=False, write_df_col='cluster', write_df_key=False,

           class_label='class', weight_label='weight'

          ):

    

    if type(sims) is not np.matrix:

        raise Exception('Requires matrix of type np.matrix')

        

    if type(lookup) == pd.core.frame.DataFrame or type(lookup) == dict:

        print("creating graph from dataframe")

        G = nx.Graph()

        edges = []

        edge_cnt = 0.

        for n, s in enumerate(sims):

            src = lookup.iloc[n].name

            for t in np.nonzero(s >= threshold)[1]:

                if t <= n:

                    continue

                if lookup_id:

                    if type(lookup_id) is int:

                        src = lookup.iat[n,lookup_id]

                        trg = lookup.iat[t,lookup_id]

                    else:

                        src = lookup.iloc[n][lookup_id]

                        trg = lookup.iloc[t][lookup_id]

                else:

                    trg = lookup.iloc[t].name

                edges.append((src,trg,s.A[0][t]))

                if edge_cnt % 50000 == 0.:

                    print('Edge count {}'.format(edge_cnt))

                edge_cnt += 1



        #return edges

        G.add_weighted_edges_from(edges)

        #print(nx.info(G))

    else:

        print("creating graph from matrix")

        G = nx.from_numpy_matrix(sims)

    

    prune_graph(G, 0)

    print('running CW')



    G = cw(G, iterations=iterations)

    

    if type(write_df) == pd.core.frame.DataFrame:

        print("writing to DF")

        if not write_df_key:

            if lookup_id:

                write_df_key = lookup_id

            else:

                return G

            return G

            

        write_df[write_df_col] = write_df[write_df_key].apply(lambda x: G.node[x][class_label] if x in G.node.keys() else -1)



    print("creating stats")

    cw_cls = np.array([G.node[n][class_label] for n in G.node])

    unique, counts = np.unique(cw_cls, return_counts=True)



    stats = pd.DataFrame(list(zip(unique, counts )))

    stats.sort_values(1, ascending=False)

    stats.columns = ['cluster','count']

    

    return G, stats
# Create a df with Sentence ID as index

df_main = pd.DataFrame(rev_sent_lrg_sample['Sentence ID'])

#df_main.set_index(0, inplace=True)



df_main.reset_index(drop=True, inplace=True)

df_v = pd.DataFrame(vector_space_norm)

df_main = pd.concat([df_main, df_v], axis=1)

df_master = df_main.copy()

df_main.set_index('Sentence ID', inplace=True)

df_main
w2v_G, w2v_stats = cw_all(np.asmatrix(sims),

                          lookup=df_main,

                          lookup_id=False, 

                          threshold=0.95,

                          iterations=10,

                          write_df=df_master, 

                          write_df_key='Sentence ID')
print(nx.info(w2v_G))
w2v_stats.sort_values(by='count', ascending=False, inplace=True)

w2v_stats
rev_sent_lrg_sample['CW Cluster'] = df_master['cluster']
inquiry_cluster = 353
rev_sent_lrg_sample['Review Sent'][rev_sent_lrg_sample['CW Cluster'] == inquiry_cluster][0:10]
cluster_wordrank(rev_sent_lrg_sample['Review Scrub'][rev_sent_lrg_sample['CW Cluster'] == inquiry_cluster])
make_wordcloud(rev_sent_lrg_sample['Review Scrub'][rev_sent_lrg_sample['CW Cluster'] == inquiry_cluster], 20)
inquiry_cluster = 1430

make_wordcloud(rev_sent_lrg_sample['Review Scrub'][rev_sent_lrg_sample['CW Cluster'] == inquiry_cluster], 20)
rev_sent_lrg_sample['Review Sent'][rev_sent_lrg_sample['CW Cluster'] == inquiry_cluster][0:10]
inquiry_cluster = 30

make_wordcloud(rev_sent_lrg_sample['Review Scrub'][rev_sent_lrg_sample['CW Cluster'] == inquiry_cluster], 20)
rev_sent_lrg_sample['Review Sent'][rev_sent_lrg_sample['CW Cluster'] == inquiry_cluster][0:10]