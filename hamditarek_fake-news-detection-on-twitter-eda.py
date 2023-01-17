import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_columns', 50)

pd.set_option('display.max_rows', 150)

import os

import gc

gc.enable()

import time

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from scipy.sparse import hstack

from scipy import stats

%matplotlib inline

from datetime import timedelta

import datetime as dt

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [16, 10]

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from sklearn.cluster import MiniBatchKMeans

import warnings

warnings.filterwarnings('ignore')

import urllib        #for url stuff



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction import text

from IPython.display import display

from tqdm import tqdm

from collections import Counter

import ast





from sklearn.feature_extraction.text import CountVectorizer

from textblob import TextBlob

import scipy.stats as stats



import seaborn as sns

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA, TruncatedSVD

import matplotlib.patches as mpatches

import time



import seaborn as sns #for making plots

import matplotlib.pyplot as plt # for plotting

import os  # for os commands



import gensim

from gensim import corpora, models, similarities

import logging

import tempfile

from nltk.corpus import stopwords

from string import punctuation

from collections import OrderedDict



from sklearn.decomposition import TruncatedSVD

from sklearn.decomposition import LatentDirichletAllocation

from sklearn.manifold import TSNE





from bokeh.plotting import figure, output_file, show

from bokeh.models import Label

from bokeh.io import output_notebook

output_notebook()

## Common Variables for Notebook 

ROOT = '/kaggle/input/nlp-getting-started/'



## load the data 

df_train = pd.read_csv(ROOT+'train.csv')

df_test = pd.read_csv(ROOT+'test.csv')

df_sub = pd.read_csv(ROOT+'sample_submission.csv')
#Looking data format and types

print(df_train.info())

print(df_test.info())

print(df_sub.info())
#Some Statistics

df_train.describe()
#Take a look at the data

df_train.head()
target = df_train['target']

sns.set_style('whitegrid')

sns.countplot(target)
df_train["text"].head()
#To check the text content we can use a list

df_train["text"].tolist()[:5]
t = df_train["text"].to_list()

for i in range(5):

    print('Tweet Number '+str(i+1)+': '+t[i])
l = df_train["location"].to_list()

print('There is '+ str(len(set(l)))+" different loction")
df_train['location'].value_counts().head(n=20)
# Plotting a bar graph of the number of tweets in each location, for the first ten locations listed

# in the column 'location'

location_count  = df_train['location'].value_counts()

location_count = location_count[:10,]

plt.figure(figsize=(10,5))

sns.barplot(location_count.index, location_count.values, alpha=0.8)

plt.title('Top 10 locations')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('location', fontsize=12)

plt.show()

# Plotting a bar graph of the number of tweets in each location, for the first ten locations listed

# in the column 'location'

location_count  = df_test['location'].value_counts()

location_count = location_count[:10,]

plt.figure(figsize=(10,5))

sns.barplot(location_count.index, location_count.values, alpha=0.8)

plt.title('Top 10 locations')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('location', fontsize=12)

plt.show()

df = df_train[df_train['location'].notnull()]
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df['location'] = le.fit_transform(df.location.values)
# tsne code from this great kernel: https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets

# New_df is from the random undersample data (fewer instances)

X = df['location']

y = df['target']





# T-SNE Implementation

t0 = time.time()

X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values.reshape(-1, 1))

t1 = time.time()

print("T-SNE took {:.2} s".format(t1 - t0))
f, (ax1) = plt.subplots(1, 1, figsize=(12,8))

# labels = ['Not Fake', 'Fake']

f.suptitle('Clusters using Dimensionality Reduction', fontsize=14)





blue_patch = mpatches.Patch(color='#0A0AFF', label='Not Fake')

red_patch = mpatches.Patch(color='#AF0000', label='Fake')





# t-SNE scatter plot

ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 0), cmap='coolwarm', label='Not Fake', linewidths=2)

ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 1), cmap='coolwarm', label='Fake', linewidths=2)

ax1.set_title('t-SNE', fontsize=14)



ax1.grid(True)



ax1.legend(handles=[blue_patch, red_patch])





plt.show()
df_train['keyword'].value_counts().head(n=20)
keyword_count  = df_train['keyword'].value_counts()

keyword_count = keyword_count[:10,]

plt.figure(figsize=(10,5))

sns.barplot(keyword_count.index, keyword_count.values, alpha=0.8)

plt.title('Top 10 keywords')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('keyword', fontsize=12)

plt.show()

df_test['keyword'].value_counts().head(n=20)
keyword_count  = df_test['keyword'].value_counts()

keyword_count = keyword_count[:10,]

plt.figure(figsize=(10,5))

sns.barplot(keyword_count.index, keyword_count.values, alpha=0.8)

plt.title('Top 10 keywords')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('keyword', fontsize=12)

plt.show()

keyword_train  = list(set(df_train['keyword']))

keyword_test  = list(set(df_test['keyword']))



print(len(keyword_train))

print(len(keyword_test))
def intersection(lst1, lst2): 

    lst3 = [value for value in lst1 if value in lst2] 

    return lst3 



len(intersection(keyword_train, keyword_test))
df = df_train[df_train['keyword'].notnull()]

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df['keyword'] = le.fit_transform(df.keyword.values)
# tsne code from this great kernel: https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets

# New_df is from the random undersample data (fewer instances)

X = df['keyword']

y = df['target']





# T-SNE Implementation

t0 = time.time()

X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values.reshape(-1, 1))

t1 = time.time()

print("T-SNE took {:.2} s".format(t1 - t0))
f, (ax1) = plt.subplots(1, 1, figsize=(12,8))

# labels = ['Not Fake', 'Fake']

f.suptitle('Clusters using Dimensionality Reduction', fontsize=14)





blue_patch = mpatches.Patch(color='#0A0AFF', label='Not Fake')

red_patch = mpatches.Patch(color='#AF0000', label='Fake')





# t-SNE scatter plot

ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 0), cmap='coolwarm', label='Not Fake', linewidths=2)

ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 1), cmap='coolwarm', label='Fake', linewidths=2)

ax1.set_title('t-SNE', fontsize=14)



ax1.grid(True)



ax1.legend(handles=[blue_patch, red_patch])





plt.show()
len(set(df_train['text']))
len(set(df_test['text']))
html_tags = ['<P>', '</P>', '<Table>', '</Table>', '<Tr>', '</Tr>', '<Ul>', '<Ol>', '<Dl>', '</Ul>', '</Ol>', \

             '</Dl>', '<Li>', '<Dd>', '<Dt>', '</Li>', '</Dd>', '</Dt>']

r_buf = ['It', 'is', 'are', 'do', 'does', 'did', 'was', 'were', 'will', 'can', 'the', 'a', 'of', 'in', 'and', 'on', \

         'what', 'where', 'when', 'which'] + html_tags



def clean(x):

    x = x.lower()

    for r in r_buf:

        x = x.replace(r, '')

    x = re.sub(' +', ' ', x)

    return x



bin_question_tokens = ['it', 'is', 'are', 'do', 'does', 'did', 'was', 'were', 'will', 'can']

stop_words = text.ENGLISH_STOP_WORDS.union(["book"])



def predict(json_data, annotated=False):

    # Parse JSON data

    candidates = json_data['long_answer_candidates']

    candidates = [c for c in candidates if c['top_level'] == True]

    doc_tokenized = json_data['document_text'].split(' ')

    question = json_data['question_text']

    question_s = question.split(' ') 

    if annotated:

        ann = json_data['annotations'][0]



    # TFIDF for the document

    tfidf = TfidfVectorizer(ngram_range=(1,1), stop_words=stop_words)

    tfidf.fit([json_data['document_text']])

    q_tfidf = tfidf.transform([question]).todense()



    # Find the nearest answer from candidates

    distances = []

    scores = []

    i_ann = -1

    for i, c in enumerate(candidates):

        s, e = c['start_token'], c['end_token']

        t = ' '.join(doc_tokenized[s:e])

        distances.append(levenshtein_distance(clean(question), clean(t)))

        

        t_tfidf = tfidf.transform([t]).todense()

        score = 1 - spatial.distance.cosine(q_tfidf, t_tfidf)

        

#         score = 0

        

#         for w in doc_tokenized[s:e]:

#             if w in q_s:

#                 score += 0.1



        scores.append(score)



    # Format results

#     ans = candidates[np.argmin(distances)]

    ans = candidates[np.argmax(scores)]

    if np.max(scores) < 0.2:

        ans_long = '-1:-1'

    else:

        ans_long = str(ans['start_token']) + ':' + str(ans['end_token'])

    if question_s[0] in bin_question_tokens:

        ans_short = 'YES'

    else:

        ans_short = ''

        

    # Preparing data for debug

    if annotated:

        ann_long_text = ' '.join(doc_tokenized[ann['long_answer']['start_token']:ann['long_answer']['end_token']])

        if ann['yes_no_answer'] == 'NONE':

            if len(json_data['annotations'][0]['short_answers']) > 0:

                ann_short_text = ' '.join(doc_tokenized[ann['short_answers'][0]['start_token']:ann['short_answers'][0]['end_token']])

            else:

                ann_short_text = ''

        else:

            ann_short_text = ann['yes_no_answer']

    else:

        ann_long_text = ''

        ann_short_text = ''

        

    ans_long_text = ' '.join(doc_tokenized[ans['start_token']:ans['end_token']])

    if len(ans_short) > 0 or ans_short == 'YES':

        ans_short_text = ans_short

    else:

        ans_short_text = '' # Fix when short answers will work

                    

    return ans_long, ans_short, question, ann_long_text, ann_short_text, ans_long_text, ans_short
reindexed_data = df_train['text'] #.str.decode("utf-8", errors="ignore")
# Define helper functions

def get_top_n_words(n_top_words, count_vectorizer, text_data):

    '''

    returns a tuple of the top n words in a sample and their 

    accompanying counts, given a CountVectorizer object and text sample

    '''

    vectorized_headlines = count_vectorizer.fit_transform(text_data.values)

    vectorized_total = np.sum(vectorized_headlines, axis=0)

    word_indices = np.flip(np.argsort(vectorized_total)[0,:], 1)

    word_values = np.flip(np.sort(vectorized_total)[0,:],1)

    

    word_vectors = np.zeros((n_top_words, vectorized_headlines.shape[1]))

    for i in range(n_top_words):

        word_vectors[i,word_indices[0,i]] = 1



    words = [word[0].encode('ascii',errors="ignore").decode('utf-8',errors="ignore") for 

             word in count_vectorizer.inverse_transform(word_vectors)]



    return (words, word_values[0,:n_top_words].tolist()[0])
count_vectorizer = CountVectorizer(stop_words='english')

words, word_values = get_top_n_words(n_top_words=25,

                                     count_vectorizer=count_vectorizer, 

                                     text_data=reindexed_data)



fig, ax = plt.subplots(figsize=(10,4))

ax.bar(range(len(words)), word_values);

ax.set_xticks(range(len(words)));

ax.set_xticklabels(words, rotation='vertical');

ax.set_title('Top words in headlines dataset (excluding stop words)');

ax.set_xlabel('Word');

ax.set_ylabel('Number of occurences');

plt.show()
tagged_headlines = [TextBlob(reindexed_data[i]).pos_tags for i in range(reindexed_data.shape[0])]
tagged_headlines_df = pd.DataFrame({'tags':tagged_headlines})



word_counts = [] 

pos_counts = {}



for headline in tagged_headlines_df[u'tags']:

    word_counts.append(len(headline))

    for tag in headline:

        if tag[1] in pos_counts:

            pos_counts[tag[1]] += 1

        else:

            pos_counts[tag[1]] = 1

            

print('Total number of words: ', np.sum(word_counts))

print('Mean number of words per tweet: ', np.mean(word_counts))
y = stats.norm.pdf(np.linspace(0,14,50), np.mean(word_counts), np.std(word_counts))



fig, ax = plt.subplots(figsize=(8,4))

ax.hist(word_counts, bins=range(1,14), density=True);

ax.plot(np.linspace(0,14,50), y, 'r--', linewidth=1);

ax.set_title('Headline word lengths');

ax.set_xticks(range(1,14));

ax.set_xlabel('Number of words');

plt.show()
pos_sorted_types = sorted(pos_counts, key=pos_counts.__getitem__, reverse=True)

pos_sorted_counts = sorted(pos_counts.values(), reverse=True)



fig, ax = plt.subplots(figsize=(14,4))

ax.bar(range(len(pos_counts)), pos_sorted_counts);

ax.set_xticks(range(len(pos_counts)));

ax.set_xticklabels(pos_sorted_types);

ax.set_title('Part-of-Speech Tagging for questions Corpus');

ax.set_xlabel('Type of Word');
small_count_vectorizer = CountVectorizer(stop_words='english', max_features=40000)

small_text_sample = reindexed_data.sample(n=500, random_state=0).values



print('Tweets before vectorization: {}'.format(small_text_sample[123]))



small_document_term_matrix = small_count_vectorizer.fit_transform(small_text_sample)



print('Tweets after vectorization: \n{}'.format(small_document_term_matrix[123]))
#number of topics

n_topics = 5
lsa_model = TruncatedSVD(n_components=n_topics)

lsa_topic_matrix = lsa_model.fit_transform(small_document_term_matrix)
# Define helper functions

def get_keys(topic_matrix):

    '''

    returns an integer list of predicted topic 

    categories for a given topic matrix

    '''

    keys = topic_matrix.argmax(axis=1).tolist()

    return keys



def keys_to_counts(keys):

    '''

    returns a tuple of topic categories and their 

    accompanying magnitudes for a given list of keys

    '''

    count_pairs = Counter(keys).items()

    categories = [pair[0] for pair in count_pairs]

    counts = [pair[1] for pair in count_pairs]

    return (categories, counts)
lsa_keys = get_keys(lsa_topic_matrix)

lsa_categories, lsa_counts = keys_to_counts(lsa_keys)
# Define helper functions

def get_top_n_words(n, keys, document_term_matrix, count_vectorizer):

    '''

    returns a list of n_topic strings, where each string contains the n most common 

    words in a predicted category, in order

    '''

    top_word_indices = []

    for topic in range(n_topics):

        temp_vector_sum = 0

        for i in range(len(keys)):

            if keys[i] == topic:

                temp_vector_sum += document_term_matrix[i]

        temp_vector_sum = temp_vector_sum.toarray()

        top_n_word_indices = np.flip(np.argsort(temp_vector_sum)[0][-n:],0)

        top_word_indices.append(top_n_word_indices)   

    top_words = []

    for topic in top_word_indices:

        topic_words = []

        for index in topic:

            temp_word_vector = np.zeros((1,document_term_matrix.shape[1]))

            temp_word_vector[:,index] = 1

            the_word = count_vectorizer.inverse_transform(temp_word_vector)[0][0]

            topic_words.append(the_word.encode('ascii',errors="ignore").decode('utf-8',errors="ignore"))

        top_words.append(" ".join(topic_words))         

    return top_words
top_n_words_lsa = get_top_n_words(10, lsa_keys, small_document_term_matrix, small_count_vectorizer)



for i in range(len(top_n_words_lsa)):

    print("Topic {}: ".format(i+1), top_n_words_lsa[i])
top_3_words = get_top_n_words(3, lsa_keys, small_document_term_matrix, small_count_vectorizer)

labels = ['Topic {}: \n'.format(i) + top_3_words[i] for i in lsa_categories]



fig, ax = plt.subplots(figsize=(8,4))

ax.bar(lsa_categories, lsa_counts);

ax.set_xticks(lsa_categories);

ax.set_xticklabels(labels);

ax.set_ylabel('Number of tweets');

ax.set_title('LSA topic counts');

plt.show()
tsne_lsa_model = TSNE(n_components=2, perplexity=50, learning_rate=100, 

                        n_iter=2000, verbose=1, random_state=0, angle=0.75)

tsne_lsa_vectors = tsne_lsa_model.fit_transform(lsa_topic_matrix)
# Define helper functions

def get_mean_topic_vectors(keys, two_dim_vectors):

    '''

    returns a list of centroid vectors from each predicted topic category

    '''

    mean_topic_vectors = []

    for t in range(n_topics):

        articles_in_that_topic = []

        for i in range(len(keys)):

            if keys[i] == t:

                articles_in_that_topic.append(two_dim_vectors[i])    

        

        articles_in_that_topic = np.vstack(articles_in_that_topic)

        mean_article_in_that_topic = np.mean(articles_in_that_topic, axis=0)

        mean_topic_vectors.append(mean_article_in_that_topic)

    return mean_topic_vectors
colormap = np.array([

    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",

    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",

    "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",

    "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5" ])

colormap = colormap[:n_topics]
top_3_words_lsa = get_top_n_words(3, lsa_keys, small_document_term_matrix, small_count_vectorizer)

lsa_mean_topic_vectors = get_mean_topic_vectors(lsa_keys, tsne_lsa_vectors)



plot = figure(title="t-SNE Clustering of {} LSA Topics".format(n_topics), plot_width=700, plot_height=700)

plot.scatter(x=tsne_lsa_vectors[:,0], y=tsne_lsa_vectors[:,1], color=colormap[lsa_keys])



for t in range(n_topics):

    label = Label(x=lsa_mean_topic_vectors[t][0], y=lsa_mean_topic_vectors[t][1], 

                  text=top_3_words_lsa[t], text_color=colormap[t])

    plot.add_layout(label)

    

show(plot)
lda_model = LatentDirichletAllocation(n_components=n_topics, learning_method='online', 

                                          random_state=0, verbose=0)

lda_topic_matrix = lda_model.fit_transform(small_document_term_matrix)
lda_keys = get_keys(lda_topic_matrix)

lda_categories, lda_counts = keys_to_counts(lda_keys)
top_n_words_lda = get_top_n_words(10, lda_keys, small_document_term_matrix, small_count_vectorizer)



for i in range(len(top_n_words_lda)):

    print("Topic {}: ".format(i+1), top_n_words_lda[i])
top_3_words = get_top_n_words(3, lda_keys, small_document_term_matrix, small_count_vectorizer)

labels = ['Topic {}: \n'.format(i) + top_3_words[i] for i in lda_categories]



fig, ax = plt.subplots(figsize=(10,4))

ax.bar(lda_categories, lda_counts);

ax.set_xticks(lda_categories);

ax.set_xticklabels(labels);

ax.set_title('LDA topic counts');

ax.set_ylabel('Number of tweets');
tsne_lda_model = TSNE(n_components=2, perplexity=50, learning_rate=100, 

                        n_iter=2000, verbose=1, random_state=0, angle=0.75)

tsne_lda_vectors = tsne_lda_model.fit_transform(lda_topic_matrix)
top_3_words_lda = get_top_n_words(3, lda_keys, small_document_term_matrix, small_count_vectorizer)

lda_mean_topic_vectors = get_mean_topic_vectors(lda_keys, tsne_lda_vectors)



plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics), plot_width=600, plot_height=600)

plot.scatter(x=tsne_lda_vectors[:,0], y=tsne_lda_vectors[:,1], color=colormap[lda_keys])



for t in range(n_topics):

    label = Label(x=lda_mean_topic_vectors[t][0], y=lda_mean_topic_vectors[t][1], 

                  text=top_3_words_lda[t], text_color=colormap[t])

    plot.add_layout(label)



show(plot)
# Preparing a corpus for analysis and checking the first 5 entries

corpus=[]



corpus = df_train['text'].to_list()



corpus[:5]
TEMP_FOLDER = tempfile.gettempdir()

print('Folder "{}" will be used to save temporary dictionary and corpus.'.format(TEMP_FOLDER))



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# removing common words and tokenizing

# google-quest-challenge

stoplist = stopwords.words('english') + list(punctuation) + list("([)]?") + [")?"]



texts = [[word for word in str(document).lower().split() if word not in stoplist] for document in corpus]



dictionary = corpora.Dictionary(texts)

dictionary.save(os.path.join(TEMP_FOLDER, 'tweets.dict'))  # store the dictionary
corpus = [dictionary.doc2bow(text) for text in texts]

corpora.MmCorpus.serialize(os.path.join(TEMP_FOLDER, 'tweets.mm'), corpus) 
tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
corpus_tfidf = tfidf[corpus]  # step 2 -- use the model to transform vectors
#I will try 10 topics

total_topics = 10



lda = models.LdaModel(corpus, id2word=dictionary, num_topics=total_topics)

corpus_lda = lda[corpus_tfidf] # create a double wrapper over the original corpus: bow->tf
lda.show_topics(total_topics,5)
data_lda = {i: OrderedDict(lda.show_topic(i,25)) for i in range(total_topics)}
df_lda = pd.DataFrame(data_lda)

df_lda = df_lda.fillna(0).T

print(df_lda.shape)
df_lda
g=sns.clustermap(df_lda.corr(), center=0, standard_scale=1, cmap="OrRd", metric='cosine', linewidths=.75, figsize=(12, 12))

plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

plt.show()

#plt.setp(ax_heatmap.get_yticklabels(), rotation=0)  # For y axis
import pyLDAvis.gensim

pyLDAvis.enable_notebook()

panel = pyLDAvis.gensim.prepare(lda, corpus_lda, dictionary, mds='tsne')

panel