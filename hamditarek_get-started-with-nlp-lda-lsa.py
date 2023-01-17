import os

import json

import numpy as np 

import pandas as pd

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

import re

from IPython.display import display

from tqdm import tqdm

from collections import Counter

import ast

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly as py

import plotly.graph_objs as go

import gensim

from gensim import corpora, models, similarities

import logging

import tempfile

from nltk.corpus import stopwords

from string import punctuation

from collections import OrderedDict

import seaborn as sns

import pyLDAvis.gensim

import matplotlib.pyplot as plt

%matplotlib inline



init_notebook_mode(connected=True) #do not miss this line



import warnings

warnings.filterwarnings("ignore")



import matplotlib.pyplot as plt

import matplotlib.mlab as mlab

import seaborn as sb



from sklearn.feature_extraction.text import CountVectorizer

from textblob import TextBlob

import scipy.stats as stats



from sklearn.decomposition import TruncatedSVD

from sklearn.decomposition import LatentDirichletAllocation

from sklearn.manifold import TSNE



from bokeh.plotting import figure, output_file, show

from bokeh.models import Label

from bokeh.io import output_notebook

output_notebook()



%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

from sklearn.metrics import accuracy_score, f1_score

from tqdm import tqdm_notebook as tqdm

from Levenshtein import ratio as levenshtein_distance



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction import text



from scipy import spatial
## Common Variables for Notebook 

ROOT = '/kaggle/input/google-quest-challenge/'



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
df_sub.describe()
#Take a look at the data

df_train.head()
df_train["question_title"].head()
q = df_train["question_title"].to_list()

for i in range(5):

    print('Question title '+str(i+1)+': '+q[i])
q = df_train["question_body"].to_list()

for i in range(5):

    print('==> Question body '+str(i+1)+': '+q[i])

    print('****************************************************************************************************')

    print('****************************************************************************************************')
q = df_train["answer"].to_list()

for i in range(5):

    print('==> Answer '+str(i+1)+': '+q[i])
#defining the figure size of our graphic

plt.figure(figsize=(12,5))



#Plotting the result

sns.countplot(x='category', data=df_train, palette="hls")

plt.xlabel("category", fontsize=16) #seting the xtitle and size

plt.ylabel("Count", fontsize=16) # Seting the ytitle and size

plt.title("Category Name Count", fontsize=20) 

plt.xticks(rotation=45)

plt.show()
df_train['host_type'] = df_train.host.apply(lambda x: x.split('.')[0])
#defining the figure size of our graphic

plt.figure(figsize=(12,8))



#Plotting the result

sns.countplot(x='host_type', data=df_train, palette="hls")

plt.xlabel("host_type", fontsize=16) #seting the xtitle and size

plt.ylabel("Count", fontsize=16) # Seting the ytitle and size

plt.title("Host Type Name Count", fontsize=20) 

plt.xticks(rotation=45)

plt.show()
## Check the scoring for questions

all_train_columns = list(df_train.columns)

question_answer_cols = all_train_columns[:11]

question_target_cols = all_train_columns[11:32]

answer_target_cols  = all_train_columns[32:41]



## Check target scoring for question

df_train[question_target_cols].loc[0]
## Check target scoring for answer

df_train[answer_target_cols].loc[0]
print('There is '+str(len(set(df_train['question_user_name'].to_list())))+' unique user asked a questions')

print('There is '+str(len(set(df_train['answer_user_name'].to_list())))+' unique user answer a questions')
## What is the distribution of all question ranking columns 



df_train[question_target_cols]
## lets see some distributions of questions targets

plt.figure(figsize=(10, 5))



sns.distplot(df_train[question_target_cols[0]], hist= False , rug= False ,kde=True, label =question_target_cols[0],axlabel =False )

sns.distplot(df_train[question_target_cols[1]], hist= False , rug= False,label =question_target_cols[1],axlabel =False)

sns.distplot(df_train[question_target_cols[2]], hist= False , rug= False,label =question_target_cols[2],axlabel =False)

sns.distplot(df_train[question_target_cols[3]], hist= False , rug= False,label =question_target_cols[3],axlabel =False)

sns.distplot(df_train[question_target_cols[4]], hist= False , rug= False,label =question_target_cols[4],axlabel =False)

plt.show()
## lets see some distributions of answer targets

plt.figure(figsize=(10, 5))



sns.distplot(df_train[answer_target_cols[0]], hist= False , rug= False ,kde=True, label =answer_target_cols[0],axlabel =False )

sns.distplot(df_train[answer_target_cols[1]], hist= False , rug= False,label =answer_target_cols[1],axlabel =False)

#sns.distplot(train[answer_target_cols[2]], hist= False , rug= False,label =answer_target_cols[2],axlabel =False)

#sns.distplot(train[answer_target_cols[3]], hist= False , rug= False,label =answer_target_cols[3],axlabel =False)

sns.distplot(df_train[answer_target_cols[4]], hist= False , rug= False,label =answer_target_cols[4],axlabel =False)

plt.show()

## Removed two columns as value was quite high and other graphs were not visible .
# Lets see how the mean value of one target feature for questions changes based on category

for idx in range(20):

    df = df_train.groupby('category')[question_target_cols[idx]].mean()

        

    fig, axes = plt.subplots(1, 1, figsize=(10,10))

    axes.set_title(question_target_cols[idx])

    df.plot(label=question_target_cols[idx])

    plt.show()
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

                    

    return ans_long, ans_short, question, ann_long_text, ann_short_text, ans_long_text, ans_short_text
reindexed_data = df_train['question_body']

reindexed_data1 = df_train['answer']
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



    words = [word[0].encode('ascii').decode('utf-8') for 

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
count_vectorizer = CountVectorizer(stop_words='english')

words, word_values = get_top_n_words(n_top_words=25,

                                     count_vectorizer=count_vectorizer, 

                                     text_data=reindexed_data1)



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

print('Mean number of words per question: ', np.mean(word_counts))
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



print('Questions before vectorization: {}'.format(small_text_sample[123]))



small_document_term_matrix = small_count_vectorizer.fit_transform(small_text_sample)



print('Questions after vectorization: \n{}'.format(small_document_term_matrix[123]))
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

            topic_words.append(the_word.encode('ascii').decode('utf-8'))

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

ax.set_ylabel('Number of questions');

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

ax.set_ylabel('Number of questions');
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



corpus = df_train['question_body'].to_list()



corpus[:5]
TEMP_FOLDER = tempfile.gettempdir()

print('Folder "{}" will be used to save temporary dictionary and corpus.'.format(TEMP_FOLDER))



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# removing common words and tokenizing

# google-quest-challenge

stoplist = stopwords.words('english') + list(punctuation) + list("([)]?") + [")?"]



texts = [[word for word in str(document).lower().split() if word not in stoplist] for document in corpus]



dictionary = corpora.Dictionary(texts)

dictionary.save(os.path.join(TEMP_FOLDER, 'google-quest-challenge.dict'))  # store the dictionary,
corpus = [dictionary.doc2bow(text) for text in texts]

corpora.MmCorpus.serialize(os.path.join(TEMP_FOLDER, 'google-quest-challenge.mm'), corpus) 
tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
corpus_tfidf = tfidf[corpus]  # step 2 -- use the model to transform vectors
#I will try 15 topics

total_topics = 15



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
pyLDAvis.enable_notebook()

panel = pyLDAvis.gensim.prepare(lda, corpus_lda, dictionary, mds='tsne')

panel
train = df_train.copy()

test = df_test.copy()
target_cols = ['question_asker_intent_understanding',

       'question_body_critical', 'question_conversational',

       'question_expect_short_answer', 'question_fact_seeking',

       'question_has_commonly_accepted_answer',

       'question_interestingness_others', 'question_interestingness_self',

       'question_multi_intent', 'question_not_really_a_question',

       'question_opinion_seeking', 'question_type_choice',

       'question_type_compare', 'question_type_consequence',

       'question_type_definition', 'question_type_entity',

       'question_type_instructions', 'question_type_procedure',

       'question_type_reason_explanation', 'question_type_spelling',

       'question_well_written', 'answer_helpful',

       'answer_level_of_information', 'answer_plausible', 'answer_relevance',

       'answer_satisfaction', 'answer_type_instructions',

       'answer_type_procedure', 'answer_type_reason_explanation',

       'answer_well_written']
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD
tfidf = TfidfVectorizer(ngram_range=(1, 3))

tsvd = TruncatedSVD(n_components = 50)

question_title = tfidf.fit_transform(train["question_title"].values)

question_title_test = tfidf.transform(test["question_title"].values)

question_title = tsvd.fit_transform(question_title)

question_title_test = tsvd.transform(question_title_test)



question_body = tfidf.fit_transform(train["question_body"].values)

question_body_test = tfidf.transform(test["question_body"].values)

question_body = tsvd.fit_transform(question_body)

question_body_test = tsvd.transform(question_body_test)



answer = tfidf.fit_transform(train["answer"].values)

answer_test = tfidf.transform(test["answer"].values)

answer = tsvd.fit_transform(answer)

answer_test = tsvd.transform(answer_test)
train["len_user_name"]= train.question_user_name.apply(lambda x : len(x.split()))

test["len_user_name"]= test.question_user_name.apply(lambda x : len(x.split()))
train["cat_host"]= train["category"]+train["host"]+str(train["len_user_name"])

test["cat_host"]= test["category"]+test["host"]+str(test["len_user_name"])





category_means_map = train.groupby("len_user_name")[target_cols].mean().T.to_dict()

category_te = train["len_user_name"].map(category_means_map).apply(pd.Series)

category_te_test = test["len_user_name"].map(category_means_map).apply(pd.Series)
train_features = np.concatenate([question_title, question_body, answer#, category_te.values

                                ], axis = 1)

test_features = np.concatenate([question_title_test, question_body_test, answer_test#, category_te_test.values

                               ], axis = 1)
from keras.models import Sequential

from keras.layers import Dense, Activation

from sklearn.model_selection import KFold

from keras.callbacks.callbacks import EarlyStopping

from scipy.stats import spearmanr



num_folds = 5

fold_scores = []

kf = KFold(n_splits = num_folds, shuffle = True, random_state = 42)

test_preds = np.zeros((len(test_features), len(target_cols)))

for train_index, val_index in kf.split(train_features):

    train_X = train_features[train_index, :]

    train_y = train[target_cols].iloc[train_index]

    

    val_X = train_features[val_index, :]

    val_y = train[target_cols].iloc[val_index]

    

    model = Sequential([

        Dense(128, input_shape=(train_features.shape[1],)),

        Activation('relu'),

        Dense(64),

        Activation('relu'),

        Dense(len(target_cols)),

        Activation('sigmoid'),

    ])

    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto', baseline=None, restore_best_weights=True)

    model.compile(optimizer='adam',

                  loss='binary_crossentropy')

    

    model.fit(train_X, train_y, epochs = 50, validation_data=(val_X, val_y), callbacks = [es])

    preds = model.predict(val_X)

    overall_score = 0

    for col_index, col in enumerate(target_cols):

        overall_score += spearmanr(preds[:, col_index], val_y[col].values).correlation/len(target_cols)

        print(col, spearmanr(preds[:, col_index], val_y[col].values).correlation)

    fold_scores.append(overall_score)



    test_preds += model.predict(test_features)/num_folds

    

print(fold_scores)
sub = df_sub.copy()
for col_index, col in enumerate(target_cols):

    sub[col] = test_preds[:, col_index]
sub.to_csv("submission.csv", index = False)