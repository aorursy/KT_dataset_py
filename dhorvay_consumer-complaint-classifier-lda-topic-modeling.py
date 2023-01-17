import numpy as np

import pandas as pd

import re

import pickle

from bokeh.plotting import figure, show

from bokeh.io import output_notebook, save

from bokeh.models import ColumnDataSource

from bokeh.transform import cumsum

from bokeh.io import curdoc

curdoc().theme = 'dark_minimal'
ca_df = pd.read_csv('/kaggle/input/comcastcomplaints/comcast_consumeraffairs_complaints.csv')

ca_df.head()
fcc_df = pd.read_csv('/kaggle/input/comcastcomplaints/comcast_fcc_complaints_2015.csv')

fcc_df.head()
# Try to make it Comcast (new logo after NBC merge) themed

peacock_theme = ['#cc004c','#f37021','#fcb711','#6460aa','#0089d0','#0db14b','#ffc0cb','#00ffff']
# Calculate CSAT

csat = len(ca_df.loc[ca_df['rating']>=4])/len(ca_df)

print("Comcast's Customer Satisfaction Score calculated from ConsumerAffairs is {:.2f}%".format(csat*100))
rating_df = pd.DataFrame({'class': ['0', '1', '2', '3', '4', '5'],

                          'percent' : ca_df['rating'].value_counts(normalize=True).sort_index()*100,

                          'angle': ca_df['rating'].value_counts(normalize=True).sort_index() * 2 * np.pi,

                          'color': peacock_theme[0:6]})

rating_df
rating_src = ColumnDataSource(rating_df)

# Date range of ratings

sorted_dates = ca_df['posted_on'].sort_values().reset_index(drop=True).tolist()

rating_plt_title = 'Comcast ConsumerAffairs Ratings {} - {}'.format(sorted_dates[0], sorted_dates[-1])



# Create the Figure object "rating_plt"

rating_plt = figure(title=rating_plt_title, tools=['save', 'hover'], tooltips='@percent{0.00}%')



# Add circular sectors to "rating_plt"

rating_plt.wedge(x=0, y=0, radius=0.8, source=rating_src, start_angle=cumsum('angle', include_zero=True),

                 end_angle=cumsum('angle'), fill_color='color', line_color=None, legend_field='class')



# Change parameters of "rating_plt"

rating_plt.axis.visible = False

rating_plt.grid.grid_line_color = None

rating_plt.legend.orientation = 'horizontal'

rating_plt.legend.location = 'top_center'

output_notebook()

show(rating_plt)
ca_df['posted_on'] = pd.to_datetime(ca_df['posted_on'])



groupby_posted_on = ca_df.groupby('posted_on').count()



ts_src = ColumnDataSource(groupby_posted_on)



ts_plt_title = 'Number of reviews per day {} - {}'.format(sorted_dates[0], sorted_dates[-1])

ts_plt = figure(title=ts_plt_title, x_axis_type='datetime', tools=['save', 'hover'], tooltips=[('Count', '@rating')])



ts_plt.line(x='posted_on', y='rating', line_width=2, source=ts_src, color=peacock_theme[0])



ts_plt.yaxis.axis_label = 'Number of Reviews'



show(ts_plt)
# What the hell happened?

groupby_posted_on.loc[groupby_posted_on['rating'] > 50]



# Change pandas settings to allow max rows be 100

ca_df.loc[ca_df['posted_on'] == '2016-02-24']
fcc_df['Date'] = pd.to_datetime(fcc_df['Date'])



ca_df['Count'] = 0

ca_df2 = ca_df.loc[ca_df['posted_on'] >= '2015-01-01']

groupby_posted_on = ca_df2.groupby('posted_on').count()



ts_ca_src = ColumnDataSource(groupby_posted_on)



fcc_df['Count'] = 0

groupby_date = fcc_df.groupby('Date').count()



ts_fcc_src = ColumnDataSource(groupby_date)



ts_fcc_plt = figure(title="Number of Customer FCC Complaints and ConsumerAffairs Reviews Per Day",

                    x_axis_type='datetime', tools=['save', 'hover'], tooltips=[('Count', '@Count')])



ts_fcc_plt.line(x='Date', y='Customer Complaint', line_width=2, source=ts_fcc_src, color=peacock_theme[1],

                legend_label=' # of FCC Customer Complaints')



ts_fcc_plt.line(x='posted_on', y='rating', line_width=2, source=ts_ca_src, color=peacock_theme[0],

                legend_label='# of ConsumerAffairs Reviews')



show(ts_fcc_plt)



fcc_df = fcc_df.drop(columns="Count")
fcc_df['Customer Complaint'].value_counts()
def get_simple_topic_percentage(topic):

    """

    Returns a percentage of rows that this particular topic is found

    in using simple string manipulation. Note: this can have overlaps,

    for example if you have two topics, one 'Internet' and one 'Speed',

    you will get duplicate findings if the customer has 'Internet Speed'

    as their topic.

    

    topic: the customer complaint category entered by the customer.

    """

    return fcc_df[fcc_df['Customer Complaint'].str.contains(topic, case=False)].shape[0] / len(fcc_df['Customer Complaint']) * 100

    



print('Comcast:', get_simple_topic_percentage('comcast'))

print('Data cap:', get_simple_topic_percentage('data'))

print('Speed:', get_simple_topic_percentage('speed'))

print('Internet:', get_simple_topic_percentage('internet'))

print('Price:', get_simple_topic_percentage('price'))

print('Bill:', get_simple_topic_percentage('bill'))

print('Customer Service:', get_simple_topic_percentage('customer service'))
from spacy.lang.en import English

nlp = English()



customize_stop_words = ['comcast', 'i', 'fcc', 'hello', 'service', 'services', 'issue',

                        'issues', 'problem', 'problems', 'xfinity', 'customer', 'complaint', '$']

for w in customize_stop_words:

    nlp.vocab[w].is_stop = True



def preprocess(verbatim):

    """

    Tokenizes, removes stopwords, and lemmatizes a verbatim text

    

    verbatim: a free-form text complaint

    """

    # Every verbatim ends with the FCC follow up, let's remove this.

    verbatim = verbatim.split('\n')[0].lower()

    doc = nlp(verbatim)

    sent = []

    for word in doc:

        # If it's not a stop word or punctuation mark, add it to our article!

        if word.text != 'n' and not word.is_stop and not word.is_punct and not word.like_num:

            # We add the lematized version of the word

            sent.append(word.lemma_.lower())

    return sent



# Tokenize each complaint

docs = fcc_df['Description'].apply(lambda verbatim: preprocess(verbatim))
docs[0]
import nltk

from nltk import FreqDist

cats = fcc_df['Customer Complaint'].apply(lambda verbatim: preprocess(verbatim))

filtered_complaints = [c for cl in cats for c in cl]

fdist = FreqDist(filtered_complaints)

print(fdist.most_common(30))

fdist.plot(30)
import gensim

from gensim.corpora import Dictionary



dictionary = Dictionary(docs)



print('Distinct words in initial documents:', len(dictionary))



# Filter out words that occur less than 10 documents, or more than 30% of the documents.

dictionary.filter_extremes(no_below=10, no_above=0.3)



print('Distinct words after removing rare and common words:', len(dictionary))
from gensim.models import CoherenceModel, LdaModel

import pyLDAvis.gensim



corpus = [dictionary.doc2bow(doc) for doc in docs]

num_topics = 8



# Check for .pickle

filename = '/kaggle/input/lda-modelpickle/lda_model.pickle'

model = []

found = False

try: 

    infile = open(filename,'rb')

    model = pickle.load(infile)

    infile.close()

    found = True

    print('Model found..loaded.')

except:

    print('Model not found!')



if not found:

    %time model = gensim.models.LdaMulticore(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=20, workers=2)

    outfile = open(filename,'wb')

    pickle.dump(model, outfile)

    outfile.close()
pyLDAvis.enable_notebook()

pyLDAvis.gensim.prepare(model, corpus, dictionary)
from sklearn.metrics.pairwise import cosine_similarity



fcc_df_super = fcc_df.copy()



fcc_df_super['Tokens'] = docs



docs_upper = fcc_df_super['Tokens'].apply(lambda l: l[:int(len(l)/2)])

docs_lower = fcc_df_super['Tokens'].apply(lambda l: l[int(len(l)/2):])
corpus_upper = [dictionary.doc2bow(doc) for doc in docs_upper]

corpus_lower = [dictionary.doc2bow(doc) for doc in docs_lower]



# Using the corpus LDA model tranformation

lda_corpus_upper = model[corpus_upper]

lda_corpus_lower = model[corpus_lower]
from collections import OrderedDict

def get_doc_topic_dist(model, corpus, kwords=False): 

    '''

    LDA transformation, for each doc only returns topics with non-zero weight

    This function makes a matrix transformation of docs in the topic space.

    

    model: the LDA model

    corpus: the documents

    kwords: if True adds and returns the keys

    '''

    top_dist =[]

    keys = []

    for d in corpus:

        tmp = {i:0 for i in range(num_topics)}

        tmp.update(dict(model[d]))

        vals = list(OrderedDict(tmp).values())

        top_dist += [np.asarray(vals)]

        if kwords:

            keys += [np.asarray(vals).argmax()]



    return np.asarray(top_dist), keys
top_dist_upper, _ = get_doc_topic_dist(model, lda_corpus_upper)

top_dist_lower, _ = get_doc_topic_dist(model, lda_corpus_lower)



print("Intra-similarity:", np.mean([cosine_similarity(c1.reshape(1, -1), c2.reshape(1, -1))[0][0] for c1, c2 in zip(top_dist_upper, top_dist_lower)]))



random_pairs = np.random.randint(0, len(fcc_df_super['Description']), size=(400, 2))



print("Inter-similarity:", np.mean([cosine_similarity(top_dist_upper[0].reshape(1, -1), top_dist_lower[1].reshape(1, -1))]))
print(u'{:20} {}'.format(u'term', u'frequency') + u'\n')

for i in range(num_topics):

    print('\nTopic {}\n'.format(str(i)))

    for term, frequency in model.show_topic(i, topn=10):

        print(u'{:20} {:.3f}'.format(term, round(frequency, 3)))
top_labels = {0: 'Pricing', 1:'Billing', 2:'Data Caps', 3:'Missed Appointments', 4:'Moving Services', 5: 'Customer Services', 6:'Internet Speed', 7: 'Business Contracts'}
from sklearn.feature_extraction.text import TfidfVectorizer



tvectorizer = TfidfVectorizer(input='content', analyzer = 'word', lowercase=True, stop_words='english',

                                  tokenizer=preprocess, ngram_range=(1,3), min_df=40, max_df=0.20,

                                  norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True)



dtm = tvectorizer.fit_transform(fcc_df_super['Description']).toarray()
top_dist, lda_keys= get_doc_topic_dist(model, corpus, True)

features = tvectorizer.get_feature_names()
top_words = []

for n in range(len(dtm)):

    inds = np.int0(np.argsort(dtm[n])[::-1][:4])

    top_words += [', '.join([features[i] for i in inds])]

    

fcc_df_super['Description Top Words'] = pd.DataFrame(top_words)

fcc_df_super['Topic'] = pd.DataFrame(lda_keys)

# Fill missing values with dummy

fcc_df_super['Topic'].fillna(-1, inplace=True)

fcc_df_super.head()
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2)

X_tsne = tsne.fit_transform(top_dist)



fcc_df_super['Description Truncated'] = fcc_df_super['Description'].apply(lambda x: x[0:140])

fcc_df_super['X_tsne'] = X_tsne[:,0]

fcc_df_super['Y_tsne'] = X_tsne[:,1]



fcc_df_super['Colors'] = fcc_df_super['Topic'].apply(lambda topic_num: peacock_theme[topic_num])



source = ColumnDataSource(dict(

    x=fcc_df_super['X_tsne'],

    y=fcc_df_super['Y_tsne'],

    color=fcc_df_super['Colors'],

    label=fcc_df_super['Topic'].apply(lambda t: top_labels[t]),

    old_topic=fcc_df_super['Customer Complaint'],

    top_words=fcc_df_super['Description Top Words'],

    description=fcc_df_super['Description Truncated']))
title = 'T-SNE Visualization of Topics'

plot_tsne = figure(plot_width=1000, plot_height=600, title=title,

                   tools=['pan', 'wheel_zoom', 'save', 'hover'], tooltips=[("Old Topic","@old_topic"),

                                                                           ("Description","@description"),

                                                                           ("Top Words","@top_words")])



plot_tsne.scatter(x='x', y='y', legend_field='label', source=source, color='color', alpha=0.6, size=5.0)

plot_tsne.legend.location = "top_right"



show(plot_tsne)