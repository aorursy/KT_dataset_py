import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import spacy

import itertools as it

from gensim.models import Phrases

from gensim.models.word2vec import LineSentence

from gensim.corpora import Dictionary, MmCorpus

from gensim.models.ldamulticore import LdaMulticore

import pyLDAvis

import pyLDAvis.gensim

import warnings

import _pickle as pickle

from gensim.models import Word2Vec

from sklearn.manifold import TSNE

from bokeh.plotting import figure, show, output_notebook

from bokeh.models import HoverTool, ColumnDataSource, value

import matplotlib.pyplot as plt

%matplotlib inline

output_notebook()



from subprocess import check_output
train = pd.read_csv('../input/train.csv')

train['author'].value_counts().plot.pie(autopct='%.2f', fontsize=20, figsize=(6, 6))

plt.title('Authorwise distribution')

None
train.isnull().any()  # sanity check for null values
nlp = spacy.load('en')
sample_sent = train.loc[1000, 'text']

print(sample_sent)
parsed_sent = nlp(sample_sent)

print(parsed_sent)
for num, sentence in enumerate(parsed_sent.sents):

    print('Sentence {}:'.format(num + 1))

    print(sentence)

    print()
for num, entity in enumerate(parsed_sent.ents):

    print('Entity {}:'.format(num + 1), entity, '-', entity.label_)

    print()
token_text = [token.orth_ for token in parsed_sent]

token_pos = [token.pos_ for token in parsed_sent]



pd.DataFrame({'token_text': token_text, 'part_of_speech': token_pos})
token_lemma = [token.lemma_ for token in parsed_sent]

token_shape = [token.shape_ for token in parsed_sent]



pd.DataFrame({'token_text': token_text, 'token_lemma': token_lemma, 'token_shape': token_shape})
token_entity_type = [token.ent_type_ for token in parsed_sent]

token_entity_iob = [token.ent_iob_ for token in parsed_sent]



pd.DataFrame({'token_text': token_text, 'token_entity_type': token_entity_type,

              'token_entity_iob': token_entity_iob})
token_attributes = [(token.orth_,

                     token.prob,

                     token.is_stop,

                     token.is_punct,

                     token.is_space,

                     token.like_num,

                     token.is_oov)

                    for token in parsed_sent]



df = pd.DataFrame(token_attributes,

                  columns=['text',

                           'log_probability',

                           'stop?',

                           'punctuation?',

                           'whitespace?',

                           'number?',

                           'out of vocab.?'])



df.loc[:, 'stop?':'out of vocab.?'] = (df.loc[:, 'stop?':'out of vocab.?']

                                       .applymap(lambda x: u'Yes' if x else u''))

                                               

df
def punct_space(token):

    """

    helper function to eliminate tokens

    that are pure punctuation or whitespace

    """

    

    return token.is_punct or token.is_space



def lemmatized_sentence(sent):

    """

    helper function to use spaCy to parse sentences,

    lemmatize the text

    """

    return u' '.join([token.lemma_ for token in nlp(sent)

                             if not punct_space(token)])
train['unigram_text'] = train['text'].map(lambda x: lemmatized_sentence(x))
print(train.loc[1000, 'unigram_text'])
bigram_model = Phrases(train.loc[:, 'unigram_text'])
train['bigram_text'] = train['unigram_text'].map(lambda x: u''.join(bigram_model[x]))
print(train.loc[1000, 'bigram_text'])
for idx, sent in train['bigram_text'].iteritems():

    bigram_dictionary = Dictionary([sent.split()])

    

bigram_dictionary.compactify()
bigram_bow = bigram_dictionary.doc2bow(train['bigram_text'])
lda = LdaMulticore(bigram_bow,

                   num_topics=3,

                   id2word=bigram_dictionary)
def explore_topic(topic_number, topn=25):

    """

    accept a user-supplied topic number and

    print out a formatted list of the top terms

    """

        

    print(u'{:20} {}'.format(u'term', u'frequency') + u'\n')



    for term, frequency in lda.show_topic(topic_number, topn=25):

        print(u'{:20} {:.3f}'.format(term, round(frequency, 3)))
explore_topic(2)
LDAvis_prepared = pyLDAvis.gensim.prepare(lda, bigram_bow, bigram_dictionary)

pyLDAvis.display(LDAvis_prepared)
topic_names={0: 'EAP',

             1: 'MWS',

             2: 'HPL'}

def lda_description(text, min_topic_freq=0.08):

    """

    accept the original text of a review and (1) parse it with spaCy,

    (2) apply text pre-proccessing steps, (3) create a bag-of-words

    representation, (4) create an LDA representation, and

    (5) print a sorted list of the top topics in the LDA representation

    """

    

    # parse the review text with spaCy

    parsed_sent = nlp(text)

    

    # lemmatize the text and remove punctuation and whitespace

    unigram_sent = [token.lemma_ for token in parsed_sent

                      if not punct_space(token)]

    

    # apply the first-order models

    bigram_sent = bigram_model[unigram_sent]

    

    # remove any remaining stopwords

    bigram_sent = [term for term in bigram_sent

                      if not term in spacy.en.English.Defaults.stop_words]

    

    # create a bag-of-words representation

    sent_bow = bigram_dictionary.doc2bow(bigram_sent)

    

    # create an LDA representation

    sent_lda = lda[sent_bow]

    

    # sort with the most highly related topics first

    sent_lda = sorted(sent_lda, key=lambda x: -x[1])

    

    for topic_number, freq in sent_lda:

        if freq < min_topic_freq:

            break

            

        # print the most highly related topic names and frequencies

        print('{:25} {}'.format(topic_names[topic_number],

                                round(freq, 3)))
print('Probabilities:')

print(lda_description(train.loc[5, 'text']))

print('Actual:', train.loc[5, 'author'])
print('Probabilities:')

print(lda_description(train.loc[1000, 'text']))

print('Actual:', train.loc[1000, 'author'])
train['vec_inp'] = train['bigram_text'].map(lambda x: x.split(' '))
import sys



word2vec = Word2Vec(train['vec_inp'], size=20, window=5,

                        min_count=5, sg=0)



# perform another 100 epochs of training

for i in range(1,200):

    sys.stderr.write('\rOn {}'.format(i))

    word2vec.train(train['vec_inp'], total_examples=word2vec.corpus_count, 

                   epochs=word2vec.iter)
print(u'{:,} terms in the word2vec vocabulary.'.format(len(word2vec.wv.vocab)))
# build a list of the terms, integer indices,

# and term counts from the food2vec model vocabulary

ordered_vocab = [(term, voc.index, voc.count)

                 for term, voc in word2vec.wv.vocab.items()]



# sort by the term counts, so the most common terms appear first

ordered_vocab = sorted(ordered_vocab, key=lambda x: -x[2])



# unzip the terms, integer indices, and counts into separate lists

ordered_terms, term_indices, term_counts = zip(*ordered_vocab)

# create a DataFrame with the word2vec vectors as data,

# and the terms as row labels

word_vectors = pd.DataFrame(word2vec.wv.syn0[:],

                            index=ordered_terms)



word_vectors
def get_related_terms(token, topn=10):

    """

    look up the topn most similar terms to token

    and print them as a formatted list

    """



    for word, similarity in word2vec.most_similar(positive=[token], topn=topn):



        print(u'{:20} {}'.format(word, round(similarity, 3)))
get_related_terms(u'owl')
get_related_terms(u'fear')
get_related_terms(u'blood')
get_related_terms(u'jealous')
def word_algebra(add=[], subtract=[], topn=1):

    """

    combine the vectors associated with the words provided

    in add= and subtract=, look up the topn most similar

    terms to the combined vector, and print the result(s)

    """

    answers = word2vec.most_similar(positive=add, negative=subtract, topn=topn)

    

    for term, similarity in answers:

        print(term)
word_algebra(add=[u'night', u'fear'])
word_algebra(add=[u'fear'], subtract=[u'night'])
word_algebra(add=[u'night'], subtract=[u'fear'])
tsne_input = word_vectors.drop(spacy.en.English.Defaults.stop_words, errors=u'ignore')
tsne = TSNE()

tsne_vectors = tsne.fit_transform(tsne_input.values)
tsne_vectors = pd.DataFrame(tsne_vectors,

                            index=pd.Index(tsne_input.index),

                            columns=[u'x_coord', u'y_coord'])

tsne_vectors.head()
tsne_vectors[u'word'] = tsne_vectors.index
# add our DataFrame as a ColumnDataSource for Bokeh

plot_data = ColumnDataSource(tsne_vectors)



# create the plot and configure the

# title, dimensions, and tools

tsne_plot = figure(title=u't-SNE Word Embeddings',

                   plot_width = 800,

                   plot_height = 800,

                   tools= (u'pan, wheel_zoom, box_zoom,'

                           u'box_select, resize, reset'),

                   active_scroll=u'wheel_zoom')



# add a hover tool to display words on roll-over

tsne_plot.add_tools( HoverTool(tooltips = u'@word') )



# draw the words as circles on the plot

tsne_plot.circle(u'x_coord', u'y_coord', source=plot_data,

                 color=u'blue', line_alpha=0.2, fill_alpha=0.1,

                 size=10, hover_line_color=u'black')



# configure visual elements of the plot

tsne_plot.title.text_font_size = value(u'16pt')

tsne_plot.xaxis.visible = False

tsne_plot.yaxis.visible = False

tsne_plot.grid.grid_line_color = None

tsne_plot.outline_line_color = None



# engage!

show(tsne_plot);