import pickle

import re

import string

from ast import literal_eval

from collections import Counter



import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import spacy

from PIL import Image

from sklearn.feature_extraction.text import CountVectorizer

from wordcloud import WordCloud



%matplotlib inline
# load CSV and construct DataFrame

df = pd.read_csv('../input/ted-ultimate-dataset/2020-05-01/ted_talks_en.csv')



print(f'Shape: {df.shape}')
df = df.loc[:, ['talk_id', 'topics', 'transcript']]

df.head()
def find_topic(topic):

    """Returns a list of booleans for talks that contain a topic by index.

    

    :param topic: Topics or related topics of a talk

    """

    has_topic = []

    for t_list in df['topics']:

        if topic.lower() in literal_eval(t_list):

            has_topic.append(1)

        else:

            has_topic.append(0)

    return has_topic
# add columns for selected topics

df['is_sex'] = find_topic('sex')

df['is_religion'] = find_topic('religion')

df['is_politics'] = find_topic('politics')

df.head()
# filter DataFrame to only include talks about sex, religion, and politics

df = df.loc[(df['is_sex']==1) | (df['is_religion']==1) | 

            (df['is_politics']==1), : ].reset_index(drop=True)



# create new DataFrames for each topic (for later use)

sex_df = df.loc[(df['is_sex']==1), 'talk_id':'transcript'].reset_index(drop=True)

religion_df = df.loc[(df['is_religion']==1), 'talk_id':'transcript'].reset_index(drop=True)

politics_df = df.loc[(df['is_politics']==1), 'talk_id':'transcript'].reset_index(drop=True)



print('Sex', sex_df.shape)

print('Religion', religion_df.shape)

print('Politics', politics_df.shape)
def combine_transcripts(transcript_list):

    """Input a list of transcripts and return them as a corpus.

    :param list_of_text: Transcript list"""

    corpus = ' '.join(transcript_list)

    return corpus
def transcripts_to_dict(df, topic_list):

    """Returns a dictionary of transcripts for each topic.

    

    :param df: DataFrame

    :param topic_list: List of topics

    """

    ted_dict = {}



    for topic in topic_list:

        # filter DataFrame to specific series and convert it to a list

        filter_string = 'is_' + str(topic)

        text_list = df.loc[(df[filter_string]==1), 'transcript'].to_list()



        # call combine_transcripts function to return combined text

        combined_text = combine_transcripts(text_list)



        # add combined text to dict

        ted_dict[topic] = combined_text

    return ted_dict
# create dictionary from the DataFrame

transcript_dict = transcripts_to_dict(df, ['sex', 'religion', 'politics'])



# construct DataFrame from dictionary

df = pd.DataFrame.from_dict(transcript_dict, orient='index')

df.rename({0: 'transcript'}, axis=1, inplace=True)
df.head()
def clean_text(text):

    """Returns clean text.

    Removes:

        *text in square brackets & parenthesis

        *punctuation

        *words containing numbers

        *double-quotes, dashes

    """

#     text = text.lower()

    text = re.sub('[\[\(].*?[\)\]]', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\w*\d\w*', '', text)

    text = re.sub('[\“\–]', '', text)

    return text
# clean text

df['transcript'] = pd.DataFrame(df['transcript'].apply(lambda x: clean_text(x)))

sex_df['transcript'] = pd.DataFrame(sex_df['transcript'].apply(lambda x: clean_text(x)))

religion_df['transcript'] = pd.DataFrame(religion_df['transcript'].apply(lambda x: clean_text(x)))

politics_df['transcript'] = pd.DataFrame(politics_df['transcript'].apply(lambda x: clean_text(x)))
# create 'data' directory

!mkdir data



# pickle DataFrame (checkpoint)

df.to_pickle('data/sex_religion_politics_corpus.pkl')

sex_df.to_pickle('data/sex_corpus.pkl')

religion_df.to_pickle('data/religion_corpus.pkl')

politics_df.to_pickle('data/politics_corpus.pkl')
# load SpaCy English model

nlp = spacy.load('en_core_web_sm')



# transform DataFrames to list of docs (corpus)

all_srp_texts = df.loc[:, 'transcript'].to_list()

sex_texts = sex_df.loc[:, 'transcript'].to_list()

religion_texts = religion_df.loc[:, 'transcript'].to_list()

politics_texts = politics_df.loc[:, 'transcript'].to_list()



# process each corpus

all_srp_docs = list(nlp.pipe(all_srp_texts))

sex_docs = list(nlp.pipe(sex_texts))

religion_docs = list(nlp.pipe(religion_texts))

politics_docs = list(nlp.pipe(politics_texts))
# print first 1000 characters

sex_docs[0].text[:1000]
def get_all_text(spaCy_doc, topics):

    """Returns a dictionary of lemmatized text.

    Keeps alphanumeric characters and non stop words.



    :param spaCy_doc: spaCy Doc object (corpus) and topic list

    """

    my_dict = {}

    for ix, doc in enumerate(spaCy_doc):

        tag = topics[ix]

        token_list = []

        for token in doc:

            if (token.is_alpha==1) & (token.is_stop==0):

                token_list.append((token.lemma_).lower())

        my_dict[tag] = ' '.join(token_list)

    return my_dict
# create dictionary of transcripts with all text

all_srp_text_dict = get_all_text(all_srp_docs, ['sex', 'religion', 'politics'])
# construct DataFrame

all_srp_text_df = pd.DataFrame.from_dict(all_srp_text_dict, orient='index')

all_srp_text_df.rename({0: 'transcript'}, axis=1, inplace=True)

all_srp_text_df.head()
# pickle DataFrame for later use

all_srp_text_df.to_pickle('data/all_srp_text_df.pkl')
def get_nouns_adj(spaCy_doc, topic_list):

    """

    Input a spaCy Doc object (corpus) and topic list.

    Return a dictionary of lemmatized nouns and adjectives per doc.

    Keep alphanumeric characters and non stop words.

    """

    topics = topic_list

    my_dict = {}

    for ix, doc in enumerate(spaCy_doc):

        topic = topics[ix]

        token_list = []

        for token in doc:

            if (token.is_alpha==1) & (token.is_stop==0) & (token.pos_ in ['NOUN', 'ADJ']):

                token_list.append((token.lemma_).lower())

        my_dict[topic] = ' '.join(token_list)

    return my_dict
# create dictionary of transcripts with nouns and adjectives

all_srp_nouns_adj_dict = get_nouns_adj(all_srp_docs, ['sex', 'religion', 'politics'])
# construct DataFrame

all_srp_nouns_adj_df = pd.DataFrame.from_dict(all_srp_nouns_adj_dict, orient='index')

all_srp_nouns_adj_df.rename({0: 'transcript'}, axis=1, inplace=True)
all_srp_nouns_adj_df
# pickle DataFrame for later use

all_srp_nouns_adj_df.to_pickle('data/all_srp_nouns_adj_df.pkl')
def get_stop_words(spaCy_doc_obj):

    """Returns a list of stop words from doc object.

    

    :param spaCy_doc_obj: spacy.tokens.doc.Doc object

    """

    stop_words = []

    for doc in spaCy_doc_obj:

        for token in doc:

            if token.is_stop:

                stop_words.append(token.text.lower())

    return set(stop_words)
# initial stop word list

stop_words_spacy = list(get_stop_words(all_srp_docs))



# pickle for later use

with open('data/stop_words_spacy.pkl', 'wb') as f:

    pickle.dump(stop_words_spacy, f)

    f.close()
def create_document_term_matrix(df):

    """Input a DataFrame and return a document-term matrix with initial stop words"""

    cv = CountVectorizer(stop_words=stop_words_spacy)

    data_cv = cv.fit_transform(df['transcript'])

    dtm_df = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())

    dtm_df.index = df.index

    return dtm_df
# create document-term matrices for corpora

all_srp_all_text_dtm = create_document_term_matrix(all_srp_text_df)

all_srp_nouns_adj_dtm = create_document_term_matrix(all_srp_nouns_adj_df)
# pickle document-term matrices (for later use)

all_srp_all_text_dtm.to_pickle('data/all_srp_all_text_dtm.pkl')

all_srp_nouns_adj_dtm.to_pickle('data/all_srp_nouns_adj_dtm.pkl')
# load pickled document-term matrix

all_srp_all_text_dtm = pd.read_pickle('data/all_srp_all_text_dtm.pkl')



# transpose document-term matrix

all_srp_all_text_dtm_transposed = all_srp_all_text_dtm.transpose()
# find the top words said by each topic

n_words = 10

top_dict = {}

for topic in all_srp_all_text_dtm_transposed.columns:

    top = all_srp_all_text_dtm_transposed[topic].sort_values(ascending=False).head(n_words)

    top_dict[topic]= list(zip(top.index, top.values))

    

# print the top words said by each topic

for topic, top_words in top_dict.items():

    print(topic)

    print(', '.join([word for word, count in top_words[0:n_words]]))

    print('---')
# look at the most common top words --> add them to the stop word list



# let's first pull out the top words for each topic

words = []

for topic in all_srp_all_text_dtm_transposed.columns:

    top = [word for (word, count) in top_dict[topic]]

    for t in top:

        words.append(t)
# let's aggregate this list and identify the most common words

print(Counter(words).most_common())
# if all three topics have the top word, exclude it

add_stop_words = [word for word, count in Counter(words).most_common() if count >= 3]

add_stop_words
# add custom stop words

custom_stop_words = [

#     'sex',

    'world',

#     'religion',

#     'god',

#     'religious',

#     'political',

#     'not',

    'know',

    'thing',

    'know',

    'think',

    'come',

#     'people'

]
# load initial stop words

with open('data/stop_words_spacy.pkl', 'rb') as f:

    initial_stop_words = list(pickle.load(f))

    f.close()
def update_stop_words(list_to_update, add_stop_words, custom_stop_words):

    """Add custom stop words to stop word list"""

    stop_words = list_to_update

    for word in add_stop_words:

        stop_words.append(word)

    for word in custom_stop_words:

        stop_words.append(word)

    return stop_words
# add new stop words

stop_words_curated = update_stop_words(initial_stop_words, add_stop_words, custom_stop_words)
# recreate document-term matrix

cv = CountVectorizer(stop_words=stop_words_curated)



# load DataFrame

all_srp_text_df = pd.read_pickle('data/all_srp_text_df.pkl')



# count vectorize the DataFrame

data_cv = cv.fit_transform(all_srp_text_df['transcript'])



# construct new DataFrame

all_srp_text_dtm_stop = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())

all_srp_text_dtm_stop.index = all_srp_text_df.index
word_cloud = WordCloud(stopwords=stop_words_curated, background_color='white', colormap='Dark2',

                       max_font_size=150, random_state=2020, max_words=50)



plt.rcParams['figure.figsize'] = [25, 10]



topics = ['sex', 'religion', 'politics']



# create subplots for each topic

for index, topic in enumerate(all_srp_all_text_dtm_transposed.columns):

    word_cloud.generate(all_srp_text_df.transcript[topic])

    

    plt.subplot(1, 3, index+1)

    plt.imshow(word_cloud, interpolation="bilinear")

    plt.axis("off")

#     plt.title(topics[index])

    

plt.show()
# load pickled document-term matrix

all_srp_nouns_adj_dtm = pd.read_pickle('data/all_srp_nouns_adj_dtm.pkl')



# transpose document-term matrix

all_srp_nouns_adj_dtm_transposed = all_srp_nouns_adj_dtm.transpose()
# find the top words said by each topic

n_words = 10

top_dict = {}

for topic in all_srp_nouns_adj_dtm_transposed.columns:

    top = all_srp_nouns_adj_dtm_transposed[topic].sort_values(ascending=False).head(n_words)

    top_dict[topic]= list(zip(top.index, top.values))

    

# print the top words said by each topic

for topic, top_words in top_dict.items():

    print(topic)

    print(', '.join([word for word, count in top_words[0:n_words]]))

    print('---')
# look at the most common top words --> add them to the stop word list



# let's first pull out the top words for each topic

words = []

for topic in all_srp_nouns_adj_dtm_transposed.columns:

    top = [word for (word, count) in top_dict[topic]]

    for t in top:

        words.append(t)
# let's aggregate this list and identify the most common words

print(Counter(words).most_common())
# if all three topics have the top word, exclude it

add_stop_words = [word for word, count in Counter(words).most_common() if count >= 3]

add_stop_words
# Add custom stop words

custom_stop_words = [

#     'sex',

    'world',

#     'religion',

#     'god',

#     'religious',

#     'political',

#     'not',

#     'know',

#     'thing',

#     'know',

#     'think',

#     'come',

#     'people'

]
# load initial stop words

with open('data/stop_words_spacy.pkl', 'rb') as f:

    initial_stop_words = list(pickle.load(f))

    f.close()
def update_stop_words(list_to_update, add_stop_words, custom_stop_words):

    """Add custom stop words to stop word list"""

    stop_words = list_to_update

    for word in add_stop_words:

        stop_words.append(word)

    for word in custom_stop_words:

        stop_words.append(word)

    return stop_words
# add new stop words

stop_words_curated = update_stop_words(initial_stop_words, add_stop_words, custom_stop_words)
# recreate document-term matrix

cv = CountVectorizer(stop_words=stop_words_curated)



# load DataFrame

all_srp_nouns_adj_df = pd.read_pickle('data/all_srp_nouns_adj_df.pkl')



# count vectorize the DataFrame

data_cv = cv.fit_transform(all_srp_nouns_adj_df['transcript'])



# construct new DataFrame

all_srp_nouns_adj_dtm_stop = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())

all_srp_nouns_adj_dtm_stop.index = all_srp_nouns_adj_df.index
word_cloud = WordCloud(stopwords=stop_words_curated, background_color='white', colormap='Dark2',

                       max_font_size=150, random_state=2020, max_words=50)



plt.rcParams['figure.figsize'] = [25, 10]



topics = ['sex', 'religion', 'politics']



# create subplots for each topic

for index, topic in enumerate(all_srp_nouns_adj_dtm_transposed.columns):

    word_cloud.generate(all_srp_nouns_adj_df.transcript[topic])

    

    plt.subplot(1, 3, index+1)

    plt.imshow(word_cloud, interpolation="bilinear")

    plt.axis("off")

#     plt.title(topics[index])

    

plt.show()
ted_mask = np.array(Image.open("../input/images/ted_mask.png"))
ted_df = all_srp_nouns_adj_df.copy()
all_transcripts = (ted_df.loc['sex', 'transcript']+ ted_df.loc['religion', 'transcript'] 

                   + ted_df.loc['politics', 'transcript'])
# create a word cloud image

wc = WordCloud(background_color="white", mask=ted_mask, max_words=500,

               stopwords=stop_words_curated, contour_width=3, 

               colormap='Dark2', contour_color='white')



# generate a wordcloud

wc.generate(all_transcripts)



# store to file

wc.to_file("ted_shaped_word_cloud.png")



# show

plt.figure(figsize=[20,10])

plt.imshow(wc, interpolation='bilinear')

plt.axis("off")

plt.show()