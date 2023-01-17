# Import needed python packages

import pandas as pd

import numpy as np

import gensim

import seaborn as sns

import textblob

from gensim.parsing.preprocessing import preprocess_documents, preprocess_string

from gensim.models.doc2vec import TaggedDocument

from gensim.models import ldamodel, LdaModel

from gensim import corpora, models

import nltk.data

from nltk.corpus import stopwords

import re

#import pyldavis

#import pattern

%matplotlib inline
# Making a function to identify non-text lines in the transcripts

def identify_nontext(text):

    '''

    Identifies non-text text column rows. 

    text = Text item

    returns:

      1: If text is non-text and is contained entirely in parentheses

      0: Text is text. 

    '''

    if text.startswith('(') and text.endswith(')'):

        return 1

    else:

        return 0
# Define a function to indicate the candidates vs speakers

def speaker_type(speaker):

    '''

    Returns a label for speaker type. Deliberate excluding candidates' crosstalk

    items since the transcript seems to try to pick out what they said anyway. 

    '''

    if (speaker == 'Trump') | (speaker == 'Clinton'):

        return 'Candidate'

    if (speaker == 'Kaine') | (speaker == 'Pence'):

        return 'VP Candidate'

    if (speaker == 'Holt') | (speaker == 'Quijano') | (speaker == 'Cooper') | (speaker == 'Raddatz') | (speaker == 'Wallace'):

        return 'Moderator'

    if (speaker == 'QUESTION') | (speaker == 'Audience'):

        return 'Audience'

    else:

        return 'Unlabeled'
# Define a function to indicate the candidates vs speakers

def general_speaker_type(speaker):

    '''

    Returns a label for speaker type. Deliberate excluding candidates' crosstalk

    items since the transcript seems to try to pick out what they said anyway. 

    Combines VP's with presidential candidates for percentage calculation

    '''

    if (speaker == 'Trump') | (speaker == 'Clinton') | (speaker == 'Kaine') | (speaker == 'Pence'):

        return 'Candidate'

    if (speaker == 'Holt') | (speaker == 'Quijano') | (speaker == 'Cooper') | (speaker == 'Raddatz') | (speaker == 'Wallace'):

        return 'Moderator'

    if (speaker == 'QUESTION') | (speaker == 'Audience'):

        return 'Audience'

    else:

        return 'Unlabeled'
# Define a function to indicate an utterance's polarity score based upon textblob

def utterance_polarity(utterance):

    '''

    Returns a textblob polarity score for text passed to this function

    '''

    # Turn the string into a text blob

    blob = textblob.TextBlob(utterance)

    

    # Return the polarity metric

    return blob.sentiment.polarity
# Define a function to indicate an utterance's subjectivity score based upon textblob

def utterance_subjectivity(utterance):

    '''

    Returns a textblob polarity score for text passed to this function

    '''

    # Turn the string into a text blob

    blob = textblob.TextBlob(utterance)

    

    # Return the polarity metric

    return blob.sentiment.subjectivity
# Convert text to lower-case and strip punctuation/symbols from words

# Borrowed from doc2vec tutorial: https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb

def normalize_text(text):

    norm_text = text.lower()



    # Replace breaks with spaces

    norm_text = norm_text.replace('<br />', ' ')



    # Pad punctuation with spaces on both sides

    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:

        norm_text = norm_text.replace(char, ' ' + char + ' ')



    return norm_text
# Making a function to identify non-text lines in the transcripts

def identify_nontext(text):

    '''

    Identifies non-text text column rows. 

    text = Text item

    returns:

      1: If text is non-text and is contained entirely in parentheses

      0: Text is text. 

    '''

    if text.startswith('(') and text.endswith(')'):

        return 1

    else:

        return 0
def utterance_to_wordlist(utterance, remove_stopwords=False ):

    '''

    Derived from the Kaggle Bag-of-Words-Meets-Bags-Of-Popcorn Tutorial: 

    https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-2-word-vectors

    Function to convert a document to a sequence of words,

    optionally removing stop words.  Returns a list of words.

    '''

    # 0. Remove non-letters

    review_text = re.sub("[^a-zA-Z0-9]"," ", utterance)

    #

    # 1. Convert words to lower case and split them

    words = review_text.lower().split()

    #

    # 4. Optionally remove stop words (false by default)

    if remove_stopwords:

        stops = set(stopwords.words("english"))

        words = [w for w in words if not w in stops]

    #

    # 5. Return a list of words

    return(words)
# Define a function to split a review into parsed sentences

def utterance_to_sentences( utterance, tokenizer, remove_stopwords=False ):

    '''

    Derived from the Kaggle Bag-of-Words-Meets-Bags-Of-Popcorn Tutorial: 

    https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-2-word-vectors

    Function to split a review into parsed sentences. Returns a 

    list of sentences, where each sentence is a list of words

    '''

    # 1. Use the NLTK tokenizer to split the paragraph into sentences

    raw_sentences = tokenizer.tokenize(utterance.strip())

    #

    # 2. Loop over each sentence

    sentences = []

    for raw_sentence in raw_sentences:

        # If a sentence is empty, skip it

        if len(raw_sentence) > 0:

            # Otherwise, call review_to_wordlist to get a list of words

            sentences.append( utterance_to_wordlist( raw_sentence, \

              remove_stopwords=True))

    #

    # Return the list of sentences (each sentence is a list of words,

    # so this returns a list of lists

    return sentences
# Define a function to get the list of topic weights for each 'document', which is each utterance in this case

def get_utterance_topic_weights(bagofwords, ldamodel):

    '''

    Use an lda model to get the list of topic weights for a bag-of-words object

    '''

    return ldamodel.get_document_topics(bagofwords)
# Define a function to get the different topic weights separated into their own columns

def get_individual_topic_weights(topic_weights, topic):

    '''

    Inputs:

        topic_weights: List of topics

        k: Number of topics

    '''

    # Make the list into a dict for easier lookup

    topic_dict = dict(topic_weights)

    

    # Return the value for the specified topic

    return topic_dict.get(topic)
# Import the dataset

df = pd.read_csv('../input/debate.csv',encoding = 'iso-8859-1')
# Insert needed categorical indicators, and also the polarity and subjectivity metrics from TextBlob

df.insert(df.shape[1], 'nontext_ind', df.Text.apply(identify_nontext))

df.insert(df.shape[1], 'speaker_type', df.Speaker.apply(speaker_type))

df.insert(df.shape[1], 'general_speaker_type', df.Speaker.apply(general_speaker_type))

df.insert(df.shape[1], 'polarity', df.Text.apply(utterance_polarity))

df.insert(df.shape[1], 'subjectivity', df.Text.apply(utterance_subjectivity))
# Check what the first lines of the dataframe are, to make sure it loaded

df.head(15)
# Check the shape of the dataframe--- also to assess whether loaded accurately--- number of rows x number of columns

df.shape
# Check how many rows there are per the different dates present in the 'Date' column

df.Date.value_counts().plot(kind='bar')
# Show candidates' utterances compared with one another

g = sns.factorplot("Speaker", col="Date", col_wrap=3, palette='Blues_r',

                  data = df[(df['Speaker'] == 'Trump') | (df['Speaker'] == 'Clinton')], 

                   kind="count")
# Show the moderators' overall utterances

g = sns.countplot(x='Date', data=df[df['speaker_type'] == 'Moderator'], palette='Reds_r')
# Show the VPs' overall utterances

g = sns.countplot(x='Speaker', data=df[df['speaker_type'] == 'VP Candidate'], palette='Blues_r')
# Compare the different speaker types over the different dates

g = sns.factorplot("general_speaker_type", col="Date", col_wrap=3, palette='Reds_r',

                  data = df[df['general_speaker_type'] != 'Unlabeled'], kind="count")
df2 = df[df['general_speaker_type'] != 'Unlabeled']
df2_ct = pd.crosstab(df2['Date'], df2['general_speaker_type'], margins=True, normalize='index')
df2_ct
# This graph shows the percentages--- the candidates spoke the most during the vice presidential debate,

# but had the least audience input. The moderators had their highest percentage of input on the 2nd

# presidential debate. 

df2_ct.plot.bar(stacked=True)
# Histograms of polarity

df[df['Speaker'] == 'Trump'].polarity.hist()
# Histograms of polarity

df[df['Speaker'] == 'Clinton'].polarity.hist()
# Histograms of polarity

df[df['Speaker'] == 'Pence'].polarity.hist()
# Histograms of polarity

df[df['Speaker'] == 'Kaine'].polarity.hist()
# Histograms of polarity ---- all moderators together

df[df['general_speaker_type'] == 'Moderator'].polarity.hist()
# Histograms of polarity ---- all moderators together

df[df['Speaker'] == 'Audience'].polarity.hist()
# Histograms of subjectivity ---- Trump

df[df['Speaker'] == 'Trump'].subjectivity.hist()
# Histograms of subjectivity ---- Clinton

df[df['Speaker'] == 'Clinton'].subjectivity.hist()
# Histograms of subjectivity ---- Pence

df[df['Speaker'] == 'Pence'].subjectivity.hist()
# Histograms of subjectivity ---- Kaine

df[df['Speaker'] == 'Kaine'].subjectivity.hist()
# Histograms of subjectivity ---- Moderators

df[df['general_speaker_type'] == 'Moderator'].subjectivity.hist()
# Histograms of subjectivity ---- Audience

df[df['Speaker'] == 'Audience'].subjectivity.hist()
df[df['Speaker'] == 'Audience'].subjectivity.value_counts()
df[df['Speaker'] == 'Audience'].polarity.value_counts()
# Lol, and the audience is the least subjective and polar group in the lot. They didn't get to 

# speak very much though. 
# How many utterances did each party end up making for all dates?

# In this case and at this early stage of analysis, I'm going to define 'utterance' as one line.

df.Speaker.value_counts()
df.Speaker.value_counts().plot(kind='bar')
# Ok, so without any other preprocessing, Trump has 224 and Clinton has 158. Percentage-wise,

# How many more utterances does he have?

# After running once, dividing comes out to 142%, so about 42% more than Clinton.

224/158
# So--- how do both presidential candidates rate for utterance count in both debates?

# This line of code excludes (!=) the 10-04 VP debate lines

df[df['Date'] != '2016-10-04'].Speaker.value_counts()
df[df['Date'] != '2016-10-04'].Speaker.value_counts().plot(kind='bar')
# How does this break down by the individual debate?

# Since Trump complained of problems with his microphone later, my expectation before

# running this code is that in the first debate, he would not have spoken as frequently as Clinton

# due to the microphone issues, and thus made up the utterance discrepancy in the 2nd debate

df[df['Date'] == '9/26/16'].Speaker.value_counts()
df[df['Date'] == '9/26/16'].Speaker.value_counts().plot(kind='bar')
df.Date.value_counts()
# Note--- the expectation did not pan out--- Trump actually talked the most of anyone in the first debate

# Now--- let's check out the utterance count in the 2nd debate

df[df['Date'] == '10/9/16'].Speaker.value_counts()
df[df['Date'] == '10/9/16'].Speaker.value_counts().plot(kind='bar')
# Utterance counts in the 2nd debate

df[df['Date'] == '10/19/2016'].Speaker.value_counts()
df[df['Date'] == '10/19/2016'].Speaker.value_counts().plot(kind='bar')
df.Speaker.value_counts()
'''

So, not only did Clinton make over 30% fewer utterances than Trump in all debates

considered together, she also typically made fewer utterances than the moderators. 

'''
# Let's see who had the most utterances in the VP debate

df[df['Date'] == '10/4/16'].Speaker.value_counts()
df[df['Date'] == '10/4/16'].Speaker.value_counts().plot(kind='bar')
# Both VP candidates had more utterances than the moderator. Interesting.

# In this case, Trump's VP candidate, Pence, also did most of the talking, but

# only had about 8% more utterances than Kaine did. 
# Just to illustrate what the polarity and subjectivity measures will and won't do, let's see how the

# infamous 'bad hombres' quote from Trump rated

df[df['Text'].str.contains('hombre')]
# However, looking over the dataframe, when some of the speech was spoken in a manner

# that was disjointed or heavily  repeated in audio recordings, the transcription 

# tends to consider these statements as two separate utterances... may need to go back

# and correct the utterance counts for this, but on first glance it doesn't look like

# this happened a lot. But first, the text itself, since the content is more value 

# than the basic utterance count. 
x
# Make another dataframe entirely out of just the text fields--- I may want to look

# at the non-text fields again, so I'll write these into a new variable.

df_text = df[df['nontext_ind'] == 0]
# Check the first 5 lines of this new one to make sure it looks right

df_text.head()
# Add another column to the dataframe in which the text is pre-processed by the

# gensim package's 'preprocess_string' function. 

# This converts everything to lowercase, removes non-informative 'stopwords' (words

# that are necessary for English language but don't lend any meaning for analysis),

# 'tokenizes' the sentences by splitting them out into individual words,

# and 'stems' the words--- i.e., takes the 'stem' of a word only and removing any

# endings... this allows things like 'grand' and 'grandly', which typically denote

# the same or very similar things via their root word to be counted as the same

# word in order to more clearly pick out relative topics without accidentally splitting

# useful information. 

df_text.insert(df_text.shape[1], 'PreprocessedText', df_text['Text'].apply(preprocess_string))
# Check the first 5 lines, or the 'head', to see if that worked as expected

df_text['PreprocessedText'].head()
# Use gensim to tag the text. Might re-do this later using some sort of 

# categrory label based on a sentiment analysis from text blob, but for now,

# I'm just going to label with the speakers, since the doc2vec tutorial

# Labeling with speakers may allow analysis of seeing if you can tell which statements

# were uttered by whom. 

# I'm following says the tagging is required---- note: starting with LDA Model instead

# Tutorial: https://linanqiu.github.io/2015/10/07/word2vec-sentiment/

df_text.insert(df_text.shape[1], 'TaggedText', df_text['PreprocessedText'].apply(TaggedDocument, args=(df_text['Speaker'],)))
# RANDOM CODE BLOCK! Started playing with textblob and got sidetracked :-)

# Maybe won't use this quite yet, but it's neat. 

# What this does is tags the part of speech. PRP = preposition, VB = verb, etc.

# I'm sure I will use this later on, but for now, I want to finish out the doc2vec stuff first. 

zz = textblob.taggers.NLTKTagger()

zzz = zz.tag(df_text['Text'][8])

zzz[0:5] # Equivalent to a 'head' statement--- this object is a list and therefore doesn't have the 

# Head function like a pandas dataframe does. 
# Sort the text

text = df_text['Text'].sort_values()
text.head()
sentences = []
import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# Convert the text to sentences and tokenize

print("Parsing sentences from unlabeled set")

for utterance in text:

    sentences += utterance_to_sentences(utterance, tokenizer)
len(sentences)
# Create lDA's vocab dictionary

dictionary = corpora.Dictionary(sentences)
# Make a bag-of-words corpus from the sentences object

corpus = [dictionary.doc2bow(text) for text in sentences]
# Make a model object - start with a default number of topics of 10

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=20)
# Show the top 10 words in each of the ten topics

dict(ldamodel.print_topics(num_topics=10, num_words=10))
# Interesting topics. I have requested pyLDAvis for the kaggle docker container and will complete the 

# visualization when it has been installed. 
# Ok... figure out how to get topic distributions for the 10 topics. 

df_text.insert(df_text.shape[1],'bow_column', df_text.PreprocessedText.apply(dictionary.doc2bow))
# Insert the list of document topics into the 'master' text dataframe

df_text.insert(df_text.shape[1], 'topic_weights', df_text['bow_column'].apply(get_utterance_topic_weights, args=(ldamodel,)))
# Loop through the topics, and make a new column for each topic

for topic in range(0, 10):

    col_name = 'topic_weight_' + str(topic)

    df_text.insert(df_text.shape[1], col_name, df_text['topic_weights'].apply(get_individual_topic_weights, args=(topic,)))
# Loop through the topics again, and this time, fillna

for topic in range(0, 10):

    col_name = 'topic_weight_' + str(topic)

    df_text[col_name].fillna(0.0, inplace=True)
# On this, to explore, I just went through all of the topic dists. Looks like a gamma dist for each, just eyeballing it.

df_text['topic_weight_0'].hist()
# Set up a groupby

df_text_heat = df_text.groupby('Speaker')
df_text_heat_med = df_text_heat.aggregate({'topic_weight_0':'median',

                                          'topic_weight_1':'median',

                                          'topic_weight_2':'median',

                                          'topic_weight_3':'median',

                                          'topic_weight_4':'median',

                                          'topic_weight_5':'median',

                                          'topic_weight_6':'median',

                                          'topic_weight_7':'median',

                                          'topic_weight_8':'median',

                                          'topic_weight_9':'median',

                                          'polarity':'median',

                                          'subjectivity':'median'})
df_heat_med = speak = df_text_heat_med.T
df_heat_med.sort_index(axis=0, inplace=True)
g = sns.heatmap(df_heat_med, annot=True, linewidths=.5, cmap='RdBu_r')
# So, how far IS Trump from Clinton on subjectivity --- not terribly different, since

# the difference is quite a bit less than one standard deviation. 

df_text.subjectivity.std()
# May play around with this a bit more

# To demonstrate though, I noticed that topic 7 might have a bi-modal distribution, 

# so let's see what that looks like relative to subjectivity



'''

Topic Six Words:

6: '0.030*taxes,

    0.029*trump,

    0.028*clinton,

    0.025*mr,

    0.023*tax,

    0.023*let,

    0.021*two,

    0.019*thank,

    0.019*want,

    0.017*secretary'

'''

sns.jointplot('topic_weight_7', 'subjectivity', data=df_text[df_text['topic_weight_7'] > 0.1], 

              kind="kde", color="Blue")
# Actually, maxes by speaker of the aggregates may be more useful

df_text_heat_max = df_text_heat.aggregate({'topic_weight_0':'max',

                                          'topic_weight_1':'max',

                                          'topic_weight_2':'max',

                                          'topic_weight_3':'max',

                                          'topic_weight_4':'max',

                                          'topic_weight_5':'max',

                                          'topic_weight_6':'max',

                                          'topic_weight_7':'max',

                                          'topic_weight_8':'max',

                                          'topic_weight_9':'max',

                                          'polarity':'max',

                                          'subjectivity':'max'})

df_heat_max = speak = df_text_heat_max.T

df_heat_max.sort_index(axis=0, inplace=True)
g = sns.heatmap(df_heat_max, annot=True, linewidths=.5, cmap='RdBu_r')
# Show the top 10 words in each of the ten topics

dict(ldamodel.print_topics(num_topics=10, num_words=10))
# Let's try mins --- the topics and subjectivity were a big bunch of 0's, so I'll just do polarity

df_text_heat_min = df_text_heat.aggregate({'polarity':'min'})

df_heat_min = df_text_heat_min.T

df_heat_min.sort_index(axis=0, inplace=True)
g = sns.heatmap(df_heat_min, annot=True, linewidths=.5, cmap='RdBu_r')
# On the minimum heatmap, the polarity line is pretty interesting. 

# The extremes kind of re-illustrate how the VP running-mates balance the candidates on polarity,

# but the candidates and VPs match better along party lines in terms of subjectivity extremes.

# I was kind of surprised that moderators are ranking as low as -.5 though... my

# assumption before doing any analysis was that they would be closer to neutral than

# they ended up being.

# On that bent too... on subjectivity, everyone ran the gamut from 0 to 1, except Cooper. He

# was the only one that didn't hit a full 1 on subjectivity, even including the questions. 