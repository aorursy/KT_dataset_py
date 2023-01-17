import numpy as np
import pandas as pd
import nltk
from nltk import word_tokenize, sent_tokenize, pos_tag, pos_tag_sents
import re
import gensim
import gensim.corpora as corpora
import gensim.models.ldamodel as ldamodel
from gensim.summarization import summarize
import pyLDAvis.gensim
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.probability import FreqDist
from IPython.utils.text import columnize
import warnings
warnings.filterwarnings("ignore")
import textwrap

%matplotlib notebook
plt.style.use('seaborn-darkgrid')
# Loading the UN General Debates Dataset
df = pd.read_csv('../input/un-general-debates/un-general-debates.csv'
                ).drop('session', axis=1)
# The dataset represents country names as 3-letter ISO-alpha3 Codes.
# To convert these codes into country names, I merged df with the 
# UN country names dataset, which I obtained from:
# https://unstats.un.org/unsd/methodology/m49/overview/ 
# This dataset also specifies the region (continent), which I will need later.
country_names = pd.read_excel('../input/un-country-names/UNSD  Methodology.xlsx')
df = pd.merge(df, country_names[['Region Name','Country or Area','ISO-alpha3 Code']],
             how='left', left_on='country', right_on='ISO-alpha3 Code')
df.drop('ISO-alpha3 Code',axis=1, inplace=True) #removing a duplicate column
df.head()
# For convenience (processing speed), I will reduce the size of the dataset by:
# 1) Looking at debates that happened this century (years 2001-2015)
df = df.loc[df.year > 2000]
# 2) Limiting the countries of interest to Asian countries:
df = df.loc[df['Region Name'] == 'Asia']
# 3) Removing countries absent from any debates this century:
always_present = [index for (index,value) in 
                      df.country.value_counts().items() 
                  if value==15] # for 15 debates in the 21st century
df = df.loc[df['country'].isin(always_present)]

# The reduced dataframe (number of unique values per column):
df.nunique()
# Preparing texts for analysis:
def preprocess_text(text):
    text = re.sub('[^a-z]+',' ', text.lower()) # remove all non-letter characters
    tokens = word_tokenize(text) # returns a list of individual words
    # Removing unhelpfull, ubiquitous words ('stop words', e.g. ‘the’, ‘is’, ‘are’):
    tokens = [token for token in tokens if len(token) > 4 and
             token not in nltk.corpus.stopwords.words('english')]
    # Lemmatizing removes inflectional endings and gets the root word (lemma):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmas

# processed text as list of lemmas:
df['lemmatized_tokens'] = df['text'].apply(preprocess_text)
# processed text in continous form:
df['lemmatized_text'] = [' '.join(x) for x in df['lemmatized_tokens']]
# calculating the number of times each words appears in a speech:
df['freq_dist'] = df['lemmatized_tokens'].apply(FreqDist)
df[['text','lemmatized_tokens','lemmatized_text']].head()
# I will need a picture to prettify my figure.
# This picture is a black silhouette on a white background.
# I have it saved as a table, which I will load into a numpy array:
peace_dove = np.genfromtxt('../input/peace-dove-picture/peace_dove.csv',
                           delimiter=',', dtype='uint8')

# Mearging all speeches into a single text:
text = df.lemmatized_text.sum()

# Plotting:
fig = plt.figure(figsize = (10,10))
# I used Word Cloud to represents the frequency of words with font size.
# These words can be printed within a shape, in this provided by my picture
wordcloud = WordCloud(background_color='white',
                      mask=peace_dove,
                      max_words=2000).generate(text);
plt.imshow(wordcloud, interpolation='bilinear') # plotting the image
plt.axis("off")
# tight_layout automatically adjusts subplot params
# so that the subplot(s) fits in to the figure area:
plt.tight_layout()
# For each word count in how many documents does it appear:

texts = df.lemmatized_tokens.tolist() # I will be working on lemmatized words
all_words = list(set(df.lemmatized_tokens.sum())) # All unique words

texts = [list(set(t)) for t in texts] # Unique words in each text
word_in_texts = {} # keys=unique words, values=the number of texts in which they appear
for word in all_words:
    word_in_texts[word] = 0 
    for text in texts:
        if word in text:
            word_in_texts[word] += 1

# This is a slow implementation. Any suggestions on how to speed it up?
no_of_speeches = len(texts)

fig, ax = plt.subplots(nrows=2, ncols=1) # figure with two subplots

# converting the dicrionary to a list of tupples and sorting them from most common,
# to rarest words
wit = sorted(list(word_in_texts.items()), key=lambda x: x[1], reverse=True)
# for plotting: numbering words from most to liest common:
x_wit = [x for x in range(len(wit))]
y_wit = [y for (x,y) in wit]
ax[0].plot(x_wit, y_wit, color='slateblue')
ax[0].fill_between(x_wit, y_wit, color="slateblue", alpha=0.3)
ax[0].set_title('All words')
ax[0].set_ylabel('Speeches')

# very few words appear in most documents
# on the other hand out of >14,000 unique words
# about 10,000 occur in no more than 10 documents

wit_10 = [x for x in wit if x[1]>10] # remove words appearing in 10 or less documents
x_wit_10 = [x for x in range(len(wit_10))]
y_wit_10 = [y for (x,y) in wit_10]
ax[1].plot(x_wit_10, y_wit_10, color='slateblue')
ax[1].fill_between(x_wit_10, y_wit_10, color="slateblue", alpha=0.3)
ax[1].set_title('Words that appear in over 10 speeches')
ax[1].set_ylabel('Speeches')
plt.tight_layout()
# By removing words that appear in at most 10 speeches, I end up with a much more
# managable number of >4,000 words.
# Let's display every 25th of those >4,000 words to get a feeling of
# how word's characted changes with it's commonness.
# Subjectively, I find words that occur in >200 speeches (i.e. over 30% of speeches)
# too general to be useful, and will remove them from further analysis.

print(columnize([str(x) for x in wit_10[0::25]])) # for a prettier display,
# I put the list into multiple columns
# LDA model

def find_topics(no_above, no_below, num_topics, data_frame):
    texts = data_frame['lemmatized_tokens'].tolist()
    id2word = corpora.Dictionary(texts)
    # ignore words with document frequency > than the given threshold
    # keep tokens which are contained in at least no_below documents
    id2word.filter_extremes(no_above=no_above, no_below=no_below)
    # Gensim creates a unique id for each word.
    # The produced corpus is a mapping of (word_id, word_frequency).
    corpus = [id2word.doc2bow(text) for text in texts]
    
    # Estimating LDA model parameters on the corpus.
    lda = ldamodel.LdaModel(corpus, num_topics=num_topics,
                        passes=50, random_state=0,
                        id2word=id2word)
    return lda, corpus, id2word
# The results are very sensitive to the number o topics, with some unintuitional results.
# 14 topics may e.g. give more similar results to 11 topics, than to 13.
# The results are also hugely sensitive to no_above. Including ubiquitous words resulted in
# pretty much just one topic, simmilar to the peace dove picture, repeted in different
# variations. As I played with the parameters, I captured a lot of interesting topics,
# but unfortunatelly never all of them at once.

num_topics = 12
lda_30, corpus_30, id2word_30 = find_topics(0.3, 10, num_topics, df)
for topic in lda_30.show_topics():
    print('topic', topic[0])
    print(textwrap.fill(topic[1], 75))
# pyLDAvis is a neat, interactive way to visualize an LDA model

# Among other things, it shows the overlap between topic, hence may help to pick 
# the number of topics that we want to find. 
# LDA is often used to clasify new documents based of topic,
# thus overlaps are better avoided, e.g. by limiting the number of topics.
# In this case however, I'm more interested in finding what is being discussed,
# than how to classify the speeches, hence I can accept some overlap if I that's
# the price for also finding more original topics.

# pyLDAvis also shows which words contribute to which topic and by how much
# Note that the numbers assigned to topics here, do not correspond to those
# from the figure above, or other plots to follow!

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_30, corpus_30, id2word_30)
vis
# I wished to know which topics does each of our speeches contain.
# Here, I create a dataframe with the list of topics contributing to each speech in
# one column ('topics'), and the main topic in another column ('main topic')

def text_topics(ldamodel, corpus):
    topics_df = df.drop(['country','text','Region Name'], axis=1, inplace=False)
    topics_df['topics'] = [sorted(ldamodel.get_document_topics(corpus[i]), 
                               key=lambda x: x[1], reverse=True) 
                        for i in range(df.shape[0])]
    topics_df['Main topic'] = [x[0][0] for x in topics_df.topics]   
    return topics_df
# Did topic distribution change over years?
# Did any topics become more/less popular?
# For each topic I counted in how many speeches was it the main one
# in each year.
# Although the counts fluctuate from year to year, and topic 9 clearly
# peaks in 2008, non of the topics suddenly appears or dissapears
# within the time frame. I wonder whether this is an artifact of the
# method or representation of facts?

def plot_by_year(ldamodel, corpus):
    topics_df = text_topics(ldamodel, corpus)
    
    # Listing words contribution to each topic for the
    # purpose of graph labelling:
    topics = ldamodel.show_topics(formatted=False, num_topics=num_topics)
    topics_words = [(tp[0], [wd[0] for wd in tp[1]])[:] for tp in topics]
    
    # A dataframe with years as rows and each topic as a column.
    # The value in row x and column y is the number of speeches in year x
    # with main topic y.
    df1 = pd.crosstab(topics_df['year'], topics_df['Main topic'])
    df1.plot.bar(subplots=True, sharey=True, figsize=(9,12),
            use_index=True, legend=False, title=topics_words,
            color='slateblue', width=0.7)
    plt.tight_layout()

plot_by_year(lda_30, corpus_30)
# Do countries have their favourite topics?
# (The answer is very, very much yes)

# I couted the number of times country x gave speech with main topic y
# and ploted the results as a heatmap.
# Each country gave exacly 15 speeches in the selected time frame.
# Interestingly most countries give 13-15 speeches on the same topic.

def plot_by_country(ldamodel, corpus):
    topics_df = text_topics(ldamodel, corpus)
    # A dataframe with countries as rows and each topic as a column.
    # The value in row x and column y is the number of speeches given by country x
    # with main topic y.
    df2 = pd.crosstab(topics_df['Country or Area'], topics_df['Main topic'])
    df2.index.name = None
    f = plt.figure(figsize=(8, 10))
    ax = sns.heatmap(df2, cmap='bone', cbar=False, annot=True)
    plt.tight_layout()
    
plot_by_country(lda_30, corpus_30)
# While the above approach gives some interesting insights, there are
# things I would like to improve upon.
# Mostly, there is the matter of resolution. I would like to
# get to more detailed, specific issues.
# One of the approaches I tried was to split each speech into individual
# paragraphs and then treat each of those paragraphs as a separate text.
# I did it on the assumpiton that each paragraph is likely to have
# one predominant topic, which may be easier for the LDA to fish out.

def texts_to_paragraphs(text):
    text = text.lower().split('\n\n') # create a list of paragraphs
    text = [par for par in text if len(par) > 1] # removing empty paragraphs
    text = [re.sub('[^a-z]+',' ', par) for par in text] # remove all non-letter characters
    tokens = [word_tokenize(par) for par in text] # returns a list of individual words
    # Removing unhelpfull, ubiquitous words ('stop words', e.g. ‘the’, ‘is’, ‘are’):
    tokens = [[token for token in par if len(token) > 4 and
             token not in nltk.corpus.stopwords.words('english')] for par in tokens]
    # Lemmatizing removes inflectional endings and gets the root word (lemma):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmas = [[lemmatizer.lemmatize(token) for token in par] for par in tokens]
    return lemmas

all_paragraphs = texts_to_paragraphs(df.text.sum())

# Building an LDA model:
id2word_par = corpora.Dictionary(all_paragraphs)
id2word_par.filter_extremes(no_above=0.1, no_below=20) # Having more text, I can lower
# the no_above value
corpus_par = [id2word_par.doc2bow(par) for par in all_paragraphs]

lda_par = ldamodel.LdaModel(corpus_par, num_topics=12,
                    passes=50, random_state=0,
                    id2word=id2word_par)

for topic in lda_par.show_topics():
    print('topic', topic[0])
    print(textwrap.fill(topic[1], 75))
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_par, corpus_par, id2word_par)
vis
# Dividing texts into paragraphs identified some specific topics, e.g. grouping
# China with Yemeni affairs. Unfortunately, not being a specialist in global politics,
# I cannot judge how much sense such grouping makes. Still, a quick internet search suggests
# that we may be onto something.

# Before I move on to other methods, I would like to test one more approach for building
# LDA models.

# This time, instead of increasing the number of texts, I will apply the models to only
# a selected group of speeches at a time.
# Specifically, I will collect all speeches given by a single country and run the LDA model
# only on them. In this way, I am hoping to find some specific issues in which those countries
# are particularly interested, rather than the general issues the entire World talks about.
# I am showing a single example below.

df_country = df.loc[df['Country or Area'] == 'China']
num_topics = 3
# There are only 15 speeches in this dataframe so no no_below filtering and just a
# mild no_above filter:
lda_country, corpus_country, id2word_country = find_topics(0.8, 1, num_topics, df_country)
for topic in lda_country.show_topics():
    print('topic', topic[0])
    print(textwrap.fill(topic[1], 75))
# ... and using individual paragraphs:
country_paragraphs = texts_to_paragraphs(df_country.text.sum())

# Building an LDA model:
id2word_c = corpora.Dictionary(country_paragraphs)
id2word_c.filter_extremes(no_above=0.5, no_below=1) # Having more texts, I can lower
# the no_above value
corpus_c = [id2word_c.doc2bow(par) for par in country_paragraphs]

lda_c = ldamodel.LdaModel(corpus_c, num_topics=6,
                    passes=50, random_state=0,
                    id2word=id2word_c)

for topic in lda_c.show_topics():
    print('topic', topic[0])
    print(textwrap.fill(topic[1], 75))
# I will need to split each speech into sentences.
# For compatibility with my lists of words,
# I lowercased and lemmatized all words in sentences.
def to_sentences_lemmas(text):
    sentences = sent_tokenize(text.lower()) # returns a list of sentences
    # Lemmatize each word:
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmas = [[lemmatizer.lemmatize(word) for word in word_tokenize(sentence)]
              for sentence in sentences]
    return lemmas

# a list of sentences in which each sentence is itself a list:
df['sentences_lemmas'] = df.text.apply(to_sentences_lemmas)
# a list of sentences in a continous form:
df['sentences'] = df.text.str.replace('\n+',' ').apply(sent_tokenize)
df[['sentences_lemmas','sentences']].head()
# Let's see how long are the speeches (measured in number of sentences):
plt.violinplot([len(i) for i in df.sentences.tolist()], showmeans=True,
               showextrema=False);
# I think that shortening the text tenfold seems a good compromise between
# reducing the length and retaining information.
# This will give me mostly 5-10 sentence summaries.

def summarize(x):
    m = x['freq_dist'].most_common(1)[0][1] # get the highest count
    scores = [sum([x['freq_dist'][word]/m for word in sentence]
                 ) for sentence in x['sentences_lemmas']]
    indexes = [i for i in range(len(scores))]
    zipped = list(zip(scores,indexes))
    zipped = sorted(zipped, key=lambda x: x[0], reverse=True)
    # how many sentences to return?
    n = int(len(indexes)/10)
    # get the index of sentences with n highest scores:
    i = [zipped[item][1] for item in range(n)]
    return [x['sentences'][index] for index in i]

df['summary'] = df.apply(summarize, axis=1)
# Let's see how well it worked:
print(df.iloc[3]['year'], df.iloc[3]['Country or Area'])
print('\nOriginal lemmatized text:', str(len(df.iloc[3]['sentences'])), 'sentences')
for sentence in df.iloc[3]['sentences']:
    print(textwrap.fill(sentence, 75))
print('\nSummary:', str(len(df.iloc[3]['summary'])), 'sentences')
for sentence in df.iloc[3]['summary']:
    print(textwrap.fill(sentence, 75))
# The analyser returns a dictionary of scores, in the form:
# {'neg': 0.152, 'neu': 0.848, 'pos': 0.0, 'compound': -0.5267}
# 'compound' represents the general sentiment of each sentence.
# Each word in the lexicon is associated with a strength of sentiment,
# and this information is reflected in the 'compound' score.

def strongest_sentiments(text):
    analyser = SentimentIntensityAnalyzer()
    scores = [analyser.polarity_scores(sentence)['compound'] for sentence in text]
    indexes = [i for i in range(len(scores))]
    zipped = list(zip(scores,indexes))
    zipped_positive = sorted(zipped, key=lambda x: x[0], reverse=True)
    zipped_negative = sorted(zipped, key=lambda x: x[0], reverse=False)
    # how many sentences to return?
    n = int(len(indexes)/250)
    # get the index of sentences with n highest scores:
    most_positive = [zipped_positive[item][1] for item in range(n)]
    most_negative = [zipped_negative[item][1] for item in range(n)]
    return [text[index] for index in most_positive], [text[index] 
                                                      for index in most_negative]

all_sentences = df.sentences.sum()
positive , negative = strongest_sentiments(all_sentences)
# ten most positive sentences
for sentence in positive[:10]:
    print(textwrap.fill(sentence, 75), '\n')
# ten most negative sentences
for sentence in negative[:10]:
    print(textwrap.fill(sentence, 75), '\n')
# And a visual...
text_positive = ' '.join(positive)
text_negative = ' '.join(negative)

# Plotting:
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,8))
wordcloud_positive = WordCloud(background_color='white',
                      max_words=2000).generate(text_positive);
ax[0].imshow(wordcloud_positive, interpolation='bilinear')
ax[0].axis("off")

wordcloud_negative = WordCloud(background_color='black',
                      max_words=2000).generate(text_negative);
ax[1].imshow(wordcloud_negative, interpolation='bilinear')
ax[1].axis("off")

plt.tight_layout()
