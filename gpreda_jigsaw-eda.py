import gc

import os

import warnings

import operator

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

from wordcloud import WordCloud, STOPWORDS

import gensim

from gensim.utils import simple_preprocess

from gensim.parsing.preprocessing import STOPWORDS

from nltk.stem import WordNetLemmatizer, SnowballStemmer

from nltk.stem.porter import *

import nltk

from gensim import corpora, models

import pyLDAvis

import pyLDAvis.gensim

from keras.preprocessing.text import Tokenizer



pyLDAvis.enable_notebook()

np.random.seed(2018)

warnings.filterwarnings('ignore')
%%time

JIGSAW_PATH = "../input/jigsaw-unintended-bias-in-toxicity-classification/"

train = pd.read_csv(os.path.join(JIGSAW_PATH,'train.csv'), index_col='id')

test = pd.read_csv(os.path.join(JIGSAW_PATH,'test.csv'), index_col='id')
train.head()
test.head()
print("Train and test shape: {} {}".format(train.shape, test.shape))
plt.figure(figsize=(12,6))

plt.title("Distribution of target in the train set")

sns.distplot(train['target'],kde=True,hist=False, bins=120, label='target')

plt.legend(); plt.show()
def plot_features_distribution(features, title):

    plt.figure(figsize=(12,6))

    plt.title(title)

    for feature in features:

        sns.distplot(train.loc[~train[feature].isnull(),feature],kde=True,hist=False, bins=120, label=feature)

    plt.xlabel('')

    plt.legend()

    plt.show()
features = ['severe_toxicity', 'obscene','identity_attack','insult','threat']

plot_features_distribution(features, "Distribution of additional toxicity features in the train set")
features = ['asian', 'black', 'jewish', 'latino', 'other_race_or_ethnicity', 'white']

plot_features_distribution(features, "Distribution of race and ethnicity features values in the train set")
features = ['female', 'male', 'transgender', 'other_gender']

plot_features_distribution(features, "Distribution of gender features values in the train set")
features = ['bisexual', 'heterosexual', 'homosexual_gay_or_lesbian', 'other_sexual_orientation']

plot_features_distribution(features, "Distribution of sexual orientation features values in the train set")
features = ['atheist','buddhist',  'christian', 'hindu', 'muslim', 'other_religion']

plot_features_distribution(features, "Distribution of religion features values in the train set")
features = ['intellectual_or_learning_disability', 'other_disability', 'physical_disability', 'psychiatric_or_mental_illness']

plot_features_distribution(features, "Distribution of disability features values in the train set")
def plot_count(feature, title,size=1):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(train))

    g = sns.countplot(train[feature], order = train[feature].value_counts().index[:20], palette='Set3')

    g.set_title("Number and percentage of {}".format(title))

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()   
plot_count('rating','rating')
plot_count('funny','funny votes given',3)
plot_count('wow','wow votes given',3)
plot_count('sad','sad votes given',3)
plot_count('likes','likes given',3)
plot_count('disagree','disagree given',3)
features = ['sexual_explicit']

plot_features_distribution(features, "Distribution of sexual explicit values in the train set")
stopwords = set(STOPWORDS)



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='white',

        stopwords=stopwords,

        max_words=50,

        max_font_size=40, 

        scale=5,

        random_state=1

    ).generate(str(data))



    fig = plt.figure(1, figsize=(10,10))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()
show_wordcloud(train['comment_text'].sample(20000), title = 'Prevalent words in comments - train data')
show_wordcloud(train.loc[train['insult'] < 0.25]['comment_text'].sample(20000), 

               title = 'Prevalent comments with insult score < 0.25')
show_wordcloud(train.loc[train['insult'] > 0.75]['comment_text'].sample(20000), 

               title = 'Prevalent comments with insult score > 0.75')
show_wordcloud(train.loc[train['threat'] < 0.25]['comment_text'], 

               title = 'Prevalent words in comments with threat score < 0.25')
show_wordcloud(train.loc[train['threat'] > 0.75]['comment_text'], 

               title = 'Prevalent words in comments with threat score > 0.75')
show_wordcloud(train.loc[train['obscene']< 0.25]['comment_text'], 

               title = 'Prevalent words in comments with obscene score < 0.25')
show_wordcloud(train.loc[train['obscene'] > 0.75]['comment_text'], 

               title = 'Prevalent words in comments with obscene score > 0.75')
show_wordcloud(train.loc[train['target'] > 0.75]['comment_text'], 

               title = 'Prevalent words in comments with target score > 0.75')
show_wordcloud(train.loc[train['target'] < 0.25]['comment_text'], 

               title = 'Prevalent words in comments with target score < 0.25')
def preprocess(text):

    result = []

    for token in gensim.utils.simple_preprocess(text):

        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2:

            result.append(token)

    return result
comment_sample = train['comment_text'][:1].values[0]

print('Original comment: {}'.format(comment_sample))

print('Tokenized comment: {}'.format(preprocess(comment_sample)))
%%time

preprocessed_comments = train['comment_text'].sample(200000).map(preprocess)
preprocessed_comments.sample(3)
%%time

dictionary = gensim.corpora.Dictionary(preprocessed_comments)

dictionary.filter_extremes(no_below=10, no_above=0.5, keep_n=75000)
%%time

bow_corpus = [dictionary.doc2bow(doc) for doc in preprocessed_comments]

tfidf = models.TfidfModel(bow_corpus)

corpus_tfidf = tfidf[bow_corpus]
%%time

lda_model = gensim.models.LdaMulticore(corpus_tfidf, num_topics=20,

                                    id2word=dictionary, passes=2, workers=2)
topics = lda_model.print_topics(num_words=5)

for i, topic in enumerate(topics[:10]):

    print("Train topic {}: {}".format(i, topic))
bd5 = bow_corpus[5]

for i in range(len(bd5)):

    print("Word {} (\"{}\") appears {} time.".format(bd5[i][0], dictionary[bd5[i][0]],bd5[i][1]))
for index, score in sorted(lda_model[bd5], key=lambda tup: -1*tup[1]):

    print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 5)))
vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dictionary)
pyLDAvis.save_html(vis, "LDAVis_train.html")
#vis
def topic_sentences(ldamodel=lda_model, corpus=bow_corpus, \

                        texts=preprocessed_comments):

    # initialization

    sent_topics_df = pd.DataFrame()



    # get main topic in each comment

    for i, row in enumerate(ldamodel[corpus]):

        row = sorted(row, key=lambda x: (x[1]), reverse=True)

        # get the dominanttopic, % contribution and keywords for each document

        for j, (topic_num, prop_topic) in enumerate(row):

            if j == 0:  # => dominant topic

                wp = ldamodel.show_topic(topic_num)

                topic_keywords = ", ".join([word for word, prop in wp])

                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4),\

                                                                topic_keywords]), ignore_index=True)

            else:

                break

    text = pd.Series(texts)

    sent_topics_df = pd.concat([sent_topics_df, text], axis=1)

    return(sent_topics_df)
topic_sents_keywords = topic_sentences(ldamodel=lda_model, corpus=bow_corpus, \

                                                  texts=preprocessed_comments)

dominant_topic =topic_sents_keywords.reset_index()

dominant_topic.columns = ['Comment', 'Dominant Topic', 'Topic Percent Contribution', 'Keywords','Text']

dominant_topic.head(5)
%%time

preprocessed_comments = test['comment_text'].map(preprocess)

dictionary = gensim.corpora.Dictionary(preprocessed_comments)

dictionary.filter_extremes(no_below=10, no_above=0.5, keep_n=75000)

bow_corpus = [dictionary.doc2bow(doc) for doc in preprocessed_comments]

tfidf = models.TfidfModel(bow_corpus)

corpus_tfidf = tfidf[bow_corpus]
%%time

lda_model = gensim.models.LdaMulticore(corpus_tfidf, num_topics=20,

                                    id2word=dictionary, passes=2, workers=2)
topics = lda_model.print_topics(num_words=5)

for i, topic in enumerate(topics[:10]):

    print("Test topic {}: {}".format(i, topic))
vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dictionary)

pyLDAvis.save_html(vis, "LDAVis_test.html")
#vis
EMBED_SIZE = 300 # size of word vector; this should be set to 300 to match the embedding source

MAX_FEATURES = 100000 # how many unique words to use (i.e num rows in embedding vector)

MAXLEN = 220 # max length of comments text
def build_vocabulary(texts):

    """

    credits to: https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings 

    credits to: https://www.kaggle.com/anebzt/quora-preprocessing-model

    input: list of list of words

    output: dictionary of words and their count

    """

    sentences = texts.apply(lambda x: x.split()).values

    vocab = {}

    for sentence in tqdm_notebook(sentences):

        for word in sentence:

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1

    return vocab
# populate the vocabulary

df = pd.concat([train ,test], sort=False)

vocabulary = build_vocabulary(df['comment_text'])
# display the first 10 elements and their count

print({k: vocabulary[k] for k in list(vocabulary)[:10]})
def load_embeddings(file):

    """

    credits to: https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings 

    credits to: https://www.kaggle.com/anebzt/quora-preprocessing-model

    input: embeddings file

    output: embedding index

    """

    def get_coefs(word,*arr): 

        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))

    return embeddings_index
%%time

GLOVE_PATH = '../input/glove840b300dtxt/'

print("Extracting GloVe embedding started")

embed_glove = load_embeddings(os.path.join(GLOVE_PATH,'glove.840B.300d.txt'))

print("Embedding completed")
len(embed_glove)
def embedding_matrix(word_index, embeddings_index):

    '''

    credits to: https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings 

    credits to: https://www.kaggle.com/anebzt/quora-preprocessing-model

    input: word index, embedding index

    output: embedding matrix

    '''

    all_embs = np.stack(embeddings_index.values())

    emb_mean, emb_std = all_embs.mean(), all_embs.std()

    EMBED_SIZE = all_embs.shape[1]

    nb_words = min(MAX_FEATURES, len(word_index))

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, EMBED_SIZE))

    for word, i in tqdm_notebook(word_index.items()):

        if i >= MAX_FEATURES:

            continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[i] = embedding_vector

    return embedding_matrix
def check_coverage(vocab, embeddings_index):

    '''

    credits to: https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings 

    credits to: https://www.kaggle.com/anebzt/quora-preprocessing-model

    input: vocabulary, embedding index

    output: list of unknown words; also prints the vocabulary coverage of embeddings and the % of comments text covered by the embeddings

    '''

    known_words = {}

    unknown_words = {}

    nb_known_words = 0

    nb_unknown_words = 0

    for word in tqdm_notebook(vocab.keys()):

        try:

            known_words[word] = embeddings_index[word]

            nb_known_words += vocab[word]

        except:

            unknown_words[word] = vocab[word]

            nb_unknown_words += vocab[word]

            pass

    print('Found embeddings for {:.3%} of vocabulary'.format(len(known_words)/len(vocab)))

    print('Found embeddings for {:.3%} of all text'.format(nb_known_words/(nb_known_words + nb_unknown_words)))

    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]

    return unknown_words
print("Verify the intial vocabulary coverage")

oov_glove = check_coverage(vocabulary, embed_glove)
oov_glove[:10]
def add_lower(embedding, vocab):

    '''

    credits to: https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings 

    credits to: https://www.kaggle.com/anebzt/quora-preprocessing-model

    input: vocabulary, embedding matrix

    output: modify the embeddings to include the lower case from vocabulary

    '''

    count = 0

    for word in tqdm_notebook(vocab):

        if word in embedding and word.lower() not in embedding:  

            embedding[word.lower()] = embedding[word]

            count += 1

    print(f"Added {count} words to embedding")
train['comment_text'] = train['comment_text'].apply(lambda x: x.lower())

test['comment_text'] = test['comment_text'].apply(lambda x: x.lower())
print("Check coverage for vocabulary with lower case")

oov_glove = check_coverage(vocabulary, embed_glove)

add_lower(embed_glove, vocabulary) # operates on the same vocabulary

oov_glove = check_coverage(vocabulary, embed_glove)
oov_glove[:10]
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}

len(contraction_mapping)
def known_contractions(embed):

    '''

    credits to: https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings 

    credits to: https://www.kaggle.com/anebzt/quora-preprocessing-model

    input: embedding matrix

    output: known contractions (from embeddings)

    '''

    known = []

    for contract in tqdm_notebook(contraction_mapping):

        if contract in embed:

            known.append(contract)

    return known
print("Known contractions in GloVe embeddings:")

print(known_contractions(embed_glove))
def clean_contractions(text, mapping):

    '''

    credits to: https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings 

    credits to: https://www.kaggle.com/anebzt/quora-preprocessing-model

    input: current text, contraction mappings

    output: modify the comments to use the base form from contraction mapping

    '''

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        text = text.replace(s, "'")

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    return text
train['comment_text'] = train['comment_text'].apply(lambda x: clean_contractions(x, contraction_mapping))

test['comment_text'] = test['comment_text'].apply(lambda x: clean_contractions(x, contraction_mapping))
df = pd.concat([train ,test], sort=False)

vocab = build_vocabulary(df['comment_text'])

print("Check embeddings after applying contraction mapping")

oov_glove = check_coverage(vocab, embed_glove)
oov_glove[:10]
punct_mapping = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

punct_mapping += '©^®` <→°€™› ♥←×§″′Â█½à…“★”–●â►−¢²¬░¶↑±¿▾═¦║―¥▓—‹─▒：¼⊕▼▪†■’▀¨▄♫☆é¯♦¤▲è¸¾Ã⋅‘∞∙）↓、│（»，♪╩╚³・╦╣╔╗▬❤ïØ¹≤‡√'



def unknown_punct(embed, punct):

    '''

    credits to: https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings 

    credits to: https://www.kaggle.com/anebzt/quora-preprocessing-model

    input: current text, contraction mappings

    output: unknown punctuation

    '''

    unknown = ''

    for p in punct:

        if p not in embed:

            unknown += p

            unknown += ' '

    return unknown
print("Find unknown punctuation:")

print(unknown_punct(embed_glove, punct_mapping))
puncts = {"‘": "'", "´": "'", "°": "", "€": "e", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', '…': ' '}



def clean_special_chars(text, punct, mapping):

    '''

    credits to: https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings 

    credits to: https://www.kaggle.com/anebzt/quora-preprocessing-model

    input: current text, punctuations, punctuation mapping

    output: cleaned text

    '''

    for p in mapping:

        text = text.replace(p, mapping[p])

    for p in punct:

        text = text.replace(p, f' {p} ') 

    return text
train['comment_text'] = train['comment_text'].apply(lambda x: clean_special_chars(x, punct_mapping, puncts))

test['comment_text'] = test['comment_text'].apply(lambda x: clean_special_chars(x, punct_mapping, puncts))
%%time

df = pd.concat([train ,test], sort=False)

vocab = build_vocabulary(df['comment_text'])

print("Check coverage after punctuation replacement")

oov_glove = check_coverage(vocab, embed_glove)
oov_glove[:10]
%%time

tokenizer = Tokenizer(num_words=MAX_FEATURES)

tokenizer.fit_on_texts(list(train))

train = tokenizer.texts_to_sequences(train)

test = tokenizer.texts_to_sequences(test)