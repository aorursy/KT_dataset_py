# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from tqdm import tqdm

import os

print(os.listdir("../input"))

tqdm.pandas()



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/cvpr-papers-to-csv/cvpr2019.csv')
df.head()
# Oh I mistakingly save index as well. I can remove there but let's  delete it.

df.drop('Unnamed: 0', axis=1, inplace=True)
df.head()
df.describe()
df.info()
df['content'] = df['content'].apply(str)

df['abstract'] = df['abstract'].apply(str)

df['authors'] = df['authors'].apply(str)

df['title'] = df['title'].apply(str)
df.info()
# Now the first question is what is the distribution of number of authors.  Let's find out

sns.distplot(df['authors'].str.split(',').apply(len))
print('The mean of the distribution is', df['authors'].str.split(',').apply(len).mean(), 'and the standard deviation is', df['authors'].str.split(',').apply(len).std())
a = pd.Series([item for sublist in df['authors'].str.split(',') for item in sublist])

a = a.str.strip()

a.value_counts()[:10]
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
def plot_wordcloud(text, mask=None, max_words=400, max_font_size=120, figure_size=(24.0,16.0), title = None, title_size=40, image_color=False):

    """

    Function Credit: https://www.kaggle.com/aashita/word-clouds-of-various-shapes

    """

    stopwords = set(STOPWORDS)

    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}

    stopwords = stopwords.union(more_stopwords)



    wordcloud = WordCloud(background_color='white',

                    stopwords = stopwords,

                    max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 42,

                    mask = mask)

    wordcloud.generate(text)

    

    plt.figure(figsize=figure_size)

    if image_color:

        image_colors = ImageColorGenerator(mask);

        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");

        plt.title(title, fontdict={'size': title_size,  

                                  'verticalalignment': 'bottom'})

    else:

        plt.imshow(wordcloud);

        plt.title(title, fontdict={'size': title_size, 'color': 'green', 

                                  'verticalalignment': 'bottom'})

    plt.axis('off');

    plt.tight_layout()

        

def plot_the_author(name):

    author_paper_abstract = df[df['authors'].str.contains(name)]['abstract']

    plot_wordcloud(str(author_paper_abstract), max_words=600, max_font_size=120,  title = 'What ' + name + ' is upto?', title_size=20, figure_size=(10,12))
plot_the_author('Wei Liu')
plot_the_author('Xiaogang Wang')
plot_the_author('Ling Shao')
import networkx as nx  

from tqdm import tqdm
def relation_graph(name):

    a = df[df['authors'].str.contains(name)]['authors'].str.split(', ')

    a = a.tolist()

    edge_list = set()

    for l in a:

        n = len(l)

        for i in range(n):

            for j in range(i+1, n):

                edge_list.add((l[i].strip(), l[j].strip()))

    edge_list = list(edge_list)

    G = nx.DiGraph()

    G.add_edges_from(edge_list)

    f, ax = plt.subplots(figsize=(18, 12))

    nx.draw(G.to_undirected(),  with_labels=True, font_weight='bold', ax=ax)

    plt.show()
relation_graph('Wei Liu')
relation_graph('Xiaogang Wang')
relation_graph('Ling Shao')
sns.distplot(df['abstract'].str.len())
print('The mean of the distribution is', df['abstract'].str.len().mean(), 'and the standard deviation is', df['abstract'].str.len().std())
plot_wordcloud('\n'.join(df['abstract'].tolist()), title_size=20, figure_size=(10,12), title="Abstract Wordcloud", max_words=800)
plot_wordcloud('\n'.join(df['title'].tolist()), title_size=20, figure_size=(10,12), title="Title Wordcloud", max_words=800)
sns.distplot(df['content'].str.len())
print(df['content'].iloc[0][:1000])
df['content'] = df['content'].str.lower()
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'u.s':'america', 'e.g':'for example'}
def clean_contractions(text, mapping):

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        text = text.replace(s, "'")

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    return text
df['content'] = df['content'].progress_apply(lambda x: clean_contractions(x, contraction_mapping))
punct = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 

 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 

 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 

 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 

 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', '[', ']', '.', ',']
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', '!':' '}
def clean_special_chars(text, punct, mapping):

    for p in mapping:

        text = text.replace(p, mapping[p])

    

    for p in punct:

        text = text.replace(p, f' ')    

    return text
df['content'] = df['content'].progress_apply(lambda x: clean_special_chars(x, punct, punct_mapping))
print(df['content'].iloc[0][:1000])
import re
def remove_citations(text):

    

    return re.sub("\[ [0-9]+[,  [0-9]+]*\]", "", text)
a = 'asdas[ 4, 6 ]asds'

re.findall("\[ [0-9]+[,  [0-9]+]*\]", a)
print(remove_citations(df['content'].iloc[0])[-1000:])
df['content'] = df['content'].progress_apply(remove_citations)
def remove_ref_sec(text):

    idx = text.rfind('references')

    return text[:idx].strip()
df['content'] = df['content'].progress_apply(remove_ref_sec)
def rem_author_names(text):

    return text[3] + '\n\n' + text[0][text[0].find('abstract'):]
df['content'] = df.progress_apply(rem_author_names, axis=1)
df.shape
print(df['content'].iloc[0][:1000])
import nltk

from tqdm import tqdm

tqdm.pandas()
df['content'] = df['content'].progress_apply(nltk.word_tokenize)
en_stopwords = set(nltk.corpus.stopwords.words('english'))

df['content'] = df['content'].progress_apply(lambda x: [item for item in x if item not in en_stopwords])
#function to filter for ADJ/NN bigrams

def rightTypes(ngram):

    if '-pron-' in ngram or 't' in ngram:

        return False

    for word in ngram:

        if word in en_stopwords or word.isspace():

            return False

    acceptable_types = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')

    second_type = ('NN', 'NNS', 'NNP', 'NNPS')

    tags = nltk.pos_tag(ngram)

    if tags[0][1] in acceptable_types and tags[1][1] in second_type:

        return True

    else:

        return False

#filter bigrams

#filtered_bi = bigramFreqTable[bigramFreqTable.bigram.map(lambda x: rightTypes(x))]

#function to filter for trigrams

def rightTypesTri(ngram):

    if '-pron-' in ngram or 't' in ngram:

        return False

    for word in ngram:

        if word in en_stopwords or word.isspace():

            return False

    first_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')

    third_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')

    tags = nltk.pos_tag(ngram)

    if tags[0][1] in first_type and tags[2][1] in third_type:

        return True

    else:

        return False

#filter trigrams

#filtered_tri = trigramFreqTable[trigramFreqTable.trigram.map(lambda x: rightTypesTri(x))]
def get_bigram_trigrams(tokens, title, return_finders = False):

    bigrams = nltk.collocations.BigramAssocMeasures()

    trigrams = nltk.collocations.TrigramAssocMeasures()

    bigramFinder = nltk.collocations.BigramCollocationFinder.from_words(tokens)

    trigramFinder = nltk.collocations.TrigramCollocationFinder.from_words(tokens)

    #bigrams

    bigram_freq = bigramFinder.ngram_fd.items()

    bigramFreqTable = pd.DataFrame(list(bigram_freq), columns=['bigram','freq']).sort_values(by='freq', ascending=False)

    #trigrams

    trigram_freq = trigramFinder.ngram_fd.items()

    trigramFreqTable = pd.DataFrame(list(trigram_freq), columns=['trigram','freq']).sort_values(by='freq', ascending=False)

    filtered_bi = bigramFreqTable[bigramFreqTable.bigram.map(lambda x: rightTypes(x))]

    filtered_tri = trigramFreqTable[trigramFreqTable.trigram.map(lambda x: rightTypesTri(x))]

    if return_finders:

        return bigramFinder, trigramFinder, bigrams, trigrams

    print('Exploring bigrams and trigrams of ' + title)

    print(filtered_bi.head(10))

    print(filtered_tri.head(10))

    return filtered_bi, filtered_tri
id = 45

_, __ = get_bigram_trigrams(df['content'].iloc[id], df['title'].iloc[id])
id = 23

_, __ = get_bigram_trigrams(df['content'].iloc[id], df['title'].iloc[id])
id = 11

_, __ = get_bigram_trigrams(df['content'].iloc[id], df['title'].iloc[id])
def get_pointwise_mi_scores(content, title):

    bigramFinder, trigramFinder, bigrams, trigrams = get_bigram_trigrams(content, title, True)

    #filter for only those with more than 20 occurences

    bigramFinder.apply_freq_filter(20)

    trigramFinder.apply_freq_filter(20)

    bigramPMITable = pd.DataFrame(list(bigramFinder.score_ngrams(bigrams.pmi)), columns=['bigram','PMI']).sort_values(by='PMI', ascending=False)

    trigramPMITable = pd.DataFrame(list(trigramFinder.score_ngrams(trigrams.pmi)), columns=['trigram','PMI']).sort_values(by='PMI', ascending=False)

    print('Exploring Point wise Mututal Information in bigrams and trigrams of ' + title)

    print(bigramPMITable.head(10))

    print(trigramPMITable.head(10))

    return bigramPMITable, trigramPMITable
id = 45

_, __ = get_pointwise_mi_scores(df['content'].iloc[id], df['title'].iloc[id])
id = 23

_, __ = get_pointwise_mi_scores(df['content'].iloc[id], df['title'].iloc[id])
id = 11

_, __ = get_pointwise_mi_scores(df['content'].iloc[id], df['title'].iloc[id])
def get_t_scores(content, title):

    bigramFinder, trigramFinder, bigrams, trigrams = get_bigram_trigrams(content, title, True)

    bigramTtable = pd.DataFrame(list(bigramFinder.score_ngrams(bigrams.student_t)), columns=['bigram','t']).sort_values(by='t', ascending=False)

    trigramTtable = pd.DataFrame(list(trigramFinder.score_ngrams(trigrams.student_t)), columns=['trigram','t']).sort_values(by='t', ascending=False)

    #filters

    filteredT_bi = bigramTtable[bigramTtable.bigram.map(lambda x: rightTypes(x))]

    filteredT_tri = trigramTtable[trigramTtable.trigram.map(lambda x: rightTypesTri(x))]

    print('Exploring t scores between the words in bigrams and trigrams of ' + title)

    print(filteredT_bi.head(10))

    print(filteredT_tri.head(10))

    return filteredT_bi, filteredT_tri
id = 45

_, __ = get_t_scores(df['content'].iloc[id], df['title'].iloc[id])
id = 23

_, __ = get_t_scores(df['content'].iloc[id], df['title'].iloc[id])
id = 11

_, __ = get_t_scores(df['content'].iloc[id], df['title'].iloc[id])
import heapq

from operator import itemgetter
def get_n_most_freq(tokens, n=10):

    allWordDist = nltk.FreqDist(w.lower() for w in tokens)

    topitems = heapq.nlargest(iterable=allWordDist.items(), key=itemgetter(1), n=n)

    topitemsasdict = dict(topitems)

    return list(topitemsasdict.keys())
df['ten_most_freq'] = df['content'].progress_apply(get_n_most_freq)
df.head()
plot_wordcloud('\n'.join(np.concatenate(df['ten_most_freq']).tolist()), title_size=20, figure_size=(12,18), title="Most frequent words in the papers", max_words=1200)