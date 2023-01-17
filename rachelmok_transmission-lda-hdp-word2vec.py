# Load in libraries 

import re

import csv

import codecs

import numpy as np

import pandas as pd

import operator

import string



import nltk

from nltk import word_tokenize

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer

from string import punctuation

from collections import defaultdict

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer

eng_stopwords = set(stopwords.words("english"))

import sys



import gensim

from gensim import corpora, models

from gensim.corpora.dictionary import Dictionary
#Read in metadata 

data = pd.read_csv("../input/CORD-19-research-challenge/metadata.csv") 



#Select the columns that you need 

meta_data = data[["cord_uid","sha", "title", "abstract"]]

#check na

meta_data.isna().sum()



#Remove NA 

complete_cases = meta_data.dropna()

print(complete_cases.isna().sum())

complete_cases.shape
def tokenize(text):

    '''

    Convert the text corpus to lower case, remove all punctuations and numbers which lead to

    a final cleaned corpus with only tokens where all characters in the string are alphabets.

    '''

    # convert the text to lower case and replace all new line characters by an empty string

    lower_text = text.lower().replace('\n', ' ')

    # replace all the punctuations in text by an empty string

    table = str.maketrans('', '', string.punctuation)

    punct_text = lower_text.translate(table)

    # use NLTK's word tokenization to tokenize the text 

    # remove numbers and empty tokens, only keep tokens with all characters in the string are alphabets

    tokens = [word for word in word_tokenize(punct_text) if word.isalpha()]

    return tokens
def remove_stopwords(word_list, sw=stopwords.words('english')):

    """ 

    Filter out all stop words from the text corpus.

    """

    # It is important to keep words like no and not. Since the meaning of the text will change oppositely

    # if they are removed.

    if 'not' in sw:

        sw.remove('not')

    if 'no' in sw:

        sw.remove('no')

    

    cleaned = []

    for word in word_list:

        if word not in sw:

            cleaned.append(word)

    return cleaned
def stem_words(word_list):

    stemmer = SnowballStemmer('english')

    stemmed_words = [stemmer.stem(word) for word in word_list]

    text = " ".join(stemmed_words)

    return stemmed_words, text
def preprocess(text):

    """

    Combine all preprocess steps together.

    Clean each text into tokenized stemmed word list and also return concatenated string

    """

    tokenized = tokenize(text)

    stopword_removed = remove_stopwords(tokenized)

    tokenized_str, cleaned_str = stem_words(stopword_removed)

    return stopword_removed, tokenized_str, cleaned_str
# clean titles

titles = complete_cases["title"].values.tolist()

tokenized_titles = []

str_titles = []

word_dict = {}

for title in titles:

  result = preprocess(title)

  tokenized = result[0]

  stemmed = result[1]

  for i in range(0,len(stemmed)):

    if stemmed[i] not in word_dict:

      word_dict[stemmed[i]] = tokenized[i]

  tokenized_titles.append(stemmed)

  str_titles.append(result[2])

    

print(tokenized_titles[0])

print(str_titles[0])

print(titles[0])

print(len(word_dict))  # there are 23209 unique words in titles 
# clean abstracts

abstracts = complete_cases["abstract"].values.tolist()

tokenized_abstracts = []

str_abstracts = []

word_dict_abstract = {}

for abstract in abstracts:

  result = preprocess(abstract)

  tokenized = result[0]

  stemmed = result[1]

  for i in range(0,len(stemmed)):

    if stemmed[i] not in word_dict_abstract:

      word_dict_abstract[stemmed[i]] = tokenized[i]

  tokenized_abstracts.append(stemmed)

  str_abstracts.append(result[2])

    

print(tokenized_abstracts[0])

print(str_abstracts[0])

print(abstracts[0])

print(len(word_dict_abstract))  # there are 92667 unique words in abstracts
complete_cases["tokenized_abstract"] = tokenized_abstracts

complete_cases["cleaned_abstracts"] = str_abstracts

# append to data

complete_cases["tokenized_title"] = tokenized_titles

complete_cases["cleaned_titles"] = str_titles
# Use the Vectorizer to identify the keywords with highest frequency

from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style('whitegrid')

%matplotlib inline



# Helper function

def plot_10_most_common_words(count_data, count_vectorizer):

    import matplotlib.pyplot as plt

    words = count_vectorizer.get_feature_names()

    total_counts = np.zeros(len(words))

    for t in count_data:

        total_counts+=t.toarray()[0]

    

    count_dict = (zip(words, total_counts))

    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]

    words = [w[0] for w in count_dict]

    counts = [w[1] for w in count_dict]

    x_pos = np.arange(len(words)) 

    

    plt.figure(2, figsize=(15, 15/1.6180))

    plt.subplot(title='10 most common words')

    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})

    sns.barplot(x_pos, counts, palette='husl')

    plt.xticks(x_pos, words, rotation=90) 

    plt.xlabel('words')

    plt.ylabel('counts')

    plt.show()

    

# Initialise the count vectorizer with the English stop words

count_vectorizer = CountVectorizer(stop_words='english')



# Fit and transform the processed titles

count_data = count_vectorizer.fit_transform(complete_cases['cleaned_titles'])



# Visualise the 10 most common words

plot_10_most_common_words(count_data, count_vectorizer)
del_list = ('virus', 'infect','respiratori','coronavirus','diseas','cell','protein','viral','influenza')

tokenized_titles = complete_cases['tokenized_title']

tokenized_titles2 = [[ele for ele in sub if ele not in del_list] for sub in tokenized_titles]
dictionary = gensim.corpora.Dictionary(tokenized_titles2)

bow_corpus = [dictionary.doc2bow(doc) for doc in tokenized_titles2]
tfidf = models.TfidfModel(bow_corpus)

corpus_tfidf = tfidf[bow_corpus]

from pprint import pprint

for doc in corpus_tfidf:

    pprint(doc)

    break
lda_model_tfidf = gensim.models.ldamodel.LdaModel(corpus_tfidf, num_topics=10, id2word=dictionary)

for idx, topic in lda_model_tfidf.print_topics(-1):

    print('Topic: {} Word: {}'.format(idx, topic))
for idx, topic in lda_model_tfidf.show_topics(formatted=False, num_words= 10):

    print('Topic: {} \nWords: {}'.format(idx, [w[0] for w in topic]))
from gensim.models import CoherenceModel

lda_cm = CoherenceModel(model = lda_model_tfidf, corpus = corpus_tfidf, dictionary= dictionary, texts = tokenized_titles2, coherence = "c_v")

lda_CM = lda_cm.get_coherence()

lda_CM
wordDict = Dictionary(tokenized_titles2)

abCorpus = [wordDict.doc2bow(text) for text in tokenized_titles2]
from gensim.models import HdpModel

hdp = HdpModel(corpus = abCorpus, id2word = wordDict)

hdp_topics = hdp.print_topics()

for topic in hdp_topics: 

    print(topic)
for idx, topic in hdp.show_topics(formatted=False, num_words= 10):

    print('Topic: {} \nWords: {}'.format(idx, [w[0] for w in topic]))
from gensim.models import CoherenceModel

hdp_cm = CoherenceModel(model = hdp, corpus = abCorpus, dictionary= wordDict, texts = tokenized_titles2, coherence = "c_v")

hdp_CM = hdp_cm.get_coherence()

hdp_CM
# Keyword Search

selection = ['contagion','contagious','shedding','transmission','transmittable','airbourne','incubation','asymptomatic', 'seasonality']

selection_stemmed = stem_words(selection)

# 'detection','antibody','bolvine','immunizaton','infection','hydrophilic','hydrophobic','calves','environmental' 

    

# Scan with abstract

aScan = complete_cases.tokenized_abstract.apply(lambda x: any(item for item in selection if item in x))

complete_cases3 = complete_cases[aScan]

print(complete_cases3.shape)



# Scan with title 

tScan = complete_cases.tokenized_title.apply(lambda x: any(item for item in selection if item in x))

complete_cases2 = complete_cases[tScan]

print(complete_cases2.shape)
"""Read in the full text and filter to the selected list based on keyword search""" 

"""posted code to generate csv: https://www.kaggle.com/xhlulu/cord-19-eda-parse-json-and-generate-clean-csv"""

tokenized_txt = pd.read_csv("../input/tokenized-text/text_tokens.csv") 

smList = pd.DataFrame(complete_cases3)

filtered_df = pd.DataFrame(pd.merge(tokenized_txt,smList, left_on = "paper_id", right_on = "sha"))

filtered_df.shape
"""Create extended stopwords list (Ref- https://www.kaggle.com/vrushank12/covid-analysis-using-doc2vec-and-sent2vec)""" 

# pos_words and extend words are some common words to be removed from abstract

stop_words = nltk.corpus.stopwords.words('english')

pos_words = ['highest','among','either','seven','six','plus','strongest','worst','doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure', 

    'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.', 

    'al.', 'Elsevier', 'PMC', 'CZI', 'www'

,'greatest','every','better','per','across','throughout','except','fewer','trillion','fewest','latest','least','manifest','unlike','eight','since','toward','largest','despite','via','finest','besides','easiest','must','million','oldest','behind','outside','smaller','nest','longest','whatever','stronger','worse','two','another','billion','best','near','nine','around','nearest','wechat','lowest','smallest','along','higher','three','older','greater','neither','inside','newest','lower','may','although','though','earlier','upon','five','ca','larger','us','whether','beyond','onto','might','one','out','unless','four','whose','can','fastest','without','ecobooth','broadest','easier','within','like', 'could','biggest','bigger','would','thereby','yet','timely','thus','also','avoid','know','usually','time','year','go','welcome','even','date',

             'used', 'following', 'go', 'instead', 'fundamentally', 'first', 'second', 'alone',

               'everything', 'end', 'also', 'year', 'made', 'many', 'towards', 'truly', 'last','introduction', 'abstract', 'section', 'edition', 'chapter','and', 'the', 'is', 'any', 'to', 'by', 'of', 'on','or', 'with', 'which', 'was','be','we', 'are', 'so',

                    'for', 'it', 'in', 'they', 'were', 'as','at','such', 'no', 'that', 'there', 'then', 'those',

                    'not', 'all', 'this','their','our', 'between', 'have', 'than', 'has', 'but', 'why', 'only', 'into',

                    'during', 'some', 'an', 'more', 'had', 'when', 'from', 'its', "it's", 'been', 'can', 'further',

                    'above', 'before', 'these', 'who', 'under', 'over', 'each', 'because', 'them', 'where', 'both',

                     'just', 'do', 'once', 'through', 'up', 'down', 'other', 'here', 'if', 'out', 'while', 'same',

                    'after', 'did', 'being', 'about', 'how', 'few', 'most', 'off', 'should', 'until', 'will', 'now',

                    'he', 'her', 'what', 'does', 'itself', 'against', 'below', 'themselves','having', 'his', 'am', 'whom',

                    'she', 'nor', 'his', 'hers', 'too', 'own', 'ma', 'him', 'theirs', 'again', 'doing', 'ourselves',

                     're', 'me', 'ours', 'ie', 'you', 'your', 'herself', 'my', 'et', 'al', 'may', 'due', 'de',

                     'one','two', 'three', 'four', 'five','six','seven','eight','nine','ten', 'however',

                     'i', 'ii', 'iii','iv','v', 'vii', 'viii', 'ix', 'x', 'xi', 'xii','xiii', 'xiv' 

               'often', 'called', 'new', 'date', 'fully', 'thus', 'new', 'include', 'http', 

               'www','doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et',

               'al', 'author', 'figure','rights', 'reserved', 'permission', 'used', 'using', 'biorxiv',

               'medrxiv', 'license', 'fig', 'fig.', 'al.', 'Elsevier', 'PMC', 'CZI','-PRON-']



stop_words.extend(pos_words)

filtered_df['body_text_processed'] = filtered_df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

filtered_df['body_text_processed_1'] = filtered_df['body_text_processed'].str.replace('[^\w\s]','')
filtered_tokenized_doc = []

for d in filtered_df['body_text_processed_1']:

    filtered_tokenized_doc.append(word_tokenize(d.lower()))
# find most similar doc 

#Read in model 

import numpy

import gensim

from gensim.test.utils import common_texts

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

tagged_data = [gensim.models.doc2vec.TaggedDocument(d, [i]) for i, d in enumerate(filtered_tokenized_doc)]
max_epochs = 100

vec_size = 100

alpha = 0.025



model = Doc2Vec(vector_size=vec_size,

                alpha=alpha, 

                min_alpha=0.00005,

                min_count=1,

                dm =0,

                dbow_words = 1)

  

model.build_vocab(tagged_data)



for epoch in range(max_epochs):

    print('iteration {0}'.format(epoch))

    model.train(tagged_data,

                total_examples=model.corpus_count,

                epochs=model.iter)

    # decrease the learning rate

    model.alpha -= 0.0002

    # fix the learning rate, no decay

    model.min_alpha = model.alpha
"""Subtopic 1: Transmission"""



import heapq

import re

mod = model

test_doc = word_tokenize("transmission".lower())

similar=mod.docvecs.most_similar(positive=[mod.infer_vector(test_doc)],topn=10)



def Extract(lst): 

    return list(list(zip(*lst))[0])

temp=Extract(similar)



treated_text = filtered_df.iloc[temp].iloc[0:,5].str.cat(sep=', ')

treated_text = re.sub(r'\[[0-9]*\]', ' ', treated_text)

treated_text = re.sub(r'\s+', ' ', treated_text)

formatted_article_text = re.sub('[^a-zA-Z]', ' ', treated_text)

formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

sentence_list = nltk.sent_tokenize(treated_text)



stopwords = nltk.corpus.stopwords.words('english')



word_frequencies = {}

for word in nltk.word_tokenize(formatted_article_text):

    if word not in stopwords:

        if word not in word_frequencies.keys():

            word_frequencies[word] = 1

        else:

            word_frequencies[word] += 1

maximum_frequncy = max(word_frequencies.values())



for word in word_frequencies.keys():

    word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

sentence_scores = {}

for sent in sentence_list:

    for word in nltk.word_tokenize(sent.lower()):

        if word in word_frequencies.keys():

            if len(sent.split(' ')) < 30:

                if sent not in sentence_scores.keys():

                    sentence_scores[sent] = word_frequencies[word]

                else:

                    sentence_scores[sent] += word_frequencies[word]



summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)



summary = ' '.join(summary_sentences)

summary = summary.replace("Wang et al.", "").replace("Chen et al.", "").replace("Liu et al.", "").replace("Prakash et al.", "")



print(summary)
"""Subtopic 1: Incubation"""

test_doc = word_tokenize("contagious".lower())

similar=mod.docvecs.most_similar(positive=[mod.infer_vector(test_doc)],topn=10)



def Extract(lst): 

    return list(list(zip(*lst))[0])

temp=Extract(similar)



treated_text = filtered_df.iloc[temp].iloc[0:,5].str.cat(sep=', ')

treated_text = re.sub(r'\[[0-9]*\]', ' ', treated_text)

treated_text = re.sub(r'\s+', ' ', treated_text)

formatted_article_text = re.sub('[^a-zA-Z]', ' ', treated_text)

formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

sentence_list = nltk.sent_tokenize(treated_text)



stopwords = nltk.corpus.stopwords.words('english')



word_frequencies = {}

for word in nltk.word_tokenize(formatted_article_text):

    if word not in stopwords:

        if word not in word_frequencies.keys():

            word_frequencies[word] = 1

        else:

            word_frequencies[word] += 1

maximum_frequncy = max(word_frequencies.values())



for word in word_frequencies.keys():

    word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

sentence_scores = {}

for sent in sentence_list:

    for word in nltk.word_tokenize(sent.lower()):

        if word in word_frequencies.keys():

            if len(sent.split(' ')) < 30:

                if sent not in sentence_scores.keys():

                    sentence_scores[sent] = word_frequencies[word]

                else:

                    sentence_scores[sent] += word_frequencies[word]



summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)



summary = ' '.join(summary_sentences)

summary = summary.replace("Wang et al.", "").replace("Chen et al.", "").replace("Liu et al.", "").replace("Prakash et al.", "")

print(summary)
"""Subtopic 3: Infection"""

test_doc = word_tokenize("infection".lower())

similar=mod.docvecs.most_similar(positive=[mod.infer_vector(test_doc)],topn=10)



def Extract(lst): 

    return list(list(zip(*lst))[0])

temp=Extract(similar)



treated_text = filtered_df.iloc[temp].iloc[0:,5].str.cat(sep=', ')

treated_text = re.sub(r'\[[0-9]*\]', ' ', treated_text)

treated_text = re.sub(r'\s+', ' ', treated_text)

formatted_article_text = re.sub('[^a-zA-Z]', ' ', treated_text)

formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

sentence_list = nltk.sent_tokenize(treated_text)



stopwords = nltk.corpus.stopwords.words('english')



word_frequencies = {}

for word in nltk.word_tokenize(formatted_article_text):

    if word not in stopwords:

        if word not in word_frequencies.keys():

            word_frequencies[word] = 1

        else:

            word_frequencies[word] += 1

maximum_frequncy = max(word_frequencies.values())



for word in word_frequencies.keys():

    word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

sentence_scores = {}

for sent in sentence_list:

    for word in nltk.word_tokenize(sent.lower()):

        if word in word_frequencies.keys():

            if len(sent.split(' ')) < 30:

                if sent not in sentence_scores.keys():

                    sentence_scores[sent] = word_frequencies[word]

                else:

                    sentence_scores[sent] += word_frequencies[word]



summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)



summary = ' '.join(summary_sentences)

print(summary)
from nltk.corpus import stopwords

from nltk.cluster.util import cosine_distance

import numpy as np

import networkx as nx



def sentence_similarity(sent1, sent2, stopwords=None):

    if stopwords is None:

        stopwords = []

 

    sent1 = [w.lower() for w in sent1]

    sent2 = [w.lower() for w in sent2]

 

    all_words = list(set(sent1 + sent2))

 

    vector1 = [0] * len(all_words)

    vector2 = [0] * len(all_words)

 

    # build the vector for the first sentence

    for w in sent1:

        if w in stopwords:

            continue

        vector1[all_words.index(w)] += 1

 

    # build the vector for the second sentence

    for w in sent2:

        if w in stopwords:

            continue

        vector2[all_words.index(w)] += 1

 

    return 1 - cosine_distance(vector1, vector2) 
def build_similarity_matrix(sentences, stop_words):

    # Create an empty similarity matrix

    similarity_matrix = np.zeros((len(sentences), len(sentences)))

 

    for idx1 in range(len(sentences)):

        for idx2 in range(len(sentences)):

            if idx1 == idx2: #ignore if both are same sentences

                continue 

            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

            return similarity_matrix
def generate_summary(sentences, top_n=5):

    stop_words = stopwords.words('english')

    summarize_text = []

    # Step 1 - Generate Similary Martix across sentences

    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 2 - Rank sentences in similarity martix

    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)

    scores = nx.pagerank(sentence_similarity_graph)

    # Step 3 - Sort the rank and pick top sentences

    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    

    for i in range(top_n):

        summarize_text.append("".join(ranked_sentence[i][1]))

        # Step 4 - print summarized text

    print("Summarize Text: \n", ".".join(summarize_text))
"""Subtopic: Transmission"""

test_doc = word_tokenize("transmission channel".lower())

similar=mod.docvecs.most_similar(positive=[mod.infer_vector(test_doc)],topn=10)

temp=Extract(similar)



treated_text = filtered_df.iloc[temp].iloc[0:,5].str.cat(sep=', ')

treated_text = re.sub(r'\[[0-9]*\]', ' ', treated_text)

treated_text = re.sub(r'\s+', ' ', treated_text)

formatted_article_text = re.sub('[^a-zA-Z]', ' ', treated_text)

formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

sentence_list = nltk.sent_tokenize(treated_text)





generate_summary(sentence_list, 10)
"""Subtopic: Incubation"""

test_doc = word_tokenize("duration of incubation period".lower())

similar=mod.docvecs.most_similar(positive=[mod.infer_vector(test_doc)],topn=10)

temp=Extract(similar)



treated_text = filtered_df.iloc[temp].iloc[0:,5].str.cat(sep=', ')

treated_text = re.sub(r'\[[0-9]*\]', ' ', treated_text)

treated_text = re.sub(r'\s+', ' ', treated_text)

formatted_article_text = re.sub('[^a-zA-Z]', ' ', treated_text)

formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

sentence_list = nltk.sent_tokenize(treated_text)





generate_summary(sentence_list, 10)
"""Subtopic: Contagious"""

test_doc = word_tokenize("contagious".lower())

similar=mod.docvecs.most_similar(positive=[mod.infer_vector(test_doc)],topn=10)

temp=Extract(similar)



treated_text = filtered_df.iloc[temp].iloc[0:,5].str.cat(sep=', ')

treated_text = re.sub(r'\[[0-9]*\]', ' ', treated_text)

treated_text = re.sub(r'\s+', ' ', treated_text)

formatted_article_text = re.sub('[^a-zA-Z]', ' ', treated_text)

formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

sentence_list = nltk.sent_tokenize(treated_text)





generate_summary(sentence_list, 10)