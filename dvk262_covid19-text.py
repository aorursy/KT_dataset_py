import os

import json

from pprint import pprint

from copy import deepcopy



import numpy as np

import pandas as pd

from tqdm.notebook import tqdm
"""posted code to generate csv: https://www.kaggle.com/xhlulu/cord-19-eda-parse-json-and-generate-clean-csv"""

def format_name(author):

    middle_name = " ".join(author['middle'])

    

    if author['middle']:

        return " ".join([author['first'], middle_name, author['last']])

    else:

        return " ".join([author['first'], author['last']])

    

def format_affiliation(affiliation):

    text = []

    location = affiliation.get('location')

    if location:

        text.extend(list(affiliation['location'].values()))

    

    institution = affiliation.get('institution')

    if institution:

        text = [institution] + text

    return ", ".join(text)



def format_authors(authors, with_affiliation=False):

    name_ls = []

    

    for author in authors:

        name = format_name(author)

        if with_affiliation:

            affiliation = format_affiliation(author['affiliation'])

            if affiliation:

                name_ls.append(f"{name} ({affiliation})")

            else:

                name_ls.append(name)

        else:

            name_ls.append(name)

    

    return ", ".join(name_ls)



def format_body(body_text):

    texts = [(di['section'], di['text']) for di in body_text]

    texts_di = {di['section']: "" for di in body_text}

    

    for section, text in texts:

        texts_di[section] += text



    body = ""



    for section, text in texts_di.items():

        body += section

        body += "\n\n"

        body += text

        body += "\n\n"

    

    return body



def format_bib(bibs):

    if type(bibs) == dict:

        bibs = list(bibs.values())

    bibs = deepcopy(bibs)

    formatted = []

    

    for bib in bibs:

        bib['authors'] = format_authors(

            bib['authors'], 

            with_affiliation=False

        )

        formatted_ls = [str(bib[k]) for k in ['title', 'authors', 'venue', 'year']]

        formatted.append(", ".join(formatted_ls))



    return "; ".join(formatted)



def load_files(dirname):

    filenames = os.listdir(dirname)

    raw_files = []



    for filename in tqdm(filenames):

        filename = dirname + filename

        file = json.load(open(filename, 'rb'))

        raw_files.append(file)

    

    return raw_files



def generate_clean_df(all_files):

    cleaned_files = []

    

    for file in tqdm(all_files):

        features = [

            file['paper_id'],

            file['metadata']['title'],

            format_authors(file['metadata']['authors']),

            format_authors(file['metadata']['authors'], 

                           with_affiliation=True),

            format_body(file['abstract']),

            format_body(file['body_text']),

            format_bib(file['bib_entries']),

            file['metadata']['authors'],

            file['bib_entries']

        ]



        cleaned_files.append(features)

        

    col_names = ['paper_id', 'title', 'authors',

                 'affiliations', 'abstract', 'text', 

                 'bibliography','raw_authors','raw_bibliography']



    clean_df = pd.DataFrame(cleaned_files, columns=col_names)

    clean_df.head()

    

    return clean_df
biorxiv_dir = '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/'

bio_files = load_files(biorxiv_dir)

bio_df = generate_clean_df(bio_files)
pmc_dir = '/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/pdf_json/'

pmc_files = load_files(pmc_dir)

pmc_df = generate_clean_df(pmc_files)
comm_dir = '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json/'

comm_files = load_files(comm_dir)

comm_df = generate_clean_df(comm_files)
noncomm_dir = '/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json/'

noncomm_files = load_files(noncomm_dir)

noncomm_df = generate_clean_df(noncomm_files)
data = pd.concat([bio_df,pmc_df,comm_df,noncomm_df])
"""I have created a list of indecies for social articles, i created this list using LDA tagging"""

social_index=pd.read_csv('/kaggle/input/index-social/index_social.csv')

social_index
data2=data.iloc[social_index['Unnamed: 0'],:].copy()
import nltk

from nltk.tokenize import word_tokenize

from nltk.stem import PorterStemmer

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

from tensorflow import keras
# Gensim

import gensim

import gensim.corpora as corpora

from gensim.utils import simple_preprocess

from gensim.models import CoherenceModel
stop_words={'ourselves', 'hers', 'between', 'yourself', 'but', 'again', \

            'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with',\

            'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such',\

            'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who',\

            'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we',\

            'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more',\

            'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above',\

            'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', \

            'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will',\

            'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over',\

            'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', \

            'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which',\

            'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', \

            'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than',\

            ')','(',',',';','et','al','.'   }
"""

https://pythonhealthcare.org/category/natural-language-processing/

Here we process the data in the following ways:

  1) change all text to lower case

  2) tokenize (breaks text down into a list of words)

  3) remove punctuation and non-word text

  4) find word stems (e.g. runnign, run and runner will be converted to run)

  5) removes stop words (commonly occuring words of little value, e.g. 'the')

"""

stemming = PorterStemmer()

stops = stop_words



def clean_text(raw_text):

    """This function works on a raw text string, and:

        1) changes to lower case

        2) tokenizes (breaks text down into a list of words)

        3) removes punctuation and non-word text

        4) finds word stems

        5) removes stop words

        6) rejoins meaningful stem words"""

    

    # Convert to lower case

    text = raw_text.lower()

    

    # Tokenize

    tokens = nltk.word_tokenize(text)

    

    # Keep only words (removes punctuation + numbers)

    # use .isalnum to keep also numbers

    token_words = [w for w in tokens if w.isalpha()]

    

    # Stemming

    stemmed_words = [stemming.stem(w) for w in token_words]

    

    # Remove stop words

    meaningful_words = [w for w in stemmed_words if not w in stops]

      

    # Return cleaned data

    return meaningful_words

def apply_cleaning_function_to_list(X):

    cleaned_X = []

    for element in X:

        cleaned_X.append(clean_text(element))

    return cleaned_X



"""

The frequency of all words is counted. Words are then given an index number so

that th emost commonly occurring words hav ethe lowest number (so the 

dictionary may then be truncated at any point to keep the most common words).

We avoid using the index number zero as we will use that later to 'pad' out

short text.

"""



def training_text_to_numbers(text, cutoff_for_rare_words = 1):

    """Function to convert text to numbers. Text must be tokenzied so that

    test is presented as a list of words. The index number for a word

    is based on its frequency (words occuring more often have a lower index).

    If a word does not occur as many times as cutoff_for_rare_words,

    then it is given a word index of zero. All rare words will be zero.

    """

    # Flatten list if sublists are present

    if len(text) > 1:

        flat_text = [item for sublist in text for item in sublist]

    else:

        flat_text = text

    

    # get word freuqncy

    fdist = nltk.FreqDist(flat_text)



    # Convert to Pandas dataframe

    df_fdist = pd.DataFrame.from_dict(fdist, orient='index')

    df_fdist.columns = ['Frequency']



    # Sort by word frequency

    df_fdist.sort_values(by=['Frequency'], ascending=False, inplace=True)



    # Add word index

    number_of_words = df_fdist.shape[0]

    df_fdist['word_index'] = list(np.arange(number_of_words)+1)

    # Convert pandas to dictionary

    word_dict = df_fdist['Frequency'].to_dict()

    

    # Use dictionary to convert words in text to numbers

    text_numbers = []

    for string in text:

        string_numbers = [word for word in string if 2<word_dict[word]<10000]

        text_numbers.append(string_numbers)

    

    return (text_numbers, df_fdist)
data2['cleaned_text'] = apply_cleaning_function_to_list(data2.text)

numbered_text, dict_df = training_text_to_numbers(data2['cleaned_text'].values)

data2=data2.assign(numbered_text=numbered_text)
#1 sentence split

def sentance(text):

    sentences=list()

    sentence_list=nltk.sent_tokenize(text)

    for sent in sentence_list:

        if sent.find('copyright') != -1:

            continue

        elif sent.find('https') != -1:

            continue

        else:             

            sentences.append(sent)

    return(sentences)

data2['sentences']= [ sentance(i) for i in data2.text ]
data2['sentences2']= [' '.join(i) for i in data2.sentences ]
sentences=list()

for lst in data2.sentences:

    for sent in lst:

        sentences.append(sent)

sentences2 = apply_cleaning_function_to_list(sentences)
from gensim.test.utils import common_texts

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(list(data2.numbered_text))]

#model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

#model.save('model')
model=Doc2Vec.load('/kaggle/input/doc2vec-model/model')
#to create a new vector

vector = model.infer_vector(apply_cleaning_function_to_list(['ethical and social'])[0])



# to find the siilarity with vector

model.similar_by_vector(vector)



# to find the most similar word to words in 2 document

model.wv.most_similar(documents[1][0])



#find similar documents to document 1

sim_docs=model.docvecs.most_similar(1)
# Create Dictionary

id2word = corpora.Dictionary(data2.numbered_text)

# Create Corpus

texts = data2.numbered_text



# Term Document Frequency

corpus = [id2word.doc2bow(text) for text in texts]



# Build LDA model

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,

                                           id2word=id2word,

                                           num_topics=5, 

                                           random_state=100,

                                           update_every=1,

                                           chunksize=100,

                                           passes=10,

                                           alpha='auto',

                                           per_word_topics=True)
# Print the Keyword in the 10 topics

print(lda_model.print_topics())

doc_lda = lda_model[corpus]
# Visualize the topics

# Plotting tools

import pyLDAvis

import pyLDAvis.gensim  # don't skip this

import matplotlib.pyplot as plt

%matplotlib inline

pyLDAvis.enable_notebook()

vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)

vis
from gensim.models import Word2Vec

model = Word2Vec(data2.numbered_text, min_count=1,size= 50,workers=3, window =3, sg = 1, seed=1)

words = list(model.wv.vocab)
from sklearn.decomposition import PCA

import matplotlib.pyplot as pyplot

X = model[model.wv.vocab]

pca = PCA(n_components=2)

result = pca.fit_transform(X)

pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)

for i, word in enumerate(words):

	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
model.most_similar(positive=['social'], topn=10)
import heapq

maximum_frequncy = max(dict_df['Frequency'])

dict_df['Frequency']=dict_df['Frequency']/maximum_frequncy



#https://stackabuse.com/text-summarization-with-nltk-in-python/

def sentance_summary(text):

    sentence_scores = {}

    sentence_list=nltk.sent_tokenize(text)

    for sent in sentence_list:

        if sent.find('copyright') != -1:

            continue

        else:             

            for word in clean_text(sent):

                if word in dict_df.index:

                    if len(sent.split(' ')) < 30:

                        if sent not in sentence_scores.keys():

                            sentence_scores[sent] = dict_df[dict_df.index==word]["Frequency"].tolist()[0]

                        elif word == 'copyright':

                            sentence_scores[sent] += -10

                        else:

                            sentence_scores[sent] += dict_df[dict_df.index==word]["Frequency"].tolist()[0]

    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)

    return(summary)
data3=data2.reset_index()

data3['summary']= [ sentance_summary(i) for i in data3.sentences2 ]
data3['abstract'][sim_docs[0][0]]
data3['summary'][sim_docs[0][0]]
import spacy

# Load English tokenizer, tagger, parser, NER and word vectors

nlp = spacy.load("en_core_web_sm")

data2['clean_text']=[' '.join(i) for i in data2.numbered_text]
#https://www.kaggle.com/shivamb/spacy-text-meta-features-knowledge-graphs

from collections import Counter 

feats = ['char_count', 'word_count', 'word_count_cln',

       'stopword_count', '_NOUN', '_VERB', '_ADP', '_ADJ', '_DET', '_PROPN',

       '_INTJ', '_PUNCT', '_NUM', '_PRON', '_ADV', '_PART', '_amod', '_ROOT',

       '_punct', '_advmod', '_auxpass', '_nsubjpass', '_ccomp', '_acomp',

       '_neg', '_nsubj', '_aux', '_agent', '_det', '_pobj', '_prep', '_csubj',

       '_nummod', '_attr', '_acl', '_relcl', '_dobj', '_pcomp', '_xcomp',

       '_cc', '_conj', '_mark', '_prt', '_compound', '_dep', '_advcl',

       '_parataxis', '_poss', '_intj', '_appos', '_npadvmod', '_predet',

       '_case', '_expl', '_oprd', '_dative', '_nmod']

class AutomatedTextFE:

    def __init__(self):

        self.pos_tags = ['NOUN', 'VERB', 'ADP', 'ADJ', 'DET', 'PROPN', 'INTJ', 'PUNCT',\

                         'NUM', 'PRON', 'ADV', 'PART']

        self.dep_tags = ['amod', 'ROOT', 'punct', 'advmod', 'auxpass', 'nsubjpass',\

                         'ccomp', 'acomp', 'neg', 'nsubj', 'aux', 'agent', 'det', 'pobj',\

                         'prep', 'csubj', 'nummod', 'attr', 'acl', 'relcl', 'dobj', 'pcomp', \

                         'xcomp', 'cc', 'conj', 'mark', 'prt', 'compound', 'dep', 'advcl',\

                         'parataxis', 'poss', 'intj', 'appos', 'npadvmod', 'predet', 'case',\

                         'expl', 'oprd', 'dative', 'nmod']



def _spacy_cleaning(doc):

    toks = [t for t in doc if (t.is_stop == False)]

    toks = [t for t in toks if (t.is_punct == False)]

    words = [t.lemma_ for token in toks]

    return " ".join(words)

def _spacy_features(df):

    df["clean_text"] = df[c].apply(lambda x : _spacy_cleaning(x))

    df["char_count"] = df[textcol].apply(len)

    df["word_count"] = df[c].apply(lambda x : len([_ for _ in x]))

    df["word_count_cln"] = df["clean_text"].apply(lambda x : len(x.split()))

    df["stopword_count"] = df[c].apply(lambda x : len([_ for _ in x if _.is_stop]))

    df["pos_tags"] = df[c].apply(lambda x : dict(Counter([_.head.pos_ for _ in x])))

    df["dep_tags"] = df[c].apply(lambda x : dict(Counter([_.dep_ for _ in x])))

    return df 



class AutomatedTextFE:

    def __init__(self, df, textcol):

        self.df = df

        self.textcol = textcol

        self.c = "spacy_" + textcol

        self.df[self.c] = self.df[self.textcol].apply( lambda x : nlp(x))

        

        self.pos_tags = ['NOUN', 'VERB', 'ADP', 'ADJ', 'DET', 'PROPN', 'INTJ', 'PUNCT',\

                         'NUM', 'PRON', 'ADV', 'PART']

        self.dep_tags = ['amod', 'ROOT', 'punct', 'advmod', 'auxpass', 'nsubjpass',\

                         'ccomp', 'acomp', 'neg', 'nsubj', 'aux', 'agent', 'det', 'pobj',\

                         'prep', 'csubj', 'nummod', 'attr', 'acl', 'relcl', 'dobj', 'pcomp', \

                         'xcomp', 'cc', 'conj', 'mark', 'prt', 'compound', 'dep', 'advcl',\

                         'parataxis', 'poss', 'intj', 'appos', 'npadvmod', 'predet', 'case',\

                         'expl', 'oprd', 'dative', 'nmod']

        

    def _spacy_cleaning(self, doc):

        tokens = [token for token in doc if (token.is_stop == False)\

                  and (token.is_punct == False)]

        words = [token.lemma_ for token in tokens]

        return " ".join(words)

        

    def _spacy_features(self):

        self.df["clean_text"] = self.df[self.c].apply(lambda x : self._spacy_cleaning(x))

        self.df["char_count"] = self.df[self.textcol].apply(len)

        self.df["word_count"] = self.df[self.c].apply(lambda x : len([_ for _ in x]))

        self.df["word_count_cln"] = self.df["clean_text"].apply(lambda x : len(x.split()))

        

        self.df["stopword_count"] = self.df[self.c].apply(lambda x : 

                                                          len([_ for _ in x if _.is_stop]))

        self.df["pos_tags"] = self.df[self.c].apply(lambda x :

                                                    dict(Counter([_.head.pos_ for _ in x])))

        self.df["dep_tags"] = self.df[self.c].apply(lambda x :

                                                    dict(Counter([_.dep_ for _ in x])))

        

    def _flatten_features(self):

        for key in self.pos_tags:

            self.df["_" + key] = self.df["pos_tags"].apply(lambda x : \

                                                           x[key] if key in x else 0)

        

        for key in self.dep_tags:

            self.df["_" + key] = self.df["dep_tags"].apply(lambda x : \

                                                           x[key] if key in x else 0)

                

    def generate_features(self):

        self._spacy_features()

        self._flatten_features()

        self.df = self.df.drop([self.c, "pos_tags", "dep_tags", "clean_text"], axis=1)

        return self.df

    

def spacy_features(df, tc):

    fe = AutomatedTextFE(df, tc)

    return fe.generate_features()
textcol = "clean_text"

feats_df=spacy_features(data2, textcol)
# document level spacy labeling

Noun=[]

Verb=[]

for article in data2.spacy_clean_text:

    #ents = [(e.text, e.start_char, e.end_char, e.label_) for e in article.ents]

    rel=[(token.text, token.dep_, token.head.text, token.head.pos_,[child for child in token.children])\

         for token in article]

    rel = pd.DataFrame(rel,columns =["TEXT",'DEP','HEAD TEXT','HEAD POS','CHILDREN']) 

    #Get verbs and no null children

    rel2=rel[([(i!=[]) for i in rel['CHILDREN']]) &  (rel['HEAD POS'] =='VERB')]

    tmp=[[(i, row[1][0]) for i in  row[1][1]] for row in rel2[["HEAD TEXT","CHILDREN"]].iterrows() ]

    for i in tmp:

        for n,v in i:

            Noun.append(n.text)

            Verb.append(v)

tmp=pd.DataFrame([Noun, Verb]).transpose()

tmp.columns=['Noun','Verb']

tmp3=tmp.groupby(['Noun','Verb']).size().reset_index()
tmp3=tmp3[(tmp3[0]>5) & (tmp3[0]<10)]
!pip install pyvis
Noun=tmp3.groupby('Noun', as_index = False).size().reset_index()

Verb=tmp3.groupby('Verb', as_index = False).size().reset_index()
words=Verb[Verb[0]>500]['Verb'].tolist() + Noun[Noun[0]>500]['Noun'].tolist()

tmp4=tmp3[(tmp3['Noun'].isin(words) | tmp3['Verb'].isin(words) )]

words=Verb[Verb[0]<10]['Verb'].tolist() + Noun[Noun[0]<=10]['Noun'].tolist()

tmp5=tmp3[(tmp3['Noun'].isin(words) | tmp3['Verb'].isin(words) )]
# pypi

import networkx

from networkx import (

    draw,

    DiGraph,

    Graph,

)

%matplotlib inline
from pyvis.network import Network
tmp3["Weight"]=tmp3[0]

sources = tmp3['Noun']

targets = tmp3['Verb']

weights = tmp3['Weight']



got_net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", \

                  notebook=True)

# set the physics layout of the network

got_net.barnes_hut()



edge_data = zip(sources, targets, weights)



for e in edge_data:

    src = e[0]

    dst = e[1]

    w = e[2]



    got_net.add_node(str(src), str(src), title=src)

    got_net.add_node(dst, dst, title=dst)

    got_net.add_edge(str(src), str(dst), value=w)



neighbor_map = got_net.get_adj_list()



# add neighbor data to node hover data

for node in got_net.nodes:

    node["title"] += " Neighbors:<br>" + "<br>".join(neighbor_map[node["id"]])

    node["value"] = len(neighbor_map[node["id"]])

    

got_net.show("Article3.html")
from pprint import pprint as print

from gensim.summarization import summarize
print(summarize(data3.sentences2[sim_docs[0][0]], ratio=0.05 ))