from IPython.display import Image
# doing it this way because have been trying to post it on image hoster and unable to open it from here



Image("../input/scheme-covid/scheme.png", width = 800, height = 800)
# Loading modules



# general module

import os

import collections

import pandas as pd

import pickle

pd.options.display.max_colwidth = 500

import numpy as np

import re

import json

from tqdm.notebook import tqdm as tqdm

from pprint import pprint

from copy import deepcopy

import requests

from requests.exceptions import HTTPError, ConnectionError

import logging

import math



# visualization module for topic modelling

import pyLDAvis.gensim

import pickle 

import pyLDAvis



# modelling module

import nltk

from nltk.tokenize import RegexpTokenizer

from nltk.stem.wordnet import WordNetLemmatizer

import gensim

from gensim.corpora import Dictionary

from gensim.utils import simple_preprocess

from gensim.models import CoherenceModel

from nltk.corpus import stopwords

from gensim.models import Word2Vec

from gensim.models import LdaModel

from gensim.corpora import Dictionary

import spacy

from sklearn.decomposition import PCA

pca = PCA(2)  # setting PCA to 2 axis

import torch

from transformers import BertForQuestionAnswering

from transformers import BertTokenizer

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

from torch.utils.data import DataLoader, SequentialSampler, TensorDataset



# search engine module bm25

!pip install rank_bm25 nltk

from rank_bm25 import BM25L
# Helper function



# function to structurize the dataframe from json format

# the function is taken from this kernel https://www.kaggle.com/xhlulu/cord-19-eda-parse-json-and-generate-clean-csv

# but with a little editing (we only need the author, title, and abstract)

def format_name(author):

    middle_name = " ".join(author['middle'])

    if author['middle']:

        return " ".join([author['first'], middle_name, author['last']])

    else:

        return " ".join([author['first'], author['last']])

def format_authors(authors, with_affiliation=False):

    name_ls = []

    for author in authors:

        name = format_name(author)

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

# function to process the json file based on structurizing function above

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

            format_body(file['abstract'])

        ]

        cleaned_files.append(features)

    col_names = ['paper_id', 'title', 'authors', 'abstract']

    clean_df = pd.DataFrame(cleaned_files, columns=col_names)

    clean_df.head() 

    return clean_df



# function for pre-processing text  

def sent_to_words(sentences):

    for sentence in sentences:

        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

# initialize stopwords

english_stopwords = stopwords.words('english')

# initialize spacy 'en' model, keeping only tagger component (for efficiency)

nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

# define functions for stopwords, bigrams, trigrams and lemmatization

def remove_stopwords(texts):

    return [[word for word in simple_preprocess(str(doc)) if word not in english_stopwords] for doc in texts]

def make_bigrams(texts):

    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):

    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):

    """https://spacy.io/api/annotation"""

    texts_out = []

    for sent in texts:

        doc = nlp(" ".join(sent)) 

        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

    return texts_out

# tokenizing the text

list_stopwords = list(set(english_stopwords))

def tokenize(text):

    words = nltk.word_tokenize(text)

    return list(set([word for word in words if word.isalnum() 

                                             and not word in list_stopwords

                                             and not (word.isnumeric() and len(word) < 4)]))

def preprocess(string):

    return tokenize(string.lower())



# class for creating "search engine"

class Paper:

    def __init__(self, item):

        self.paper = item.to_frame().fillna('')

        self.paper.columns = ['Value']

    def abstract(self):

        return self.paper.loc['abstract'].values[0]

    def title(self):

        return self.paper.loc['title'].values[0]



class SearchResults:

    def __init__(self, 

                 data: pd.DataFrame,

                 columns = None):

        self.results = data

        if columns:

            self.results = self.results[columns]        

    def __getitem__(self, item):

        return Paper(self.results.loc[item])

    def __len__(self):

        return len(self.results)    

    def _repr_html_(self):

        return self.results._repr_html_()



SEARCH_DISPLAY_COLUMNS = ['title','abstract']



class WordTokenIndex:

    def __init__(self, 

                 corpus: pd.DataFrame, 

                 columns=SEARCH_DISPLAY_COLUMNS):

        self.corpus = corpus

        raw_search_str = self.corpus.abstract.fillna('') + ' ' + self.corpus.title.fillna('')

        self.index = raw_search_str.apply(preprocess).to_frame()

        self.index.columns = ['terms']

        self.index.index = self.corpus.index

        self.columns = columns

    def search(self, search_string):

        search_terms = preprocess(search_string)

        result_index = self.index.terms.apply(lambda terms: any(i in terms for i in search_terms))

        results = self.corpus[result_index].copy().reset_index().rename(columns={'index':'paper'})

        return SearchResults(results, self.columns + ['paper'])



class RankBM25Index(WordTokenIndex):

    def __init__(self, corpus: pd.DataFrame, columns=SEARCH_DISPLAY_COLUMNS):

        super().__init__(corpus, columns)

        self.bm25 = BM25L(self.index.terms.tolist())  

    def search(self, search_string, n=10):

        search_terms = preprocess(search_string)

        doc_scores = self.bm25.get_scores(search_terms)

        ind = np.argsort(doc_scores)[::-1][:n]

        results = self.corpus.iloc[ind][self.columns]

        results['Score'] = doc_scores[ind]

        results = results[results.Score > 0]

        return SearchResults(results.reset_index(), self.columns + ['Score'])
# Generating data frame



prepared_dataframe = 1 # change this to zero if you dont have any prepared dataframe



if not prepared_dataframe :

    # data from biorxiv

    biorxiv_dir = '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/'

    biorxiv_files = load_files(biorxiv_dir)

    biorxiv_df = generate_clean_df(biorxiv_files)



    # data from commercial use

    comm_dir = '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json/'

    comm_files = load_files(comm_dir)

    comm_df = generate_clean_df(comm_files)





    # data from non commercial use

    noncomm_dir = '/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json/'

    noncomm_files = load_files(noncomm_dir)

    noncomm_df = generate_clean_df(noncomm_files)



    # data from custom use

    custom_dir = '/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/pdf_json/'

    custom_files = load_files(custom_dir)

    custom_df = generate_clean_df(custom_files)



    df = pd.concat([biorxiv_df, comm_df, noncomm_df, custom_df], axis  = 0)



    # deleting duplicated abstract because we focused on abstract

    df.drop_duplicates(subset = 'abstract', keep = 'first', inplace = True)

    df.to_csv('data.csv', index = False)

    

else :

    df = pd.read_csv('../input/data-covid-paper/data.csv')



bm25 = RankBM25Index(df)
# Preprocessed the abstract



pre_trained = 1 # change this to 0 if you dont have the embeddings



df['abstract'] = df['abstract'].fillna('').apply(str)

# remove punctuation

df['abstract_processed'] = df['abstract'].map(lambda x: re.sub('[,\.!?]', '', x))

# convert to lowercase

df['abstract_processed'] = df['abstract_processed'].map(lambda x: x.lower())

data_words = list(sent_to_words(df.abstract_processed.values.tolist()))

del df['abstract_processed']

# build the bigram and trigram models

bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.

trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

# faster way to get a sentence clubbed as a trigram/bigram

bigram_mod = gensim.models.phrases.Phraser(bigram)

trigram_mod = gensim.models.phrases.Phraser(trigram)

# remove stop words

data_words_nostops = remove_stopwords(data_words)

# form bigrams

data_words_bigrams = make_bigrams(data_words_nostops)
# Create dictionary of W2V embeddings



if not pre_trained :

    # do lemmatization for unbigram data (keeping only noun, adj, vb, adv)

    data_lemmatized = lemmatization(data_words_nostops, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # making the pre trained w2v_embeddings's vocabulary and embeddings to dictionary

    w2v_embeddings = Word2Vec(data_lemmatized, size=200, sg=1, min_count=1, window=8, hs=0, negative=15, workers=1)

    ordered_vocab = [(term, voc.index, voc.count) for term, voc in w2v_embeddings.wv.vocab.items()]

    ordered_vocab = sorted(ordered_vocab, key=lambda k: k[2])

    ordered_terms, term_indices, term_counts = zip(*ordered_vocab)

    word_vectors = pd.DataFrame(w2v_embeddings.wv.syn0[term_indices, :], index=ordered_terms)

    word_vectors = word_vectors.T.to_dict()

    f = open("word_vectors","wb")

    pickle.dump(word_vectors,f)

    f.close()

else :

    with open('../input/word-embeddings-covid-paper/word_vectors', 'rb') as embedding:

        word_vectors = pickle.load(embedding)
# Create LDA model

# code taken from https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0



# do lemmatization for bigram data (keeping only noun, adj, vb, adv)

data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

id2word = Dictionary(data_lemmatized)

id2word.filter_extremes(no_below=20, no_above=0.5)

# create corpus of bag of words

corpus = [id2word.doc2bow(text) for text in data_lemmatized]

if not pre_trained :

    # create LDA model

    lda_model = gensim.models.LdaMulticore(corpus=corpus,id2word=id2word,num_topics=17, 

                                           random_state=100,chunksize=100,passes=10,

                                           per_word_topics=True)

    lda_model.save('lda_model')

else :

    lda_model = LdaModel.load('../input/lda-model-covid-paper/ldamodel')

# compute Coherence Score

coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')

coherence_lda = coherence_model_lda.get_coherence()

print('\nCoherence Score: ', coherence_lda)



# print the topic model

pprint(lda_model.print_topics())
# Visualize the topics

pyLDAvis.enable_notebook()

LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)

LDAvis_prepared
# Listing topics of each paper



# listing the topic's probability for each paper

t = []

for text in data_lemmatized:

    bow = id2word.doc2bow(text)

    t.append(lda_model.get_document_topics(bow))

    

parsing_t = []

for topics in t :

    topic = [a[0] for a in topics]

    parsing_t.append(topic)

df['topic'] = parsing_t  



# creating dictionary of topics for each title

list_title = list(df['title'])

list_topic = list(df['topic'])

topic_dict = {}

for i in range(len(list_title)) :

    topic_dict.update(dict({list_title[i] : list_topic[i]}))
# Find the similar word by comparing Euclidian Distance between transformed word embeddings to PCA



keywords = ["coordination and communication during pandemic","resource distribution strategies","infrastructure for pandemic","institutional collaboration","government strategy","facemask distribution","hospital utilization", 

            "npi assessment","intervention effectiveness during outbreak","execution of planning","isolation implementation","school closure","travel ban","mass gathering effect","social distancing result","lockdown effectivity","emigration ban","testing of mass scale","facemask wearing", "nursing education","city quarantine","mathematical modelling",

            "web based surveillance","tracking of community source","digital technology","media effectivity on reporting","campaign strategy","influencing in social media","policy during pandemic", "public knowledge", "spread controlling",

            "economic strategy during pandemic","cost analysis","housing price","occupation rate","household price",

            "homeless compliance", "quarantine of city","mathematical modelling"]



title_dict = {}



for keyword in keywords:

    

    keyword_init = keyword.split()[0]

    print("Most similar words to the keyword '" + keyword_init + "' :")

    print("")

    

    # getting keyword's embeddings

    vec1 = []

    for val in word_vectors[keyword_init].values() :

        vec1.append(val)

    vec1 = np.array(tuple(vec1))

        

    distances = []

    words = []

    

    for word in word_vectors.keys() :

        if word != keyword_init :

            # getting compared word's embeddings

            vec2 = []

            for val in word_vectors[word].values() :

                vec2.append(val)

            vec2 = np.array(tuple(vec2))

        # calculating Euclidian Distance between two vector

        distances.append(np.linalg.norm(vec1 - vec2))

        words.append(word)

    

    # sorting the distance

    indices = np.argsort(distances)

    top_words = [words[j] for j in indices[0:7]]

    

    for idx, word in enumerate(top_words):

        print(str(idx+1) + ". " + word)

        print("")

    

    # making list of the paper related to the keywords

    

    top_words.insert(0,keyword)

    title_list = []

    abstract_dict = {}

    for word in top_words :

        paper_result = bm25.search(word)

        if word != keyword : 

            for i in range(len(paper_result)) :

                try : 

                    if (paper_result[i].title() not in title_list) & (len(set(topic_dict[paper_result[i].title()]) \

                                                                           & set(topic_keyword)) >= 2) :   # ensuring the topic is similar

                        title_list.append(paper_result[i].title())

                except :

                    abstract = paper_result[i].abstract()

                    topic = list(df[df['abstract'] == abstract]['topic'])[0]

                    if (paper_result[i].title() not in title_list) & (len(set(topic) \

                                                                           & set(topic_keyword)) >= 2) :   # ensuring the topic is similar

                        title_list.append(paper_result[i].title())

        else :

            topic_keyword = []

            for i in range(len(paper_result)) :

                try : 

                    title = paper_result[i].title()

                    topic = topic_dict[title]

                    for j in topic :

                        if j not in topic_keyword :

                            topic_keyword.append(j)

                    title_list.append(paper_result[i].title())

                except :

                    abstract = paper_result[i].abstract()

                    topic = list(df[df['abstract'] == abstract]['topic'])[0]

                    for j in topic :

                        if j not in topic_keyword :

                            topic_keyword.append(j)

                    title_list.append(paper_result[i].title())

                    

    title_list = title_list[:30]

                    

    # updating list of paper for every keyword

    

    title_dict.update(dict({keyword : title_list}))
from pytorch_transformers import (WEIGHTS_NAME, BertConfig,

                                  BertForQuestionAnswering, BertTokenizer)

from pytorch_transformers.tokenization_bert import (BasicTokenizer,

                                                    whitespace_tokenize)



class SquadExample(object):

    """

    A single training/test example for the Squad dataset.

    For examples without an answer, the start and end position are -1.

    """



    def __init__(self,

                 qas_id,

                 question_text,

                 doc_tokens,

                 orig_answer_text=None,

                 start_position=None,

                 end_position=None):

        self.qas_id = qas_id

        self.question_text = question_text

        self.doc_tokens = doc_tokens

        self.orig_answer_text = orig_answer_text

        self.start_position = start_position

        self.end_position = end_position



    def __str__(self):

        return self.__repr__()



    def __repr__(self):

        s = ""

        s += "qas_id: %s" % (self.qas_id)

        s += ", question_text: %s" % (

            self.question_text)

        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))

        if self.start_position:

            s += ", start_position: %d" % (self.start_position)

        if self.end_position:

            s += ", end_position: %d" % (self.end_position)

        return s



class InputFeatures(object):

    """A single set of features of data."""



    def __init__(self,

                 unique_id,

                 example_index,

                 doc_span_index,

                 tokens,

                 token_to_orig_map,

                 token_is_max_context,

                 input_ids,

                 input_mask,

                 segment_ids,

                 paragraph_len,

                 start_position=None,

                 end_position=None,):

        self.unique_id = unique_id

        self.example_index = example_index

        self.doc_span_index = doc_span_index

        self.tokens = tokens

        self.token_to_orig_map = token_to_orig_map

        self.token_is_max_context = token_is_max_context

        self.input_ids = input_ids

        self.input_mask = input_mask

        self.segment_ids = segment_ids

        self.paragraph_len = paragraph_len

        self.start_position = start_position

        self.end_position = end_position



def input_to_squad_example(passage, question):

    """Convert input passage and question into a SquadExample."""



    def is_whitespace(c):

        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:

            return True

        return False



    paragraph_text = passage

    doc_tokens = []

    char_to_word_offset = []

    prev_is_whitespace = True

    for c in paragraph_text:

        if is_whitespace(c):

            prev_is_whitespace = True

        else:

            if prev_is_whitespace:

                doc_tokens.append(c)

            else:

                doc_tokens[-1] += c

            prev_is_whitespace = False

        char_to_word_offset.append(len(doc_tokens) - 1)



    qas_id = 0

    question_text = question

    start_position = None

    end_position = None

    orig_answer_text = None



    example = SquadExample(

        qas_id=qas_id,

        question_text=question_text,

        doc_tokens=doc_tokens,

        orig_answer_text=orig_answer_text,

        start_position=start_position,

        end_position=end_position)

                

    return example



def _check_is_max_context(doc_spans, cur_span_index, position):

    """Check if this is the 'max context' doc span for the token."""



    # Because of the sliding window approach taken to scoring documents, a single

    # token can appear in multiple documents. E.g.

    #  Doc: the man went to the store and bought a gallon of milk

    #  Span A: the man went to the

    #  Span B: to the store and bought

    #  Span C: and bought a gallon of

    #  ...

    #

    # Now the word 'bought' will have two scores from spans B and C. We only

    # want to consider the score with "maximum context", which we define as

    # the *minimum* of its left and right context (the *sum* of left and

    # right context will always be the same, of course).

    #

    # In the example the maximum context for 'bought' would be span C since

    # it has 1 left context and 3 right context, while span B has 4 left context

    # and 0 right context.

    best_score = None

    best_span_index = None

    for (span_index, doc_span) in enumerate(doc_spans):

        end = doc_span.start + doc_span.length - 1

        if position < doc_span.start:

            continue

        if position > end:

            continue

        num_left_context = position - doc_span.start

        num_right_context = end - position

        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length

        if best_score is None or score > best_score:

            best_score = score

            best_span_index = span_index



    return cur_span_index == best_span_index



def squad_examples_to_features(example, tokenizer, max_seq_length,

                                 doc_stride, max_query_length,cls_token_at_end=False,

                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,

                                 sequence_a_segment_id=0, sequence_b_segment_id=1,

                                 cls_token_segment_id=0, pad_token_segment_id=0,

                                 mask_padding_with_zero=True):

    """Loads a data file into a list of `InputBatch`s."""



    unique_id = 1000000000

    example_index = 0

    features = []

    query_tokens = tokenizer.tokenize(example.question_text)

    if len(query_tokens) > max_query_length:

        query_tokens = query_tokens[0:max_query_length]



    tok_to_orig_index = []

    orig_to_tok_index = []

    all_doc_tokens = []

    for (i, token) in enumerate(example.doc_tokens):

        orig_to_tok_index.append(len(all_doc_tokens))

        sub_tokens = tokenizer.tokenize(token)

        for sub_token in sub_tokens:

            tok_to_orig_index.append(i)

            all_doc_tokens.append(sub_token)



    # The -3 accounts for [CLS], [SEP] and [SEP]

    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3



    # We can have documents that are longer than the maximum sequence length.

    # To deal with this we do a sliding window approach, where we take chunks

    # of the up to our max length with a stride of `doc_stride`.

    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name

        "DocSpan", ["start", "length"])

    doc_spans = []

    start_offset = 0

    while start_offset < len(all_doc_tokens):

        length = len(all_doc_tokens) - start_offset

        if length > max_tokens_for_doc:

            length = max_tokens_for_doc

        doc_spans.append(_DocSpan(start=start_offset, length=length))

        if start_offset + length == len(all_doc_tokens):

            break

        start_offset += min(length, doc_stride)



    for (doc_span_index, doc_span) in enumerate(doc_spans):

        tokens = []

        token_to_orig_map = {}

        token_is_max_context = {}

        segment_ids = []



        # CLS token at the beginning

        if not cls_token_at_end:

            tokens.append(cls_token)

            segment_ids.append(cls_token_segment_id)



        # Query

        for token in query_tokens:

            tokens.append(token)

            segment_ids.append(sequence_a_segment_id)



        # SEP token

        tokens.append(sep_token)

        segment_ids.append(sequence_a_segment_id)



        # Paragraph

        for i in range(doc_span.length):

            split_token_index = doc_span.start + i

            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]



            is_max_context = _check_is_max_context(doc_spans, doc_span_index,

                                                    split_token_index)

            token_is_max_context[len(tokens)] = is_max_context

            tokens.append(all_doc_tokens[split_token_index])

            segment_ids.append(sequence_b_segment_id)

        paragraph_len = doc_span.length



        # SEP token

        tokens.append(sep_token)

        segment_ids.append(sequence_b_segment_id)



        # CLS token at the end

        if cls_token_at_end:

            tokens.append(cls_token)

            segment_ids.append(cls_token_segment_id)



        input_ids = tokenizer.convert_tokens_to_ids(tokens)



        # The mask has 1 for real tokens and 0 for padding tokens. Only real

        # tokens are attended to.

        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)



        # Zero-pad up to the sequence length.

        while len(input_ids) < max_seq_length:

            input_ids.append(pad_token)

            input_mask.append(0 if mask_padding_with_zero else 1)

            segment_ids.append(pad_token_segment_id)



        assert len(input_ids) == max_seq_length

        assert len(input_mask) == max_seq_length

        assert len(segment_ids) == max_seq_length



        start_position = None

        end_position = None



        features.append(

            InputFeatures(

                unique_id=unique_id,

                example_index=example_index,

                doc_span_index=doc_span_index,

                tokens=tokens,

                token_to_orig_map=token_to_orig_map,

                token_is_max_context=token_is_max_context,

                input_ids=input_ids,

                input_mask=input_mask,

                segment_ids=segment_ids,

                paragraph_len=paragraph_len,

                start_position=start_position,

                end_position=end_position))

        unique_id += 1



    return features



def to_list(tensor):

    return tensor.detach().cpu().tolist()



def _get_best_indexes(logits, n_best_size):

    """Get the n-best logits from a list."""

    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)



    best_indexes = []

    for i in range(len(index_and_score)):

        if i >= n_best_size:

            break

        best_indexes.append(index_and_score[i][0])

    return best_indexes



RawResult = collections.namedtuple("RawResult",["unique_id", "start_logits", "end_logits"])



def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):

    """Project the tokenized prediction back to the original text."""



    # When we created the data, we kept track of the alignment between original

    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So

    # now `orig_text` contains the span of our original text corresponding to the

    # span that we predicted.

    #

    # However, `orig_text` may contain extra characters that we don't want in

    # our prediction.

    #

    # For example, let's say:

    #   pred_text = steve smith

    #   orig_text = Steve Smith's

    #

    # We don't want to return `orig_text` because it contains the extra "'s".

    #

    # We don't want to return `pred_text` because it's already been normalized

    # (the SQuAD eval script also does punctuation stripping/lower casing but

    # our tokenizer does additional normalization like stripping accent

    # characters).

    #

    # What we really want to return is "Steve Smith".

    #

    # Therefore, we have to apply a semi-complicated alignment heuristic between

    # `pred_text` and `orig_text` to get a character-to-character alignment. This

    # can fail in certain cases in which case we just return `orig_text`.



    def _strip_spaces(text):

        ns_chars = []

        ns_to_s_map = collections.OrderedDict()

        for (i, c) in enumerate(text):

            if c == " ":

                continue

            ns_to_s_map[len(ns_chars)] = i

            ns_chars.append(c)

        ns_text = "".join(ns_chars)

        return (ns_text, ns_to_s_map)



    # We first tokenize `orig_text`, strip whitespace from the result

    # and `pred_text`, and check if they are the same length. If they are

    # NOT the same length, the heuristic has failed. If they are the same

    # length, we assume the characters are one-to-one aligned.

    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)



    tok_text = " ".join(tokenizer.tokenize(orig_text))



    start_position = tok_text.find(pred_text)

    if start_position == -1:

        return orig_text

    end_position = start_position + len(pred_text) - 1



    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)

    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)



    if len(orig_ns_text) != len(tok_ns_text):

        return orig_text



    # We then project the characters in `pred_text` back to `orig_text` using

    # the character-to-character alignment.

    tok_s_to_ns_map = {}

    for (i, tok_index) in tok_ns_to_s_map.items():

        tok_s_to_ns_map[tok_index] = i



    orig_start_position = None

    if start_position in tok_s_to_ns_map:

        ns_start_position = tok_s_to_ns_map[start_position]

        if ns_start_position in orig_ns_to_s_map:

            orig_start_position = orig_ns_to_s_map[ns_start_position]



    if orig_start_position is None:

        return orig_text



    orig_end_position = None

    if end_position in tok_s_to_ns_map:

        ns_end_position = tok_s_to_ns_map[end_position]

        if ns_end_position in orig_ns_to_s_map:

            orig_end_position = orig_ns_to_s_map[ns_end_position]



    if orig_end_position is None:

        return orig_text



    output_text = orig_text[orig_start_position:(orig_end_position + 1)]

    return output_text



def _compute_softmax(scores):

    """Compute softmax probability over raw logits."""

    if not scores:

        return []



    max_score = None

    for score in scores:

        if max_score is None or score > max_score:

            max_score = score



    exp_scores = []

    total_sum = 0.0

    for score in scores:

        x = math.exp(score - max_score)

        exp_scores.append(x)

        total_sum += x



    probs = []

    for score in exp_scores:

        probs.append(score / total_sum)

    return probs



def get_answer(example, features, all_results, n_best_size,

                max_answer_length, do_lower_case):

    example_index_to_features = collections.defaultdict(list)

    for feature in features:

        example_index_to_features[feature.example_index].append(feature)

    

    unique_id_to_result = {}

    for result in all_results:

        unique_id_to_result[result.unique_id] = result

    

    _PrelimPrediction = collections.namedtuple( "PrelimPrediction",["feature_index", "start_index", "end_index", "start_logit", "end_logit"])



    example_index = 0

    features = example_index_to_features[example_index]



    prelim_predictions = []



    for (feature_index, feature) in enumerate(features):

        result = unique_id_to_result[feature.unique_id]

        start_indexes = _get_best_indexes(result.start_logits, n_best_size)

        end_indexes = _get_best_indexes(result.end_logits, n_best_size)

        for start_index in start_indexes:

            for end_index in end_indexes:

                # We could hypothetically create invalid predictions, e.g., predict

                # that the start of the span is in the question. We throw out all

                # invalid predictions.

                if start_index >= len(feature.tokens):

                    continue

                if end_index >= len(feature.tokens):

                    continue

                if start_index not in feature.token_to_orig_map:

                    continue

                if end_index not in feature.token_to_orig_map:

                    continue

                if not feature.token_is_max_context.get(start_index, False):

                    continue

                if end_index < start_index:

                    continue

                length = end_index - start_index + 1

                if length > max_answer_length:

                    continue

                prelim_predictions.append(

                    _PrelimPrediction(

                        feature_index=feature_index,

                        start_index=start_index,

                        end_index=end_index,

                        start_logit=result.start_logits[start_index],

                        end_logit=result.end_logits[end_index]))

    prelim_predictions = sorted(prelim_predictions,key=lambda x: (x.start_logit + x.end_logit),reverse=True)

    _NbestPrediction = collections.namedtuple("NbestPrediction",

                        ["text", "start_logit", "end_logit","start_index","end_index"])

    seen_predictions = {}

    nbest = []

    for pred in prelim_predictions:

        if len(nbest) >= n_best_size:

            break

        feature = features[pred.feature_index]

        orig_doc_start = -1

        orig_doc_end = -1

        if pred.start_index > 0:  # this is a non-null prediction

            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]

            orig_doc_start = feature.token_to_orig_map[pred.start_index]

            orig_doc_end = feature.token_to_orig_map[pred.end_index]

            orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]

            tok_text = " ".join(tok_tokens)



            # De-tokenize WordPieces that have been split off.

            tok_text = tok_text.replace(" ##", "")

            tok_text = tok_text.replace("##", "")



            # Clean whitespace

            tok_text = tok_text.strip()

            tok_text = " ".join(tok_text.split())

            orig_text = " ".join(orig_tokens)



            final_text = get_final_text(tok_text, orig_text,do_lower_case)

            if final_text in seen_predictions:

                continue



            seen_predictions[final_text] = True

        else:

            final_text = ""

            seen_predictions[final_text] = True



        nbest.append(

            _NbestPrediction(

                text=final_text,

                start_logit=pred.start_logit,

                end_logit=pred.end_logit,

                start_index=orig_doc_start,

                end_index=orig_doc_end))



    if not nbest:

        nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0,start_index=-1,

                end_index=-1))



    assert len(nbest) >= 1



    total_scores = []

    for entry in nbest:

        total_scores.append(entry.start_logit + entry.end_logit)



    probs = _compute_softmax(total_scores)

    

    answer = {"answer" : nbest[0].text,

               "start" : nbest[0].start_index,

               "end" : nbest[0].end_index,

               "confidence" : probs[0],

               "document" : example.doc_tokens

             }

    return answer



RawResult = collections.namedtuple("RawResult",

                                   ["unique_id", "start_logits", "end_logits"])



def makeBERTSQuADPrediction(model, document, question):

    input_ids = tokenizer.encode(question, document)

    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    sep_index = input_ids.index(tokenizer.sep_token_id)

    num_seg_a = sep_index + 1

    num_seg_b = len(input_ids) - num_seg_a

    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    assert len(segment_ids) == len(input_ids)

    n_ids = len(segment_ids)

    if n_ids < 512:

        start_scores, end_scores = model(torch.tensor([input_ids]), 

                                 token_type_ids=torch.tensor([segment_ids]))

    else:

        #this cuts off the text if its more than 512 words so it fits in model space

        #need run multiple inferences for longer text. add to the todo

        start_scores, end_scores = model(torch.tensor([input_ids[:512]]), 

                                 token_type_ids=torch.tensor([segment_ids[:512]]))

    answer_start = torch.argmax(start_scores)

    answer_end = torch.argmax(end_scores)

    answer = tokens[answer_start]

    



    for i in range(answer_start + 1, answer_end + 1):

        if tokens[i][0:2] == '##':

            answer += tokens[i][2:]

        else:

            answer += ' ' + tokens[i]

            

    full_txt = ''

    

    for t in tokens:

        if t[0:2] == '##':

            full_txt += t[2:]

        else:

            full_txt += ' ' + t

            

    abs_returned = full_txt.split('[SEP] ')[1]

            

    ans={}

    ans['answer'] = answer

    #print(answer)

    if answer.startswith('[CLS]') or answer.endswith('[SEP]'):

        ans['confidence'] = -1.0

    else:

        confidence = torch.max(start_scores) + torch.max(end_scores)

        confidence = np.log(confidence.item())

        ans['confidence'] = confidence/(1.0+confidence)

    ans['start'] = answer_start

    ans['end'] = answer_end

    ans['abstract_bert'] = abs_returned

    return ans



def searchAbstracts(data, model, question):

    abstractResults = {}

    for i in tqdm(range(len(df_choosen))):  

        abstract = data.iloc[i]['abstract']

        emptyToken = -1

        if abstract:

            ans = makeBERTSQuADPrediction(model, abstract, question)

            confidence = ans['confidence']

            abstractResults[confidence]={}

            abstractResults[confidence]['answer'] = ans['answer']

            abstractResults[confidence]['start'] = ans['start']

            abstractResults[confidence]['end'] = ans['end']

            abstractResults[confidence]['idx'] = i

        else:

            abstractResults[emptyToken]={}

            abstractResults[emptyToken]['answer'] = []

            abstractResults[emptyToken]['start'] = []

            abstractResults[emptyToken]['end'] = []

            abstractResults[confidence]['idx'] = i

            emptyToken -= 1

    return abstractResults



from IPython.core.display import display, HTML



def displayResults(df_choosen, answers, question):

    display(HTML('<div style="font-family: Times New Roman; font-size: 20px; padding-bottom:7px ; line-height: 1.5"><b>Question</b>: '+question+'</div>')) 

    display(HTML('<div style="font-family: Times New Roman; font-size: 16px; padding-bottom:7px ;  line-height: 1.5"><b>Answer</b>:</div>'))

    

    confidence = list(answers.keys())

    confidence.sort(reverse=True)

    

    HTML_list_text = '<div style="font-family: Times New Roman; font-size: 14px; padding-bottom:7px ;  line-height:1.5"> <ul style="list-style-type:disc;">' 

    

    list_id = []

    for i,c in enumerate(confidence):

        if (i < 8) & (c > 0):

            if 'idx' not in  answers[c]:

                continue

            list_id.append(i)

            bert_ans = answers[c]['answer']

            HTML_list_text += '<li>'+bert_ans+'</li>'

    HTML_list_text += '</ul> </div>'

            

    display(HTML(HTML_list_text))

            

    for i,c in enumerate(confidence):

        if (i in list_id):

            idx = answers[c]['idx']

            title = df_choosen.iloc[idx]['title']

            authors = df_choosen.iloc[idx]['authors']



            display(HTML('<div style="font-family: Times New Roman; font-size: 14px; padding-bottom:7px ; line-height: 1.5"><b>Author</b>: '+

                         F'{authors}' + '<br>' + '<br>' + '<b>Title</b>: ' + '<br>' + F'{title}. ' + '</div>'))



            full_abs = df_choosen.iloc[idx]['abstract']

            bert_ans = answers[c]['answer']

            display(HTML('<div style="font-family: Times New Roman; font-size: 13px; padding-bottom:7px ; line-height: 1.5"><b>Answer</b>: '

                         +" <font color='red'>"+bert_ans+"</font> "+'</div>'))

            display(HTML('<div style="font-family: Times New Roman; font-size: 13px; padding-bottom:7px ; line-height: 1.5"><b>Abstract</b>: '

                         +full_abs+'</div>'))
query = "What is the best option of coordination and communication during pandemic strategy?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)
query = "What is better and optimum resource distribution strategies during pandemic?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)
query = "How is the infrastructure for pandemic preparation?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)
query = "How is the optimum institutional collaboration to prepare for pandemics?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)
query = "What is effective government strategy during pandemic?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)
query = "What is the better strategy to optimize facemask distribution during pandemic?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)
query = "What is the suggestion to manage hospital utilization during outbreak?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)
query = "What is the method to do npi assessment?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)
query = "How to optmize intervention effectiveness during outbreak?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)
query = "What is the best execution of planning during pandemic?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)
query = "Is isolation implementation effective to prevent outbreak?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)
query = "How effective is school closure during pandemic?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)
query = "How effective is travel ban during pandemic?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)
query = "How significant is mass gathering effect during pandemic?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)
query = "How is social distancing result in reducing transmission during pandemic?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)
query = "How is the lockdown effectivity in reducing transmission rate during pandemic?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)
query = "How is the emigration ban effectivity in reducing transmission rate during pandemic?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)
query = " How effective is testing of mass scale during pandemic?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)
query = "How effective is proper facemask wearing during pandemic?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)
query = "How effective is web based surveillance to reduce transmission during pandemic?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)
query = "How is the effectivity of self nursing education during pandemic?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)
query = "How to do tracking of community source during pandemic?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)
query = "What is the role of digital technology during pandamic?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)
query = "How is media effectivity on reporting during pandemic?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)
query = "How to optimize campaign strategy during pandemic?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)
query = "What is the effect of influencing in social media during pandemic?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)
query = "How to optimize policy during pandemic to reduce transmission?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)
query = "How to optimize public knowledge during pandemic?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)
query = "What is the optimum method to do spread controlling during pandemic?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)
query = "How to optimize the economic strategy during pandemic?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)
query = "How much cost during outbreak from cost analysis?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)
query = "What is the impact of pandemic on housing price?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)
query = "What is the effect of pandemic on household price?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)
query = "What causes citizen fail to comply the law during pandemic?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)
query = "How is the homeless compliance during pandemic?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)
query = "How is the effectivity of quarantine of city during pandemic?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)
query = "How mathematical modelling helps during pandemic?"

for i in keywords : 

    if i in query :

        title_choosen = title_dict[i]

        df_choosen = df[df['title'].isin(title_choosen)]

        break

answers = searchAbstracts(df_choosen, model, query)

displayResults(df_choosen, answers, query)