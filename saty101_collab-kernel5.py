import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
df_train = pd.read_csv("../input/learn-ai-bbc/BBC News Train.csv")
df_train['category_id'] = df_train['Category'].factorize()[0]
df_train.groupby('Category').category_id.count()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df_train.Text).toarray()
labels = df_train.category_id
category_to_id = {'business':0, 'tech':1, 'politics':2, 'sport':3, 'entertainment':4}
id_to_category = {0: 'business', 1: 'tech', 2: 'politics', 3: 'sport', 4: 'entertainment'}

from sklearn.model_selection import train_test_split

model = RandomForestClassifier()

#Split Data 
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df_train.index, test_size=0.33, random_state=0)

#Train Algorithm
model.fit(X_train, y_train)

# Make Predictions
y_pred_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)
category_id_df = df_train[['Category', 'category_id']].drop_duplicates().sort_values('category_id')
model2 = RandomForestClassifier()
model2.fit(features, labels)
text = 'worldcom ex-boss launches defence lawyers defending former worldcom chief bernie ebbers against a battery of fraud charges have called a company whistleblower as their first witness.  cynthia cooper  worldcom s ex-head of internal accounting  alerted directors to irregular accounting practices at the us telecoms giant in 2002. her warnings led to the collapse of the firm following the discovery of an $11bn (£5.7bn) accounting fraud. mr ebbers has pleaded not guilty to charges of fraud and conspiracy.  prosecution lawyers have argued that mr ebbers orchestrated a series of accounting tricks at worldcom  ordering employees to hide expenses and inflate revenues to meet wall street earnings estimates. but ms cooper  who now runs her own consulting business  told a jury in new york on wednesday that external auditors arthur andersen had approved worldcom s accounting in early 2001 and 2002. she said andersen had given a  green light  to the procedures and practices used by worldcom. mr ebber s lawyers have said he was unaware of the fraud  arguing that auditors did not alert him to any problems.  ms cooper also said that during shareholder meetings mr ebbers often passed over technical questions to the company s finance chief  giving only  brief  answers himself. the prosecution s star witness  former worldcom financial chief scott sullivan  has said that mr ebbers ordered accounting adjustments at the firm  telling him to  hit our books . however  ms cooper said mr sullivan had not mentioned  anything uncomfortable  about worldcom s accounting during a 2001 audit committee meeting. mr ebbers could face a jail sentence of 85 years if convicted of all the charges he is facing. worldcom emerged from bankruptcy protection in 2004  and is now known as mci. last week  mci agreed to a buyout by verizon communications in a deal valued at $6.75bn.'
text
test_features = tfidf.transform([text])
prediction = model.predict(test_features)
id_to_category = {0: 'business', 1: 'tech', 2: 'politics', 3: 'sport', 4: 'entertainment'}
for i in range(len(prediction)):
    print(id_to_category[prediction[i]])
from sklearn import preprocessing
import math
import spacy
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import re
from bs4 import BeautifulSoup
import unicodedata
nltk.download('all', halt_on_error=False)
from nltk.corpus import sentiwordnet as swn
nlp = spacy.load('en', parse = False, tag=False, entity=False)
tokenizer = ToktokTokenizer()

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def remove_special_characters(text):
    text = re.sub(r'[^a-zA-z0-9\s]', '', text)
    return text

def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')

def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text
CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text
def normalize_corpus(corpus, html_stripping=True, contraction_expansion=True,
                     accented_char_removal=True, text_lower_case=True, 
                     text_lemmatization=True, special_char_removal=True, 
                     stopword_removal=True):
    
    normalized_corpus = []
    for doc in corpus:        
        if accented_char_removal:
            doc = remove_accented_chars(doc)            
        if contraction_expansion:
            doc = expand_contractions(doc)           
        if text_lower_case:
            doc = doc.lower()        
        doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)          
        special_char_pattern = re.compile(r'([{.(-)!}])')
        doc = special_char_pattern.sub(" \\1 ", doc)        
        if text_lemmatization:
            doc = lemmatize_text(doc)          
        if special_char_removal:
            doc = remove_special_characters(doc)        
        doc = re.sub(' +', ' ', doc)        
        if stopword_removal:
            doc = remove_stopwords(doc, is_lower_case=text_lower_case)            
        normalized_corpus.append(doc)        
    return normalized_corpus
nptext = np.array([text])
norm_articles = normalize_corpus(nptext)
def analyze_sentiment_sentiwordnet_lexicon(article):
    tagged_text = [(token.text, token.tag_) for token in nlp(article)]
    pos_score = neg_score = token_count = obj_score = 0
    for word, tag in tagged_text:
        ss_set = None
        if 'NN' in tag and list(swn.senti_synsets(word, 'n')):
            ss_set = list(swn.senti_synsets(word, 'n'))[0]
#             print(ss_set)
        elif 'VB' in tag and list(swn.senti_synsets(word, 'v')):
            ss_set = list(swn.senti_synsets(word, 'v'))[0]
        elif 'JJ' in tag and list(swn.senti_synsets(word, 'a')):
            ss_set = list(swn.senti_synsets(word, 'a'))[0]
#             ss_set.pos_score()*= 3
        elif 'RB' in tag and list(swn.senti_synsets(word, 'r')):
            ss_set = list(swn.senti_synsets(word, 'r'))[0]
        if ss_set:
            if 'JJ' in tag:
                pos_score += ss_set.pos_score()*4
                neg_score += ss_set.neg_score()*4
#                 token_count +=3
            elif 'RB' in tag:
                pos_score += ss_set.pos_score()*3
                neg_score += ss_set.neg_score()*3
#                 token_count +=2
            elif 'VB' in tag:
                pos_score += ss_set.pos_score()*2
                neg_score += ss_set.neg_score()*2
            else:
                pos_score += ss_set.pos_score()
                neg_score +=ss_set.neg_score()
                token_count += 1
            obj_score += ss_set.obj_score()
            token_count +=1
    final_score = pos_score - neg_score
#     print(str(token_count)+'words which are being considered as tokens')
    norm_final_score = round(float(final_score) / token_count, 4)
#     final_sentiment = math.exp(final_sentiment)
#     final_sentiment = math.log(final_score)
#     final_sentiment = 'positive' if norm_final_score >= 0.05 else 'negative'
#     norm_final_score = (20 * (norm_final_score)/2) - 10
    norm_final_score = math.exp(norm_final_score)
    norm_final_score = (20 * (norm_final_score)/2.35) - 10
    if norm_final_score>=1:
        final_sentiment = 'positive'
    elif norm_final_score<=-1:
        final_sentiment = 'negative'
    else:
        final_sentiment = 'neutral'
    if norm_final_score>10:
        norm_final_score = 10
    if norm_final_score<-10:
        norm_final_score = -10
    #################  NOTE  ##########################
    # Please change this if statement to suit the needs because I've only done it using an arbitrary 
    #statement which might be wrong
    #So please change the if statement
    #Please please see if we can do it for a specific named entity 
#     (20*math.exp(norm_final_score)/(2.35)) - 10
#     exp_sentiment = math.exp(norm_final_score)
#     minmax_scale = preprocessing.MinMaxScaler(feature_range=(-10, 10))
#     scaled_feature = minmax_scale.fit_transform(exp_sentiment)
    return final_sentiment, (norm_final_score), final_score
predicted_sentiments = [analyze_sentiment_sentiwordnet_lexicon(article) for article in norm_articles]
print('predicted sentiment score:'+ str(predicted_sentiments[0][1]))
print('predicted sentiment:' + str(predicted_sentiments[0][0]))
# print('predicted sentiment scaled feature:' + str(predicted_sentiments[0][3]))
import warnings
warnings.filterwarnings("ignore")
import spacy
nlp = spacy.load("en_core_web_sm")
from collections import defaultdict, OrderedDict
from math import sqrt
from operator import itemgetter
from spacy.tokens import Doc
import graphviz
import json
import logging
import networkx as nx
import os
import os.path
import re
import string
import sys
import time
import unicodedata
# doc = nlp(text)

######################################################################
## utility functions
######################################################################

PAT_FORWARD = re.compile("\n\-+ Forwarded message \-+\n")
PAT_REPLIED = re.compile("\nOn.*\d+.*\n?wrote\:\n+\>")
PAT_UNSUBSC = re.compile("\n\-+\nTo unsubscribe,.*\nFor additional commands,.*")


def split_grafs (lines):
    """
    segment raw text, given as a list of lines, into paragraphs
    """
    graf = []

    for line in lines:
        line = line.strip()

        if len(line) < 1:
            if len(graf) > 0:
                yield "\n".join(graf)
                graf = []
        else:
            graf.append(line)

    if len(graf) > 0:
        yield "\n".join(graf)


def filter_quotes (text, is_email=True):
    """
    filter the quoted text out of a message
    """
    global PAT_FORWARD, PAT_REPLIED, PAT_UNSUBSC

    if is_email:
        text = filter(lambda x: x in string.printable, text)

        # strip off quoted text in a forward
        m = PAT_FORWARD.split(text, re.M)

        if m and len(m) > 1:
            text = m[0]

        # strip off quoted text in a reply
        m = PAT_REPLIED.split(text, re.M)

        if m and len(m) > 1:
            text = m[0]

        # strip off any trailing unsubscription notice
        m = PAT_UNSUBSC.split(text, re.M)

        if m:
            text = m[0]

    # replace any remaining quoted text with blank lines
    lines = []

    for line in text.split("\n"):
        if line.startswith(">"):
            lines.append("")
        else:
            lines.append(line)

    return list(split_grafs(lines))


def maniacal_scrubber (text):
    """
    it scrubs the garble from its stream...
    or it gets the debugger again
    """
    x = " ".join(map(lambda s: s.strip(), text.split("\n"))).strip()

    x = x.replace('“', '"').replace('”', '"')
    x = x.replace("‘", "'").replace("’", "'").replace("`", "'")
    x = x.replace("…", "...").replace("–", "-")

    x = str(unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("utf-8"))

    # some web content returns "not string" ?? ostensibly no longer
    # possibl in Py 3.x but crazy "mixed modes" of character encodings
    # have been found in the wild -- YMMV

    try:
        assert type(x).__name__ == "str"
    except AssertionError:
        print("not a string?", type(line), line)

    return x


def default_scrubber (text):
    """
    remove spurious punctuation (for English)
    """
    return text.lower().replace("'", "")


######################################################################
## class definitions
######################################################################

class CollectedPhrase:
    """
    represents one phrase during the collection process
    """

    def __init__ (self, chunk, scrubber):
        self.sq_sum_rank = 0.0
        self.non_lemma = 0
        
        self.chunk = chunk
        self.text = scrubber(chunk.text)


    def __repr__ (self):
        return "{:.4f} ({},{}) {} {}".format(
            self.rank, self.chunk.start, self.chunk.end, self.text, self.key
        )


    def range (self):
        """
        generate the index range for the span of tokens in this phrase
        """
        return range(self.chunk.start, self.chunk.end)


    def set_key (self, compound_key):
        """
        create a unique key for the the phrase based on its lemma components
        """
        self.key = tuple(sorted(list(compound_key)))


    def calc_rank (self):
        """
        since noun chunking is greedy, we normalize the rank values
        using a point estimate based on the number of non-lemma
        tokens within the phrase
        """
        chunk_len = self.chunk.end - self.chunk.start + 1
        non_lemma_discount = chunk_len / (chunk_len + (2.0 * self.non_lemma) + 1.0)

        # normalize the contributions of all the kept lemma tokens
        # within the phrase using root mean square (RMS)

        self.rank = sqrt(self.sq_sum_rank / (chunk_len + self.non_lemma)) * non_lemma_discount


class Phrase:
    """
    represents one extracted phrase
    """

    def __init__ (self, text, rank, count, phrase_list):
        self.text = text
        self.rank = rank
        self.count = count
        self.chunks = [p.chunk for p in phrase_list]


    def __repr__ (self):
        return self.text


class TextRank:
    """
    Python impl of TextRank by Milhacea, et al., as a spaCy extension,
    used to extract the top-ranked phrases from a text document
    """
    _EDGE_WEIGHT = 1.0
    _POS_KEPT = ["ADJ", "NOUN", "PROPN", "VERB"]
    _TOKEN_LOOKBACK = 3
    

    def __init__ (
            self,
            edge_weight=_EDGE_WEIGHT,
            logger=None,
            pos_kept=_POS_KEPT,
            scrubber=default_scrubber,
            token_lookback=_TOKEN_LOOKBACK
    ):
        self.edge_weight = edge_weight
        self.logger = logger
        self.pos_kept = pos_kept
        self.scrubber = scrubber
        self.stopwords = defaultdict(list)
        self.token_lookback = token_lookback

        self.doc = None
        self.reset()


    def reset (self):
        """
        initialize the data structures needed for extracting phrases
        removing any state
        """
        self.elapsed_time = 0.0
        self.lemma_graph = nx.Graph()
        self.phrases = defaultdict(list)
        self.ranks = {}
        self.seen_lemma = OrderedDict()


    def load_stopwords (self, path="stop.json"):
        """
        load a list of "stop words" that get ignored when constructing
        the lemma graph -- NB: be cautious when using this feature
        """
        stop_path = None

        # check if the path is fully qualified, or if the file is in
        # the current working directory

        if os.path.isfile(path):
            stop_path = path
        else:
            cwd = os.getcwd()
            stop_path = os.path.join(cwd, path)

            if not os.path.isfile(stop_path):
                loc = os.path.realpath(os.path.join(cwd, os.path.dirname(__file__)))
                stop_path = os.path.join(loc, path)

        try:
            with open(stop_path, "r") as f:
                data = json.load(f)

                for lemma, pos_list in data.items():
                    self.stopwords[lemma] = pos_list
        except FileNotFoundError:
            pass


    def increment_edge (self, node0, node1):
        """
        increment the weight for an edge between the two given nodes,
        creating the edge first if needed
        """
        if self.logger:
            self.logger.debug("link {} {}".format(node0, node1))
    
        if self.lemma_graph.has_edge(node0, node1):
            self.lemma_graph[node0][node1]["weight"] += self.edge_weight
        else:
            self.lemma_graph.add_edge(node0, node1, weight=self.edge_weight)


    def link_sentence (self, sent):
        """
        link nodes and edges into the lemma graph for one parsed sentence
        """
        visited_tokens = []
        visited_nodes = []

        for i in range(sent.start, sent.end):
            token = self.doc[i]

            if token.pos_ in self.pos_kept:
                # skip any stop words...
                lemma = token.lemma_.lower().strip()

                if lemma in self.stopwords and token.pos_ in self.stopwords[lemma]:
                    continue

                # ...otherwise proceed
                key = (token.lemma_, token.pos_)

                if key not in self.seen_lemma:
                    self.seen_lemma[key] = set([token.i])
                else:
                    self.seen_lemma[key].add(token.i)

                node_id = list(self.seen_lemma.keys()).index(key)

                if not node_id in self.lemma_graph:
                    self.lemma_graph.add_node(node_id)

                if self.logger:
                    self.logger.debug("visit {} {}".format(
                        visited_tokens, visited_nodes
                    ))
                    self.logger.debug("range {}".format(
                        list(range(len(visited_tokens) - 1, -1, -1))
                    ))
            
                for prev_token in range(len(visited_tokens) - 1, -1, -1):
                    if self.logger:
                        self.logger.debug("prev_tok {} {}".format(
                            prev_token, (token.i - visited_tokens[prev_token])
                        ))
                
                    if (token.i - visited_tokens[prev_token]) <= self.token_lookback:
                        self.increment_edge(node_id, visited_nodes[prev_token])
                    else:
                        break

                if self.logger:
                    self.logger.debug(" -- {} {} {} {} {} {}".format(
                        token.i, token.text, token.lemma_, token.pos_, visited_tokens, visited_nodes
                    ))

                visited_tokens.append(token.i)
                visited_nodes.append(node_id)


    def collect_phrases (self, chunk):
        """
        collect instances of phrases from the lemma graph
        based on the given chunk
        """
        phrase = CollectedPhrase(chunk, self.scrubber)
        compound_key = set([])

        for i in phrase.range():
            token = self.doc[i]
            key = (token.lemma_, token.pos_)
        
            if key in self.seen_lemma:
                node_id = list(self.seen_lemma.keys()).index(key)
                rank = self.ranks[node_id]
                phrase.sq_sum_rank += rank
                compound_key.add(key)
        
                if self.logger:
                    self.logger.debug(" {} {} {} {}".format(
                        token.lemma_, token.pos_, node_id, rank
                    ))
            else:
                phrase.non_lemma += 1
    
        phrase.set_key(compound_key)
        phrase.calc_rank()

        self.phrases[phrase.key].append(phrase)

        if self.logger:
            self.logger.debug(phrase)


    def calc_textrank (self):
        """
        iterate through each sentence in the doc, constructing a lemma graph
        then returning the top-ranked phrases
        """
        self.reset()
        t0 = time.time()

        for sent in self.doc.sents:
            self.link_sentence(sent)

        if self.logger:
            self.logger.debug(self.seen_lemma)

        # to run the algorithm, we use PageRank – i.e., approximating
        # eigenvalue centrality – to calculate ranks for each of the
        # nodes in the lemma graph

        self.ranks = nx.pagerank(self.lemma_graph)

        # collect the top-ranked phrases based on both the noun chunks
        # and the named entities

        for chunk in self.doc.noun_chunks:
            self.collect_phrases(chunk)

        for ent in self.doc.ents:
            self.collect_phrases(ent)

        # since noun chunks can be expressed in different ways (e.g., may
        # have articles or prepositions), we need to find a minimum span
        # for each phrase based on combinations of lemmas

        min_phrases = {}

        for phrase_key, phrase_list in self.phrases.items():
            phrase_list.sort(key=lambda p: p.rank, reverse=True)
            best_phrase = phrase_list[0]
            min_phrases[best_phrase.text] = (best_phrase.rank, len(phrase_list), phrase_key)

        # yield results

        results = sorted(min_phrases.items(), key=lambda x: x[1][0], reverse=True)

        phrase_list = [
            Phrase(p, r, c, self.phrases[k]) for p, (r, c, k) in results
        ]

        t1 = time.time()
        self.elapsed_time = (t1 - t0) * 1000.0

        return phrase_list


    def write_dot (self, path="graph.dot"):
        """
        output the lemma graph in Dot file format
        """
        keys = list(self.seen_lemma.keys())
        dot = graphviz.Digraph()

        for node_id in self.lemma_graph.nodes():
            text = keys[node_id][0].lower()
            rank = self.ranks[node_id]
            label = "{} ({:.4f})".format(text, rank)
            dot.node(str(node_id), label)

        for edge in self.lemma_graph.edges():
            dot.edge(str(edge[0]), str(edge[1]), constraint="false")

        with open(path, "w") as f:
            f.write(dot.source)


    def summary (self, limit_phrases=10, limit_sentences=4):
        """
        run extractive summarization, based on vector distance 
        per sentence from the top-ranked phrases
        """
        unit_vector = []

        # construct a list of sentence boundaries with a phrase set
        # for each (initialized to empty)

        sent_bounds = [ [s.start, s.end, set([])] for s in self.doc.sents ]

        # iterate through the top-ranked phrases, added them to the
        # phrase vector for each sentence

        phrase_id = 0

        for p in self.doc._.phrases:
            unit_vector.append(p.rank)

            if self.logger:
                self.logger.debug(
                    "{} {} {}".format(phrase_id, p.text, p.rank)
                )
    
            for chunk in p.chunks:
                for sent_start, sent_end, sent_vector in sent_bounds:
                    if chunk.start >= sent_start and chunk.start <= sent_end:
                        sent_vector.add(phrase_id)

                        if self.logger:
                            self.logger.debug(
                                " {} {} {} {}".format(sent_start, chunk.start, chunk.end, sent_end)
                                )

                        break

            phrase_id += 1

            if phrase_id == limit_phrases:
                break

        # construct a unit_vector for the top-ranked phrases, up to
        # the requested limit

        sum_ranks = sum(unit_vector)
        unit_vector = [ rank/sum_ranks for rank in unit_vector ]

        # iterate through each sentence, calculating its euclidean
        # distance from the unit vector

        sent_rank = {}
        sent_id = 0

        for sent_start, sent_end, sent_vector in sent_bounds:
            sum_sq = 0.0
    
            for phrase_id in range(len(unit_vector)):
                if phrase_id not in sent_vector:
                    sum_sq += unit_vector[phrase_id]**2.0

            sent_rank[sent_id] = sqrt(sum_sq)
            sent_id += 1

        # extract the sentences with the lowest distance

        sent_text = {}
        sent_id = 0

        for sent in self.doc.sents:
            sent_text[sent_id] = sent
            sent_id += 1

        # yield results, up to the limit requested

        num_sent = 0

        for sent_id, rank in sorted(sent_rank.items(), key=itemgetter(1)):
            yield sent_text[sent_id]
            num_sent += 1

            if num_sent == limit_sentences:
                break


    def PipelineComponent (self, doc):
        """
        define a custom pipeline component for spaCy and extend the
        Doc class to add TextRank
        """
        self.doc = doc
        Doc.set_extension("phrases", force=True, default=[])
        Doc.set_extension("textrank", force=True, default=self)
        doc._.phrases = self.calc_textrank()

        return doc

length=3
tr = TextRank()
nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)
doc = nlp(text)
sent_bounds = [ [s.start, s.end, set([])] for s in doc.sents ]
limit_phrases = 4

phrase_id = 0
unit_vector = []

for p in doc._.phrases:
#     print(phrase_id, p.text, p.rank)
    
    unit_vector.append(p.rank)
    
    for chunk in p.chunks:
#         print(" ", chunk.start, chunk.end)
        
        for sent_start, sent_end, sent_vector in sent_bounds:
            if chunk.start >= sent_start and chunk.start <= sent_end:
#                 print(" ", sent_start, chunk.start, chunk.end, sent_end)
                sent_vector.add(phrase_id)
                break

    phrase_id += 1

    if phrase_id == limit_phrases:
        break
sum_ranks = sum(unit_vector)
unit_vector = [ rank/sum_ranks for rank in unit_vector ]

from math import sqrt

sent_rank = {}
sent_id = 0

for sent_start, sent_end, sent_vector in sent_bounds:
#     print(sent_vector)
    sum_sq = 0.0
    
    for phrase_id in range(len(unit_vector)):
#         print(phrase_id, unit_vector[phrase_id])
        
        if phrase_id not in sent_vector:
            sum_sq += unit_vector[phrase_id]**2.0

    sent_rank[sent_id] = sqrt(sum_sq)
    sent_id += 1
from operator import itemgetter

sorted(sent_rank.items(), key=itemgetter(1)) 
limit_sentences = length

sent_text = {}
sent_id = 0

for sent in doc.sents:
    sent_text[sent_id] = sent.text
    sent_id += 1

num_sent = 0

for sent_id, rank in sorted(sent_rank.items(), key=itemgetter(1)):
    print(sent_text[sent_id])
    num_sent += 1
    
    if num_sent == limit_sentences:
        break
