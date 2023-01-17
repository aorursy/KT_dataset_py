import pandas as pd;
import numpy as np;
import scipy as sp;
import sklearn;
import sys;
from nltk.corpus import stopwords;
import nltk;
from gensim.models import ldamodel
import gensim.corpora
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer;
from sklearn.decomposition import NMF;
from sklearn.preprocessing import normalize;
import pickle;
import string
PUNCT_STRING = string.punctuation
from string import digits
import re
import time 
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
from pprint import pprint

from gensim.models import CoherenceModel
from gensim.models.hdpmodel import HdpModel
class TextClean:
    
    def __init__(self, stop_list=None, remove_punct=True, remove_nums=True,
                 remove_unames=True, remove_stop_words=True, lowercase=True,
                 langauge='english', extra_stop_list=None, do_lemma=True, do_stem=True,
                allowed_pos_tags=['N', 'J', 'V', 'R']):
        st  = stopwords.words(langauge)
        if stop_list is None:
            self.stop_list = st
        else:
            self.stop_list = stop_list
            
        if extra_stop_list:
            self.stop_list += extra_stop_list 
            
        self.remove_punct = remove_punct
        self.remove_nums = remove_nums
        self.remove_unames = remove_unames
        self.remove_stop_words = stopwords
        self.lowercase = lowercase
        self.langauge = langauge
        self.do_lemma = do_lemma
        self.do_stem = do_stem
        self.allowed_pos_tags = allowed_pos_tags

    def clean_text(self, sent):
        if self.lowercase:
            sent = str(sent).lower()
        
        if self.remove_unames:
            # Remove username, urls
            sent = re.sub("https?://(?:[-\w.\/.\?]|(?:%[\da-fA-F]{2}))+|@([A-Za-z0-9_]+)", " ", sent)
            
        sent_tokens = nltk.word_tokenize(sent)
        
        if self.allowed_pos_tags:
            sent_tokens = [w_t[0] for w_t in nltk.pos_tag(sent_tokens) if w_t[1][0] in self.allowed_pos_tags]
            
        if self.remove_stop_words:
            sent_tokens = [s for s in sent_tokens if s not in self.stop_list]
            sent = " ".join(sent_tokens)
                            
        if self.remove_punct:
            table = str.maketrans('', '', PUNCT_STRING)
            sent = sent.translate(table) 
            
        if self.remove_nums:
            remove_digits = str.maketrans('', '', digits)
            sent = sent.translate(remove_digits)
        
        sent = sent.strip()
        
        if self.do_lemma:
            sent_tokens = nltk.word_tokenize(sent)
            sent = " ".join([lemmatizer.lemmatize(w) for w in sent_tokens])
            
        if self.do_stem:
            sent_tokens = nltk.word_tokenize(sent)
            sent = " ".join([stemmer.stem(w) for w in sent_tokens])
        
        if self.remove_stop_words:
            sent_tokens = [s for s in sent_tokens if s not in self.stop_list]
            sent = " ".join(sent_tokens)
        
        return sent 
    
data = pd.read_csv('../input/unstructured-l0-nlp-hackathon/data.csv')
data.index = data['Id'].tolist()
try:
    del data['Id']
except:
    pass

text_clean_obj = TextClean(extra_stop_list=['s', 'nt', 'ca', "n't", "'s", "they", "us", "'ve", "said", "sunday", "monday", "tuesday",
                                "wednesday", "thursday",  "friday", "saturday", "would", "will", "from", "subject", "writes",
                                 "one", "two", "three", "four",  "five",  "six", "seven", "eight", "nine", "ten",
                                "first", "second", "third", "fourth",  "fifth", "sixth", "seventh",  "eighth", "ninth", "tenth",
                                "re", "to", "also",  "cur",  "email", "reply", "replyto", "de", "awd", "s0",  "la",  "en",
                                "dont", "article", 'youre', 'get',  'year', 'month', 'day',  'like', 'id', 'go', 'im', 
                                'may', 'could', 'line'],
                remove_punct=True, 
                remove_nums=False,
                remove_unames=True, 
                remove_stop_words=True,
                lowercase=True,
                langauge='english',
                do_lemma=True, 
                do_stem=True,
                allowed_pos_tags=None)
# Check text clean
text_clean_obj.clean_text("disney running diving they've ashvjh 5757$^%$^ @jvsj https://abc.com .,.,.,. does car's very doing")
def clip_from_line(text):
    if str(text).startswith('from'):
        result = " ".join(text.split(">")[1:])
        if len(result):
            return result
        return text
    else:
        return text
    
data['clean_text'] = data['text'].apply(clip_from_line)
data['clean_text'] = data['clean_text'].apply(lambda x: text_clean_obj.clean_text(x))
data['Tokenized'] = data['clean_text'].apply(lambda x: nltk.word_tokenize(x)) 
data['Tokenized'].head()
def cleaner_func(text):
    clean_text = clip_from_line(text)
    clean_text= text_clean_obj.clean_text(clean_text)
    

# data.index = data['Id'].tolist()
# del data['Id']
from sklearn.model_selection import train_test_split
trn_data, tst_data = train_test_split(data, test_size=20, random_state=1729)
trn_data.shape, tst_data.shape
class TopicModel:
    def __init__(self, method='lda', use_bi_gm=True, use_tri_gm=False, min_bi_gm_count=5, min_tri_count=3, bi_gm_thrs=10, tri_gm_thrs=10,
                n_topics=5, random_state=1729):
        self.use_bi_gm = use_bi_gm
        self.use_tri_gm = use_tri_gm
        self.min_bi_gm_count = min_bi_gm_count
        self.min_tri_count = min_tri_count
        self.bi_gm_thrs = bi_gm_thrs 
        self.tri_gm_thrs = tri_gm_thrs
        self.bigram_mod = None
        self.trigram_mod = None
        if self.use_tri_gm:
            self.n_gram = 3
        elif self.use_bi_gm:
            self.n_gram = 2
        else:
            self.n_gram = 1
        self.n_topics = n_topics    
        self.random_state = random_state
        self.topic_model = None
        self.coherence_model = None
        self.topic_label_dict = None
        self.method = method
        self.id2word = None
       
    
    def create_n_gm_models(self, tokenized_data):
        """
        Using training corpus train Bi-gram & Trigram models
        """
        
        if self.n_gram == 3:
            bigram = gensim.models.Phrases(tokenized_data,
                                           min_count=self.min_bi_gm_count,
                                           threshold=self.bi_gm_thrs) # higher threshold fewer phrases.
            bigram_mod = gensim.models.phrases.Phraser(bigram)
            self.bigram_mod = bigram_mod
            
            trigram = gensim.models.Phrases(bigram[tokenized_data],
                                            threshold=self.tri_gm_thrs)
            trigram_mod = gensim.models.phrases.Phraser(trigram)
            self.trigram_mod = trigram_mod
            return 
            
        if self.n_gram == 2:
            bigram = gensim.models.Phrases(tokenized_data,
                                           min_count=self.min_bi_gm_count,
                                           threshold=self.bi_gm_thrs) # higher threshold fewer phrases.
            bigram_mod = gensim.models.phrases.Phraser(bigram)
            self.bigram_mod = bigram_mod
            return             
        
            
    def get_n_gm_tokens(self, texts):
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        if self.n_gram==2:
            return [self.bigram_mod[doc] for doc in texts]
        elif self.n_gram==3:
            return [self.trigram_mod[self.bigram_mod[doc]] for doc in texts]
        elif self.n_gram==1:
            return texts
        else:
            raise Exception("Not Implemented")
            
    def create_corpus(self, texts):
        data_words_ngrams = self.get_n_gm_tokens(texts)
#         id2word = gensim.corpora.Dictionary(data_words_ngrams)
        corpus = [self.id2word.doc2bow(text) for text in data_words_ngrams]
        return corpus
    
    def create_word_dict(self, X):
        if isinstance(X, pd.Series):
            X = X.tolist()
        data_words_ngrams = self.get_n_gm_tokens(X)
        self.id2word = gensim.corpora.Dictionary(data_words_ngrams)
        
    
    def fit(self, X):
        
        if isinstance(X, pd.Series):
            X = X.tolist()
        
        # Create ngram models
        self.create_n_gm_models(tokenized_data=X)
        
        # Create corpus
        corpus = self.create_corpus(X)
        
        # Fit topic model
        if self.method == 'lda':
            self.topic_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=self.id2word,
                                           num_topics=self.n_topics, 
                                           random_state=self.random_state,
                                           chunksize=100,
                                           passes=10,
                                           per_word_topics=True)
        elif self.method == "hdp":
            self.topic_model = HdpModel(corpus=corpus,id2word=self.id2word, )
            
        # Print the Keyword in the 10 topics
        pprint(self.topic_model.print_topics())
        
    def set_topic_labels(self, topic_label_dict):
        self.topic_label_dict = topic_label_dict
        
    def evaluate(self, X):
        
        if isinstance(X, pd.Series):
            X = X.tolist()
        
        # Create corpus
        corpus = self.create_corpus(X)
        
        # Build Coherence Model using trained topic model and tokenised data 
        self.coherence_model = CoherenceModel(model=self.topic_model,
                                     texts=X, 
                                     dictionary=self.id2word, 
                                     coherence='c_v')
        
        coherence_score = self.coherence_model.get_coherence()
        return coherence_score
    
    def transform(self, X):
        
        if isinstance(X, pd.Series):
            data_index_list = X.index.tolist()
            X = X.tolist()
        else:
            data_index_list = [i for i in range(len(X))]
            
        test_corpus = self.create_corpus(X)
        preds = list(self.topic_model.get_document_topics(test_corpus))
        if len(test_corpus) == 1:
            preds = [preds]
        else:
            preds = list(preds)
        
        pred_proba_df = pd.DataFrame(columns=[i for i in range(self.n_topics)])
        for idx, pred in enumerate(preds):
            record = pd.DataFrame.from_records(pred).set_index(0).T
            record.index = [data_index_list[idx]]
            pred_proba_df = pred_proba_df.append(record)
        
        if self.topic_label_dict:
            pred_proba_df = pred_proba_df.rename(columns=self.topic_label_dict)
        
        pred_proba_df = pred_proba_df.fillna(0)
        return pred_proba_df
# Baseline model
topic_model = TopicModel(method='lda',
                         use_bi_gm=True, 
                         use_tri_gm=False,
                         min_bi_gm_count=5, 
                         min_tri_count=5, 
                         bi_gm_thrs=10, 
                         tri_gm_thrs=10,
                         n_topics=5, 
                         random_state=17290)
# fit corpus
topic_model.create_n_gm_models(trn_data['Tokenized'])
topic_model.create_word_dict(trn_data['Tokenized'])
trn_data.head()
topic_model.fit(trn_data['Tokenized'])
topic_model.transform([['google', 'iphone', 'siri', 'launch',  'app'], 
                       ['good', 'place', 'near', 'sea'], 
                      ['player', 'club', 'game'],
                      ['player']])
topic_model.evaluate(trn_data['Tokenized'])
"""
"glassdoor_reviews"
"tech_news"
"room_rentals"
"sports_news"
"Automobiles"
"""

topic_model.set_topic_labels({0: "room_rentals",
                             1: "sports_news",
                             2: "tech_news",
                             3: "Automobiles",
                             4: "glassdoor_reviews"})
# tst_data
predict_on = tst_data
predictions = topic_model.transform(predict_on['Tokenized'])
result = pd.concat([predict_on[['text']], predictions], axis=1)
result.head()
predict_on = trn_data
predictions = topic_model.transform(predict_on['Tokenized'])
result = pd.concat([predict_on[['text']], predictions], axis=1)
result.head()

# Create submission
predict_on = data
predictions = topic_model.transform(predict_on['Tokenized'])
result = pd.concat([predict_on[['text']], predictions], axis=1)
result.head()
submission = result.apply(lambda x: x[['room_rentals', 'sports_news', 
                          'tech_news', 'Automobiles', 'glassdoor_reviews']].sort_values().tail(1).index.tolist()[0] , axis=1)
submission.head()
submission =  submission.reset_index()
submission.columns = ['Id', 'topic'] #['Id', 'topic']
submission.head()
submission.to_csv("Hackathon_Submission_June20.csv")
submission.columns = ['Id', 'topic']
sample_submission = pd.read_csv("../input/unstructured-l0-nlp-hackathon/sample_submission.csv")
sample_submission.columns
submission2 = sample_submission.merge(submission, on=['Id'], how='left')
submission2.head()
del submission2['topic_x']
submission2.columns = ['Id', 'topic']
submission2.head()
submission2.to_csv("HackathonSubmission1_June20.csv")


# import pickle
# with open('Hackathon_baseline_model_June20.pkl', 'wb') as F:
#     pickle.dump(topic_model, F)






