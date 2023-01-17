import pandas as pd 
from collections import defaultdict
import string
from gensim.models import CoherenceModel
import gensim
from pprint import pprint
import spacy,en_core_web_sm
from nltk.stem import PorterStemmer
import os
import json
from gensim.models import Word2Vec
import nltk
import re
import collections
from sklearn.metrics import cohen_kappa_score
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict
import string
class MetaData:
    def __init__(self):
        """Define varibles."""
        # path and data
        self.path = '../input/CORD-19-research-challenge/'
        self.meta_data = pd.read_csv(self.path + 'metadata.csv')

    def data_dict(self):
        """Convert df to dictionary. """
        mydict = lambda: defaultdict(mydict)
        meta_data_dict = mydict()

        for cord_uid, abstract, title, sha in zip(self.meta_data['cord_uid'], self.meta_data['abstract'], self.meta_data['title'], self.meta_data['sha']):
            meta_data_dict[cord_uid]['title'] = title
            meta_data_dict[cord_uid]['abstract'] = abstract
            meta_data_dict[cord_uid]['sha'] = sha

        return meta_data_dict
class ExtractText:
    """Extract text according to keywords or phrases"""

    def __init__(self, metaDict, keyword, variable):
        """Define varibles."""
        self.path = '../input/CORD-19-research-challenge/'
        self.metadata = metaDict
        self.keyword = keyword
        self.variable = variable


    def simple_preprocess(self):
        """Simple text process: lower case, remove punc. """
        mydict = lambda: defaultdict(mydict)
        cleaned = mydict()
        for k, v in self.metadata.items():
            sent = v[self.variable]
            sent = str(sent).lower().translate(str.maketrans('', '', string.punctuation))
            cleaned[k]['processed_text'] = sent
            cleaned[k]['sha'] = v['sha']
            cleaned[k]['title'] = v['title']

        return cleaned

    def very_simple_preprocess(self):
        """Simple text process: lower case only. """
        mydict = lambda: defaultdict(mydict)
        cleaned = mydict()
        for k, v in self.metadata.items():
            sent = v[self.variable]
            sent = str(sent)
            #sent = str(sent).lower()
            cleaned[k]['processed_text'] = sent
            cleaned[k]['sha'] = v['sha']
            cleaned[k]['title'] = v['title']

        return cleaned
     

    def extract_w_keywords(self):
        """Select content with keywords."""
        ps = PorterStemmer()
        mydict = lambda: defaultdict(mydict)
        selected = mydict()
        textdict = self.simple_preprocess()
        
        for k, v in textdict.items():
            if self.keyword in v['processed_text'].split():
                #print(ps.stem(str(self.keyword)))
                selected[k]['processed_text'] = v['processed_text']
                selected[k]['sha'] = v['sha']
                selected[k]['title'] = v['title']
        return selected

    def extract_w_keywords_punc(self):
        """Select content with keywords, with punctuations in text"""
        ps = PorterStemmer()
        mydict = lambda: defaultdict(mydict)
        selected = mydict()
        textdict = self.very_simple_preprocess()
        
        for k, v in textdict.items():
            #keywords are stemmed before matching
            if ps.stem(str(self.keyword)) in ps.stem(str(v['processed_text'].split())):
                selected[k]['processed_text'] = v['processed_text']
                selected[k]['sha'] = v['sha']
                selected[k]['title'] = v['title']
        return selected

    def get_noun_verb(self, text):
        """get noun trunks for the lda model,
        change noun and verb part to decide what
        you want to use as input for LDA"""
        ps = PorterStemmer()
      
        #find nound trunks
        nlp = en_core_web_sm.load()
        all_extracted = {}
        for k, v in text.items():
            #v = v.replace('incubation period', 'incubation_period')
            doc = nlp(v)
            nouns = ' '.join(str(v) for v in doc if v.pos_ is 'NOUN').split()
            verbs = ' '.join(ps.stem(str(v)) for v in doc if v.pos_ is 'VERB').split()
            adj = ' '.join(str(v) for v in doc if v.pos_ is 'ADJ').split()
            all_w = nouns + verbs + adj
            all_extracted[k] = all_w
      
        return all_extracted

    def get_noun_verb2(self, text):
        """get noun trunks for the lda model,
        change noun and verb part to decide what
        you want to use as input for LDA"""
        ps = PorterStemmer()
      
        #find nound trunks
        nlp = en_core_web_sm.load()
        all_extracted = {}
        for k, v in text.items():
            #v = v.replace('incubation period', 'incubation_period')
            doc = nlp(v['processed_text'])
            nouns = ' '.join(ps.stem(str(v)) for v in doc if v.pos_ is 'NOUN').split()
            verbs = ' '.join(ps.stem(str(v)) for v in doc if v.pos_ is 'VERB').split()
            adj = ' '.join(str(v) for v in doc if v.pos_ is 'ADJ').split()
            all_w = nouns + verbs + adj
            all_extracted[k] = all_w
      
        return all_extracted

    def tokenization(self, text):
        """get noun trunks for the lda model,
        change noun and verb part to decide what
        you want to use as input for the next step"""
        nlp = spacy.load("en_core_web_sm")

        all_extracted = {}
        for k, v in text.items():
            doc = nlp(v)
            all_extracted[k] = [w.text for w in doc]
      
        return all_extracted


class LDATopic:
    def __init__(self, processed_text, topic_num, alpha, eta):
        """Define varibles."""
        self.path = '../input/CORD-19-research-challenge/'
        self.text = processed_text
        self.topic_num = topic_num
        self.alpha = alpha
        self.eta = eta

    def get_lda_score_eval(self, dictionary, bow_corpus):
        """LDA model and coherence score."""

        lda_model = gensim.models.ldamodel.LdaModel(bow_corpus, num_topics=self.topic_num, id2word=dictionary, passes=10,  update_every=1, random_state = 300, alpha=self.alpha, eta=self.eta)
        #pprint(lda_model.print_topics())

        # get coherence score
        cm = CoherenceModel(model=lda_model, corpus=bow_corpus, coherence='u_mass')
        coherence = cm.get_coherence()
        print('coherence score is {}'.format(coherence))

        return lda_model, coherence

    def get_score_dict(self, bow_corpus, lda_model_object):
        """
        get lda score for each document
        """
        all_lda_score = {}
        for i in range(len(bow_corpus)):
            lda_score ={}
            for index, score in sorted(lda_model_object[bow_corpus[i]], key=lambda tup: -1*tup[1]):
                lda_score[index] = score
                od = collections.OrderedDict(sorted(lda_score.items()))
            all_lda_score[i] = od
        return all_lda_score


    def topic_modeling(self):
        """Get LDA topic modeling."""
        # generate dictionary
        dictionary = gensim.corpora.Dictionary(self.text.values())
        bow_corpus = [dictionary.doc2bow(doc) for doc in self.text.values()]
        # modeling
        model, coherence = self.get_lda_score_eval(dictionary, bow_corpus)

        lda_score_all = self.get_score_dict(bow_corpus, model)

        all_lda_score_df = pd.DataFrame.from_dict(lda_score_all)
        all_lda_score_dfT = all_lda_score_df.T
        all_lda_score_dfT = all_lda_score_dfT.fillna(0)

        return model, coherence, all_lda_score_dfT

    def get_ids_from_selected(self, text):
        """Get unique id from text """
        id_l = []
        for k, v in text.items():
            id_l.append(k)
            
        return id_l
# Now we extract articles contain the most relevant topic

def selected_best_LDA(keyword, varname):
        """Select the best lda model with extracted text """
        # convert data to dictionary format
        m = MetaData()
        metaDict = m.data_dict()

        #process text and extract text with keywords
        et = ExtractText(metaDict, keyword, varname)
        text1 = et.extract_w_keywords()


        # extract nouns, verbs and adjetives
        text = et.get_noun_verb2(text1)

        # optimized alpha and beta
        alpha = [0.1, 0.3, 0.5, 0.7, 0.9]
        beta = [0.1, 0.3, 0.5, 0.7, 0.9]

        mydict = lambda: defaultdict(mydict)
        cohere_dict = mydict()
        for a in alpha:
            for b in beta:
                lda = LDATopic(text, 20, a, b)
                model, coherence, scores = lda.topic_modeling()
                cohere_dict[coherence]['a'] = a
                cohere_dict[coherence]['b'] = b
    
        # sort result dictionary to identify the best a, b
        # select a,b with the largest coherence score 
        sort = sorted(cohere_dict.keys())[0] 
        a = cohere_dict[sort]['a']
        b = cohere_dict[sort]['b']
        
        # run LDA with the optimized values
        lda = LDATopic(text, 20, a, b)
        model, coherence, scores_best = lda.topic_modeling()
        pprint(model.print_topics())

        # select merge ids with the LDA topic scores
        id_l = lda.get_ids_from_selected(text)
        scores_best['cord_uid'] = id_l

        return scores_best




def select_text_from_LDA_results(keyword, varname, scores_best, topic_num):
        # choose papers with the most relevant topic
        # convert data to dictionary format
        m = MetaData()
        metaDict = m.data_dict()

        # process text and extract text with keywords
        et = ExtractText(metaDict, keyword, varname)
        # extract text together with punctuation
        text1 = et.extract_w_keywords_punc()
        # need to decide which topic to choose after training
        sel = scores_best[scores_best[topic_num] > 0] 
        
        mydict = lambda: defaultdict(mydict)
        selected = mydict()
        for k, v in text1.items():
            if k in sel.cord_uid.tolist():
                selected[k]['title'] = v['title']
                selected[k]['processed_text'] = v['processed_text']
                selected[k]['sha'] = v['sha']
    
        return selected

def extract_relevant_sentences(cor_dict, search_keywords, filter_title=None):
    """Extract sentences contain keyword in relevant articles. """
    #here user can also choose whether they would like to only select title contain covid keywords

    mydict = lambda: defaultdict(mydict)
    sel_sentence = mydict()
    filter_w = ['covid19','ncov','2019-ncov','covid-19','sars-cov','wuhan']
    
    for k, v in cor_dict.items():
        keyword_sentence = []
        sentences = v['processed_text'].split('.')
        for sentence in sentences:
            # for each sentence, check if keyword exist
            # append sentences contain keyword to list
            keyword_sum = sum(1 for word in search_keywords if word in sentence)
            if keyword_sum > 0:
                keyword_sentence.append(sentence)         

        # store results
        if not keyword_sentence:
            pass
        elif filter_title is not None:
            for f in filter_w:
                title = v['title'].lower().translate(str.maketrans('', '', string.punctuation))
                abstract = v['processed_text'].lower().translate(str.maketrans('', '', string.punctuation))
                if (f in title) or (f in abstract):
                    sel_sentence[k]['sentences'] = keyword_sentence
                    sel_sentence[k]['sha'] = v['sha']
                    sel_sentence[k]['title'] = v['title'] 
        else:
            sel_sentence[k]['sentences'] = keyword_sentence
            sel_sentence[k]['sha'] = v['sha']
            sel_sentence[k]['title'] = v['title'] 
            
    print('{} articles are relevant to the topic you choose'.format(len(sel_sentence)))

    path = '../input/CORD-19-research-challenge/'
    df = pd.DataFrame.from_dict(sel_sentence, orient='index')
    #df.to_csv(path + 'search_results_{}.csv'.format(search_keywords))
    #sel_sentence_df = pd.read_csv(path + 'search_results_{}.csv'.format(search_keywords))
    return sel_sentence, df

def extract_relevant_sentences2(cor_dict, search_keywords, filter_title=None):
    """Extract sentences contain keyword in relevant articles for system evaluation. """
    #here user can also choose whether they would like to only select title contain covid keywords
    #difference from the previous one is where we store the result

    mydict = lambda: defaultdict(mydict)
    sel_sentence = mydict()
    filter_w = ['covid19','ncov','2019-ncov','covid-19','sars-cov','wuhan']
    
    for k, v in cor_dict.items():
        keyword_sentence = []
        sentences = v['processed_text'].split('.')
        for sentence in sentences:
            # for each sentence, check if keyword exist
            # append sentences contain keyword to list
            keyword_sum = sum(1 for word in search_keywords if word in sentence)
            if keyword_sum > 0:
                keyword_sentence.append(sentence)         

        # store results
        if not keyword_sentence:
            pass
        
        elif filter_title is not None:
            for f in filter_w:
                title = v['title'].lower().translate(str.maketrans('', '', string.punctuation))
                abstract = v['processed_text'].lower().translate(str.maketrans('', '', string.punctuation))
                if (f in title) or (f in abstract):
                    sel_sentence[k]['sentences'] = keyword_sentence
                    sel_sentence[k]['sha'] = v['sha']
                    sel_sentence[k]['title'] = v['title'] 
        else:
            sel_sentence[k]['sentences'] = keyword_sentence
            sel_sentence[k]['sha'] = v['sha']
            sel_sentence[k]['title'] = v['title'] 
    print('{} articles contain keyword {}'.format(len(sel_sentence),  search_keywords))

    path = '../input/CORD-19-research-challenge/'
    df = pd.DataFrame.from_dict(sel_sentence, orient='index')
    df.to_csv(path + 'eval_results_{}.csv'.format(search_keywords))
    sel_sentence_df = pd.read_csv(path + 'eval_results_{}.csv'.format(search_keywords))
    return sel_sentence, sel_sentence_df

#here we select the LDA model with the lowe
scores_best_mask = selected_best_LDA('mask', 'abstract')
scores_best_mask.shape
# topic number 10 is most relevant to public wearing mask
# which topic do you think is most relevant to your search
cor_dict_mask = select_text_from_LDA_results('mask', 'abstract', scores_best_mask, 10)
print ("There are {} abstracts selected". format(len(cor_dict_mask)))
# extract relevant sentences  #search keywords can be a list
sel_sentence_mask, sel_sentence_df_mask = extract_relevant_sentences(cor_dict_mask, ['mask'])
#read extracted article
sel_sentence_df_mask.head(20)
#here we annotated a sample of 40 abstracts
path = '../input/annotation/'
annotation_mask = pd.read_csv(path + 'wear_mask.csv')
# view file
annotation_mask.head(5)
print('there are {} articles relevant to the topic'.format(annotation_mask.shape[0]))
annotation_mask['stance'].value_counts()
print('there are {} papers support using a mask during a pandemic is useful, {} assume masks as useful and examine the public’s willingness to comply the rules,  {} papers show no obvious evidence that shows using mask is protective or the protection is very little'. format(str(annotation_mask['stance'].value_counts()[1]), str(annotation_mask['stance'].value_counts()[2]), annotation_mask['stance'].value_counts()[0]) )
          
scores_best_incu = selected_best_LDA('incubation', 'abstract')
# topic number 0 is most relevant to public wearing mask
# which topic do you think is most relevant to your search
cor_dict_incu = select_text_from_LDA_results('incubation', 'abstract', scores_best_incu, 0)
print ("There are {} abstracts selected". format(len(cor_dict_incu)))
# extract relevant sentences  #search keywords can be a list
sel_sentence_incu, sel_sentence_df_incu = extract_relevant_sentences(cor_dict_incu, ['incubation','day'], 'title')
#read extracted article
sel_sentence_df_incu.head(10)
#here we need to add the stats analysis 
path = '../input/annotation/'
annotation_incubation = pd.read_csv(path + 'incubation.csv')
print('there are {} articles relevant to the topic'.format(annotation_incubation.shape[0]))
incubation = annotation_incubation['stance'].value_counts()
print('there are {} paper shows the incubation period is 2-14 days with mean 5 days, {} papers shows a different number'. format(incubation[1], incubation[0])
     )
incubation = annotation_incubation['relevance'].value_counts()
print('there are {} papers relevant to the topic, {} papers not relevant to the topic'. format(incubation[1], incubation[0]))
scores_best_asym = selected_best_LDA('asymptomatic', 'abstract')
# topic number 8 is most relevant to public wearing mask
# which topic do you think is most relevant to your search
cor_dict_asym = select_text_from_LDA_results('asymptomatic', 'abstract', scores_best_asym, 8)
print ("There are {} abstracts selected". format(len(cor_dict_asym)))
# extract relevant sentences  #search keywords can be a list
sel_sentence_asym, sel_sentence_df_asym = extract_relevant_sentences(cor_dict_asym, ['asymptomatic','transmission'], 'title')
sel_sentence_df_asym.tail(10)
#here we need to add the stats analysis 
annotation_asymptomatic = pd.read_csv(path + 'asymtomatic.csv')
print('there are {} articles relevant to the topic'.format(annotation_asymptomatic.shape[0]))
asymptomatic = annotation_asymptomatic['stance'].value_counts()
print('{} papers show that there is clear evidence show that asymtomatic cases contribute to the spread of the virus, {} papers show that it is unlikely that asymtomatic cases contribute to the spread of the virus'.format(asymptomatic[1], asymptomatic[0]))

