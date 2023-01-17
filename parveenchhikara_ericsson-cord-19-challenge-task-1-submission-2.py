# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install rank_bm25


import json

import re

import sys

import pandas as pd

import os

import pprint

import datetime as dt

from nltk.stem import WordNetLemmatizer

from nltk.stem import PorterStemmer

from nltk.corpus import wordnet

import numpy as np



import gensim

from gensim.parsing.preprocessing import strip_tags, strip_punctuation,strip_non_alphanum, strip_multiple_whitespaces,stem_text, strip_numeric, remove_stopwords 

from gensim.summarization import summarize

from gensim import similarities

from gensim.summarization.textcleaner import split_sentences

import pyLDAvis.gensim

import collections



import joblib

from rank_bm25 import BM25Okapi, BM25Plus

import timeit

import ipywidgets as widgets

from IPython.display import display , HTML

from jinja2 import Template

import json

from json import JSONEncoder

from scipy import spatial



import warnings

warnings.filterwarnings('ignore')
current_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

data_dir =  current_dir + "/input/data-covid/structured"

version = 1

model_dir = current_dir + "/input/model-covid/model/v"+str(version)

view_dir = current_dir + "/input/view-covid/view"

print(view_dir)
start_time = timeit.default_timer()

base_df = pd.read_parquet(data_dir + "/all_articles.parquet")

journal_df = pd.read_csv(data_dir + "/journal_ranking_info.csv")

complete_index_list = base_df.index

bm25Plus_complete = joblib.load(model_dir + "/bm25Plus_complete.mdl")

word2Vec = joblib.load(model_dir + "/word2Vec_full_corpus.mdl")

index2word_set = set(word2Vec.wv.index2word)

elapsed = timeit.default_timer() - start_time

print(elapsed)
# stop words related to corpus

corpus_stop_words = [

    'doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure', 'introduction', 'section', 'abstract', 'summary',

    'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'fig', 'fig.', 'al.', 'al'

    'di', 'la', 'il', 'del', 'le', 'della', 'dei', 'delle', 'una', 'da',  'dell',  'non', 'si','also', 'too',

    'que', 'qui', 'le', 'la', 'les', 'un', 'une', 'si','de', 'des', 'est', 'sont', 'plus', 'dans', 'par', 'ici',

    'para', 'por', 'lo', 'sera', 'caso', 'entre', 'avec', 'sur', 'ont', 'pour', 'pa', 'ce', 'ca', 'ces', 'cehz', 'son', 'moi', 'toi',

]



# general stop words of english

generic_stop_words = ["a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", 

               "any", "are", "aren", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", 

               "both", "but", "by", "can", "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't",

               "doing", "don", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "hadn't", "has", 

               "hasn", "hasn't", "have", "haven", "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", 

               "how", "i", "if", "in", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "ll", "m", "ma", "me", "mightn", 

               "mightn't", "more", "most", "mustn", "mustn't", "my", "myself", "needn", "needn't", "no", "nor", "not", "now", "o", "of", "off", 

               "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shan't", "she", "she's", "should", "should've", "shouldn", "shouldn't", "so", "some", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm", "i've", "let's", "ought", "she'd", "she'll", "that's", "there's", "they'd", "they'll", "they're", "they've", "we'd", "we'll", "we're", "we've", "what's", "when's", "where's", "who's", "why's", "would", "able", "abst", "accordance", "according", "accordingly", "across", "act", "actually", "added", "adj", "affected", "affecting", "affects", "afterwards", "ah", "almost", "alone", "along", "already", "also", "although", "always", "among", "amongst", "announce", "another", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "apparently", "approximately", "arent", "arise", "around", "aside", "ask", "asking", "auth", "available", "away", "awfully", "b", "back", "became", "become", "becomes", "becoming", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "believe", "beside", "besides", "beyond", "biol", "brief", "briefly", "c", "ca", "came", "cannot", "can't", "cause", "causes", "certain", "certainly", "co", "com", "come", "comes", "contain", "containing", "contains", "couldnt", "date", "different", "done", "downwards", "due", "e", "ed", "edu", "effect", "eg", "eight", "eighty", "either", "else", "elsewhere", "end", "ending", "enough", "especially", "et", "etc", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "except", "f", "far", "ff", "fifth", "first", "five", "fix", "followed", "following", "follows", "former", "formerly", "forth", "found", "four", "furthermore", "g", "gave", "get", "gets", "getting", "give", "given", "gives", "giving", "go", "goes", "gone", "got", "gotten", "h", "happens", "hardly", "hed", "hence", "hereafter", "hereby", "herein", "heres", "hereupon", "hes", "hi", "hid", "hither", "home", "howbeit", "however", "hundred", "id", "ie", "im", "immediate", "immediately", "importance", "important", "inc", "indeed", "index", "information", "instead", "invention", "inward", "itd", "it'll", "j", "k", "keep", "keeps", "kept", "kg", "km", "know", "known", "knows", "l", "largely", "last", "lately", "later", "latter", "latterly", "least", "less", "lest", "let", "lets", "like", "liked", "likely", "line", "little", "'ll", "look", "looking", "looks", "ltd", "made", "mainly", "make", "makes", "many", "may", "maybe", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "million", "miss", "ml", "moreover", "mostly", "mr", "mrs", "much", "mug", "must", "n", "na", "name", "namely", "nay", "nd", "near", "nearly", "necessarily", "necessary", "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine", "ninety", "nobody", "non", "none", "nonetheless", "noone", "normally", "nos", "noted", "nothing", "nowhere", "obtain", "obtained", "obviously", "often", "oh", "ok", "okay", "old", "omitted", "one", "ones", "onto", "ord", "others", "otherwise", "outside", "overall", "owing", "p", "page", "pages", "part", "particular", "particularly", "past", "per", "perhaps", "placed", "please", "plus", "poorly", "possible", "possibly", "potentially", "pp", "predominantly", "present", "previously", "primarily", "probably", "promptly", "proud", "provides", "put", "q", "que", "quickly", "quite", "qv", "r", "ran", "rather", "rd", "readily", "really", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "respectively", "resulted", "resulting", "results", "right", "run", "said", "saw", "say", "saying", "says", "sec", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sent", "seven", "several", "shall", "shed", "shes", "show", "showed", "shown", "showns", "shows", "significant", "significantly", "similar", "similarly", "since", "six", "slightly", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specifically", "specified", "specify", "specifying", "still", "stop", "strongly", "sub", "substantially", "successfully", "sufficiently", "suggest", "sup", "sure", "take", "taken", "taking", "tell", "tends", "th", "thank", "thanks", "thanx", "thats", "that've", "thence", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "thereto", "thereupon", "there've", "theyd", "theyre", "think", "thou", "though", "thoughh", "thousand", "throug", "throughout", "thru", "thus", "til", "tip", "together", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying", "ts", "twice", "two", "u", "un", "unfortunately", "unless", "unlike", "unlikely", "unto", "upon", "ups", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "v", "value", "various", "'ve", "via", "viz", "vol", "vols", "vs", "w", "want", "wants", "wasnt", "way", "wed", "welcome", "went", "werent", "whatever", "what'll", "whats", "whence", "whenever", "whereafter", "whereas", "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "whim", "whither", "whod", "whoever", "whole", "who'll", "whomever", "whos", "whose", "widely", "willing", "wish", "within", "without", "wont", "words", "world", "wouldnt", "www", "x", "yes", "yet", "youd", "youre", "z", "zero", "a's", "ain't", "allow", "allows", "apart", "appear", "appreciate", "appropriate", "associated", "best", "better", "c'mon", "c's", "cant", "changes", "clearly", "concerning", "consequently", "consider", "considering", "corresponding", "course", "currently", "definitely", "described", "despite", "entirely", "exactly", "example", "going", "greetings", "hello", "help", "hopefully", "ignored", "inasmuch", "indicate", "indicated", "indicates", "inner", "insofar", "it'd", "keep", "keeps", "novel", "presumably", "reasonably", "second", "secondly", "sensible", "serious", "seriously", "sure", "t's", "third", "thorough", "thoroughly", "three", "well", "wonder", "a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "co", "op", "research-articl", "pagecount", "cit", "ibid", "les", "le", "au", "que", "est", "pas", "vol", "el", "los", "pp", "u201d", "well-b", "http", "volumtype", "par", "0o", "0s", "3a", "3b", "3d", "6b", "6o", "a1", "a2", "a3", "a4", "ab", "ac", "ad", "ae", "af", "ag", "aj", "al", "an", "ao", "ap", "ar", "av", "aw", "ax", "ay", "az", "b1", "b2", "b3", "ba", "bc", "bd", "be", "bi", "bj", "bk", "bl", "bn", "bp", "br", "bs", "bt", "bu", "bx", "c1", "c2", "c3", "cc", "cd", "ce", "cf", "cg", "ch", "ci", "cj", "cl", "cm", "cn", "cp", "cq", "cr", "cs", "ct", "cu", "cv", "cx", "cy", "cz", "d2", "da", "dc", "dd", "de", "df", "di", "dj", "dk", "dl", "do", "dp", "dr", "ds", "dt", "du", "dx", "dy", "e2", "e3", "ea", "ec", "ed", "ee", "ef", "ei", "ej", "el", "em", "en", "eo", "ep", "eq", "er", "es", "et", "eu", "ev", "ex", "ey", "f2", "fa", "fc", "ff", "fi", "fj", "fl", "fn", "fo", "fr", "fs", "ft", "fu", "fy", "ga", "ge", "gi", "gj", "gl", "go", "gr", "gs", "gy", "h2", "h3", "hh", "hi", "hj", "ho", "hr", "hs", "hu", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ic", "ie", "ig", "ih", "ii", "ij", "il", "in", "io", "ip", "iq", "ir", "iv", "ix", "iy", "iz", "jj", "jr", "js", "jt", "ju", "ke", "kg", "kj", "km", "ko", "l2", "la", "lb", "lc", "lf", "lj", "ln", "lo", "lr", "ls", "lt", "m2", "ml", "mn", "mo", "ms", "mt", "mu", "n2", "nc", "nd", "ne", "ng", "ni", "nj", "nl", "nn", "nr", "ns", "nt", "ny", "oa", "ob", "oc", "od", "of", "og", "oi", "oj", "ol", "om", "on", "oo", "oq", "or", "os", "ot", "ou", "ow", "ox", "oz", "p1", "p2", "p3", "pc", "pd", "pe", "pf", "ph", "pi", "pj", "pk", "pl", "pm", "pn", "po", "pq", "pr", "ps", "pt", "pu", "py", "qj", "qu", "r2", "ra", "rc", "rd", "rf", "rh", "ri", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "rv", "ry", "s2", "sa", "sc", "sd", "se", "sf", "si", "sj", "sl", "sm", "sn", "sp", "sq", "sr", "ss", "st", "sy", "sz", "t1", "t2", "t3", "tb", "tc", "td", "te", "tf", "th", "ti", "tj", "tl", "tm", "tn", "tp", "tq", "tr", "ts", "tt", "tv", "tx", "ue", "ui", "uj", "uk", "um", "un", "uo", "ur", "ut", "va", "wa", "vd", "wi", "vj", "vo", "wo", "vq", "vt", "vu", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y2", "yj", "yl", "yr", "ys", "yt", "zi", "zz"]



# Complete set of stopwords

customize_stop_words_set = set(corpus_stop_words + generic_stop_words)



# Text columns of Articles

# By combining all these columns, we get complete text present in a article 

text_columns = ['title', 'abstract', 'text', 'introduction', 'method', 'result', 'discussion', 'conclusion']

                      

# initializing of Singleton objects 



# Lemmatizer Object

lemmatizer = WordNetLemmatizer()

                      

#Stemmer Object to find the root of the word

stemmer = PorterStemmer()
class Article:

    

    def __init__(self,df_row):

        self.paper_id = str(df_row['paper_id'])

        self.title = str(df_row['title'])

        self.link = str(df_row['url'])

        self.abstract = str(df_row['abstract'])

        self.text = str(df_row['text'])

        self.publish_time = str(df_row['publish_time'])

        self.search_score = str(df_row['score'])

        self.authors = [ x for x in zip(list(df_row['authors']),list(df_row['author_institutions']))]

        self.journal = str(df_row['journal'])

        self.document_type = str(df_row['document_type'])

        #self.document_type = "COMMERCIAL USE SUBSET"

        self.introduction = str(df_row['introduction'])

        self.method = str(df_row['method'])

        self.result = str(df_row['result'])

        self.discussion = str(df_row['discussion'])

        self.conclusion = str(df_row['conclusion'])

        #self.institution = str(df_row['institution'])

        self.country_tags = list(df_row['country_tags'])

        self.row_num = str(df_row['row_num'])

        

    

    def __str__(self):

        return f"title = {self.title}, link ={self.link}, publish_year ={self.publish_year}"

    

    def set_country_tags(self):

        loc_list = GeoText(self.complete_text()).country_mentions

        location_list = [reverse_country_dict[x].title() for x,y in collections.Counter(loc_list).most_common()]

        self.country_tags = location_list

    

    def set_journal_info(self):

        row_df = journal_df[journal_df['orig_title'] == self.journal.lower()]

        if len(row_df) > 0:

            self.journal_rank = str(row_df.iloc[0]['rank'])

            self.journal_title = str(row_df.iloc[0]['title'])

        else:

            self.journal_rank = 100000

            self.journal_title = self.journal

    

    def set_search_rank(self,rank):

        self.search_rank =  int(rank)

    

    def set_cluster_id(self,id):

        self.cluster_id =  int(id)

    

    def set_semantic_tags(self,tags):

        self.semantic_tags =  tags

    

    def set_publish_year(self):

        date_str = self.publish_time

        if date_str == "nan" or date_str == None or date_str == "None":

            self.publish_year = int(dt.datetime.today().year)

        else:

            self.publish_year = int(date_str[0:4])

    

    def complete_text(self):

        text_list = [self.title,self.abstract,self.text,self.introduction,self.method,self.result,self.discussion,self.conclusion]

        texts_present = [text for text in text_list if not check_pd_nan(text)]

        return " ".join(texts_present)

    

    def summary(self):

        #return summarize(self.text,word_count=200)

        try:

            if not check_pd_nan(self.text):

                if len(split_sentences(self.text)) > 1 and len(self.text) > 800:

                    return summarize(self.text,word_count=200)

                else:

                    return self.text

            else:

                text_list = [self.introduction,self.discussion,self.method,self.result,self.conclusion]

                texts_present = [text for text in text_list if not check_pd_nan(text)]

                if len(split_sentences(texts_present)) > 1 and len(texts_present) > 800:

                    return summarize(texts_present,word_count=200)

        except:

            print("row_number : "+ self.row_num)

            return "Data is not sufficient in this Article for generating valid summary."

        print("row_number 2 : "+ self.row_num)

        return  "Data is not sufficient in this Article for generating valid summary."

    

    def set_search_score(self,score):

        self.search_score = float(score)

        

    def set_query_sim_score(self,score):

        self.query_sim_score = float(score)

    

    def marked_complete_html(self,query,query_avg_vector,word2Vec,index2word_set):

        

        section_list = ["Introduction" , "Discussion", "Content","Method", "Result", "Conclusion"]

        section_data = [self.introduction,self.discussion,self.text,

                        self.method,self.result,self.conclusion]

        

        marked_section_data =[]

        for index,data in enumerate(section_data):

            if check_pd_nan(data):

                marked_section_data.append(data)

                continue

                

            section_type = section_list[index]

            num_sent_per_para = 6

            num_mark_per_para = 2

            if section_type != "Content":

                num_sent_per_para=2

                num_mark_per_para = 1

            custom_section = custom_paragraph_in_document(data,num_sent_per_para)

            marked_data = mark_best_paragraph_in_document(custom_section,query_avg_vector,word2Vec,index2word_set,num_mark_per_para)

            marked_section_data.append(marked_data)

        

        css_template_str = custom_article_css_template.render()

        

        # fetching the article list html and turn into jinja2 template to render

        article_html = custom_article_template.render(article=self,

                        check_pd_nan = check_pd_nan,

                        section_list=section_list,

                        section_data =marked_section_data,

                        css = css_template_str)



        self.html_str = article_html.replace("\n","").replace("'", r"\'")



        

# Json Encoder class for generating Json object corresponding to Article class

class ArticleEncoder(JSONEncoder):

    def default(self, o):

        return o.__dict__
def check_pd_nan(value):

    """ func check_pd_nan(value) ->



        Usage : to check null value from panda dataframe

        Input : value to check

        Output : Boolean - true if ti is none or nan

    """

    return value == "nan" or value == None or value == "None"   





def custom_paragraph_in_document(document,num_sentences=6):

    """ func custom_paragraph_in_document(document,num_sentences=6) -->

    

        Usage : Creating paragraph based on number of sentences from document.

                Used for the purpose of finding the similiar section in document

                instead of paragraph(more abstract) or sentence (fine-grain)

                

        Input :  document : document or section of document 

                        num_sentences : number of sentences

        Output : List of paragraph in document

    """

    

    section_list =[]

    count=0;

    current_section=""

    for x,sentence in enumerate(split_sentences(document)):

        count=count+1

        if count <= num_sentences:

            current_section = current_section + " "+ sentence

            if count == num_sentences:

                section_list.append(current_section)

                count=0

                current_section=""

    if count>0 :

        section_list.append(current_section)

    return section_list



def avg_feature_vector(document, model, num_features, index2word_set):

    """func avg_feature_vector(document, model, num_features, index2word_set) -->

    

        Usage : finding the average of word vector(feature vecture) using word2vec model

                Using this average vector to find cosine similiartiy between sentence or document

                

        Input: document :  document or section or sentence

                model : word2Vec model

                num_features : should be same as number of features in word2vec model

                index2word_set : index mapping of words present in word2vec model

        Output: Word Vector (numpy array)

    """

    words = create_tokens(document)

    feature_vec = np.zeros((num_features, ), dtype='float32')

    n_words = 0

    for word in words:

        if word in index2word_set:

            n_words += 1

            feature_vec = np.add(feature_vec, model[word])

    if (n_words > 0):

        feature_vec = np.divide(feature_vec, n_words)

    return feature_vec



def mark_best_paragraph_in_document(document,query_avg_vector,word2Vec,index2word_set,num_marks=2):

    """ func mark_best_paragraph_in_document(document, model, num_features, index2word_set) -->

    

        Usage : finding the cosine similarity between query and document(list of sentence)

                return the most similiar sections in document to query with marking

                

        Input: document :  list of section

                query_avg_vector : average word vector of query

                model : word2Vec model

                index2word_set : index mapping of words present in word2vec model

                num_marks : number of marks need in the document

                

        Output: Document in str with marked content similiar to query

    """

    sim_scores =[]

    for index,paragraph in enumerate(document):

        para_avg_vector = avg_feature_vector(paragraph, model=word2Vec, num_features=300, index2word_set=index2word_set)

        sim = 1 - spatial.distance.cosine(query_avg_vector, para_avg_vector)

        sim_scores.append((index,sim))

    sorted_sim_scores = sorted(sim_scores,key=lambda x: x[1],reverse=True)

    top_indexes = [para_index for para_index,score in sorted_sim_scores[0:num_marks]]

    marked_corpus=""

    for index,para in enumerate(document):

        if index in top_indexes:

            marked_corpus +=  '<span class="mark-sentence">'+para +'</span>';

        else:

            marked_corpus +=  para

    return marked_corpus   

 



def create_tokens(text_str):

    """ func create_tokens(text_str) ->

        

        Usage : create tokens using following processing

                     strip_tags,

                     strip_non_alphanum,

                     strip_punctuation,

                     strip_multiple_whitespaces,

                     strip_numeric,

                     remove_stopwords,

                     stem_text

        Input : text_str : document or paragraph or query

        Output : List of tokens

    """

    

    tokens_list = gensim.parsing.preprocess_string(text_str, filters =

                                            [strip_tags,

                                             strip_non_alphanum,

                                             strip_punctuation,

                                             strip_multiple_whitespaces,

                                             strip_numeric,

                                             remove_stopwords,

                                             stem_text])

    

    cleaned_tokens = []

    for token in tokens_list:

            if token == token[0] * len(token) :

                continue

            if token  in customize_stop_words_set:

                continue

            else:

                cleaned_tokens.append(token)   

    return cleaned_tokens



def create_tm_tokens(text_str, lemmatizer):

    """ func create_tm_tokens(text_str,lemmatizer) ->

        

        Usage : create tokens for topic modelling.

                For topic modelling we used following preprocessing.

                Different from above tokenization in last step. Here we use lemmatizer

                     strip_tags,

                     strip_non_alphanum,

                     strip_punctuation,

                     strip_multiple_whitespaces,

                     strip_numeric,

                     remove_stopwords,

                     Lemmatizer

        Input : text_str : document or paragraph or query

                Lemmatizer : Singleton Object 

        Output : List of tokens

        """

    tokens_list = gensim.parsing.preprocess_string(text_str, filters =

                                            [strip_tags,

                                             strip_non_alphanum,

                                             strip_punctuation,

                                             strip_multiple_whitespaces,

                                             strip_numeric,

                                             remove_stopwords])

    

    cleaned_tokens = []

    for token in tokens_list:

        final_token = lemmatizer.lemmatize(token)

        final_token = final_token.lower() if final_token.istitle() else final_token

        token_lower = final_token.lower()

        if token_lower == token_lower[0] * len(token_lower) :

            continue

        if token_lower  in customize_stop_words_set:

            continue

        else:

            cleaned_tokens.append(final_token)   

    return cleaned_tokens



def create_complete_text(row,text_columns):

    """ func create_complete_text(row,text_columns) ->

        

        Usage : Create complete text of an article which is row of dataframe

                It is user defined function for Pandas dataframe to apply row wise

            

        Input : row : row of dataframe or an article in pandas row format

                text_columns : list of columns in row to be part of return text

        Output : Complete text in str

    """

    

    texts_present = [row[col] for col in text_columns if not pd.isna(row[col])]

    return " ".join(texts_present)



def create_stem_dictionary(doc_tokens_list,stemmer):

    """ func create_stem_dictionary(doc_tokens_list,stemmer) ->

        

        Usage : Create stem words dictionary

                dictionary will be used to display purpose of LDA tokens

                becuase stem tokens will be difficult to understand 

                so we find stem_word -> [all the words have same root word(stem_word)]

                later we replace stem_word with word of minimum length from the list

            

        Input : doc_tokens : list of tokens

                stemmer : Singelton object to do stemming

        Output : Dictionary { stem_word -> human_understandable_word}

    """

    

    dictionary={}

    for doc_tokens in doc_tokens_list:

        for token in doc_tokens:

            stem_token = stemmer.stem(token)

            if stem_token in dictionary:

                prev_val = dictionary[stem_token]

                if len(prev_val) > len(token):

                    dictionary[stem_token] = token

            else:

                dictionary[stem_token] = token

    return dictionary



def create_stem_dictionary_tokens(tokens_list,stemmer,dictionary=None):

    """ func create_stem_dictionary_tokens(doc_tokens_list,stemmer) ->

        

        Usage : Create stem words dictionary and stem tokens 

                dictionary will be used to display purpose of LDA tokens

                becuase stem tokens will be difficult to understand 

                so we find stem_word -> [all the words have same root word(stem_word)]

                later we replace stem_word with word of minimum length from the list

            

        Input : doc_tokens : list of tokens

                stemmer : Singelton object to do stemming

                dictionary : if stemming dictionary supplied, it will not create stem dictionary

        Output : List of stem tokens

    """

    # Creating the dictionary: 

    stem_dictionary = dictionary

    if stem_dictionary == None:

        stem_dictionary = create_stem_dictionary(tokens_list,stemmer)

        

    # Stemming the tokens:

    stem_tokens_list =[]

    for doc_tokens in tokens_list:

        stem_tokens_list.append([stem_dictionary[stemmer.stem(token)] for token in doc_tokens if stemmer.stem(token) in stem_dictionary])

    

    return (stem_dictionary,stem_tokens_list)
def create_topic_model(tokens_list, stem_dictionary,num_topics):

    """func create_topic_model(tokens_list, stem_dictionary,num_topics) ->

        

        Usage : Create Topic modelling using LDA to generate topics or clusters 

                Using Unigram, bigram, trigram and doc2bow

                we are generating the clusters from the search results

            

        Input : tokens_list : list of tokens

                stem_dictionary : stem dictionary

                num_topics : number of clusters or topics to be generated 

        Output : lda_model  = LDA model

                doc2bows = List of Bag of words 

                dictionary =  dictionary of final_tokens

                final_tokens = list of unigram,bigram and trigram tokens

    """

    

    unigram_tokens = tokens_list



    ''' Preparing Bigram and Trigram '''

    bigram = gensim.models.Phrases(unigram_tokens,min_count= 5,threshold=10) # higher threshold fewer phrases.

    trigram = gensim.models.Phrases(bigram[unigram_tokens], threshold=10)  

    bigram_mod = gensim.models.phrases.Phraser(bigram)

    trigram_mod = gensim.models.phrases.Phraser(trigram)



    #### Storing the bigram and trigram tokens:



    bigram_tokens = [bigram_mod[doc] for doc in unigram_tokens]

    trigram_tokens = [trigram_mod[bigram_mod[doc]] for doc in bigram_tokens]



    #### Creating the dictionary and corpus for the LDA :

    final_tokens = trigram_tokens

    dictionary = gensim.corpora.Dictionary(final_tokens)

    #dictionary.filter_extremes( no_above=0.5) 

    doc2bows = [dictionary.doc2bow(text) for text in final_tokens]



    # Build LDA model

    lda_model = gensim.models.ldamodel.LdaModel(corpus=doc2bows,

                                               id2word=dictionary,

                                               num_topics=num_topics,

                                               passes = 50,

                                               update_every=5,

                                               alpha='symmetric',

                                               iterations=50,

                                               random_state=np.random.seed(42),

                                               minimum_probability=0)

    

    return (lda_model,doc2bows,dictionary,final_tokens)





def find_doc_topic_cluster(lda_model,doc2bows):

    """func find_doc_topic_cluster(lda_model,doc2bows) ->

        

        Usage : finding the cluster for each document in the search results

            

        Input : lda_model : LDA Model 

                doc2bows : list of bag of words

        Output : topics_df :

                 columns : 'document number','dominant_topic' ,'percentage_contribution', 'topic_keywords' 

    """

    

    # Init output

    sent_topics_df = pd.DataFrame()



    # Get main topic in each document

    for i, row_list in enumerate(lda_model[doc2bows]):

        row = row_list[0] if lda_model.per_word_topics else row_list            

        # print(row)

        row = sorted(row, key=lambda x: (x[1]), reverse=True)

        # Get the Dominant topic, Perc Contribution and Keywords for each document

        for j, (topic_num, prop_topic) in enumerate(row):

            if j == 0:  # => dominant topic

                wp = lda_model.show_topic(topic_num)

                topic_keywords = [word for word, prop in wp]

                sent_topics_df = sent_topics_df.append(pd.Series([i,int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)

            else:

                break

    sent_topics_df.columns = ['doc_no','dominant_topic' ,'percentage_contribution', 'topic_keywords']



    return sent_topics_df



def query_docs_similiarity(query,query_avg_vector,result_df,word2Vec,index2word_set):

    """func query_docs_similiarity(query,query_avg_vector,result_df,word2Vec,index2word_set) ->

        

        Usage : finding the query document similarity using word2vec and cosine similarity for every document in search result

            

        Input : query : search query 

                query_avg_vector : average word vector of query

                result_df : search result as pandas dataframe

                word2Vec : word2vec model

                index2word_set : index mapping of words present in word2vec model

        Output : List of document similarity score

    """

    

    doc_sim_scores=[]

    for index, row in result_df.iterrows():

        complete_text = create_complete_text(row,text_columns)

        doc_avg_vector = avg_feature_vector(complete_text, model=word2Vec, num_features=300, index2word_set=index2word_set)

        sim = 1 - spatial.distance.cosine(query_avg_vector, doc_avg_vector)

        doc_sim_scores.append(sim)

    return doc_sim_scores
def find_query_results(query,bm25plus) :

    """func find_query_results(query,bm25plus) ->

        

        Usage : finding the search results using BM25plus model

            

        Input : query : search query 

                bm25plus : bm25plus search engine model

        Output : List of index  with highest search score

    """

    

    tokenized_query = create_tokens(query)

    doc_scores = bm25plus.get_scores(tokenized_query)

    doc_list = [(i,x) for i,x in enumerate(doc_scores)]

    top_n = sorted(doc_list,key = lambda key : key[1],reverse=True)[0:5000] 

    #doc_index = model.get_top_n(tokenized_query, index_list, n=num_top_results)

    return top_n



def find_query_df(doc_scores,is_covid_only,num_records):

    """func find_query_df(doc_scores,is_covid_only,num_records) ->

        

        Usage : finding the result df form the index list returned by find_query_results

            

        Input : doc_scores : List of index  with highest search score

                is_covid_only : True : covid related articles or False : all articles in corpus

                num_records : number of records to be fetched 

        Output : Pandas dataframe with top results

    """

    index_list = [ index for index,score in doc_scores]

    #score_df = pd.DataFrame(doc_scores, columns =['index_1', 'score'])

    df = None

    score_df = None

    if not is_covid_only:

        index_list = index_list[0:num_records]

        score_df = pd.DataFrame(doc_scores[0:num_records], columns =['index_1', 'score'])

        df = base_df[base_df.index.isin(index_list)]  

    else:

        score_df = pd.DataFrame(doc_scores, columns =['index_1', 'score'])

        df = base_df[base_df.index.isin(index_list)]  

        df = df[df['is_covid'] == True]  

    

    df['index_1'] = df['row_num']

    final_df = pd.merge(df, score_df, on='index_1', how='left')

    sort_final_df = final_df.sort_values(by=['score'], ascending=False).reset_index()

    return sort_final_df[0:num_records]



def find_num_cluster(num_records):

    """func find_num_cluster(num_records) ->

        

        Usage : finding the number of clusters for topic modelling

            

        Input : num_records (int)

        Output : number of clusters (int)

    """

    

    num_cluster = int(num_records/5)

    num_cluster = 10 if num_cluster >=10 else num_cluster

    return num_cluster



def find_final_results(query,bm25plus ,is_covid,num_top_results):

    """func find_final_results(query,is_covid,num_results,num_cluster)->

        

        Usage : finding the search result with search score and cluster id 

                with the help of BM25Plus and Topic modelling using lda

            

        Input : query : search query 

                bm25plus : bm25plus search model

                is_covid_only : True : covid related articles or False : all articles in corpus

                num_top_results : number of top ranked document need to return

        Output : lda_model  = LDA model

                doc2bows = List of Bag of words 

                dictionary =  dictionary of final_tokens

                final_tokens = list of unigram,bigram and trigram tokens

                result_df = search result dataframe with search score

                doc_topic_df = document cluster or topic dataframe

    """

    

    num_cluster = find_num_cluster(num_top_results)

    doc_scores = find_query_results(query,bm25plus)

    result_df = find_query_df(doc_scores,is_covid,num_top_results)

    result_df['complete_text'] = result_df.apply (lambda row: create_complete_text(row,text_columns), axis=1)

    result_df['tokens'] = result_df.apply (lambda row: create_tm_tokens(row['complete_text'],lemmatizer), axis=1)

    stem_dictionary,stem_tokens_list = create_stem_dictionary_tokens(result_df['tokens'].tolist(),stemmer)

    lda_model,doc2bows,dictionary,final_tokens = create_topic_model(stem_tokens_list, stem_dictionary,num_topics=num_cluster)

    doc_topic_df = find_doc_topic_cluster(lda_model,doc2bows)

    return (lda_model,doc2bows,dictionary,final_tokens,result_df,doc_topic_df)
def find_pos_tags(word):

    """func find_pos_tags(word) ->

        

        Usage : finding the pos tagging of word

                posibility of word to be noun,adjuctive and verb

            

        Input : word 

        Output : list of pos tags like 'n' for noun, 'v' for verb etc

    """

    

    syns = wordnet.synsets(word)

    check_pos = set([p.pos() for p in syns])

    return check_pos



def find_semantic_tags(doc_no,doc_topic_df,doc2bows,dictionary,lookup_dict,num_tags):

    """func find_semantic_tags(doc_no,doc_topic_df,doc2bows,dictionary,lookup_dict,num_tags) ->

        

        Usage : find the semantic tags with bag of words and pos tagging

                Sorting with the help of Bag of words(BoW)

                Return the noun only

            

        Input : doc_no : document number

                doc_topic_df : cluster related infor

                doc2bows : list of Bag of words 

                dictionary : dictionary of corpus ( index , word)

                lookup_dict : dictionary of corpus (word , dict)

                num_tags : number of tags to be retured 

        Output : list of pos tags like 'n' for noun, 'v' for verb etc

    """

    sorted_bow = sorted(doc2bows[doc_no] , key = lambda x:x[1], reverse=True)  

    sorted_bow2 = [(lookup_dict[x],y) for x,y in sorted_bow]

    sorted_bow3 = [(x,y,find_pos_tags(x)) for x,y in sorted_bow2]

    noun_bow = [x for x,y,z in sorted_bow3 if not ('v' in z or 's' in z or 'a' in z or 'r' in z)]

    return noun_bow[0:num_tags]
def get_articles_from_df(result_df,doc_topic_df,doc_sim_scores,doc2bows,dictionary,query,query_avg_vector,word2Vec,index2word_set):

    """ function get_articles_from_df(result_df,doc_topic_df,doc_sim_scores,doc2bows,dictionary,query,query_avg_vector,word2Vec,index2word_set) ->

        

        Usage : Convert search result df to list of Article object 

                Output will be used for UI purpose

        

        Input : result_df = search result dataframe with search score

                doc_topic_df = document cluster or topic dataframe

                doc_sim_scores = list of query document similarity score

                lda_model  = LDA model

                doc2bows = List of Bag of words 

                dictionary =  dictionary of final_tokens

                final_tokens = list of unigram,bigram and trigram tokens

                query = search query

                query_avg_vector : average word vector of query

                word2Vec : word2vec model

                index2word_set : index mapping of words present in word2vec model

        Output : List of Article object        

                

        

    """

    article_list=[]

    lookup_dict = { item[0]:item[1] for item in dictionary.items()}

    for index, row in result_df.iterrows() :

        article = Article(row)

        article.set_search_rank(index)

        article.set_search_score(row['score'])

        #article.set_country_tags()

        article.set_journal_info()

        article.marked_complete_html(query,query_avg_vector,word2Vec,index2word_set)

        article.set_publish_year()

        article.set_query_sim_score(get_query_sim_score(index, doc_sim_scores))

        article.set_cluster_id(find_cluster_id(index,doc_topic_df))

        article.set_semantic_tags(find_semantic_tags(index,doc_topic_df,doc2bows,dictionary,lookup_dict,50))

        article_list.append(article)

    return article_list



def get_query_sim_score(doc_no, doc_sim_scores):

    """func get_query_sim_score(doc_no, doc_sim_scores) ->

        

        Usage : finding query document similarity score for particular document by document number

            

        Input : doc_no : document number

                sim_socres = List of document similarity score

        Output : query document similarity score(float) 

    """

    return doc_sim_scores[doc_no]



def find_cluster_id(doc_no,doc_topic_df):

    """func find_cluster_id(doc_no,doc_topic_df) ->

        

        Usage : finding cluster id for particular document by document number

            

        Input : doc_no : document number

                doc_topic_df = document topic dataframe

        Output : Cluster id (int)

    """

    

    return int(doc_topic_df.iloc[doc_no].dominant_topic) + 1



def sort_articles_by_query_sim(result_articles):

    """func sort_articles_by_query_sim(result_articles) ->

        

        Usage : Sort the list of articles by query document similarity score

            

        Input : list of result_articles

        Output : sorted list of result_articles

    """

    return sorted(result_articles , key = lambda article : article.query_sim_score, reverse=True)
# reading jinja2 templates for UI purpose

article_list_css_template =  Template(open(view_dir +"/css/article_list_style.css").read())

article_list_js_template = Template(open(view_dir +"/js/search-engine-custom.js").read())

graph_banner_template  = Template(open(view_dir +"/banner_view.html").read())

article_list_template = Template(open(view_dir +"/article_list_view.html").read())





custom_article_css_template =  Template(open(view_dir +"/css/custom_publish_style.css", encoding="utf8").read())

        



custom_article_content=""

with open(view_dir +"/custom_publish.html", "rb") as f:

    custom_article_content = f.read().decode("UTF-8")

custom_article_template = Template(custom_article_content)





def find_min_max_year(result_articles):

    """func find_min_max_year(result_articles) ->

        

        Usage : finding the min and max year from the list of result articles

                Used for UI to provide search slider corresponding to publish year

        Input : list of result_articles

        Output : (min_year, max_year)

    """

    min_year = 5000

    max_year = -1

    for article in result_articles: 

        if article.publish_year < min_year:

            min_year = article.publish_year

        if article.publish_year > max_year:

            max_year = article.publish_year

    return (min_year,max_year)



def collect_countries(result_articles):

    """func collect_countries(result_articles) ->

        

        Usage : finding all the countries from the list of result articles

                Country present in the content of result articles ( not where it is published)

                we are focusing on infected countries information in articles

                Used for UI to search based on countries

                first five country in the drop down will be in this order beacuse they are most affected

                ['China','United States','Italy','Spain','Germany','Iran']

                and other countries will be present after that based on number of times they mentioned in articles.

                

        Input : list of result_articles

        Output : List of articles

    """

    countries_list=[]

    for article in result_articles:

        for tag in article.country_tags:

                countries_list.append(tag)

    #print(countries_list)

    most_common_countries = collections.Counter(countries_list).most_common()

    final_country_list = [country for country,count in most_common_countries]

    most_common_sets = set(final_country_list)



    static_countries = ['China','United States','Italy','Spain','Germany','Iran']



    filter_static_countries = [country for country in static_countries if country in most_common_sets]

    filter_static_countries_set = set(filter_static_countries)

    filter_country_list = [ country for country in final_country_list if country not in  filter_static_countries_set]

    

    return  filter_static_countries + filter_country_list





def fetch_cluster_graphical_data(lda_model,doc2bows,dictionary):

    """func fetch_cluster_graphical_data(lda_model,doc2bows,dictionary) ->

        

        Usage : preparing the pyLDAvis graphical data for interactive visualization

                

        Input : lad_model = LDA model 

                doc2bows = List of bag of words

                dictionary = corpus dictionary used for topic modelling

        Output : pyLDAvis formatted data for graph

    """

    lda_display = pyLDAvis.gensim.prepare(lda_model, doc2bows, dictionary=dictionary)

    return lda_display



def display_complete_result(query="corona virus",num_records=25,is_covid=True):

    """ display_complete_result(query="corona virus",num_records=25,is_covid=True)->

        

        Usage : Fetching and preparing the result data for UI purpose

                

        Input : query = search query 

                num_records = number of records to fetch

                is_covid = True - for covid related articles only ; False : all articles in corpus

        Output : Display output in cell output section where it is called

    """

    

    # backend logic for retreiving all the result data

    query_avg_vector =  avg_feature_vector(query, model=word2Vec, num_features=300, index2word_set=index2word_set)

    num_cluster = find_num_cluster(num_records)

    lda_model,doc2bows,dictionary,final_tokens,result_df,doc_topic_df = find_final_results(query, bm25Plus_complete,is_covid,num_records)

    #doc_sim_scores = query_docs_similiarity(query,lda_model,doc2bows,dictionary,lemmatizer,stemmer)

    doc_sim_scores = query_docs_similiarity(query,query_avg_vector,result_df,word2Vec,index2word_set)

    

    

    

    #  logic for preparing the above backend data for UI purpose 

    result_articles = get_articles_from_df(result_df,doc_topic_df,doc_sim_scores,doc2bows,dictionary,query,query_avg_vector,word2Vec,index2word_set)  

    sorted_result_articles = sort_articles_by_query_sim(result_articles)

    cluster_graph_data = fetch_cluster_graphical_data(lda_model,doc2bows,dictionary)

    

    cluster_filter_list = [str(i+1) for i in range(0,num_cluster)]

    country_filter_list = collect_countries(sorted_result_articles)

    min_year,max_year = find_min_max_year(sorted_result_articles)

    

    complete_result_json =json.dumps(sorted_result_articles,cls=ArticleEncoder)

    

    # fetching the article list css and turn into jinja2 template to render

    css_template_str = article_list_css_template.render()

    

     # fetching the javascript and turn into jinja2 template to render

    js_template_str = article_list_js_template.render(data=complete_result_json,

                                        min_year=min_year,max_year=max_year)

    # fetching the article list html and turn into jinja2 template to render

    article_list_html = article_list_template.render(result_articles=sorted_result_articles,

                    check_pd_nan = check_pd_nan,

                    min_year=min_year,max_year=max_year,

                    country_filter_list=country_filter_list,

                    cluster_filter_list=cluster_filter_list,

                    script = js_template_str,

                    css = css_template_str)

    

    # fetching the banner html and turn into jinja2 template to render

    graph_banner_title = "Topic based Representation of Relevant Papers"

    graph_banner_html = graph_banner_template.render(title = graph_banner_title)

    

    # display command

    display(HTML(graph_banner_html))

    display(pyLDAvis.display(cluster_graph_data))

    display(HTML(article_list_html))
# Run this cell(Ctrl-Enter) to see the ranked seach articles with augmented information

query = """

            Implementation of diagnostics and products to improve clinical processes

        """

num_reocds = 25  #  we recommend between 25-100 ( this is enough articles to read to fetch required information)

is_covid = True  # True - for covid related articles only ; False :  all type of articles in corpus



display_complete_result(query,num_reocds,is_covid)
# Run this cell(Ctrl-Enter) to see the ranked seach articles with augmented information

query = """

            Natural history of the virus and shedding of it from an infected person

        """

num_reocds = 25  #  we recommend between 25-100 ( this is enough articles to read to fetch required information)

is_covid = True  # True - for covid related articles only ; False :  all type of articles in corpus



display_complete_result(query,num_reocds,is_covid)
# Run this cell(Ctrl-Enter) to see the ranked seach articles with augmented information

query = """

            Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic).

        """

num_reocds = 25  #  we recommend between 25-100 ( this is enough articles to read to fetch required information)

is_covid = True  # True - for covid related articles only ; False :  all type of articles in corpus



display_complete_result(query,num_reocds,is_covid)
# Run this cell(Ctrl-Enter) to see the ranked seach articles with augmented information

query = """

            Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood).

        """

num_reocds = 25  #  we recommend between 25-100 ( this is enough articles to read to fetch required information)

is_covid = True  # True - for covid related articles only ; False :  all type of articles in corpus



display_complete_result(query,num_reocds,is_covid)