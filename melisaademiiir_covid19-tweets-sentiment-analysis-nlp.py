# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import sklearn

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix 
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, precision_score, recall_score,  accuracy_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, precision_score, recall_score,  accuracy_score, precision_recall_curve
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.layers as L
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
!pip install emot

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import seaborn as sns
import matplotlib.pyplot as plt

import scipy

import warnings
warnings.filterwarnings('ignore')

import re
from collections import Counter
from emot.emo_unicode import UNICODE_EMO, EMOTICONS
train = pd.read_csv('/kaggle/input/covid-19-nlp-text-classification/Corona_NLP_train.csv', encoding="ISO-8859-1", low_memory=False)
test = pd.read_csv('/kaggle/input/covid-19-nlp-text-classification/Corona_NLP_test.csv', encoding="ISO-8859-1", low_memory=False) 
df = train.append(test, sort = False) #getting all together.
df
df.describe()
duplicatedRow = df[df.duplicated()]
print(duplicatedRow[:5]) #remove dublicated rows
df.shape
df.info()
display(train.isnull().sum().sort_values(ascending=False))
df['Location'].fillna(value='unknown', inplace=True) #filling missing values
encoding = {'Extremely Negative': 'Negative',
            'Extremely Positive': 'Positive'
           }

labels = ['Negative', 'Positive']
           

df['Sentiment'].replace(encoding, inplace=True) #less label
df["sentiment"] = LabelEncoder().fit_transform(df["Sentiment"])
display(df[["Sentiment", "sentiment"]].head(5))
df['CleanTweet'] = df['OriginalTweet'].copy()
display(df.head(5))
a = df.corr()
plt.figure(figsize=(9,9))
sns.heatmap(a, linewidth=.5, annot=True, fmt=".2f", annot_kws={"size":10}, cmap="viridis", vmin =0, vmax=1)
def before_lowercase(tweet):
    tweet = re.sub(r" usa ", " America ", tweet)
    tweet = re.sub(r" USA ", " America ", tweet)
    tweet = re.sub(r" u s ", " America ", tweet)
    tweet = re.sub(r" uk ", " England ", tweet)
    tweet = re.sub(r" UK ", " England ", tweet)
    tweet = re.sub(r"USAgov", "USA government", tweet)
    tweet = re.sub(r"the US", "America", tweet)
    tweet = re.sub(r"Coronavirus", " covid ", tweet)
    tweet = re.sub(r"Covid19", " covid ", tweet)
    return str(tweet)
#before lowercase I replaced some important words.
df['CleanTweet'] = df['CleanTweet'].apply(before_lowercase)
display(df['CleanTweet'].head(5))
# Function for url's
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'link', text)

from bs4 import BeautifulSoup
#Function for removing html
def html(text):
    return BeautifulSoup(text, "lxml").text
df['CleanTweet'] = df['CleanTweet'].apply(remove_urls)
df['CleanTweet'] = df['CleanTweet'].apply(html)

df['CleanTweet'] = df['CleanTweet'].str.lower()
display(df['CleanTweet'].head(5))
char_list = ["don", "ain", "ain't", "aren", "arent", "aren't", "cannot", "cant", "can't", "couldn", "couldnt", "couldn't", "didn",
               "didn't", "doesn", "doesn't", "don", "don't", "hadn", "hadn't", "hasn", "hasnt", "hasn't", "haven", "haven't", "mightn", "mightn't",
               "isn", "isn't",  "mustn", "mustn't", "needn", "needn't", "nt", "shouldn", "shouldn't",  "wasn", "wasnt", "wasn't", "don't"]

def before_lowercase(tweet0):
    tweet0 =  re.sub(r"|".join(char_list), "not", tweet0) 
    return str(tweet0)

df['CleanTweet'] = df['CleanTweet'].apply(before_lowercase)
display(df['CleanTweet'].head(15))
# Function for converting emojis into word
def convert_emojis(text):
    for emot in UNICODE_EMO:
        text = text.replace(emot, "_".join(UNICODE_EMO[emot].replace(",","").replace(":","").split()))
    return text
# Function for converting emoticons into word
def convert_emoticons(text):
    for emot in EMOTICONS:
        text = re.sub(u'('+emot+')', "_".join(EMOTICONS[emot].replace(",","").split()), text)
    return text
# Example
text = "Hello :-) :-)"
convert_emoticons(text)
df['CleanTweet'] = df['CleanTweet'].apply(convert_emojis)
df['CleanTweet'] = df['CleanTweet'].apply(convert_emoticons)
display(df['CleanTweet'].head(15))
my_stopwords = ["0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", 
                     "across", "act", "actually", "ad", "added", "adj", "ae", "af", "after", "afterwards", "ag", "again", "ah", "aj", "al", "all",
                      "also", "although", "always", "am", "among", "amongst", "amoungst",  "an", "and", "announce", 
                      "ao", "ap", "apparently", "appear",  "appropriate", "to",
                     "approximately", "ar", "are",  "arise", "around", "as", "a's", "aside",  "associated", "at", "au", "auth", "av",  "aw", "away", "ax", "ay", 
                     "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning",
                     "beginnings", "begins", "behind", "being",  "below", "beside", "besides", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", 
                     "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came",  "cc", "cd", "ce", 
                      "cf", "cg", "ch", "ci", "cit", "cj", "cl", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently",
                     "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently",
                     "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did",  "different", "dj",
                     "dk", "dl", "do", "does", "doing", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", 
                     "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "en", "end", "ending", "entirely", "eo", "ep", "eq",
                     "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "ey", "f", "f2",
                     "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows",
                     "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", 
                     "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", 
                     "h3", "had", "happens", "hardly", "has",  "have",  "having", "he", "hed", "he'd", "he'll", "hello", "hence", "her", "here", "hereafter", "hereby", "herein", "heres",
                     "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho",  "hopefully", "how", "howbeit", "however", 
                     "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig",  "ih", "ii", "ij",
                     "il", "i'll", "im", "i'm", "in", "inasmuch", "inc", "index", "indicate", "indicated", "indicates", "inner", "insofar", "interest", "into", "invention",
                     "inward", "io", "ip", "iq", "ir", "is",  "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", 
                     "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely",  "lately", "later", "latter", "latterly", "lb", "lc",
                     "le", "les", "lest", "let", "lets", "let's", "lf", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd",
                     "m", "m2", "ma", "made", "mainly", "make", "makes", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "mill", "million", "mine", 
                     "ml", "mn", "mo", "more", "moreover", "move", "mr", "mrs", "ms", "mt", "mu", "mug",  "my", "myself", "n", "n2", "na", "name", "namely", "nay", 
                     "nc", "nd", "ne", "near", "nearly","new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "nos", "noted",  "novel", "now", "nr", "ns",  "ny", "o", "oa", "ob", 
                     "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ol", "old", "om", "omitted", "on", "once", "one", "ones",  "onto", 
                     "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others",  "ou", "ought", "our", "ours", "ourselves",  "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2",
                     "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "pf", "ph", "pi", "pj", "pk", "pl", "placed", 
                      "plus", "pm", "pn", "po", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously",  "promptly", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que",
                      "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily",  "ref", "refs", "regarding",  "related", "relatively", "research-articl", "respectively",
                      "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says",
                     "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", 
                     "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should",  "should've",  "si", "side", "significant",
                     "significantly", "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone",
                     "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st",
                      "sub", "substantially", "sup", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th",  "that",
                     "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein",
                     "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've",
                     "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", 
                     "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to",  "too", "took", "top", "toward", "towards", "tp", "tq", "tr",  "ts", "t's", "tt", "tv", "twelve", "twenty",
                     "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used",  "uses", "using","ut",
                     "v", "va", "value", "various", "vd", "ve", "ve",  "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was",  "way", "we", 
                     "wed", "we'd",  "went", "were", "we're",  "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter",
                     "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll",
                     "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "with", "within",  "wo",  "words", "world", "would",  "www", "x", "x1", "x2",
                     "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours",
                     "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz",',', '.', '"', ':', ')', '(', '!', '?', '|', ';', "'", '$', '&','/', '[', ']', '>', '%', '=', '#', '*', '+', 
                '\\', '•',  '~', '@', '£', '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', 
                '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', 
                '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', 
                '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',
                'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', 
                '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', 
                '¹', '≤', '‡', '√', '«', '»', '´', 'º', '¾', '¡', '§', '£', '₤']
# for adding multiple words
print(len(my_stopwords))
def remove_swords(text,s_list):
    a=[]
    for s in text.split():
        if s not in my_stopwords:
            a.append(s)
            #remove_swords(text ,my_stopwords)
    return a     

b=[]
for t in df['CleanTweet']:
    
    b.append(remove_swords(t ,my_stopwords))
df['CleanTweet2'] = b
df['CleanTweet2'].head()
# 2.8 Combine individual words
def combine_text(input):
    combined = ' '.join(input)
    return combined
df['CleanTweet'] = df['CleanTweet2'].apply(combine_text)
df['CleanTweet']
def clean(tweet): 
            
    # Special characters
    tweet = re.sub(r"\x89Û_", "", tweet)
    tweet = re.sub(r"\x89ÛÒ", "", tweet)
    tweet = re.sub(r"\x89ÛÓ", "", tweet)
    tweet = re.sub(r"\x89ÛÏWhen", "When", tweet)
    tweet = re.sub(r"\x89ÛÏ", "", tweet)
    tweet = re.sub(r"China\x89Ûªs", "China's", tweet)
    tweet = re.sub(r"let\x89Ûªs", "let's", tweet)
    tweet = re.sub(r"\x89Û÷", "", tweet)
    tweet = re.sub(r"\x89Ûª", "", tweet)
    tweet = re.sub(r"\x89Û\x9d", "", tweet)
    tweet = re.sub(r"å_", "", tweet)
    tweet = re.sub(r"\x89Û¢", "", tweet)
    tweet = re.sub(r"\x89Û¢åÊ", "", tweet)
    tweet = re.sub(r"fromåÊwounds", "from wounds", tweet)
    tweet = re.sub(r"åÊ", "", tweet)
    tweet = re.sub(r"åÈ", "", tweet)
       
    tweet = re.sub(r"Ì©", "e", tweet)
    tweet = re.sub(r"å¨", "", tweet)
    
    tweet = re.sub(r"åÇ", "", tweet)
    
    tweet = re.sub(r"åÀ", "", tweet)
    tweet = re.sub(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'mentioned', tweet)
    tweet = re.sub(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'referance', #Replace URLs with 'httpaddr'
                     tweet)
    tweet = re.sub(r'£|\$', 'money', tweet) #Replace money symbols with 'moneysymb'
    tweet = re.sub(r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', #Replace phone numbers with 'numbers'
                   ' ', tweet)
    tweet = re.sub(r'\d+(\.\d+)?', ' ', tweet)  #Replace numbers with 'numbr'
    tweet = re.sub(r'[^\w\d\s]', ' ', tweet)
    tweet = re.sub(r'\s+', ' ', tweet)
    tweet = re.sub(r'^\s+|\s+?$', '', tweet.lower())
    
    
    # Contractions
   
    tweet = re.sub(r"he'll", "he will", tweet)
    tweet = re.sub(r"Y'all", "You all", tweet)
    tweet = re.sub(r"Weren't", "Were not", tweet)
    tweet = re.sub(r"Didn't", "Did not", tweet)
    tweet = re.sub(r"they'll", "they will", tweet)
    tweet = re.sub(r"luv", "love", tweet)
    tweet = re.sub(r"they'd", "they would", tweet)
    tweet = re.sub(r"DON'T", "DO NOT", tweet)
    tweet = re.sub(r"That\x89Ûªs", "That is", tweet)
    tweet = re.sub(r"You\x89Ûªre", "You are", tweet)
    tweet = re.sub(r"where's", "where is", tweet)
    tweet = re.sub(r"Don\x89Ûªt", "Do not", tweet)
    tweet = re.sub(r"we'd", "we would", tweet)
    tweet = re.sub(r"i'll", "I will", tweet)
    tweet = re.sub(r"weren't", "were not", tweet)
    tweet = re.sub(r"They're", "They are", tweet)
    tweet = re.sub(r"Can\x89Ûªt", "Cannot", tweet)
    tweet = re.sub(r"you\x89Ûªll", "you will", tweet)
    tweet = re.sub(r"I\x89Ûªd", "I would", tweet)
    
            
    # Character entity references
    tweet = re.sub(r"&gt;", ">", tweet)
    tweet = re.sub(r"&lt;", "<", tweet)
    tweet = re.sub(r"&amp;", "&", tweet)
    
    # Typos, slang and informal abbreviations
    tweet = re.sub(r"w/e", "whatever", tweet)
    tweet = re.sub(r"w/", "with", tweet)
   
    tweet = re.sub(r"Ph0tos", "Photos", tweet)
    tweet = re.sub(r"amirite", "am I right", tweet)
    tweet = re.sub(r"exp0sed", "exposed", tweet)
  
   
    tweet = re.sub(r"Trfc", "Traffic", tweet)
    tweet = re.sub(r"lmao", "laughing my ass off", tweet)   
   
    tweet = re.sub(r"e-mail", "email", tweet)
    tweet = re.sub(r"\s{2,}", " ", tweet)
    tweet = re.sub(r"quikly", "quickly", tweet)
    
    
    
    tweet = re.sub(r" iPhone ", " phone ", tweet)
    tweet = re.sub(r"\0rs ", " rs ", tweet) 
    
    tweet = re.sub(r"ios", "operating system", tweet)
  
    tweet = re.sub(r"programing", "programming", tweet)
    tweet = re.sub(r"bestfriend", "best friend", tweet)
    
    
    tweet = re.sub(r" J K ", " JK ", tweet)
    tweet = re.sub(r"coronavirus", " covid19", tweet)
    tweet = re.sub(r"covid", " covid19", tweet)
    tweet = re.sub(r"corrona", " covid19 ", tweet)
    tweet = re.sub(r"covid1919", " covid19 ", tweet)
    tweet = re.sub(r"_", "  ", tweet)
    
    # Urls
    tweet = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", tweet)
        
    # Words with punctuations and special characters
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`"
    for p in punctuations:
        tweet = tweet.replace(p, f' {p} ')
        
    # ... and ..
    tweet = tweet.replace('...', ' ... ')
    if '...' not in tweet:
        tweet = tweet.replace('..', ' ... ') 
        
    
        
        
    return str(tweet)
df['CleanTweet'] = df['CleanTweet'].apply(clean)
display(df['CleanTweet'].head(15))
!pip install pyspellchecker
from spellchecker import SpellChecker

spell = SpellChecker()
def correct_spellings(text):
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        elif word not in misspelled_words:
            corrected_text.append(word)
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)
        
text = "raed"
correct_spellings(text)
#df['CleanTweet'] = df['CleanTweet'].apply(SpellChecker)
display(df['CleanTweet'].head(5))
import string
regular_punct = list(string.punctuation)
#all_punct = list(set(regular_punct+ my_stopwords ))
def remove_punctuation(text,punct_list):
    for punc in punct_list:
        if punc in text:
            text = text.replace(punc, ' ')
    return text.strip()
text =" advice talk to your neighbours family to excha.."
remove_punctuation(text ,regular_punct)
df.groupby('Sentiment').describe(include=['O']).T
temp = df.groupby('Sentiment').count()['CleanTweet'].reset_index().sort_values(by='CleanTweet',ascending=False)
temp.style.background_gradient(cmap='Purples')
plt.figure(figsize=(12,6))
sns.countplot(x='Sentiment',data=df)
plt.figure(figsize=(9,6))
sns.countplot(y=df.Location, order = df.Location.value_counts().iloc[:25].index)
plt.title('Top 25 locations')
plt.show()
#Optional Step: Looking into data
display(df.sample(2)) #Sample rows of dataframe

print ( '\nSample Tweet Positive :\n-------------------------------')
print ( df[df['Sentiment']=='Positive'].CleanTweet.values[0] )

print ( '\nSample Tweet Negative :\n--------------------------------------')
print ( df[df['Sentiment']=='Negative'].CleanTweet.values[0] )

print ( '\nSample Tweet Neutral:\n--------------------------------------')
print ( df[df['Sentiment']=='Neutral'].CleanTweet.values[0] )

print ( '\nTweets distribution for Disaster Tweets (1)  and Non-Disaster Tweets (0)\n------------------------------------------------------------------------')
df['Sentiment'].hist() ;
from collections import Counter
cnt = Counter()
for text in df["CleanTweet"].values:
    for word in text.split():
        cnt[word] += 1
        
cnt.most_common(10)
def get_n_words(corpus, direction, n):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    if direction == "top":
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    else:
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=False)
    return words_freq[:n]
from sklearn.feature_extraction.text import CountVectorizer
common_words = get_n_words(df['CleanTweet'], "top", 15)
rare_words = get_n_words(df['CleanTweet'], "bottom", 15)
common_words = dict(common_words)
names = list(common_words.keys())
values = list(common_words.values())
plt.subplots(figsize = (15,10))
bars = plt.bar(range(len(common_words)),values,tick_label=names)
plt.title('15 most common words:')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x(), yval + .01, yval)
plt.show()
# Get all the pozitive and negative tweets
Positive = df[df.Sentiment =='Positive']
Negative = df[df.Sentiment=='Negative']
Neutral = df[df.Sentiment=='Neutral']
# Create numpy list to visualize using wordcloud
positive_text = " ".join(Positive.CleanTweet.to_numpy().tolist())
negative_text = " ".join(Negative.CleanTweet.to_numpy().tolist())
neutral_text = " ".join(Neutral.CleanTweet.to_numpy().tolist())
# wordcloud of pozitive messages
font_path = 'path/to/font'
mask = np.array(Image.open('../input/wordcloud-mask/indir.png'))
positive_cloud = WordCloud(width =520, height =260, stopwords=my_stopwords,max_font_size=80,
                           contour_width=5, contour_color='pink', max_words=100, background_color='white',
                           colormap='Set2', collocations=False, mask=mask).generate(positive_text)
plt.figure(figsize=(16,10))
plt.imshow(positive_cloud, interpolation='bilinear')
plt.axis('off') # turn off axis
plt.show()
# wordcloud of neutral messages
font_path = 'path/to/font'
mask = np.array(Image.open('../input/wordcloud-mask/indir.png'))
positive_cloud = WordCloud(width =520, height =260, stopwords=my_stopwords,max_font_size=80, 
                           contour_width=5, contour_color='orange', max_words=100,
                            background_color='purple',colormap='Set2', collocations=False, mask=mask).generate(negative_text)
plt.figure(figsize=(16,10))
plt.imshow(positive_cloud, interpolation='bilinear')
plt.axis('off') # turn off axis
plt.show()
# wordcloud of negative messages

font_path = 'path/to/font'
mask = np.array(Image.open('../input/wordcloud-mask/indir.png'))
positive_cloud = WordCloud(width =520, height =260, stopwords=my_stopwords,max_font_size=80, 
                           contour_width=5, contour_color='red', max_words=100,
                            background_color='black',colormap='Set2', collocations=False, mask=mask).generate(negative_text)
plt.figure(figsize=(16,10))
plt.imshow(positive_cloud, interpolation='bilinear')
plt.axis('off') # turn off axis
plt.show()

df['text_length'] = df['CleanTweet'].apply(len)
#Calculate average length by label types
labels = df.groupby('Sentiment').mean()
labels
sns.catplot(x="sentiment", y="text_length",hue="sentiment", data=df);
sns.jointplot(x=df['text_length'], y=df['sentiment']);
df['CleanTweet'] = df['CleanTweet'].apply(word_tokenize)
display(df['CleanTweet'].head(5))
lem = WordNetLemmatizer()
def lemma_wordnet(input):
    return [lem.lemmatize(w) for w in input]
df['CleanTweet'] = df['CleanTweet'].apply(lemma_wordnet)
display(df['CleanTweet'].head(5))
# 2.8 Combine individual words
def combine_text(input):
    combined = ' '.join(input)
    return combined
df['CleanTweet'] = df['CleanTweet'].apply(combine_text)
df['CleanTweet']
train, test = train_test_split(df)
# Bag of words

cv = CountVectorizer()
cv.fit(train)
X_train_bow = cv.fit_transform(train['CleanTweet']) #X_train
X_test_bow = train['sentiment'] #Y_train
Y_train_bow = cv.transform(test['CleanTweet']) #X_test
Y_test = test['sentiment'] # Y_test

# 3.2 TF-IDF

vectorizer = TfidfVectorizer(norm = None)
vectorizer.fit(train)
X_train_tfidf = vectorizer.fit_transform(train['CleanTweet'])
X_test_tfidf = train['sentiment'] #
Y_train_tdidf =vectorizer.transform(test['CleanTweet']) #vectorizer.fit_transform

# 3.3 Hashing

hv = HashingVectorizer()
hv.fit(train)
X_train_hash = hv.fit_transform(train['CleanTweet'])
X_test_hash = train['sentiment']
Y_train_hash = hv.transform(test['CleanTweet']) 
display("Bow-TF:IDF :", X_train_bow.shape)
df_tfidf = pd.DataFrame(X_train_bow.toarray(), columns=cv.get_feature_names())
display(df_tfidf.head())
# Rigde with bag of word
from sklearn import linear_model
alpha = [80.0, 90.0, 100.0, 110.0, 120.0] 
for a in alpha:
    ridge = linear_model.RidgeClassifier(a)
    scores = sklearn.model_selection.cross_val_score(ridge, X_train_bow, X_test_bow, cv=5)#scoring='f1' kaldirdim multiclass hatasina karsilik
    print("alpha: ",a)
    print(scores)
    print(np.mean(scores))
    print('\n')
# MultinomialNB with bag of word
from sklearn.naive_bayes import MultinomialNB
alpha = [1e-10, 1e-5, 0.1, 1.0, 2.0, 5.0]
for a in alpha:
    mnb = MultinomialNB(a)
    scores = sklearn.model_selection.cross_val_score(mnb, X_train_bow, X_test_bow, cv=5)
    print('alpha: ', a)
    print(scores)
    print(np.mean(scores))
    print('\n')
# MultinomialNB with TF-IDF
alpha = [175.0, 200.0, 225.0, 250.0, 300.0]
for a in alpha:
    mnb = MultinomialNB(a)
    scores = sklearn.model_selection.cross_val_score(mnb, X_train_tfidf, X_test_tfidf, cv=5)
    print('alpha: ', a)
    print(scores)
    print(np.mean(scores))
    print('\n')
# Rigde with Hash
alpha = [1.1, 1.2, 1.3, 1.4, 1.5, 2.0]
for a in alpha:
    ridge = linear_model.RidgeClassifier(a)
    scores = sklearn.model_selection.cross_val_score(ridge, X_train_hash, X_test_hash, cv=5)
    print("alpha: ",a)
    print(scores)
    print(np.mean(scores))
    print('\n')
# Rigde with TF-IDF
alpha = [500.0, 1500.0, 2500.0, 3000.0]
for a in alpha:
    ridge = linear_model.RidgeClassifier(a)
    scores = sklearn.model_selection.cross_val_score(ridge, X_train_tfidf, X_test_tfidf, cv=5)
    print("alpha: ",a)
    print(scores)
    print(np.mean(scores))
    print('\n')
from sklearn.metrics import accuracy_score
ridge = linear_model.RidgeClassifier(1.4)
ridge.fit(X_train_hash, X_test_hash)
test['sentiment_pred'] = ridge.predict(Y_train_hash)
y_true = test['sentiment']
y_pred = test['sentiment_pred']
accuracy_score(y_true, y_pred)
from sklearn.metrics import classification_report,confusion_matrix 
print(classification_report(y_true, y_pred, target_names = ['Negative Tweets','Neutral Tweets', 'Positive Tweets']))
import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize = (9,9))
sns.heatmap(cm,cmap= "Blues", 
            linecolor = 'black', 
            linewidth = 1, 
            annot = True, 
            fmt='', 
            xticklabels = ['Negative Tweets','Neutral Tweets', 'Positive Tweets'], 
            yticklabels = ['Negative Tweets','Neutral Tweets', 'Positive Tweets'])
plt.xlabel("Predicted")
plt.ylabel("Actual")

dtclassifier=DecisionTreeClassifier(criterion="entropy", max_depth=None)
dtclassifier.fit(X_train_bow,train['sentiment'])
preddt = dtclassifier.predict(Y_train_bow)
accuracy= accuracy_score(preddt,Y_test)
print(accuracy)
dtclassifier=DecisionTreeClassifier(criterion="entropy", max_depth=None) 
dtclassifier.fit(X_train_tfidf,train['sentiment'])
preddt = dtclassifier.predict(Y_train_tdidf) 
accuracy= accuracy_score(preddt,Y_test)
print(accuracy)
dtclassifier=DecisionTreeClassifier(criterion="entropy", max_depth=None)
dtclassifier.fit(X_train_hash,train['sentiment'])
preddt = dtclassifier.predict(Y_train_hash) 
from sklearn.feature_extraction.text import TfidfVectorizer
# Create feature vectors
vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)
vectors = vectorizer.fit_transform(df['CleanTweet']) #all data bunu boyle yaptik cunku matrrislerde uyumsuzluk olsun isteemiyoruz. islemde sorun yasayabilirim cunku.
#test_vectors = vectorizer.transform(al['Sentiment'])
test_vectors = vectors[40000:]
train_vectors = vectors[:40000]
# Perform classification with SVM, kernel=linear
import time
from sklearn import svm
from sklearn.metrics import classification_report
classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(train_vectors, df['Sentiment'][:40000])
t1 = time.time()
prediction_linear = classifier_linear.predict(test_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1

# results
print("Results for SVC(kernel=linear)")
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
report = classification_report(df['Sentiment'][40000:], prediction_linear, output_dict=True)
print('positive: ', report['Positive'])
print('negative: ', report['Negative'])
print('notr: ', report['Neutral'])
review = """I can help"""
review_vector = vectorizer.transform([review]) # vectorizing
print(classifier_linear.predict(review_vector))
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(solver='lbfgs', multi_class="ovr")
log_reg.fit(X_train_hash, X_test_hash)
train_accuracy = log_reg.score(X_train_hash, X_test_hash)
test_accuracy = log_reg.score(Y_train_hash, Y_test)

print('One-vs.-Rest', '-'*30, 
      'Accuracy on Train Data : {:.2f}'.format(train_accuracy), 
      'Accuracy on Test Data  : {:.2f}'.format(test_accuracy), sep='\n')
log_reg_mnm = LogisticRegression(multi_class='multinomial', solver='lbfgs')
log_reg_mnm.fit(X_train_hash, X_test_hash)

train_accuracy = log_reg_mnm.score(X_train_hash, X_test_hash)
test_accuracy = log_reg_mnm.score(Y_train_hash, Y_test)

print('Multinomial (Softmax)', '-'*20, 
      'Accuracy on Train Data : {:.2f}'.format(train_accuracy), 
      'Accuracy on Test Data  : {:.2f}'.format(test_accuracy), sep='\n')

C_values = [0.001,0.01, 0.1,1,10,100, 1000]

accuracy_values = pd.DataFrame(columns=['C_values', 'Train Accuracy', 'Test Accuracy'])

for c in C_values:
    # Apply logistic regression model to training data
    lr = LogisticRegression(penalty = 'l2', C = c, random_state = 0, solver='lbfgs', multi_class='multinomial')
    lr.fit(X_train_hash, X_test_hash)
    accuracy_values = accuracy_values.append({'C_values': c,
                                              'Train Accuracy': lr.score(X_train_hash, X_test_hash),
                                              'Test Accuracy': lr.score(Y_train_hash, Y_test)
                                             }, ignore_index=True)
display(accuracy_values)    
parameters = {"C": [10 ** x for x in range (-5, 5, 1)],
              "penalty": ['l1', 'l2']
             }
from sklearn.model_selection import GridSearchCV

grid_cv = GridSearchCV(estimator=log_reg,
                       param_grid = parameters,
                       cv = 10
                      )

grid_cv.fit(X_train_hash, X_test_hash)
print("Best Parameters : ", grid_cv.best_params_)
print("Best Score      : ", grid_cv.best_score_)

%time results = grid_cv.cv_results_

df1 = pd.DataFrame(results)
display(df1.head(35))
df1.info()
df1 = df1[['param_penalty','param_C', 'mean_test_score']]
df1 = df1.sort_values(by='mean_test_score', ascending = False)
df1






