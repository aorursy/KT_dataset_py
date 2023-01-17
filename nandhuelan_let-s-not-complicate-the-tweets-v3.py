# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



!pip install --upgrade scikit-learn

from catboost import CatBoostClassifier

from sklearn.experimental import enable_hist_gradient_boosting  # noqa

from sklearn.ensemble import HistGradientBoostingClassifier,StackingClassifier





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.core.display import display, HTML



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

from plotly import tools

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

from matplotlib_venn import venn2

import re

import matplotlib.pyplot as plt

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from bs4 import BeautifulSoup

import string

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import torch

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn import metrics

import seaborn as sns

from sklearn.model_selection import cross_val_score,train_test_split,StratifiedKFold

import sys

import torch

import gc

import tensorflow as tf

from tqdm import tqdm_notebook

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Concatenate

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score,plot_roc_curve

import folium 

from folium import plugins 

import lightgbm as lgb

from lightgbm import LGBMClassifier

import tensorflow_hub as hub

from gensim.models import word2vec

import itertools

from sklearn.manifold import TSNE

import itertools

STOPWORDS = set(stopwords.words('english'))

from IPython.display import Markdown



def bold(string):

    display(Markdown(string))

%matplotlib inline

# Any results you write to the current directory are saved as output.
# !pip install simpletransformers



# from simpletransformers.classification import ClassificationModel 
inputpath='../input/nlp-getting-started'



print("Reading the data")

traindata=pd.read_csv(inputpath+'/train.csv')

testdata=pd.read_csv(inputpath+'/test.csv')

submission=pd.read_csv(inputpath+'/sample_submission.csv')
#Reference: https://www.kaggle.com/marcovasquez/basic-nlp-with-tensorflow-and-wordcloud



display(HTML(f"""

   

        <ul class="list-group">

          <li class="list-group-item disabled" aria-disabled="true"><h4>Shape of Train and Test Dataset</h4></li>

          <li class="list-group-item"><h4>Number of rows in Train dataset is: <span class="label label-primary">{ traindata.shape[0]:,}</span></h4></li>

          <li class="list-group-item"> <h4>Number of columns Train dataset is <span class="label label-primary">{traindata.shape[1]}</span></h4></li>

          <li class="list-group-item"><h4>Number of rows in Test dataset is: <span class="label label-success">{ testdata.shape[0]:,}</span></h4></li>

          <li class="list-group-item"><h4>Number of columns Test dataset is <span class="label label-success">{testdata.shape[1]}</span></h4></li>

        </ul>

  

    """))
traindata.head()
'''A Function To Plot Pie Plot using Plotly'''



def pie_plot(cnt_srs, colors, title):

    labels=cnt_srs.index

    values=cnt_srs.values

    trace = go.Pie(labels=labels, 

                   values=values, 

                   title=title, 

                   hoverinfo='percent+value', 

                   textinfo='percent',

                   textposition='inside',

                   hole=0.7,

                   showlegend=True,

                   marker=dict(colors=colors,

                               line=dict(color='#000000',

                                         width=2),

                              )

                  )

    return trace



bold("**Non disaster tweets vs Disaster tweets**")

py.iplot([pie_plot(traindata['target'].value_counts(), ['cyan', 'green'], 'Tweets')])

compare_cols = ['keyword', 'location']



def get_trace(col, df, color):

    temp = df[col].value_counts().nlargest(5)

    x = list(reversed(list(temp.index)))

    y = list(reversed(list(temp.values)))

    trace = go.Bar(x = y, y = x, width = [0.9, 0.9, 0.9], orientation='h', marker=dict(color=color))

    return trace





traintraces = []

traintitles = []

for i,col in enumerate(compare_cols):

    traintitles.append(col)

    traintraces.append(get_trace(col, traindata, '#a3dd56'))



    

testtraces = []

testtitles = []

for i,col in enumerate(compare_cols):

    testtitles.append(col)

    testtraces.append(get_trace(col, testdata, '#ef7067'))



titles = []

for each in compare_cols:

    titles.append(each)

    titles.append(each)

fig = tools.make_subplots(rows=len(compare_cols), cols=2, print_grid=False, horizontal_spacing = 0.15, subplot_titles=titles)



i = 0



for g,b in zip(traintraces, testtraces):

    i += 1

    fig.append_trace(g, i, 1);

    fig.append_trace(b, i, 2);



fig['layout'].update(height=1000, margin=dict(l=100), showlegend=False, title="Comparing Features of train and test data",template= "plotly_dark");

iplot(fig, filename='simple-subplot');   
plt.figure(figsize=(23,13))



plt.subplot(321)

venn2([set(traindata.keyword.unique()), set(testdata.keyword.unique())], set_labels = ('Train set', 'Test set') )

plt.title("Common keyword in training and test data", fontsize=15)



plt.subplot(322)

venn2([set(traindata.location.unique()), set(testdata.location.unique())], set_labels = ('Train set', 'Test set') )

plt.title("Common location in training and test data", fontsize=15)
traindata["text"] = traindata["text"].astype(str)

testdata["text"] = testdata["text"].astype(str)
add_stopwords = [

    "a", "about", "above", "after", "again", "against", "ain", "all", "am",

    "an", "and", "any", "are", "aren", "aren't", "as", "at", "be", "because",

    "been", "before", "being", "below", "between", "both", "but", "by", "can",

    "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn",

    "doesn't", "doing", "don", "don't", "down", "during", "each", "few", "for",

    "from", "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have",

    "haven", "haven't", "having", "he", "her", "here", "hers", "herself", "him",

    "himself", "his", "how", "i", "if", "in", "into", "is", "isn", "isn't",

    "it", "it's", "its", "itself", "just", "ll", "m", "ma", "me", "mightn",

    "mightn't", "more", "most", "mustn", "mustn't", "my", "myself", "needn",

    "needn't", "no", "nor", "not", "now", "o", "of", "off", "on", "once",

    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own",

    "re", "s", "same", "shan", "shan't", "she", "she's", "should", "should've",

    "shouldn", "shouldn't", "so", "some", "such", "t", "than", "that",

    "that'll", "the", "their", "theirs", "them", "themselves", "then", "there",

    "these", "they", "this", "those", "through", "to", "too", "under", "until",

    "up", "ve", "very", "was", "wasn", "wasn't", "we", "were", "weren",

    "weren't", "what", "when", "where", "which", "while", "who", "whom", "why",

    "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd",

    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",

    "could", "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm",

    "i've", "let's", "ought", "she'd", "she'll", "that's", "there's", "they'd",

    "they'll", "they're", "they've", "we'd", "we'll", "we're", "we've",

    "what's", "when's", "where's", "who's", "why's", "would", "able", "abst",

    "accordance", "according", "accordingly", "across", "act", "actually",

    "added", "adj", "affected", "affecting", "affects", "afterwards", "ah",

    "almost", "alone", "along", "already", "also", "although", "always",

    "among", "amongst", "announce", "another", "anybody", "anyhow", "anymore",

    "anyone", "anything", "anyway", "anyways", "anywhere", "apparently",

    "approximately", "arent", "arise", "around", "aside", "ask", "asking",

    "auth", "available", "away", "awfully", "b", "back", "became", "become",

    "becomes", "becoming", "beforehand", "begin", "beginning", "beginnings",

    "begins", "behind", "believe", "beside", "besides", "beyond", "biol",

    "brief", "briefly", "c", "ca", "came", "cannot", "can't", "cause", "causes",

    "certain", "certainly", "co", "com", "come", "comes", "contain",

    "containing", "contains", "couldnt", "date", "different", "done",

    "downwards", "due", "e", "ed", "edu", "effect", "eg", "eight", "eighty",

    "either", "else", "elsewhere", "end", "ending", "enough", "especially",

    "et", "etc", "even", "ever", "every", "everybody", "everyone", "everything",

    "everywhere", "ex", "except", "f", "far", "ff", "fifth", "first", "five",

    "fix", "followed", "following", "follows", "former", "formerly", "forth",

    "found", "four", "furthermore", "g", "gave", "get", "gets", "getting",

    "give", "given", "gives", "giving", "go", "goes", "gone", "got", "gotten",

    "h", "happens", "hardly", "hed", "hence", "hereafter", "hereby", "herein",

    "heres", "hereupon", "hes", "hi", "hid", "hither", "home", "howbeit",

    "however", "hundred", "id", "ie", "im", "immediate", "immediately",

    "importance", "important", "inc", "indeed", "index", "information",

    "instead", "invention", "inward", "itd", "it'll", "j", "k", "keep", "keeps",

    "kept", "kg", "km", "know", "known", "knows", "l", "largely", "last",

    "lately", "later", "latter", "latterly", "least", "less", "lest", "let",

    "lets", "like", "liked", "likely", "line", "little", "'ll", "look",

    "looking", "looks", "ltd", "made", "mainly", "make", "makes", "many", "may",

    "maybe", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might",

    "million", "miss", "ml", "moreover", "mostly", "mr", "mrs", "much", "mug",

    "must", "n", "na", "name", "namely", "nay", "nd", "near", "nearly",

    "necessarily", "necessary", "need", "needs", "neither", "never",

    "nevertheless", "new", "next", "nine", "ninety", "nobody", "non", "none",

    "nonetheless", "noone", "normally", "nos", "noted", "nothing", "nowhere",

    "obtain", "obtained", "obviously", "often", "oh", "ok", "okay", "old",

    "omitted", "one", "ones", "onto", "ord", "others", "otherwise", "outside",

    "overall", "owing", "p", "page", "pages", "part", "particular",

    "particularly", "past", "per", "perhaps", "placed", "please", "plus",

    "poorly", "possible", "possibly", "potentially", "pp", "predominantly",

    "present", "previously", "primarily", "probably", "promptly", "proud",

    "provides", "put", "q", "que", "quickly", "quite", "qv", "r", "ran",

    "rather", "rd", "readily", "really", "recent", "recently", "ref", "refs",

    "regarding", "regardless", "regards", "related", "relatively", "research",

    "respectively", "resulted", "resulting", "results", "right", "run", "said",

    "saw", "say", "saying", "says", "sec", "section", "see", "seeing", "seem",

    "seemed", "seeming", "seems", "seen", "self", "selves", "sent", "seven",

    "several", "shall", "shed", "shes", "show", "showed", "shown", "showns",

    "shows", "significant", "significantly", "similar", "similarly", "since",

    "six", "slightly", "somebody", "somehow", "someone", "somethan",

    "something", "sometime", "sometimes", "somewhat", "somewhere", "soon",

    "sorry", "specifically", "specified", "specify", "specifying", "still",

    "stop", "strongly", "sub", "substantially", "successfully", "sufficiently",

    "suggest", "sup", "sure", "take", "taken", "taking", "tell", "tends", "th",

    "thank", "thanks", "thanx", "thats", "that've", "thence", "thereafter",

    "thereby", "thered", "therefore", "therein", "there'll", "thereof",

    "therere", "theres", "thereto", "thereupon", "there've", "theyd", "theyre",

    "think", "thou", "though", "thoughh", "thousand", "throug", "throughout",

    "thru", "thus", "til", "tip", "together", "took", "toward", "towards",

    "tried", "tries", "truly", "try", "trying", "ts", "twice", "two", "u", "un",

    "unfortunately", "unless", "unlike", "unlikely", "unto", "upon", "ups",

    "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using",

    "usually", "v", "value", "various", "'ve", "via", "viz", "vol", "vols",

    "vs", "w", "want", "wants", "wasnt", "way", "wed", "welcome", "went",

    "werent", "whatever", "what'll", "whats", "whence", "whenever",

    "whereafter", "whereas", "whereby", "wherein", "wheres", "whereupon",

    "wherever", "whether", "whim", "whither", "whod", "whoever", "whole",

    "who'll", "whomever", "whos", "whose", "widely", "willing", "wish",

    "within", "without", "wont", "words", "world", "wouldnt", "www", "x", "yes",

    "yet", "youd", "youre", "z", "zero", "a's", "ain't", "allow", "allows",

    "apart", "appear", "appreciate", "appropriate", "associated", "best",

    "better", "c'mon", "c's", "cant", "changes", "clearly", "concerning",

    "consequently", "consider", "considering", "corresponding", "course",

    "currently", "definitely", "described", "despite", "entirely", "exactly",

    "example", "going", "greetings", "hello", "help", "hopefully", "ignored",

    "inasmuch", "indicate", "indicated", "indicates", "inner", "insofar",

    "it'd", "keep", "keeps", "novel", "presumably", "reasonably", "second",

    "secondly", "sensible", "serious", "seriously", "sure", "t's", "third",

    "thorough", "thoroughly", "three", "well", "wonder", "a", "about", "above",

    "above", "across", "after", "afterwards", "again", "against", "all",

    "almost", "alone", "along", "already", "also", "although", "always", "am",

    "among", "amongst", "amoungst", "amount", "an", "and", "another", "any",

    "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as",

    "at", "back", "be", "became", "because", "become", "becomes", "becoming",

    "been", "before", "beforehand", "behind", "being", "below", "beside",

    "besides", "between", "beyond", "bill", "both", "bottom", "but", "by",

    "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry",

    "de", "describe", "detail", "do", "done", "down", "due", "during", "each",

    "eg", "eight", "either", "eleven", "else", "elsewhere", "empty", "enough",

    "etc", "even", "ever", "every", "everyone", "everything", "everywhere",

    "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five",

    "for", "former", "formerly", "forty", "found", "four", "from", "front",

    "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he",

    "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers",

    "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if",

    "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself",

    "keep", "last", "latter", "latterly", "least", "less", "ltd", "made",

    "many", "may", "me", "meanwhile", "might", "mill", "mine", "more",

    "moreover", "most", "mostly", "move", "much", "must", "my", "myself",

    "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no",

    "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of",

    "off", "often", "on", "once", "one", "only", "onto", "or", "other",

    "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own",

    "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see",

    "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should",

    "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow",

    "someone", "something", "sometime", "sometimes", "somewhere", "still",

    "such", "system", "take", "ten", "than", "that", "the", "their", "them",

    "themselves", "then", "thence", "there", "thereafter", "thereby",

    "therefore", "therein", "thereupon", "these", "they", "thickv", "thin",

    "third", "this", "those", "though", "three", "through", "throughout",

    "thru", "thus", "to", "together", "too", "top", "toward", "towards",

    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",

    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",

    "whence", "whenever", "where", "whereafter", "whereas", "whereby",

    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",

    "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within",

    "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves",

    "the", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",

    "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C",

    "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R",

    "S", "T", "U", "V", "W", "X", "Y", "Z", "co", "op", "research-articl",

    "pagecount", "cit", "ibid", "les", "le", "au", "que", "est", "pas", "vol",

    "el", "los", "pp", "u201d", "well-b", "http", "volumtype", "par", "0o",

    "0s", "3a", "3b", "3d", "6b", "6o", "a1", "a2", "a3", "a4", "ab", "ac",

    "ad", "ae", "af", "ag", "aj", "al", "an", "ao", "ap", "ar", "av", "aw",

    "ax", "ay", "az", "b1", "b2", "b3", "ba", "bc", "bd", "be", "bi", "bj",

    "bk", "bl", "bn", "bp", "br", "bs", "bt", "bu", "bx", "c1", "c2", "c3",

    "cc", "cd", "ce", "cf", "cg", "ch", "ci", "cj", "cl", "cm", "cn", "cp",

    "cq", "cr", "cs", "ct", "cu", "cv", "cx", "cy", "cz", "d2", "da", "dc",

    "dd", "de", "df", "di", "dj", "dk", "dl", "do", "dp", "dr", "ds", "dt",

    "du", "dx", "dy", "e2", "e3", "ea", "ec", "ed", "ee", "ef", "ei", "ej",

    "el", "em", "en", "eo", "ep", "eq", "er", "es", "et", "eu", "ev", "ex",

    "ey", "f2", "fa", "fc", "ff", "fi", "fj", "fl", "fn", "fo", "fr", "fs",

    "ft", "fu", "fy", "ga", "ge", "gi", "gj", "gl", "go", "gr", "gs", "gy",

    "h2", "h3", "hh", "hi", "hj", "ho", "hr", "hs", "hu", "hy", "i", "i2", "i3",

    "i4", "i6", "i7", "i8", "ia", "ib", "ic", "ie", "ig", "ih", "ii", "ij",

    "il", "in", "io", "ip", "iq", "ir", "iv", "ix", "iy", "iz", "jj", "jr",

    "js", "jt", "ju", "ke", "kg", "kj", "km", "ko", "l2", "la", "lb", "lc",

    "lf", "lj", "ln", "lo", "lr", "ls", "lt", "m2", "ml", "mn", "mo", "ms",

    "mt", "mu", "n2", "nc", "nd", "ne", "ng", "ni", "nj", "nl", "nn", "nr",

    "ns", "nt", "ny", "oa", "ob", "oc", "od", "of", "og", "oi", "oj", "ol",

    "om", "on", "oo", "oq", "or", "os", "ot", "ou", "ow", "ox", "oz", "p1",

    "p2", "p3", "pc", "pd", "pe", "pf", "ph", "pi", "pj", "pk", "pl", "pm",

    "pn", "po", "pq", "pr", "ps", "pt", "pu", "py", "qj", "qu", "r2", "ra",

    "rc", "rd", "rf", "rh", "ri", "rj", "rl", "rm", "rn", "ro", "rq", "rr",

    "rs", "rt", "ru", "rv", "ry", "s2", "sa", "sc", "sd", "se", "sf", "si",

    "sj", "sl", "sm", "sn", "sp", "sq", "sr", "ss", "st", "sy", "sz", "t1",

    "t2", "t3", "tb", "tc", "td", "te", "tf", "th", "ti", "tj", "tl", "tm",

    "tn", "tp", "tq", "tr", "ts", "tt", "tv", "tx", "ue", "ui", "uj", "uk",

    "um", "un", "uo", "ur", "ut", "va", "wa", "vd", "wi", "vj", "vo", "wo",

    "vq", "vt", "vu", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn",

    "xo", "xs", "xt", "xv", "xx", "y2", "yj", "yl", "yr", "ys", "yt", "zi", "zz"

]



mispell_dict = {"aren't" : "are not",

"can't" : "cannot",

"couldn't" : "could not",

"couldnt" : "could not",

"didn't" : "did not",

"doesn't" : "does not",

"doesnt" : "does not",

"don't" : "do not",

"hadn't" : "had not",

"hasn't" : "has not",

"haven't" : "have not",

"havent" : "have not",

"he'd" : "he would",

"he'll" : "he will",

"he's" : "he is",

"i'd" : "I would",

"i'd" : "I had",

"i'll" : "I will",

"i'm" : "I am",

"isn't" : "is not",

"it's" : "it is",

"it'll":"it will",

"i've" : "I have",

"let's" : "let us",

"mightn't" : "might not",

"mustn't" : "must not",

"shan't" : "shall not",

"she'd" : "she would",

"she'll" : "she will",

"she's" : "she is",

"shouldn't" : "should not",

"shouldnt" : "should not",

"that's" : "that is",

"thats" : "that is",

"there's" : "there is",

"theres" : "there is",

"they'd" : "they would",

"they'll" : "they will",

"they're" : "they are",

"theyre":  "they are",

"they've" : "they have",

"we'd" : "we would",

"we're" : "we are",

"weren't" : "were not",

"we've" : "we have",

"what'll" : "what will",

"what're" : "what are",

"what's" : "what is",

"what've" : "what have",

"where's" : "where is",

"who'd" : "who would",

"who'll" : "who will",

"who're" : "who are",

"who's" : "who is",

"who've" : "who have",

"won't" : "will not",

"wouldn't" : "would not",

"you'd" : "you would",

"you'll" : "you will",

"you're" : "you are",

"you've" : "you have",

"'re": " are",

"wasn't": "was not",

"we'll":" will",

"didn't": "did not",

"tryin'":"trying"}



#Reference: https://www.kaggle.com/jdparsons/tweet-cleaner

slang_abbrev_dict = {

    'AFAIK': 'As Far As I Know',

    'AFK': 'Away From Keyboard',

    'ASAP': 'As Soon As Possible',

    'ATK': 'At The Keyboard',

    'ATM': 'At The Moment',

    'A3': 'Anytime, Anywhere, Anyplace',

    'BAK': 'Back At Keyboard',

    'BBL': 'Be Back Later',

    'BBS': 'Be Back Soon',

    'BFN': 'Bye For Now',

    'B4N': 'Bye For Now',

    'BRB': 'Be Right Back',

    'BRT': 'Be Right There',

    'BTW': 'By The Way',

    'B4': 'Before',

    'B4N': 'Bye For Now',

    'CU': 'See You',

    'CUL8R': 'See You Later',

    'CYA': 'See You',

    'FAQ': 'Frequently Asked Questions',

    'FC': 'Fingers Crossed',

    'FWIW': 'For What It\'s Worth',

    'FYI': 'For Your Information',

    'GAL': 'Get A Life',

    'GG': 'Good Game',

    'GN': 'Good Night',

    'GMTA': 'Great Minds Think Alike',

    'GR8': 'Great!',

    'G9': 'Genius',

    'IC': 'I See',

    'ICQ': 'I Seek you',

    'ILU': 'I Love You',

    'IMHO': 'In My Humble Opinion',

    'IMO': 'In My Opinion',

    'IOW': 'In Other Words',

    'IRL': 'In Real Life',

    'KISS': 'Keep It Simple, Stupid',

    'LDR': 'Long Distance Relationship',

    'LMAO': 'Laugh My Ass Off',

    'LOL': 'Laughing Out Loud',

    'LTNS': 'Long Time No See',

    'L8R': 'Later',

    'MTE': 'My Thoughts Exactly',

    'M8': 'Mate',

    'NRN': 'No Reply Necessary',

    'OIC': 'Oh I See',

    'OMG': 'Oh My God',

    'PITA': 'Pain In The Ass',

    'PRT': 'Party',

    'PRW': 'Parents Are Watching',

    'QPSA?': 'Que Pasa?',

    'ROFL': 'Rolling On The Floor Laughing',

    'ROFLOL': 'Rolling On The Floor Laughing Out Loud',

    'ROTFLMAO': 'Rolling On The Floor Laughing My Ass Off',

    'SK8': 'Skate',

    'STATS': 'Your sex and age',

    'ASL': 'Age, Sex, Location',

    'THX': 'Thank You',

    'TTFN': 'Ta-Ta For Now!',

    'TTYL': 'Talk To You Later',

    'U': 'You',

    'U2': 'You Too',

    'U4E': 'Yours For Ever',

    'WB': 'Welcome Back',

    'WTF': 'What The Fuck',

    'WTG': 'Way To Go!',

    'WUF': 'Where Are You From?',

    'W8': 'Wait',

    '7K': 'Sick:-D Laugher'

}

traindata['Hashtags']=traindata['text'].apply(lambda x: re.findall(r'#(\w+)',x))

testdata['Hashtags']=testdata['text'].apply(lambda x: re.findall(r'#(\w+)',x))
traindata.head()
hashes=pd.DataFrame(data=list(itertools.chain(*traindata['Hashtags'].values)),columns=['Hashes'])



temp_df =  hashes['Hashes'].value_counts()[:15]

data = go.Bar(x = temp_df.index,y = temp_df.values,text = temp_df.values,  textposition='auto')

fig = go.Figure(data = data)

fig.update_traces(marker_color='#C5197D', marker_line_color='#8E0052',marker_line_width=1.5, opacity=0.6)

fig.update_layout(barmode='stack', title={'text': "Trending hashtags",'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'},template= "plotly_dark")

fig.show()
PUNCT_TO_REMOVE = string.punctuation

def remove_punctuation(text):

    """custom function to remove the punctuation"""

    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))





def clean_text(x):

    x = str(x).replace("\n","")

    

    stops  = set(list(STOPWORDS)+add_stopwords)

    text = [w for w in word_tokenize(x) if w not in stops]    

    text = " ".join(text)

    

    return text





def clean_numbers(x):

    x = re.sub('[0-9]{5,}', '#####', x)

    x = re.sub('[0-9]{4}', '####', x)

    x = re.sub('[0-9]{3}', '###', x)

    x = re.sub('[0-9]{2}', '##', x)

    return x





def _get_mispell(mispell_dict):

    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))

    return mispell_dict, mispell_re





def replace_typical_misspell(text):

    mispellings, mispellings_re = _get_mispell(mispell_dict)



    def replace(match):

        return mispellings[match.group(0)]



    return mispellings_re.sub(replace, text)





def unslang(text):

    """Converts text like "OMG" into "Oh my God"

    """

    text = [slang_abbrev_dict[w.upper()] if w.upper() in slang_abbrev_dict.keys() else w for w in word_tokenize(text)]    

    return " ".join(text)

    



# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b

def remove_emoji(string):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', string)





def remove_urls(text):

    url_pattern = re.compile(r'https?://\S+|www\.\S+')

    return url_pattern.sub(r'', text)



def remove_html(text):

    return BeautifulSoup(text, "lxml").text





def clean_data(df, col):

        df[col] = df[col].apply(lambda x: clean_numbers(x))

        df[col] = df[col].apply(lambda x: remove_urls(x))

        df[col] = df[col].apply(lambda x: remove_html(x))

        df[col] = df[col].apply(lambda text: remove_punctuation(text))

        df[col] = df[col].apply(lambda x: clean_text(x.lower()))

        df[col] = df[col].apply(lambda x: replace_typical_misspell(x))

        df[col] = df[col].apply(lambda x: unslang(x))

        df[col] = df[col].apply(lambda x: remove_emoji(x))

        return df
traindata = clean_data(traindata, 'text')

testdata = clean_data(testdata, 'text')
def plot_wordcloud(text, mask=None, max_words=400, max_font_size=120, figure_size=(24.0,16.0), 

                   title = None, title_size=40, image_color=False):

    stopwords = set(STOPWORDS)



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

    

d = '../input/masks/masks-wordclouds/'

comments_text = str(traindata.text)

comments_mask = np.array(Image.open(d + 'upvote.png'))

plot_wordcloud(comments_text, comments_mask, max_words=2000, max_font_size=300, 

               title = 'Most common words in all of the tweets', title_size=30)
%%time

module_url = 'https://tfhub.dev/google/universal-sentence-encoder-large/4'

embed = hub.KerasLayer(module_url, trainable=False, name='USE_embedding')
USE_train_embeddings = embed(traindata.text.values)

USE_test_embeddings = embed(testdata.text.values)



del embed
lr_cross = LogisticRegression(solver='lbfgs')

dtc_cross = DecisionTreeClassifier()

rfc_cross = RandomForestClassifier(n_estimators=50)

knn_cross = KNeighborsClassifier(n_neighbors=1)
lr_scores = cross_val_score(lr_cross,USE_train_embeddings['outputs'].numpy(),traindata.target.values,cv=10,scoring='f1')

dtc_scores = cross_val_score(dtc_cross,USE_train_embeddings['outputs'].numpy(),traindata.target.values,cv=10,scoring='f1')

rfc_scores = cross_val_score(rfc_cross,USE_train_embeddings['outputs'].numpy(),traindata.target.values,cv=10,scoring='f1')

knn_scores = cross_val_score(knn_cross,USE_train_embeddings['outputs'].numpy(),traindata.target.values,cv=10,scoring='f1')

def run_lgb(reduce_train, reduce_test):

    

    kf = StratifiedKFold(n_splits=10) 

    avg_f1_score=[]

    

    oof_pred = np.zeros((len(reduce_train)))

    

    y_pred = np.zeros((len(reduce_test)))

    

    for fold, (tr_ind, val_ind) in enumerate(kf.split(reduce_train,traindata['target'].values)):

        print('Fold {}'.format(fold + 1))

        x_train, x_val = reduce_train[tr_ind,:], reduce_train[val_ind,:]

        y_train, y_val = traindata['target'][tr_ind], traindata['target'][val_ind]

        

        train_set = lgb.Dataset(x_train, y_train)#, categorical_feature=cat_features)

        val_set = lgb.Dataset(x_val, y_val)#, categorical_feature=cat_features)



        params = {

            'learning_rate': 0.04,

            'n_estimators': 1500,

            'metric':'auc',

            'colsample_bytree': 0.4,

        }

       

        model = lgb.train(params, train_set, num_boost_round = 1000, early_stopping_rounds = 5, #1000

                          valid_sets=[train_set, val_set], verbose_eval = 100)

        

        oof_pred[val_ind] = [1 if i>=0.5 else 0 for i in model.predict(x_val)]

        train_folds=[1 if i>=0.5 else 0 for i in model.predict(reduce_train)]

        

        y_pred += model.predict(reduce_test) / kf.n_splits

        

        avg_f1_score.append(f1_score(traindata['target'].values,train_folds))

        print('OOF F1:', f1_score(traindata['target'].values,oof_pred))

        

    return y_pred,model,avg_f1_score
y_pred,modelobj,avg_f1_score = run_lgb(USE_train_embeddings['outputs'].numpy(),USE_test_embeddings['outputs'].numpy())
summary_table = pd.DataFrame([lr_scores.mean(),dtc_scores.mean(),rfc_scores.mean(),knn_scores.mean(),np.mean(avg_f1_score)],index=['Logistic Regression', 'Decision Tree', 'Random Forest', 'KNN','LGBM'], columns=['F1 Score'])

summary_table=summary_table.reset_index()
trace1 = go.Bar(

                x = summary_table['index'],

                y = summary_table['F1 Score'],

                marker = dict(color = 'rgb(153,255,153)',

                              line=dict(color='rgb(0,0,0)',width=1.5)))

layout = go.Layout(template= "plotly_dark",title = 'BASE_LINE_MODELS' , xaxis = dict(title = 'Models'), yaxis = dict(title = 'Mean CV'))

fig = go.Figure(data = [trace1], layout = layout)

fig.show()
def make_classifier():

    clf = CatBoostClassifier(

                               loss_function='CrossEntropy',

                               eval_metric="F1",

                               task_type="CPU",

                               learning_rate=0.05,

                               n_estimators =100,   #5000

                               early_stopping_rounds=10,

                               random_seed=2019,

                               silent=True

                              )

        

    return clf





scoring = "f1"





HistGBM_param = {

    'l2_regularization': 0.0,

    'loss': 'auto',

    'max_bins': 255,

    'max_depth': 15,

    'max_leaf_nodes': 31,

    'min_samples_leaf': 20,

    'n_iter_no_change': 50,

    'scoring': scoring,

    'tol': 1e-07,

    'validation_fraction': 0.15,

    'verbose': 0,

    'warm_start': False   

}



folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

fold_preds = np.zeros([USE_test_embeddings['outputs'].numpy().shape[0],3])

oof_preds = np.zeros([USE_train_embeddings['outputs'].numpy().shape[0],3])

results = {}



estimators = [

        ('histgbm', HistGradientBoostingClassifier(**HistGBM_param)),

        ('catboost', make_classifier())

    ]



# Fit Folds

f, ax = plt.subplots(1,3,figsize = [14,5])

for i, (trn_idx, val_idx) in enumerate(folds.split(USE_train_embeddings['outputs'].numpy(),traindata['target'].values)):

    print(f"Fold {i} stacking....")

    clf = StackingClassifier(

            estimators=estimators,

            final_estimator=LogisticRegression(),

            )

    clf.fit(USE_train_embeddings['outputs'].numpy()[trn_idx,:], traindata['target'].loc[trn_idx])

    tmp_pred = clf.predict_proba(USE_train_embeddings['outputs'].numpy()[val_idx,:])[:,1]

    

    oof_preds[val_idx,0] = tmp_pred

    fold_preds[:,0] += clf.predict_proba(USE_test_embeddings['outputs'].numpy())[:,1] / folds.n_splits

        

    estimator_performance = {}

    estimator_performance['stack_score'] = metrics.roc_auc_score(traindata['target'].loc[val_idx], tmp_pred)

    

    for ii, est in enumerate(estimators):

            model = clf.named_estimators_[est[0]]

            pred = model.predict_proba(USE_train_embeddings['outputs'].numpy()[val_idx,:])[:,1]

            oof_preds[val_idx, ii+1] = pred

            fold_preds[:,ii+1] += model.predict_proba(USE_test_embeddings['outputs'].numpy())[:,1] / folds.n_splits

            estimator_performance[est[0]+"_score"] = metrics.roc_auc_score(traindata['target'].loc[val_idx], pred)

            

    stack_coefficients = {x+"_coefficient":y for (x,y) in zip([x[0] for x in estimators], clf.final_estimator_.coef_[0])}

    stack_coefficients['intercept'] = clf.final_estimator_.intercept_[0]

        

    results["Fold {}".format(str(i+1))] = [

            estimator_performance,

            stack_coefficients

        ]



    plot_roc_curve(clf, USE_train_embeddings['outputs'].numpy()[val_idx,:], traindata['target'].loc[val_idx], ax=ax[i])

    ax[i].plot([0.0, 1.0])

    ax[i].set_title("Fold {} - ROC AUC".format(str(i)))



plt.tight_layout(pad=2)

plt.show()



f, ax = plt.subplots(1,2,figsize = [11,5])

sns.heatmap(pd.DataFrame(oof_preds, columns = ['stack','histgbm','catboost']).corr(),

            annot=True, fmt=".2f",cbar_kws={'label': 'Correlation Coefficient'},cmap="magma",ax=ax[0])

ax[0].set_title("OOF PRED - Correlation Plot")

sns.heatmap(pd.DataFrame(fold_preds, columns = ['stack','histgbm','catboost']).corr(),

            annot=True, fmt=".2f",cbar_kws={'label': 'Correlation Coefficient'},cmap="inferno",ax=ax[1])

ax[1].set_title("TEST PRED - Correlation Plot")

plt.tight_layout(pad=3)

plt.show()
del USE_train_embeddings,USE_test_embeddings

gc.collect()



custom_args={'num_train_epochs':2,'max_seq_length':512,'fp16':False,'overwrite_output_dir': True}
# n=5

# kf = StratifiedKFold(n_splits=n, random_state=2019, shuffle=True)

# results = []



# for train_index, val_index in kf.split(traindata.text,traindata.target.values):

#     train_df = traindata[['text','target']].iloc[train_index]

#     val_df = traindata[['text','target']].iloc[val_index]

#     model = ClassificationModel('bert', 'bert-base-uncased', args=custom_args,use_cuda=False) 

#     model.train_model(train_df)

#     result, model_outputs, wrong_predictions = model.eval_model(val_df, acc=f1_score)

#     print(result['acc'])

#     results.append(result['acc'])

    

# del model

# gc.collect()
with open('../input/nlp-predictions/fold_results.txt') as f:

    results=f.readlines()



for i, result in enumerate(results, 1):

    print(f"Fold-{i}: {result}")

# model = ClassificationModel('bert', 'bert-base-uncased', args=custom_args) 

# model.train_model(traindata[['text','target']])



# gc.collect()
#predictions, raw_outputs = model.predict(testdata['text'])



# with open('../input/nlp-predictions/predictions.txt') as f:

#     predictions=f.readlines()

    

submission['target'] = [1  if stack>=0.5 else 0 for stack in fold_preds[:,0] ] #predictions

submission.to_csv('submission.csv', index=False)