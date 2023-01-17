import pandas as pd
import csv
import os
import numpy
import copy
%matplotlib inline
import numpy as np
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim import corpora, models, similarities, matutils
from gensim.corpora.dictionary import Dictionary
from sklearn.cluster import DBSCAN
from sklearn.decomposition import NMF, PCA
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import Counter, defaultdict, OrderedDict
import seaborn as sns
import matplotlib.pyplot as plt
folder_path = 'your/directory/path'

#plotting params
mpl.rcParams['figure.figsize'] = (8,5)
mpl.rcParams['lines.linewidth'] = 3
plt.style.use('ggplot')
df = pd.read_csv("../input/tweets.csv", parse_dates= [6])
df.username = df.username.str.lower()
import re
 
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
 
regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]
    
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
 
def tokenize(s):
    return tokens_re.findall(s)
 
def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens
def clean_tweet(tweet):
    ext = "http"
    text = tweet[:tweet.find(ext)].lower()
    text = re.sub("[^\S]", " ", text)
    text = re.sub("english translation ", "", text)
    textOnly = re.sub("[^a-zA-Z0-9@# ]", "", text)
    return(textOnly)
def remove_users(tweet):
    text = tweet.lower()
    textOnly = re.sub(r"@\w+", "", text)
    return(textOnly)
df.tweets = df.tweets.apply(clean_tweet)
infoDict = OrderedDict()
for r in df[['username','tweets']].iterrows():
    match = re.search('^rt', r[1][1])
    if match:
        m = list(re.findall(r"@\w+", r[1][1]))
        if m:
            username=m[0][1:]
            tweet=r[1][1][len('rt ' + m[0]):] +' @' + r[1][0]
    else:
        username=r[1][0]
        tweet=r[1][1]
    if username not in infoDict:
        user = {}
        user['affil'] = []
        user['hashtags'] = []
        user['tweets'] = []
        user['doc'] = ''
        infoDict[username] = user
    if tweet not in infoDict[username]['tweets']:
        infoDict[username]['tweets'].append(tweet)
        infoDict[username]['doc']+=' ' + tweet
    infoDict[username]['hashtags'].extend(re.findall('(?<=#)\w+', tweet))
    infoDict[username]['affil'].extend(re.findall('(?<=@)\w+', tweet))
print('We went from 112 unique users to', len(infoDict), 'users') 
docs = [remove_users(v['doc']) for k, v in infoDict.items()]
Tfidf_vectorizer = TfidfVectorizer(analyzer='word', tokenizer=tokenize,
                                  ngram_range=(1,1), stop_words='english',
                                  token_pattern='\\b[a-z][a-z]+\\b')
tfidf_docs = Tfidf_vectorizer.fit_transform(docs)
count_vectorizer = CountVectorizer(analyzer='word',
                                  ngram_range=(1,3), stop_words='english',
                                  token_pattern='\\b[a-z][a-z]+\\b')
cv_tweets = count_vectorizer.fit_transform(docs)
#treat each tweets seperately
Tfidf_vectorizer = TfidfVectorizer(analyzer='word', tokenizer=tokenize,
                                  ngram_range=(1,2), stop_words='english',
                                  token_pattern='\\b[a-z][a-z]+\\b')
tfidf_tweets = Tfidf_vectorizer.fit_transform(df.tweets)
from sklearn.utils.extmath import randomized_svd
U, Sigma, VT = randomized_svd(tfidf_docs, n_components=15,
                                      n_iter=5,
                                      random_state=None)
sigma = []
for k,v in enumerate(Sigma):
    sigma.append((k,v))
f = plt.scatter(*zip(*sigma))
tfidf_docs.shape
num_topics = 6
model = NMF(n_components=num_topics, init='random', random_state=0)
nmf = model.fit_transform(tfidf_docs)
doc_cluster = [list(r).index(max(r)) for r in nmf]
print (doc_cluster[0:20])
print (doc_cluster[41:60])
print (doc_cluster[61:80])
print( doc_cluster[-40:-20])
cluster_size = [0,0,0,0,0,0]
for v in doc_cluster:
    cluster_size[v]+=1
cluster_size
data = pd.DataFrame(infoDict).T
data.reset_index(inplace=True)
data['cluster'] = doc_cluster
# create dictionary that maps a user to their specific cluster
user_docs = {}
for k, cluster in enumerate(doc_cluster):
    user_docs[data['index'][k]] = cluster
words = sorted([(i,v) for v,i in Tfidf_vectorizer.vocabulary_.items()])
topic_words = []
for r in model.components_:
    a = sorted([(v,i) for i,v in enumerate(r)],reverse=True)[0:7]
    topic_words.append([words[e[1]] for e in a])
# Create the a list of topic words but only inlude them if they are the highest among all clusters
# they also need to be weighted appropriately within their cluster


word_cluster = [(list(r).index(max(r)),max(r)) for r in model.components_.transpose()]
for i,r in enumerate(model.components_.transpose()):
    s = sorted(r)
    if (s[0]-s[1])/s[1]<0.25:
        word_cluster[i] = (-1,-1)
topic_words = []
for c in range(6):
    a = sorted([(v[1],i) for i,v in enumerate(word_cluster) if v[0]==c], reverse=True)[0:7]
    topic_words.append([words[e[1]] for e in a])
topic_words
def is_retweet(tweet):
    match = re.search('^rt', tweet)
    if match:
        return True
    return False

def check_string(string):
    return string in df.username.unique()
# number of retweets

count=0
rt = []
for tweet in df.tweets:
    m = re.findall(r"^rt @", tweet)
    if m:
        count+=1
        rt.append(tweet)
count
#len(set(rt))
# create the nodes for the graph. Dictionary keys as usernames 
# with each item in the value list being a connection with number of times mentioned
# NOTE!! this is for every mentioned user

nodes = defaultdict(str)
for K, V in infoDict.items():
    nodes[K] = [(k, v) for k,v in Counter(list(V['affil'])).items() if k != K]
for k, v in list(nodes.items()):
    if not v:
        del nodes[k]
matches = 0
for k, v in list(nodes.items()):
    if not v:
        del nodes[k]
        
    for i in range(len(v)):
        if v[i][0] == k:
            matches+=1
            v.remove(v[i])
#create dictionary that maps usernames to a uniqe ID and the cluster they belong to
# user_docs = a list of documents per user

ID = defaultdict()
for k, v in enumerate(list(nodes.items())):
    ID[v[0]] = k, user_docs[v[0]]
    for i in range(0, len(v[1])):
        try:
            ID[v[1][i][0]] = k, user_docs[v[1][i][0]]
        except:
            pass
df2 = pd.DataFrame(ID).T
# save the nodes as a csv
# pd.DataFrame(ID).T.to_csv(folder_path+'/nodes3.csv')
# create dict with a source, target tied to a weight
edge_dict = {}
for k, v in nodes.items():
    for i in range(len(v)):
        try:
            edge_dict[ID[[k][0]][0], ID[v[i][0]][0]] = v[i][1]
        except:
            pass
# save edges as a csv
# pd.DataFrame([[k[0],k[1],v] for k, v in edge_dict.items()]).to_csv(folder_path+'/edges3.csv')
from scipy import interpolate
import seaborn as sns

df.time = pd.to_datetime(df.time)
perhr = df.set_index(df['time']).resample('D', how='count')
pd.rolling_mean(perhr, window=7).tweets['2016-01-01':].plot()
fig, ax = plt.subplots(figsize = (20,8))

perhr['2016-01-01':].numberstatuses.interpolate(method='linear').plot(ax = ax, color="black", fontsize=12, alpha=0.1)
pd.rolling_mean(perhr, window=7).tweets['2016-01-01':].plot(color ='r')

#sns.timeseries(perhr, ax=ax)

yemen = '2016-01-29'
brussels = '2016-03-22'


ax.annotate('Bombing in Brussels',xy=(brussels, 200),xytext=('2016-03-03', 310),
            arrowprops=dict(facecolor='white', shrink=0.05), size=15)

ax.annotate('Car bombing in Yemen',xy=(yemen, 200),xytext=('2016-01-10', 310),
            arrowprops=dict(facecolor='white', shrink=0.05),size=15)

ax.margins(None,0.1)
ax.legend(['Tweets Per Day','7-Day Rolling Average'], loc = 'upper right',
           numpoints = 1, labelspacing = 2.0, fontsize = 14)
ax.set_xlabel('Date')
ax.set_ylabel('Number of Tweets')
ax.set_title('Frequency of ISIS Tweets in 2016')

# ax.spines['bottom'].set_color('w')
# ax.spines['top'].set_color('w') 
# ax.spines['right'].set_color('w')
# ax.spines['left'].set_color('w')

# ax.tick_params(axis='x', colors='w')
# ax.tick_params(axis='y', colors='w')

# ax.yaxis.label.set_color('w')
# ax.xaxis.label.set_color('w')

# ax.set_axis_bgcolor('w')
fig.savefig('temp.png')
plt.show()