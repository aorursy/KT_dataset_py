# import comet_ml in the top of your file
# from comet_ml import Experiment

#Inspecting
import numpy as np 
import pandas as pd 
pd.set_option('display.max_colwidth', -1)
from time import time
import re
import string
import os
import emoji
from pprint import pprint
from collections import Counter
import pyLDAvis.gensim
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
from spacy import displacy
import gensim
import spacy
nlp = spacy.load("en_core_web_lg")

#visualisation
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(style="darkgrid")
sns.set(font_scale=1.3)
from wordcloud import WordCloud
from PIL import Image
import collections
from matplotlib import style
plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'figure.figsize': [16, 12]})
plt.style.use('seaborn-whitegrid')
sns.set_style('dark')

#Warnings
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)



# Balance data
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

#Cleaning
import nltk
from nltk.tokenize import word_tokenize, TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import SnowballStemmer, PorterStemmer, LancasterStemmer
from nltk.corpus import stopwords
from nltk.probability import FreqDist

#Modeling
from sklearn.pipeline import Pipeline
from gensim.models import Word2Vec
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV,train_test_split, RandomizedSearchCV


#metrics for analysis
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score, f1_score

# Import Pickle for streamlit Application
import pickle

# Add the following code anywhere in your machine learning file
# experiment = Experiment(api_key="ZO3kD6D1uVgIr9CHdhUPQaU3B",
#                         project_name="climate-change-belief-analysis", workspace="helloaggregator")
train = pd.read_csv('../input/dataset/train.csv')
test = pd.read_csv('../input/dataset/test.csv')
plt.title('Number of Characters Present in Tweet')
train['message'].str.len().hist()
train['message'].str.split().\
apply(lambda x : [len(i) for i in x]). \
map(lambda x: np.mean(x)).hist()
# Fetch stopwords so it doesn't take away from Ngram analysis
stop = set(stopwords.words('english'))
# Create corpus
corpus=[]
new= train['message'].str.split()
new=new.values.tolist()
corpus=[word for i in new for word in i]

from collections import defaultdict
dic=defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word]+=1
counter=Counter(corpus)
most=counter.most_common()

x, y = [], []
for word,count in most[:40]:
    if (word not in stop):
        x.append(word)
        y.append(count)
        
sns.barplot(x=y,y=x)
plt.title('Most Common Words')
plt.show()
def get_top_ngram(corpus, n=None):
    
    '''
    Takes a list of words and groups then in terms of ngrams depending on how many words you want to group, returns 
    a word count based on the number of times ngram appears
    
    Parameters
    -----------
    corpus: list
            input list of strings
    n: int
       input the number of ngrams needed
       
    Output
    ----------
    Output: Returns a tuple list with specified number of words grouped and counts the frequency
    
    '''
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) 
                  for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:10]

top_n_bigrams = get_top_ngram(train['message'],2)[:40]
x,y=map(list,zip(*top_n_bigrams))
sns.barplot(x = y,y = x)
plt.title('Bigram analysis')
plt.show()
top_tri_grams=get_top_ngram(train['message'],n=3)
x,y=map(list,zip(*top_tri_grams))
sns.barplot(x=y,y=x)
plt.title('Trigram Analysis')
plt.show()
def preprocess_train(df):
    '''
    Creates a list of lemmetized words that must have a length greater than 2 from an input of a dataframe
    
    Parameters
    -----------
    df: Dataframe
        Input needs to be dataframe
        
    Output
    -----------
    corpus: Returns a list of lemmatized words
    
    '''
    corpus=[]
    stem=PorterStemmer()
    lem=WordNetLemmatizer()
    for train in df['message']:
        words = [w for w in word_tokenize(train) if (w not in stop)]
        
        words = [lem.lemmatize(w) for w in words if len(w)>2]
        
        corpus.append(words)
    return corpus
#Create corpus
corpus = preprocess_train(train)

# Create tuple vectorised words
dic=gensim.corpora.Dictionary(corpus)
bow_corpus = [dic.doc2bow(doc) for doc in corpus]

# creat a weight for topics of vectorized words
lda_model =  gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = 10, 
                                   id2word = dic,                                    
                                   passes = 10,
                                   workers = 2)
#visual the the top ten topics
style.use('dark_background')
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dic)
vis
def ner(text):
    '''
    Takes in  text and returns entity label of text using the natural language processor on python
    
    Parameters
    ----------
    text: String
          input a string
    ent: String
         Input a string ant it will return the entity you desire
    
    Output
    ---------
    output: Entity labelled string
            Returns a label depending on the context of string
    
    '''
    doc=nlp(text)
    return [X.label_ for X in doc.ents]
#create labels for all the tweets
ent=train['message'].apply(lambda x : ner(x))
ent=[x for sub in ent for x in sub]
counter=Counter(ent)
count=counter.most_common()

#Plot the labels that occurred the most from the tweets
x,y=map(list,zip(*count))
sns.barplot(x=y,y=x)
plt.title('The most Occurring Entities')
plt.show()
def ner(text,ent="GPE"):
    doc=nlp(text)
    return [X.text for X in doc.ents if X.label_ == ent]
gpe = train['message'].apply(lambda x: ner(x,"GPE"))
gpe = [i for x in gpe for i in x]
counter = Counter(gpe)

x,y = map(list,zip(*counter.most_common(40)))
sns.barplot(y,x)
plt.title('Most Common GPE')
plt.show()
per = train['message'].apply(lambda x: ner(x,"PERSON"))
per = [i for x in per for i in x]
counter = Counter(per)

x,y = map(list,zip(*counter.most_common(40)))
sns.barplot(y,x)
plt.title('The most Common Person')
plt.show()
gpe = train['message'].apply(lambda x: ner(x,"ORG"))
gpe = [i for x in gpe for i in x]
counter = Counter(gpe)

x,y = map(list,zip(*counter.most_common(40)))
sns.barplot(y,x)
plt.title('The most Common Organisations')
plt.show()
test.head()
train.head()
sns.factorplot('sentiment',data = train, kind='count',size=6,aspect = 1.5, palette = 'PuBuGn_d') 
plt.suptitle("Climate Sentiment Bar Graph",y=1)
plt.show()
# total number of negative ,neutral, positive and news posts.
climate_sentiment  = train['sentiment'].value_counts()

# pie plot for total percentage of climate change sentiment
plt.figure(figsize=(5,5))
labels = 'Positive','News','Neutral','Negative'
sizes = climate_sentiment.tolist()
colors = ['green', 'purple', 'blue','red']
explode = (0, 0, 0,0) 

# Plot
plt.suptitle("Climate Sentiment Pie Chart",y=1)
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()

#Analyse the text tweets for cleaning
for index,text in enumerate(train['message'][35:]):
  print('Tweet %d:\n'%(index+1),text)
# number of words in a tweet
df_eda = pd.DataFrame()
df_eda['count_words'] = train['message'].apply(lambda x: len(re.findall(r'\w+',x)))

# referrals to other twiiter accounts
df_eda['count_mentions'] = train['message'].apply(lambda x: len(re.findall(r'@\w+',x)))

# number of hashtags 
df_eda['count_hashtags'] = train['message'].apply(lambda x: len(re.findall(r'#\w+',x)))

# Number of upper case words 3 or more to ignore RT
df_eda['count_capital_words'] = train['message'].apply(lambda x: len(re.findall(r'\b[A-Z]{3,}\b',x)))

#count number of exclamation marks and questions marks 
df_eda['count_exl_quest'] = train['message'].apply(lambda x: len(re.findall(r'!|\?',x)))

#count number of urls
df_eda['count_urls'] = train['message'].apply(lambda x: len(re.findall(r'http.?://[^\s]+[\s]?',x)))

#count the number of emojis
df_eda['count_emojis'] = train['message'].apply(lambda x: emoji.demojize(x)).apply(lambda x: len(re.findall(r':[a-z_&]+:',x)))

#add the dependent varaible for further analysis
df_eda['sentiment'] = train.sentiment
df_eda.head()
# Comment
column_names = [col for col in df_eda.columns if col != 'sentiment']
for i in column_names:
    bins = np.arange(df_eda[i].min(),df_eda[i].max()+1)
    g = sns.FacetGrid(data=df_eda,col='sentiment',size=5, hue = 'sentiment',palette="PuBuGn_d")
    g = g.map(sns.distplot, i, kde= False, norm_hist = True,bins = bins)
    plt.show()
#Check for duplicates
duplicate_rows_train = train['message'].duplicated().sum()
duplicate_rows_test = test['message'].duplicated().sum()
print('There are ',duplicate_rows_train,' duplicated rows for the training set')
print('There are ',duplicate_rows_test,' duplicated rows for the test set')
# Drop duplicate rows/retweets
train = train.drop_duplicates(subset='message', keep='first',)
train = train.reset_index()
train.drop('index',inplace=True,axis =1)
train.head()
# Cleaning the data 
def data_preprocessing(train,test):
    '''
    Cleaning the data based on analysis which includes removing capilised letters, changing contractions like don't to do not,
    replace urls, replace emjicons, remove digits and lastly remove any funny characters in tweets
    
    Parameters
    ----------
    train: data frame
          The data frame of training set
    test: data frame
          The data frame of test set
          
    Output
    ---------
    train: Adds column of tidy tweets to train dataframe
    test: Adds column of tidy tweets to test dataframe
    '''
    def remove_capital_words(df,column):
        df_Lower = df[column].map(lambda x: x.lower())
        return df_Lower
    train['tidy_tweet'] = remove_capital_words(train,'message')
    test['tidy_tweet'] = remove_capital_words(test,'message')
    contra_map = {
                    "ain't": "am not ",
                    "aren't": "are not ",
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
                    "i'd": "I would",
                    "i'd've": "I would have",
                    "i'll": "I will",
                    "i'll've": "I will have",
                    "i'm": "I am",
                    "i've": "I have",
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
                    "so's": "so is",
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
                    "you've": "you have"}
    contractions_re = re.compile('(%s)' % '|'.join(contra_map.keys()))
    def contradictions(s, contractions_dict=contra_map):
        def replace(match):
            return contractions_dict[match.group(0)]
        return contractions_re.sub(replace, s)
    train['tidy_tweet']=train['tidy_tweet'].apply(lambda x:contradictions(x))
    test['tidy_tweet']=test['tidy_tweet'].apply(lambda x:contradictions(x))
    def replace_url(df,column):
        df_url = df[column].str.replace(r'http.?://[^\s]+[\s]?', 'urlweb ')
        return df_url
    train['tidy_tweet'] = replace_url(train,'tidy_tweet')
    test['tidy_tweet'] = replace_url(test,'tidy_tweet')
    def replace_emoji(df,column):
        df_emoji = df[column].apply(lambda x: emoji.demojize(x)).apply(lambda x: re.sub(r':[a-z_&]+:','emoji ',x))
        return df_emoji
    train['tidy_tweet'] = replace_emoji(train,'tidy_tweet')
    test['tidy_tweet'] = replace_emoji(test,'tidy_tweet')
    def remove_digits(df,column):
        df_digits = df[column].apply(lambda x: re.sub(r'\d','',x))
        return df_digits
    train['tidy_tweet'] = remove_digits(train,'tidy_tweet')
    test['tidy_tweet'] = remove_digits(test,'tidy_tweet')	
    def remove_patterns(df,column):
        df_char = df[column].apply(lambda x:  re.sub(r'[^a-z# ]', '', x))
        return df_char
    train['tidy_tweet'] = remove_patterns(train,'tidy_tweet')
    test['tidy_tweet'] = remove_patterns(test,'tidy_tweet')   
    return train,test
(train,test) = data_preprocessing(train,test)
#Analyse the cleaned tweets
for index,text in enumerate(train['tidy_tweet'][35:]):
  print('Tweet %d:\n'%(index+1),text)
# Get Tokens of clean tweets
train['token'] = train['tidy_tweet'].apply(lambda x: x.split())
test['token'] = test['tidy_tweet'].apply(lambda x: x.split())
train['token'].head()
# use stemming process on clean tweets
stemmer = PorterStemmer()

train['stemming'] = train['token'].apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
test['stemming'] = test['token'].apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
train['stemming'].head()
#create list of all cleaned text appearing in tweets
stemma_list_all = []
for index, rows in train.iterrows():
    stemma_list_all.append(rows['stemming'])
flatlist_all = [item for sublist in stemma_list_all for item in sublist]
flatlist_all
#Count the number of words apppearing in all the tweets
frequency_dist = FreqDist(flatlist_all)
freq_dist = dict(frequency_dist)
sorted(freq_dist.items(), key= lambda x:-x[1])

#Make Data frame 
df_all = pd.DataFrame(freq_dist.items(),columns = ['Word','Occurrence'])
# Sort values 
df_all = df_all.sort_values('Occurrence', ascending=False)
fig, ax = plt.subplots(figsize=(20, 20))

# Plot horizontal bar graph
df_all.iloc[:60].sort_values(by='Occurrence').plot.barh(x='Word',
                      y='Occurrence',
                      ax=ax,
                      color="deepskyblue")

ax.set_title("Plot 4: Common Words Found in all Tweets")

plt.show()
#check for stopwords in train
stop = stopwords.words('english')
train['stopwords'] = train['stemming'].apply(lambda x: len([i for i in x if i in stop]))
train[['stemming','stopwords']].head()
#check for stopwords in test
stop = stopwords.words('english')
test['stopwords'] = test['stemming'].apply(lambda x: len([i for i in x if i in stop]))
test[['stemming','stopwords']].head()
#create my own stop words from analysis and comparing with general stopwords
stopwords_own =[ 'i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him',
                'his','himself','she','her','hers','herself','it','itself','they','them','their','theirs','themselves','what',
                'which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being','have','has',
                'had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while',
                'of','at','by','for','with','about','against','between','into','through','during','before','after','above',
                'below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here',
                'there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','only',
                'own','same','so','than','too','very','s','t','can','will','just','should','now','d','ll','m','o','re','ve','y',
               #my own stopwords found from analysis
                'u','doe','going','ha','wa','l', 'thi','becaus','rt']
# def remove_strop_words(df,column):
def remove_stopwords(df,column):
    '''
    Removing the stop words from the clean tweets
    
    Parameters
    ----------
    df: data frame
        Input a dataframe
    column: String
        name of column from data frame
        
    Output
    ----------
    output: df
            Returns a dataframe with no stopwords
    
    '''
    df_stopwords = df[column].apply(lambda x: [item for item in x if item not in stopwords_own])
    return df_stopwords
train['stem_no_stopwords'] = remove_stopwords(train,'stemming')
test['stem_no_stopwords'] = remove_stopwords(test,'stemming')
train['stem_no_stopwords'].head()
def convert_st_str(df,column):
    '''
    Changes list of strings into one string per row in dataframe
    
    Parameters
    -----------
    df: data frame
        Takes  in a dataframe
        
    Output
    -----------
    output: df_str
            Returns a dataframe with a string instead of list for each row 
    '''
    df_str = df[column].apply(lambda x: ' '.join(x))
    return df_str
train['clean_tweet'] = convert_st_str(train,'stem_no_stopwords')
test['clean_tweet'] = convert_st_str(test,'stem_no_stopwords')
train['clean_tweet'].head()
#Create WordCloud Plot
news_words =' '.join([text for text in train['clean_tweet'][train['sentiment'] == 2]])
wordcloud = WordCloud(width=2000, height=1500, random_state=21, max_font_size=200, background_color='white').generate(news_words)
print(wordcloud)
plt.figure(figsize=(12, 12))
plt.title("Word Cloud for News Sentiment")
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
#Create WordCloud Plot
pro_words =' '.join([text for text in train['clean_tweet'][train['sentiment'] == 1]])
wordcloud = WordCloud(width=2000, height=1500, random_state=21, max_font_size=200,background_color='white').generate(pro_words)
plt.figure(figsize=(12, 12))
plt.title("Word Cloud for postive Sentiment")
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
#Create WordCloud Plot
neutral_words =' '.join([text for text in train['clean_tweet'][train['sentiment'] == 0]])
wordcloud = WordCloud(width=2000, height=1500, random_state=21, max_font_size=200, background_color='white').generate(neutral_words)
plt.figure(figsize=(12, 12))
plt.title("Word Cloud for neutral Sentiment")
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
#Create WordCloud Plot
anti_words =' '.join([text for text in train['clean_tweet'][train['sentiment'] == -1]])
wordcloud = WordCloud(width=2000, height=1500, random_state=21, max_font_size=200, background_color='white').generate(anti_words)
plt.figure(figsize=(12, 12))
plt.title("Word Cloud for negative Sentiment")
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
df_news = train[train.sentiment == 2]
top_bi_grams_news=get_top_ngram(df_news['clean_tweet'],n=2)
x,y=map(list,zip(*top_bi_grams_news))
sns.barplot(x=y,y=x).set(title = 'Common Words Found in Tweets of 2 sentiment')
df_news = train[train.sentiment == 1]
top_bi_grams_pos=get_top_ngram(df_news['clean_tweet'],n=2)
x,y=map(list,zip(*top_bi_grams_pos))
sns.barplot(x=y,y=x).set(title = 'Common Words Found in Tweets of 1 sentiment')
df_news = train[train.sentiment == 0]
top_bi_grams_neutr=get_top_ngram(df_news['clean_tweet'],n=2)
x,y=map(list,zip(*top_bi_grams_neutr))
sns.barplot(x=y,y=x).set(title = 'Common Words Found in Tweets of 0 sentiment')
df_news = train[train.sentiment == -1]
top_bi_grams_neg = get_top_ngram(df_news['clean_tweet'],n=2)
x,y=map(list,zip(*top_bi_grams_neg))
sns.barplot(x=y,y=x).set(title = 'Common Words Found in Tweets of -1 sentiment')
# function to collect hashtags
def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags
# extracting hashtags from  tweets

HT_neutral = hashtag_extract(train['clean_tweet'][train['sentiment'] == 0])


HT_pro = hashtag_extract(train['clean_tweet'][train['sentiment'] == 1])

HT_news = hashtag_extract(train['clean_tweet'][train['sentiment'] == 2])


HT_anti = hashtag_extract(train['clean_tweet'][train['sentiment'] == -1])
# unnesting list
HT_neutral = sum(HT_neutral,[])
HT_pro = sum(HT_pro,[])
HT_news = sum(HT_news,[])
HT_anti = sum(HT_anti,[])
a = nltk.FreqDist(HT_news)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
# selecting top 5 most frequent hashtags  
d = d.sort_values(by = 'Count',ascending = False)
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d[0:5], x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.title("Hashtag plot for news (2) Sentiment")
plt.show()
a = nltk.FreqDist(HT_pro)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
# selecting top 5 most frequent hashtags     
d = d.nlargest(columns="Count", n = 5) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.title("Hashtag plot for positive (1) Sentiment")
plt.show()
a = nltk.FreqDist(HT_neutral)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
# selecting top 5 most frequent hashtags     
d = d.nlargest(columns="Count", n = 5) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.title("Hashtag plot for neutral (0) Sentiment")
plt.show()
a = nltk.FreqDist(HT_anti)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
# selecting top 5 most frequent hashtags     
d = d.nlargest(columns="Count", n = 5) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.title("Hashtag plot for negative (-1) Sentiment")
plt.show()
#Create Count Vector
cv = CountVectorizer(max_df = 0.90,min_df = 2, max_features = 1000)
bow = cv.fit_transform(train['clean_tweet'])
#Plot Count Vector
word_freq = dict(zip(cv.get_feature_names(), np.asarray(bow.sum(axis=0)).ravel()))
word_counter = collections.Counter(word_freq)
word_counter_df = pd.DataFrame(word_counter.most_common(20), columns = ['word', 'freq'])
fig, ax = plt.subplots(figsize=(20, 7))
sns.barplot(x ="word", y="freq", data=word_counter_df, palette="PuBuGn_d", ax=ax)
plt.title("Count Vectorizer plot")
plt.show();
#Create TF -IDF
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000)
tfidf = tfidf_vectorizer.fit_transform(train['clean_tweet'])
#Plot TF-IDF
word_freq = dict(zip(cv.get_feature_names(), np.asarray(tfidf.sum(axis=0)).ravel()))
word_counter = collections.Counter(word_freq)
word_counter_df = pd.DataFrame(word_counter.most_common(20), columns = ['word', 'freq'])
fig, ax = plt.subplots(figsize=(20, 7))
sns.barplot(x="word", y="freq", data=word_counter_df, palette="PuBuGn_d", ax=ax)
plt.title("Plot 16: TD - IDF plot")
plt.show();
#Create Word2Vec 
tokenised_tweet = train['clean_tweet'].apply(lambda x: x.split()) #tokenising
test_tokenised_tweets = test['clean_tweet'].apply(lambda x: x.split())

model_w2v = Word2Vec(            
            tokenised_tweet,
            size=200, # desired no. of features/independent variables 
            window=5, # context window size
            min_count=2,
            sg = 1, # 1 for skip-gram model
            hs = 0,
            negative = 10, # for negative sampling
            workers= 2, # no.of cores
            seed = 34) 
model_w2v.train(tokenised_tweet,total_examples= len(train['clean_tweet']), epochs=20)
model_w2v.wv.most_similar(positive="realdonaldtrump")
#The below function will be used to create a vector for each tweet by taking the average of the vectors
def word_vector(tokens, size):
    '''
    create a vector for each tweet by taking the average of the vectors of the words present in the tweet
    
    Parameters
    ----------
    tokens: list of strings
            Input of tokens per tweet
    size: int
          how many words to vectorize
          
    Output
    ---------
    output: Average vector per token in tweets
    
    '''
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += model_w2v[word].reshape((1, size))
            count += 1.
        except KeyError: # handling the case where the token is not in vocabulary
                         
            continue
    if count != 0:
        vec /= count
    return vec
#create word2vec dataframe
wordvec_arrays = np.zeros((len(tokenised_tweet), 200))

for i in range(len(tokenised_tweet)):
    wordvec_arrays[i,:] = word_vector(tokenised_tweet[i], 200)
    
wordvec_df = pd.DataFrame(wordvec_arrays)
wordvec_df.shape
X = train['message']
y = train['sentiment']
# X_vec = wordvec_df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train_vec, X_test_vec, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
# TF-IDF Features
pipe1 = Pipeline(steps = [('tfidf_vectorisation',TfidfVectorizer()),('classifier',BernoulliNB())])
pipe1.fit(X_train,y_train)
#prediction set
prediction_nb = pipe1.predict(X_test)
# adding labels to confusion matrix
confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test,prediction_nb),index=['-1','0','1','2'], columns=['-1','0','1','2'])
confusion_matrix_df
# print classification report
print(classification_report(y_test,prediction_nb))
# Print overall acuracy
print(accuracy_score(y_test,prediction_nb))
pipe2 = Pipeline(steps = [('tfidf_vectorisation',TfidfVectorizer()),('classifier',LinearSVC(random_state = 42))])
pipe2.fit(X_train,y_train)
#prediction set
prediction_lsvc = pipe2.predict(X_test)
# adding labels to confusion matrix
confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test,prediction_lsvc),index=['-1','0','1','2'], columns=['-1','0','1','2'])
confusion_matrix_df
# print classification report
print(classification_report(y_test,prediction_lsvc))
# Print overall acuracy
print(accuracy_score(y_test,prediction_lsvc))
pipe3 = Pipeline(steps = [('tfidf_vectorisation',TfidfVectorizer()),('classifier',PassiveAggressiveClassifier(random_state = 42))])
pipe3.fit(X_train,y_train)
#prediction set
prediction_pas = pipe3.predict(X_test)
# adding labels to confusion matrix
confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test,prediction_pas),index=['-1','0','1','2'], columns=['-1','0','1','2'])
confusion_matrix_df
# print classification report
print(classification_report(y_test,prediction_pas))
# Print overall acuracy
print(accuracy_score(prediction_pas,y_test))
pipe4 = Pipeline(steps = [('tfidf_vectorisation',TfidfVectorizer()),('classifier',LogisticRegression(random_state = 42))])
pipe4.fit(X_train,y_train)
#prediction set
prediction_lr = pipe4.predict(X_test)
# adding labels to confusion matrix
confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test,prediction_lr),index=['-1','0','1','2'], columns=['-1','0','1','2'])
confusion_matrix_df
# print classification report
print(classification_report(y_test,prediction_lr))
# Print overall acuracy
print(accuracy_score(y_test,prediction_lr))
pipe5 = Pipeline(steps = [('tfidf_vectorisation',TfidfVectorizer()),('classifier',KNeighborsClassifier())])
pipe5.fit(X_train,y_train)
#prediction set
prediction_knnc = pipe5.predict(X_test)
# adding labels to confusion matrix
confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test,prediction_knnc),index=['-1','0','1','2'], columns=['-1','0','1','2'])
confusion_matrix_df
# print classification report
print(classification_report(y_test,prediction_knnc))
# Print overall acuracy
print(accuracy_score(y_test,prediction_knnc))
pipe6 = Pipeline(steps = [('tfidf_vectorisation',TfidfVectorizer()),('classifier',GradientBoostingClassifier(random_state = 42))])
pipe6.fit(X_train,y_train)
#prediction set
prediction_gbc = pipe6.predict(X_test)
# adding labels to confusion matrix
confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test,prediction_gbc),index=['-1','0','1','2'], columns=['-1','0','1','2'])
confusion_matrix_df
# print classification report
print(classification_report(y_test,prediction_gbc))
# Print overall acuracy
print(accuracy_score(y_test,prediction_gbc))
#Calculating f1 - scores
nb_f1 = round(f1_score(y_test,prediction_nb, average='weighted'),2)
lsvc_f1 = round(f1_score(y_test,prediction_lsvc, average='weighted'),2)
pac_f1 = round(f1_score(y_test,prediction_pas, average='weighted'),2)
lr_f1 = round(f1_score(y_test,prediction_lr, average='weighted'),2)
knnc_f1 = round(f1_score(y_test,prediction_knnc, average='weighted'),2)
gbc_f1 = round(f1_score(y_test,prediction_gbc, average='weighted'),2)

dict_f1 = {'BernoulliNB':nb_f1,'LinearSVC':lsvc_f1,'PassiveAggressiveClassifier':pac_f1,
                      'LogisticRegression':lr_f1, 'KNeighborsClassifier':knnc_f1,'GradientBoostingClassifier':gbc_f1}
f1_df = pd.DataFrame(dict_f1,index=['f1_score'])
f1_df = f1_df.T
f1_df.sort_values('f1_score',ascending = False)
#Model LSVC
model_lsvc = LinearSVC(random_state = 42)
model_lsvc.fit(X_train_vec,y_train)

#Predict LSVC
predict_vec_lsvc = model_lsvc.predict(X_test_vec)

#Model LR
model_lr = LogisticRegression(random_state = 42)
model_lr.fit(X_train_vec,y_train)
#Predict LR
predict_vec_lr = model_lsvc.predict(X_test_vec)

#Comparing f1 score and accuracy to see if model improved
lsvc_vec_f1 = round(f1_score(predict_vec_lsvc,y_test, average='weighted'),2)
lr_vec_f1 = round(f1_score(predict_vec_lr,y_test, average='weighted'),2)
lsvc_vec_acc = accuracy_score(predict_vec_lsvc,y_test)
lr_vec_acc = accuracy_score(predict_vec_lr,y_test)

#Dict
dict1 = {'Linear SVC vec2word':[lsvc_vec_f1,lsvc_vec_acc],'Logisitc Regression vec2word':[lr_vec_f1 ,lr_vec_acc]}

#Dataframe
gs_rs_df = pd.DataFrame(dict1,index =['f1 score','accuracy']).T
gs_rs_df = gs_rs_df.sort_values('f1 score',ascending =False)
gs_rs_df
#Tuning parameters for TD-IDF first
pipeline = Pipeline([('tfidf', TfidfVectorizer()),('clf',LinearSVC(random_state = 42))])

parameters = {'tfidf': [TfidfVectorizer()],
           'tfidf__max_df': [0.25,0.5,0.75],
           'tfidf__ngram_range':[(1, 1),(1,2),(2, 2)],
           'tfidf__min_df':(1,2),
           'tfidf__norm':['l1','l2']},

grid_search_tune = RandomizedSearchCV(pipeline, parameters, cv=10, n_jobs=-1, verbose=3)
grid_search_tune.fit(X_train, y_train)

print("Best parameters set:")
print(grid_search_tune.best_estimator_.steps)
#Tuning parameters for Linear Support Vector and Passive Agressive Classifiers
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.25, min_df=2, ngram_range=(1, 2))),
    ('clf', LinearSVC(random_state = 42))])

parameters = [{'clf':[LinearSVC(random_state = 42)],
           'clf__penalty':['l1','l2'],
           'clf__C':np.logspace(0, 4, 10),
           'clf__class_weight':['balanced',None]},
           {'clf':[LogisticRegression(random_state = 42)],
            'clf__penalty' : ['l1', 'l2'],
            'clf__C' : np.logspace(0, 4, 10),
            'clf__solver' : ["newton-cg", "lbfgs", "liblinear"],
            'clf__class_weight':['balanced',None]}]

grid_search_tune = RandomizedSearchCV(pipeline, parameters, cv=10, n_jobs=-1, verbose=3)
grid_search_tune.fit(X_train, y_train)

print("Best parameters set:")
print(grid_search_tune.best_estimator_.steps)
#Tuning parameters for TD-IDF first
pipeline = Pipeline([('tfidf', TfidfVectorizer()),('clf',LinearSVC(random_state = 42))])

parameters = {'tfidf': [TfidfVectorizer()],
           'tfidf__max_df': [0.25,0.5,0.75],
           'tfidf__ngram_range':[(1, 1),(1,2),(2, 2)],
           'tfidf__min_df':(1,2),
           'tfidf__norm':['l1','l2']},

grid_search_tune = GridSearchCV(pipeline, parameters, cv=4, n_jobs=-1, verbose=3)
grid_search_tune.fit(X_train, y_train)

print("Best parameters set:")
print(grid_search_tune.best_estimator_.steps)
#Tuning parameters for Linear Support Vector and Passive Agressive Classifiers
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.75, min_df=2, ngram_range=(1, 2))),
    ('clf', LinearSVC(random_state = 42))])

parameters = [{'clf':[LinearSVC(random_state = 42)],
           'clf__penalty':['l1','l2'],
           'clf__C':np.logspace(0, 4, 10),
           'clf__class_weight':['balanced',None]},
           {'clf':[LogisticRegression(random_state = 42)],
            'clf__penalty' : ['l1', 'l2'],
            'clf__C' : np.logspace(0, 4, 10),
            'clf__solver' : ["newton-cg", "lbfgs", "liblinear"],
            'clf__class_weight':['balanced',None]}]

grid_search_tune = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=3)
grid_search_tune.fit(X_train, y_train)

print("Best parameters set:")
print(grid_search_tune.best_estimator_.steps)
#pipeline for random search cv
randomcv = Pipeline(steps = [('tfidf_vectorisation',TfidfVectorizer(max_df=0.25, min_df=2, ngram_range=(1, 2))),('classifier',LinearSVC(C=7.742636826811269, random_state=42))])
randomcv.fit(X_train,y_train)
#prediction set
prediction_lsvc_best1 = randomcv.predict(test['message'])

#pipeline for grid search cv
gridcv = Pipeline(steps = [('tfidf_vectorisation',TfidfVectorizer(max_df=0.75, min_df=2, ngram_range=(1, 2))),('classifier', LinearSVC(C=2.7825594022071245, random_state=42))])
gridcv.fit(X_train,y_train)
#prediction set
prediction_lsvc_best2 = gridcv.predict(X_test)


#Calculating f1 - scores
randomcv_f1 = round(f1_score(y_test,prediction_lsvc_best1, average='weighted'),2)
gridcv_f1 = round(f1_score(y_test,prediction_lsvc_best1, average='weighted'),2)
randomcv_acc = accuracy_score(y_test,prediction_lsvc_best1,)
gridcv_acc = accuracy_score(y_test,prediction_lsvc_best2)
dict1 = {'RandomSearch':[randomcv_f1,randomcv_acc],'GridSearch':[gridcv_f1,gridcv_acc]}

gs_rs_df = pd.DataFrame(dict1,index =['f1 score','accuracy']).T
gs_rs_df = gs_rs_df.sort_values('f1 score',ascending =False)
gs_rs_df
my_submission = pd.DataFrame({'tweetid': test.tweetid, 'sentiment': prediction_lsvc_best1})
# you could use any filename. We choose submission here
my_submission.to_csv('finalsubmission.csv', index=False)
