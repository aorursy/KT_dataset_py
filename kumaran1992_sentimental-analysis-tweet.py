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
import pandas as pd
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
df = pd.read_csv('/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv',encoding = 'latin1',header = None)
df.head(3)
df = df[[5,0]]
pd.set_option('display.max_columns', 15000)
pd.set_option('display.max_rows', 15000)
df.columns = ['twitts','sentiment']
df.head()
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')
df['sentiment'].value_counts()
#Count words
df['word_count']=df['twitts'].apply(lambda x :len(str(x).split()))
#count character
df['char_count']=df['twitts'].apply(lambda x :len(x))
df.head()
#Avg_word
def avg_word(x):
    words = x.split()
    word_len = 0
    for word in words:
        word_len = word_len + len(word)
    return word_len/len(words)  
df['Avg_count'] = df['twitts'].apply(lambda x :avg_word(x))    
df.head()    
print(STOP_WORDS)
#Stop_word_len
df['stop_word_len']= df['twitts'].apply(lambda x :len([t for t in x.split() if t in STOP_WORDS]))
df.head()
for t in df['twitts'][0].split():
    if t.startswith('@'):
        print(t)
        print(len(t))
#Count hastag & @
df['hash_len']= df['twitts'].apply(lambda x: len([t for t in x.split() if t.startswith('#')]))
df['mention_len']= df['twitts'].apply(lambda x: len([t for t in x.split() if t.startswith('@')]))
df.head()
#count numeric digit
df['numeric']= df['twitts'].apply(lambda x: len([t for t in x.split() if t.isdigit()]))
df['upper_case']= df['twitts'].apply(lambda x: len([t for t in x.split() if t.isupper() and len(x)>3]))
df['lower_case']= df['twitts'].apply(lambda x: len([t for t in x.split() if t.islower() and len(x)>3]))
df.head()
df['twitts'][1]
#Change all character to lower case
df['twitts']=df['twitts'].apply(lambda x:x.lower())
#contraction to Expansion
contractions = {
"aight": "alright",
"ain't": "am not",
"amn't": "am not",
"aren't": "are not",
"can't": "can not",
"cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"daren't": "dare not",
"daresn't": "dare not",
"dasn't": "dare not",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"d'ye": "do you",
"e'er": "ever",
"everybody's": "everybody is",
"everyone's": "everyone is",
"finna": "fixing to",
"g'day": "good day",
"gimme": "give me",
"giv'n": "given",
"gonna": "going to",
"gon't": "go not",
"gotta": "got to",
"hadn't": "had not",
"had've": "had have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had",
"he'dn't've'd": "he would not have had",
"he'll": "he will",
"he's": "he is",
"he've": "he have",
"how'd": "how would",
"howdy": "how do you do",
"how'll": "how will",
"how're": "how are",
"I'll": "I will",
"I'm": "I am",
"I'm'a": "I am about to",
"I'm'o": "I am going to",
"innit": "is it not",
"I've": "I have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"may've": "may have",
"methinks": "me thinks",
"mightn't": "might not",
"might've": "might have",
"mustn't": "must not",
"mustn't've": "must not have",
"must've": "must have",
"needn't": "need not",
"ne'er": "never",
"o'clock": "of the clock",
"o'er": "over",
"ol'": "old",
"oughtn't": "ought not",
"'s": "is",
"shalln't": "shall not",
"shan't": "shall not",
"she'd": "she would",
"she'll": "she shall",
"she'll": "she will",
"she's": "she has",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"somebody's": "somebody has",
"somebody's": "somebody is",
"someone's": "someone has",
"someone's": "someone is",
"something's": "something has",
"something's": "something is",
"so're": "so are",
"that'll": "that shall",
"that'll": "that will",
"that're": "that are",
"that's": "that has",
"that's": "that is",
"that'd": "that would",
"that'd": "that had",
"there'd": "there had",
"there'd": "there would",
"there'll": "there shall",
"there'll": "there will",
"there're": "there are",
"there's": "there has",
"there's": "there is",
"these're": "these are",
"these've": "these have",
"they'd": "they had",
"they'd": "they would",
"they'll": "they shall",
"they'll": "they will",
"they're": "they are",
"they're": "they were",
"they've": "they have",
"this's": "this has",
"this's": "this is",
"those're": "those are",
"those've": "those have",
"'tis": "it is",
"to've": "to have",
"'twas": "it was",
"wanna": "want to",
"wasn't": "was not",
"we'd": "we had",
"we'd": "we would",
"we'd": "we did",
"we'll": "we shall",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'd": "what did",
"what'll": "what shall",
"what'll": "what will",
"what're": "what are",
"what're": "what were",
"what's": "what has",
"what's": "what is",
"what's": "what does",
"what've": "what have",
"when's": "when has",
"when's": "when is",
"where'd": "where did",
"where'll": "where shall",
"where'll": "where will",
"where're": "where are",
"where's": "where has",
"where's": "where is",
"where's": "where does",
"where've": "where have",
"which'd": "which had",
"which'd": "which would",
"which'll": "which shall",
"which'll": "which will",
"which're": "which are",
"which's": "which has",
"which's": "which is",
"which've": "which have",
"who'd": "who would",
"who'd": "who had",
"who'd": "who did",
"who'd've": "who would have",
"who'll": "who shall",
"who'll": "who will",
"who're": "who are",
"who's": "who has",
"who's": "who is",
"who's": "who does",
"who've": "who have",
"why'd": "why did",
"why're": "why are",
"why's": "why has",
"why's": "why is",
"why's": "why does",
"won't": "will not",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd've": "you all would have",
"y'all'dn't've'd": "you all would not have had",
"y'all're": "you all are",
"you'd": "you had",
"you'd": "you would",
"you'll": "you shall",
"you'll": "you will",
"you're": "you are",
"you're": "you are",
"you've": "you have",
" u ": "you",
" ur ": "your",
" n ": "and"
}
def cont_to_exp(x):
    if type(x) is str:
        for key in contractions:
            value = contractions[key]
            x = x.replace(key,value)
        return x
    else:
        return x
df['twitts'] = df['twitts'].apply(lambda x: cont_to_exp(x))
df.head(3)
#email remove 
import re
df['email']=df['twitts'].apply(lambda x: re.findall(r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)',x))
df['email_count']=df['email'].apply(lambda y:len(y))
df.head()
df[df['email_count']>0]
#count URL and remove it
df['ulr_flag_count']= df['twitts'].apply(lambda x:len([re.findall(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', x)]))
df.head()
#remove the URL from the message of twitts
df['twitts']=df['twitts'].apply(lambda x: re.sub('(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?',"", x))
df.head()
#remove the RT
df['twitts']=df['twitts'].apply(lambda x: re.sub('RT',"", x))
df.head()
#Remove special character and punctuation
df['twitts'] = df['twitts'].apply(lambda x: re.sub('[^a-z A-Z 0-9-]+','', x))
df.head()
df['twitts'][0]
#remove multiple white space
df['twitts'] = df['twitts'].apply(lambda x: ' '.join(x.split()))
df.head()
#remove html tags
#from bs4 import BeautifulSoup
#df['twitts'] = df['twitts'].apply(lambda x: BeautifulSoup(x, 'lxml').get_text())
df.head()
tk = df['twitts'][1]
tk
c = tk.split()
c
print(STOP_WORDS)
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
def nltk1(text):
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    #print(tokens_without_sw)
    return ("").join(tokens_without_sw)
df['twitts'].apply(lambda x: nltk1(x))