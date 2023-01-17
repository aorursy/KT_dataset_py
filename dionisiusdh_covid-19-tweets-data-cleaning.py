import pandas as pd
import numpy as np
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline

from textblob import TextBlob as tb
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS

!pip install PySastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

!pip install googletrans
from googletrans import Translator
df = pd.read_csv('../input/covid19-indonesian-twitter-sentiment/covid-sentiment.csv')
df.shape
df = df.drop_duplicates(subset='tweet', keep='first').reset_index()
df.shape
slang_dict = pd.read_csv('../input/indonesian-abusive-and-hate-speech-twitter-text/new_kamusalay.csv', encoding='latin-1', header=None)
slang_dict = slang_dict.rename(columns={0: 'original', 
                                      1: 'replacement'})

id_stopword_dict = pd.read_csv('../input/indonesian-stoplist/stopwordbahasa.csv', header=None)
id_stopword_dict = id_stopword_dict.rename(columns={0: 'stopword'})
stopwords_new = pd.DataFrame(['sih','nya', 'iya', 'nih', 'biar', 'tau', 'kayak', 'banget'], columns=['stopword'])
id_stopword_dict = pd.concat([id_stopword_dict,stopwords_new]).reset_index()
id_stopword_dict = pd.DataFrame(id_stopword_dict['stopword'])
factory = StemmerFactory()
stemmer = factory.create_stemmer()


def lowercase(text):
    return text.lower()

def remove_unnecessary_char(text):
    text = re.sub(r'pic.twitter.com.[\w]+', '', text) # Remove every pic 
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',text) # Remove every URL
    
    text = re.sub('gue','saya',text) # Sub gue saya
    text = re.sub('\n',' ',text) # Remove every '\n'
    
    text = re.sub(r'[^\x00-\x7F]+',' ', text)
    text = re.sub(r':', '', text)
    text = re.sub(r'‚Ä¶', '', text)
    
    to_delete = ['hypertext', 'transfer', 'protocol', 'over', 'secure', 'socket', 'layer', 'dtype', 'tweet', 'name', 'object'
                 ,'twitter','com', 'pic', ' ya ']
    
    for word in to_delete:
        text = re.sub(word,'', text)
        text = re.sub(word.upper(),' ',text)
    
    retweet_user = [' rt ', ' user ']
    
    for word in retweet_user:
        text = re.sub(word,' ',text) # Remove every retweet symbol & username
        text = re.sub(word.upper(),' ',text)
        
    text = re.sub('  +', ' ', text) # Remove extra spaces
    return text
    
def remove_nonaplhanumeric(text):
    text = re.sub('[^0-9a-zA-Z]+', ' ', text) 
    return text

slang_dict_map = dict(zip(slang_dict['original'], slang_dict['replacement']))

def normalize_slang(text):
    return ' '.join([slang_dict_map[word] if word in slang_dict_map else word for word in text.split(' ')])

def remove_stopword(text):
    text = ' '.join(['' if word in id_stopword_dict.stopword.values else word for word in text.split(' ')])
    text = re.sub('  +', ' ', text) # Remove extra spaces
    text = text.strip()
    return text

def stemming(text):
    return stemmer.stem(text)
def preprocess(text):
    text = lowercase(text)
    text = remove_unnecessary_char(text)
    text = remove_nonaplhanumeric(text)
    text = normalize_slang(text)
    text = stemming(text) 
    text = remove_stopword(text)
    return text
df['tweet'] = df['tweet'].apply(preprocess).apply(preprocess)
df = df.drop_duplicates(subset='tweet', keep='first').reset_index()
df.shape
df.to_csv('covid-sentiment-preprocessed.csv', index=False)
mpl.rcParams['figure.figsize']=(12.0,12.0) 
mpl.rcParams['font.size']=12              
mpl.rcParams['savefig.dpi']=100             
mpl.rcParams['figure.subplot.bottom']=.1 


stopwords = set(STOPWORDS)

wordcloud = WordCloud(
                          background_color='white',
                          stopwords=id_stopword_dict,
                          max_words=400,
                          max_font_size=50, 
                          random_state=69
                         ).generate(str(df['tweet']))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=900)
translator = Translator()
translator.translate('nice', dest='id').text
def en_to_id(sentence):
    if tb(sentence).detect_language() == 'en':
        return tb(sentence)
    
    translator = Translator()
    
    output = translator.translate(sentence, dest='en')
    return tb(output.text)
    
def get_sentiment(sentence):
    sentence = en_to_id(sentence)
    return sentence.sentiment

def round_polarity(value):
    if value >= 0.3:
        return 1
    elif value == 0:
        return 0
    return -1

def round_subjectivity(value):
    if value >= 0:
        return 1
    elif value == 0:
        return 0
    return -1
tweets = df['tweet']
polarity = []
subjectivity = []

for tweet in tweets:
    sentiment = get_sentiment(tweet)
    
    polarity.append(round_polarity(sentiment[0]))
    subjectivity.append(round_subjectivity(sentiment[1]))