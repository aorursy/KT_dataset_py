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
import seaborn as sns

import matplotlib.pyplot as plt

import re

from collections import Counter

import operator

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize 

import re

import nltk

import pandas as pd

from collections import Counter

from itertools import chain

import tensorflow as tf

from keras.optimizers import Nadam

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer

from sklearn.metrics import confusion_matrix, classification_report

train = pd.read_csv('../input/nlp-getting-started/train.csv')

test = pd.read_csv('../input/nlp-getting-started/test.csv')
train.head()
def getNullColDict(train,columns):

    null_count = []

    for column in columns:

        null_count.append((train[column].isnull().sum())*100/(train.shape[0]))

    

    dict_percentage = dict(zip(columns,null_count))

    return dict_percentage



columns = list(train.columns[1:])

dict_percentage = getNullColDict(train,columns)   

dict_percentage = dict(sorted(dict_percentage.items(), key=lambda x: x[1]))
def barplot(X,Xlabel,Y,Ylabel,title,size=(10,10)):

    plt.figure(figsize=size)

    sns.set(style="whitegrid")

    plt.title(title,fontsize=14,fontweight="bold")

    plt.xlabel(Xlabel,fontsize=14,fontweight="bold")

    plt.ylabel(Ylabel,fontsize=14,fontweight="bold")

    ax = sns.barplot(x=X, y=Y,palette="rocket")

    plt.show()

barplot(list(dict_percentage.keys()),'Feature',list(dict_percentage.values()),'Percentage', 'Missing Feature Values Percentage',size=(8,8))
labels = ['Real Disaster', 'Not Disaster']

amount = []

amount.append(train.loc[train.target == 1].shape[0]/train.shape[0]*100)

amount.append(train.loc[train.target == 0].shape[0]/train.shape[0]*100)

barplot(labels,'Target',amount,'Percentage', 'Target Classes Percentage')
plt.figure(figsize=(20,15))

df = train.keyword.value_counts().to_frame()

barplot(list(df.keyword)[0:20],'Kewords',list(df.index)[0:20],'Count', 'Top 10 Most Frequent Keywords',(20,10))
def getDisasterDict(df,train):

    positive = []

    negative = []

    for key in list(df.index[0:20]):

        positive.append(train.loc[(train.keyword == key) & (train.target == 1)].shape[0])

        negative.append(train.loc[(train.keyword == key) & (train.target == 0)].shape[0])

    dict_top10 = {'type': list(df.index[0:20]), 'positives': positive, 'negatives': negative}

    return dict_top10



dict_top10 = getDisasterDict(df,train)
barWidth = 0.25



bars1 = dict_top10['positives']

bars2 = dict_top10['negatives']

 

r1 = np.arange(len(bars1))

r2 = [x + barWidth for x in r1]



plt.figure(figsize=(25,15))

plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Real Disasters')

plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='Not Disasters')

 

plt.title("Top 10 Disaster Ocurrances - Real or Not", fontsize=14, fontweight='bold')

plt.xlabel('Disaster Keyword', fontweight='bold')

plt.ylabel('Count', fontweight='bold')

plt.xticks([r + barWidth for r in range(len(bars1))], list(dict_top10['type']))

 

plt.legend()

plt.show()
def getNumberWords (row):

    return len(row.text.split())



def getNumberUnique(row):

    return len(set(str(row).split()))



def getMeanLength(row):

    words = row.split()

    return sum(len(word) for word in words) / len(words)



from nltk.corpus import stopwords    

stop_words = set(stopwords.words('english'))

train['stopwords'] = train['text'].str.split().apply(lambda x: len(set(x) & stop_words))

train.head()
f, axes = plt.subplots(2, 2,figsize=(20,10))

true_disaster = train[train.target == 1]['text'].str.len()

false_disaster = train[train.target == 0]['text'].str.len()

#plt.figure(figsize=(20,10))

#"Character Number in Tweets and Disaster Veracity"

axes[0][0].set_title("Character Number in Tweets and Disaster Veracity",fontsize=14,fontweight='bold')

axes[0][0].set_xlabel('Tweet Length',fontsize=14, fontweight='bold')

axes[0][0].set_ylabel('Probability',fontsize=14, fontweight='bold')

sns.distplot( true_disaster , color="blue", label='True Disaster',ax=axes[0][0])

sns.distplot( false_disaster , color="red", label="Not Disaster",ax=axes[0][0])

axes[0][0].legend()



train['number_words_real'] = train[train.target == 1].apply(lambda row: getNumberWords(row), axis=1)

train['number_words_fake'] = train[train.target == 0].apply(lambda row: getNumberWords(row), axis=1)



axes[0][1].set_title("Tweet Number of Words and Disaster Veracity", fontsize=14,fontweight='bold')

axes[0][1].set_xlabel('Number of Words',fontsize=14,fontweight='bold')

axes[0][1].set_ylabel('Probability', fontsize=14,fontweight='bold')

sns.distplot( list(train['number_words_real']) , color="blue", label='True Disaster',ax=axes[0][1])

sns.distplot( list(train['number_words_fake']) , color="red", label="Not Disaster",ax=axes[0][1])

axes[0][1].legend()





train['unique_words'] = train.text.apply(lambda row: getNumberUnique(row))



axes[1][0].set_title("Number of Unique Words Distribution Plot", fontsize=14,fontweight='bold')

axes[1][0].set_xlabel('Number of unique words', fontsize=14,fontweight='bold')

axes[1][0].set_ylabel('Probability', fontsize=14,fontweight='bold')

sns.distplot( list(train[train.target == 1].unique_words) , color="blue", label='True Disaster',ax=axes[1][0])

sns.distplot( list(train[train.target == 0].unique_words) , color="red", label="Not Disaster",ax=axes[1][0])

axes[1][0].legend()





axes[1][1].set_title("Number of Stp Words Distribution Plot", fontsize=14,fontweight='bold')

axes[1][1].set_xlabel('Number of stop words', fontsize=14,fontweight='bold')

axes[1][1].set_ylabel('Probability', fontsize=14,fontweight='bold')

sns.distplot( list(train[train.target == 1].stopwords) , color="blue", label='True Disaster',ax=axes[1][1])

sns.distplot( list(train[train.target == 0].stopwords) , color="red", label="Not Disaster",ax=axes[1][1])

axes[1][1].legend()

f.subplots_adjust(hspace=0.3)

plt.show()
import re

def elementCount(row,string):

    if string == '#':

        return len(re.findall(r"#(\w+)", row))

    else:

        return len(re.findall(r"@(\w+)", row))



train['hashtag_count'] = train.text.apply(lambda row: elementCount(row,'#'))

train['at_count'] = train.text.apply(lambda row: elementCount(row,'@'))
train.describe()
train[train.target == 1].hashtag_count
print("Correlation between target and hashtag_count ", train.target.corr(train.hashtag_count))

print("Correlation between target and at_count ", train.target.corr(train.at_count))
hashtags_real_disasters = []

hashtags_fake_disasters = []

for row in range(train.shape[0]):

    if row < train[train.target == 1].shape[0]:

        hashtags_real_disasters.extend(re.findall(r"#(\w+)", train[train.target == 1].reset_index().text[row]))

    if row < train[train.target == 0].shape[0]:

        hashtags_fake_disasters.extend(re.findall(r"#(\w+)", train[train.target == 0].reset_index().text[row]))

        

hashtags_real_disasters = [x.lower() for x in hashtags_real_disasters]

hashtags_fake_disasters = [x.lower() for x in hashtags_fake_disasters]



hashtags_real_disasters = dict(Counter(hashtags_real_disasters))

hashtags_real_disasters = dict(sorted(hashtags_real_disasters.items(), key=operator.itemgetter(1),reverse=True))

hashtags_fake_disasters = dict(Counter(hashtags_fake_disasters))

hashtags_fake_disasters = dict(sorted(hashtags_fake_disasters.items(), key=operator.itemgetter(1),reverse=True))

f, axes = plt.subplots(1, 2,figsize=(20,15))

sns.set(style="whitegrid")

axes[0].set_title('Top 20 Keywords - Real Disasters',fontsize=14,fontweight="bold")

axes[0].set_xlabel('Words',fontsize=14,fontweight="bold")

axes[0].set_ylabel('Ocrruance',fontsize=14,fontweight="bold")

sns.barplot(list(hashtags_real_disasters.values())[0:20], list(hashtags_real_disasters.keys())[0:20],ax=axes[0],palette='rocket')



axes[1].set_title('Top 20 Keywords - Not Disasters',fontsize=14,fontweight="bold")

axes[1].set_xlabel('Words',fontsize=14,fontweight="bold")

axes[1].set_ylabel('Occurance',fontsize=14,fontweight="bold")

sns.barplot(list(hashtags_fake_disasters.values())[0:20], list(hashtags_fake_disasters.keys())[0:20],ax=axes[1],palette='rocket')



plt.show()



def contractions(tweet):

    tweet = re.sub(r"he's", "he is", tweet)

    tweet = re.sub(r"there's", "there is", tweet)

    tweet = re.sub(r"We're", "We are", tweet)

    tweet = re.sub(r"That's", "That is", tweet)

    tweet = re.sub(r"won't", "will not", tweet)

    tweet = re.sub(r"they're", "they are", tweet)

    tweet = re.sub(r"Can't", "Cannot", tweet)

    tweet = re.sub(r"wasn't", "was not", tweet)

    tweet = re.sub(r"don\x89Ûªt", "do not", tweet)

    tweet = re.sub(r"aren't", "are not", tweet)

    tweet = re.sub(r"isn't", "is not", tweet)

    tweet = re.sub(r"What's", "What is", tweet)

    tweet = re.sub(r"haven't", "have not", tweet)

    tweet = re.sub(r"hasn't", "has not", tweet)

    tweet = re.sub(r"There's", "There is", tweet)

    tweet = re.sub(r"He's", "He is", tweet)

    tweet = re.sub(r"It's", "It is", tweet)

    tweet = re.sub(r"You're", "You are", tweet)

    tweet = re.sub(r"I'M", "I am", tweet)

    tweet = re.sub(r"shouldn't", "should not", tweet)

    tweet = re.sub(r"wouldn't", "would not", tweet)

    tweet = re.sub(r"i'm", "I am", tweet)

    tweet = re.sub(r"I\x89Ûªm", "I am", tweet)

    tweet = re.sub(r"I'm", "I am", tweet)

    tweet = re.sub(r"Isn't", "is not", tweet)

    tweet = re.sub(r"Here's", "Here is", tweet)

    tweet = re.sub(r"you've", "you have", tweet)

    tweet = re.sub(r"you\x89Ûªve", "you have", tweet)

    tweet = re.sub(r"we're", "we are", tweet)

    tweet = re.sub(r"what's", "what is", tweet)

    tweet = re.sub(r"couldn't", "could not", tweet)

    tweet = re.sub(r"we've", "we have", tweet)

    tweet = re.sub(r"it\x89Ûªs", "it is", tweet)

    tweet = re.sub(r"doesn\x89Ûªt", "does not", tweet)

    tweet = re.sub(r"It\x89Ûªs", "It is", tweet)

    tweet = re.sub(r"Here\x89Ûªs", "Here is", tweet)

    tweet = re.sub(r"who's", "who is", tweet)

    tweet = re.sub(r"I\x89Ûªve", "I have", tweet)

    tweet = re.sub(r"y'all", "you all", tweet)

    tweet = re.sub(r"can\x89Ûªt", "cannot", tweet)

    tweet = re.sub(r"would've", "would have", tweet)

    tweet = re.sub(r"it'll", "it will", tweet)

    tweet = re.sub(r"we'll", "we will", tweet)

    tweet = re.sub(r"wouldn\x89Ûªt", "would not", tweet)

    tweet = re.sub(r"We've", "We have", tweet)

    tweet = re.sub(r"he'll", "he will", tweet)

    tweet = re.sub(r"Y'all", "You all", tweet)

    tweet = re.sub(r"Weren't", "Were not", tweet)

    tweet = re.sub(r"Didn't", "Did not", tweet)

    tweet = re.sub(r"they'll", "they will", tweet)

    tweet = re.sub(r"they'd", "they would", tweet)

    tweet = re.sub(r"DON'T", "DO NOT", tweet)

    tweet = re.sub(r"That\x89Ûªs", "That is", tweet)

    tweet = re.sub(r"they've", "they have", tweet)

    tweet = re.sub(r"i'd", "I would", tweet)

    tweet = re.sub(r"should've", "should have", tweet)

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

    tweet = re.sub(r"let's", "let us", tweet)

    tweet = re.sub(r"it's", "it is", tweet)

    tweet = re.sub(r"can't", "cannot", tweet)

    tweet = re.sub(r"don't", "do not", tweet)

    tweet = re.sub(r"you're", "you are", tweet)

    tweet = re.sub(r"i've", "I have", tweet)

    tweet = re.sub(r"that's", "that is", tweet)

    tweet = re.sub(r"i'll", "I will", tweet)

    tweet = re.sub(r"doesn't", "does not", tweet)

    tweet = re.sub(r"i'd", "I would", tweet)

    tweet = re.sub(r"didn't", "did not", tweet)

    tweet = re.sub(r"ain't", "am not", tweet)

    tweet = re.sub(r"you'll", "you will", tweet)

    tweet = re.sub(r"I've", "I have", tweet)

    tweet = re.sub(r"Don't", "do not", tweet)

    tweet = re.sub(r"I'll", "I will", tweet)

    tweet = re.sub(r"I'd", "I would", tweet)

    tweet = re.sub(r"Let's", "Let us", tweet)

    tweet = re.sub(r"you'd", "You would", tweet)

    tweet = re.sub(r"It's", "It is", tweet)

    tweet = re.sub(r"Ain't", "am not", tweet)

    tweet = re.sub(r"Haven't", "Have not", tweet)

    tweet = re.sub(r"Could've", "Could have", tweet)

    tweet = re.sub(r"youve", "you have", tweet)  

    tweet = re.sub(r"donå«t", "do not", tweet)   

    

    return tweet




def removeNonEnglish(row):

    words = set(nltk.corpus.words.words())

    return " ".join(w for w in nltk.wordpunct_tokenize(sent) \

             if w.lower() in words or not w.isalpha())



def convertToLowerCase(row):

    return row.lower()



def removeNumbers(row):

    return re.sub('[0-9]+', '', row)



def removeURL(row):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',row)



def removeHTML(row):

    html=re.compile(r'<.*?>')

    return html.sub(r'',row)



def removeEmoji(row):

    return row.encode('ascii', 'ignore').decode('ascii')



def removeSymbols(row):

    return re.sub(r'[^\w]', ' ', row)



def removeUnderscore(row):

    return row.replace("_","")



def removeStopWords(row):

    filtered_sentence = []

    stop_words = set(stopwords.words('english')) 

    word_tokens = word_tokenize(row) 

    return ' '.join([word for word in word_tokens if word not in stop_words])



def removeSpecialChar(row):

    punctuations = '@#!_?+&*[]-%.:/();$=><|{}^' + "'`"

    for p in punctuations:

        row = row.replace(p, f' {p} ')

    return row





abbreviations = {

    "$" : " dollar ",

    "€" : " euro ",

    "4ao" : "for adults only",

    "a.m" : "before midday",

    "a3" : "anytime anywhere anyplace",

    "aamof" : "as a matter of fact",

    "acct" : "account",

    "adih" : "another day in hell",

    "afaic" : "as far as i am concerned",

    "afaict" : "as far as i can tell",

    "afaik" : "as far as i know",

    "afair" : "as far as i remember",

    "afk" : "away from keyboard",

    "app" : "application",

    "approx" : "approximately",

    "apps" : "applications",

    "asap" : "as soon as possible",

    "asl" : "age, sex, location",

    "atk" : "at the keyboard",

    "ave." : "avenue",

    "aymm" : "are you my mother",

    "ayor" : "at your own risk", 

    "b&b" : "bed and breakfast",

    "b+b" : "bed and breakfast",

    "b.c" : "before christ",

    "b2b" : "business to business",

    "b2c" : "business to customer",

    "b4" : "before",

    "b4n" : "bye for now",

    "b@u" : "back at you",

    "bae" : "before anyone else",

    "bak" : "back at keyboard",

    "bbbg" : "bye bye be good",

    "bbc" : "british broadcasting corporation",

    "bbias" : "be back in a second",

    "bbl" : "be back later",

    "bbs" : "be back soon",

    "be4" : "before",

    "bfn" : "bye for now",

    "blvd" : "boulevard",

    "bout" : "about",

    "brb" : "be right back",

    "bros" : "brothers",

    "brt" : "be right there",

    "bsaaw" : "big smile and a wink",

    "btw" : "by the way",

    "bwl" : "bursting with laughter",

    "c/o" : "care of",

    "cet" : "central european time",

    "cf" : "compare",

    "cia" : "central intelligence agency",

    "csl" : "can not stop laughing",

    "cu" : "see you",

    "cul8r" : "see you later",

    "cv" : "curriculum vitae",

    "cwot" : "complete waste of time",

    "cya" : "see you",

    "cyt" : "see you tomorrow",

    "dae" : "does anyone else",

    "dbmib" : "do not bother me i am busy",

    "diy" : "do it yourself",

    "dm" : "direct message",

    "dwh" : "during work hours",

    "e123" : "easy as one two three",

    "eet" : "eastern european time",

    "eg" : "example",

    "embm" : "early morning business meeting",

    "encl" : "enclosed",

    "encl." : "enclosed",

    "etc" : "and so on",

    "faq" : "frequently asked questions",

    "fawc" : "for anyone who cares",

    "fb" : "facebook",

    "fc" : "fingers crossed",

    "fig" : "figure",

    "fimh" : "forever in my heart", 

    "ft." : "feet",

    "ft" : "featuring",

    "ftl" : "for the loss",

    "ftw" : "for the win",

    "fwiw" : "for what it is worth",

    "fyi" : "for your information",

    "g9" : "genius",

    "gahoy" : "get a hold of yourself",

    "gal" : "get a life",

    "gcse" : "general certificate of secondary education",

    "gfn" : "gone for now",

    "gg" : "good game",

    "gl" : "good luck",

    "glhf" : "good luck have fun",

    "gmt" : "greenwich mean time",

    "gmta" : "great minds think alike",

    "gn" : "good night",

    "g.o.a.t" : "greatest of all time",

    "goat" : "greatest of all time",

    "goi" : "get over it",

    "gps" : "global positioning system",

    "gr8" : "great",

    "gratz" : "congratulations",

    "gyal" : "girl",

    "h&c" : "hot and cold",

    "hp" : "horsepower",

    "hr" : "hour",

    "hrh" : "his royal highness",

    "ht" : "height",

    "ibrb" : "i will be right back",

    "ic" : "i see",

    "icq" : "i seek you",

    "icymi" : "in case you missed it",

    "idc" : "i do not care",

    "idgadf" : "i do not give a damn fuck",

    "idgaf" : "i do not give a fuck",

    "idk" : "i do not know",

    "ie" : "that is",

    "i.e" : "that is",

    "ifyp" : "i feel your pain",

    "IG" : "instagram",

    "iirc" : "if i remember correctly",

    "ilu" : "i love you",

    "ily" : "i love you",

    "imho" : "in my humble opinion",

    "imo" : "in my opinion",

    "imu" : "i miss you",

    "iow" : "in other words",

    "irl" : "in real life",

    "j4f" : "just for fun",

    "jic" : "just in case",

    "jk" : "just kidding",

    "jsyk" : "just so you know",

    "l8r" : "later",

    "lb" : "pound",

    "lbs" : "pounds",

    "ldr" : "long distance relationship",

    "lmao" : "laugh my ass off",

    "lmfao" : "laugh my fucking ass off",

    "lol" : "laughing out loud",

    "ltd" : "limited",

    "ltns" : "long time no see",

    "m8" : "mate",

    "mf" : "motherfucker",

    "mfs" : "motherfuckers",

    "mfw" : "my face when",

    "mofo" : "motherfucker",

    "mph" : "miles per hour",

    "mr" : "mister",

    "mrw" : "my reaction when",

    "ms" : "miss",

    "mte" : "my thoughts exactly",

    "nagi" : "not a good idea",

    "nbc" : "national broadcasting company",

    "nbd" : "not big deal",

    "nfs" : "not for sale",

    "ngl" : "not going to lie",

    "nhs" : "national health service",

    "nrn" : "no reply necessary",

    "nsfl" : "not safe for life",

    "nsfw" : "not safe for work",

    "nth" : "nice to have",

    "nvr" : "never",

    "nyc" : "new york city",

    "oc" : "original content",

    "og" : "original",

    "ohp" : "overhead projector",

    "oic" : "oh i see",

    "omdb" : "over my dead body",

    "omg" : "oh my god",

    "omw" : "on my way",

    "p.a" : "per annum",

    "p.m" : "after midday",

    "pm" : "prime minister",

    "poc" : "people of color",

    "pov" : "point of view",

    "pp" : "pages",

    "ppl" : "people",

    "prw" : "parents are watching",

    "ps" : "postscript",

    "pt" : "point",

    "ptb" : "please text back",

    "pto" : "please turn over",

    "qpsa" : "what happens", #"que pasa",

    "ratchet" : "rude",

    "rbtl" : "read between the lines",

    "rlrt" : "real life retweet", 

    "rofl" : "rolling on the floor laughing",

    "roflol" : "rolling on the floor laughing out loud",

    "rotflmao" : "rolling on the floor laughing my ass off",

    "rt" : "retweet",

    "ruok" : "are you ok",

    "sfw" : "safe for work",

    "sk8" : "skate",

    "smh" : "shake my head",

    "sq" : "square",

    "srsly" : "seriously", 

    "ssdd" : "same stuff different day",

    "tbh" : "to be honest",

    "tbs" : "tablespooful",

    "tbsp" : "tablespooful",

    "tfw" : "that feeling when",

    "thks" : "thank you",

    "tho" : "though",

    "thx" : "thank you",

    "tia" : "thanks in advance",

    "til" : "today i learned",

    "tl;dr" : "too long i did not read",

    "tldr" : "too long i did not read",

    "tmb" : "tweet me back",

    "tntl" : "trying not to laugh",

    "ttyl" : "talk to you later",

    "u" : "you",

    "u2" : "you too",

    "u4e" : "yours for ever",

    "utc" : "coordinated universal time",

    "w/" : "with",

    "w/o" : "without",

    "w8" : "wait",

    "wassup" : "what is up",

    "wb" : "welcome back",

    "yr" : 'year',

    "wtf" : "what the fuck",

    "wtg" : "way to go",

    "wtpa" : "where the party at",

    "wuf" : "where are you from",

    "wuzup" : "what is up",

    "wywh" : "wish you were here",

    "yd" : "yard",

    "ygtr" : "you got that right",

    "ynk" : "you never know",

    "zzz" : "sleeping bored and tired"

}



def convert_abbrev(row):

    return abbreviations[row.lower()] if row.lower() in abbreviations.keys() else row





def cleanAll(train):

    train['text'] = train.text.apply(lambda row: convertToLowerCase(row))

    train['text'] = train.text.apply(lambda row: contractions(row))

    train['text'] = train.text.apply(lambda row: convert_abbrev(row))

    train['text'] = train.text.apply(lambda row: removeURL(row))

    train['text'] = train.text.apply(lambda row: removeNumbers(row))

    train['text'] = train.text.apply(lambda row: removeSymbols(row))

    train['text'] = train.text.apply(lambda row: removeHTML(row))

    train['text'] = train.text.apply(lambda row: removeEmoji(row))

    train['text'] = train.text.apply(lambda row: convertToLowerCase(row))

    train['text'] = train.text.apply(lambda row: removeUnderscore(row))

    train['text'] = train.text.apply(lambda row: removeStopWords(row))

    train['text'] = train.text.apply(lambda row: removeSpecialChar(row))

    #train['text'] = train.text.apply(lambda row: removeNonEnglish(row))

    return train



train = cleanAll(train)

test = cleanAll(test)
from collections import Counter

from collections import OrderedDict

from operator import itemgetter    



real_disasters = list(train[train.target == 1].text.str.cat(sep=' ').lower().split())

not_disasters = list(train[train.target == 0].text.str.cat(sep=' ').lower().split())

counts_real = dict(Counter(real_disasters))

counts_not_real= dict(Counter(not_disasters))

real_disasters_occurance = dict(OrderedDict(sorted(counts_real.items(), key = itemgetter(1), reverse = True)))

not_disasters_occurance = dict(OrderedDict(sorted(counts_not_real.items(), key = itemgetter(1), reverse = True)))
f, axes = plt.subplots(1, 2,figsize=(20,15))

sns.set(style="whitegrid")

axes[0].set_title('Real Disasters - Top 20 Words',fontsize=14,fontweight="bold")

axes[0].set_xlabel('Top Words',fontsize=14,fontweight="bold")

axes[0].set_ylabel('Ocrruance',fontsize=14,fontweight="bold")

sns.barplot(list(real_disasters_occurance.values())[0:20], list(real_disasters_occurance.keys())[0:20],ax=axes[0],palette='rocket')



axes[1].set_title('Not Disasters - Top 20 Words',fontsize=14,fontweight="bold")

axes[1].set_xlabel('Top Words',fontsize=14,fontweight="bold")

axes[1].set_ylabel('Occurance',fontsize=14,fontweight="bold")

sns.barplot(list(not_disasters_occurance.values())[0:20], list(not_disasters_occurance.keys())[0:20],ax=axes[1],palette='rocket')



plt.show()

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt



def generateWordCloud(df):

    text =df

    wordcloud = WordCloud(

        width = 3000,

        height = 2000,

        background_color = 'white',

        stopwords = STOPWORDS).generate(str(text))

    fig = plt.figure(

        figsize = (10, 6),

        facecolor = 'k',

        edgecolor = 'k')

    plt.imshow(wordcloud, interpolation = 'bilinear')

    plt.axis('off')

    plt.tight_layout(pad=0)

    plt.show()
generateWordCloud(list(real_disasters_occurance.keys())[0:100])
generateWordCloud(list(not_disasters_occurance.keys())[0:100])
def find_ngrams(input_list, n):

    return list(zip(*[input_list[i:] for i in range(n)]))



def convertTuple(tup): 

    str =  ' '.join(tup) 

    return str



train['bigrams'] = train['text'].map(lambda x: find_ngrams(x.split(" "), 2))

bigrams_real = train[train.target == 1].bigrams.tolist()

bigrams_real = list(chain(*bigrams_real))

bigrams_real = [(x.lower(), y.lower()) for x,y in bigrams_real]



bigrams_fake = train[train.target == 0].bigrams.tolist()

bigrams_fake = list(chain(*bigrams_fake))

bigrams_fake = [(x.lower(), y.lower()) for x,y in bigrams_fake]



bigram_counts_real = Counter(bigrams_real)

bigram_real_dict = dict(bigram_counts_real.most_common(20))



bigram_counts_fake = Counter(bigrams_fake)

bigram_fake_dict = dict(bigram_counts_fake.most_common(20))
bigrams_real = []

bigrams_fake = []

for bigram in range(len(bigram_real_dict)):

    bigrams_real.append(convertTuple(list(bigram_real_dict.keys())[bigram]))

    bigrams_fake.append(convertTuple(list(bigram_fake_dict.keys())[bigram]))





f, axes = plt.subplots(1, 2,figsize=(25,15))

sns.set(style="whitegrid")

axes[0].set_title('Most Common Bigrams - Real Disasters',fontsize=14,fontweight="bold")

axes[0].set_xlabel('Bigrams',fontsize=14,fontweight="bold")

axes[0].set_ylabel('Ocrruance',fontsize=14,fontweight="bold")

sns.barplot(list(bigram_real_dict.values()),bigrams_real,ax=axes[0],palette='rocket')



axes[1].set_title('Most Common Bigrams - Not Disasters',fontsize=14,fontweight="bold")

axes[1].set_xlabel('Bigrams',fontsize=14,fontweight="bold")

axes[1].set_ylabel('Occurance',fontsize=14,fontweight="bold")

sns.barplot(list(bigram_fake_dict.values()),bigrams_fake,ax=axes[1],palette='rocket')



plt.show()    

def find_ngrams(input_list, n):

    return list(zip(*[input_list[i:] for i in range(n)]))



def convertTuple(tup): 

    str =  ' '.join(tup) 

    return str



train['trigrams'] = train['text'].map(lambda x: find_ngrams(x.split(" "), 3))

trigrams_real = train[train.target == 1].trigrams.tolist()

trigrams_real = list(chain(*trigrams_real))

trigrams_real = [(x.lower(), y.lower(), z.lower()) for x,y,z in trigrams_real]



trigrams_fake = train[train.target == 0].trigrams.tolist()

trigrams_fake = list(chain(*trigrams_fake))

trigrams_fake = [(x.lower(), y.lower(), z.lower()) for x,y,z in trigrams_fake]



trigram_counts_real = Counter(trigrams_real)

trigram_real_dict = dict(trigram_counts_real.most_common(20))



trigram_counts_fake = Counter(trigrams_fake)

trigram_fake_dict = dict(trigram_counts_fake.most_common(20))



trigrams_real = []

trigrams_fake = []

for bigram in range(len(bigram_real_dict)):

    trigrams_real.append(convertTuple(list(trigram_real_dict.keys())[bigram]))

    trigrams_fake.append(convertTuple(list(trigram_fake_dict.keys())[bigram]))





f, axes = plt.subplots(1, 2,figsize=(25,15))

sns.set(style="whitegrid")

axes[0].set_title('Most Common Trigrams - Real Disasters',fontsize=14,fontweight="bold")

axes[0].set_xlabel('Bigrams',fontsize=14,fontweight="bold")

axes[0].set_ylabel('Ocrruance',fontsize=14,fontweight="bold")

sns.barplot(list(trigram_real_dict.values()),trigrams_real,ax=axes[0],palette='rocket')



axes[1].set_title('Most Common Trigrams - Not Disasters',fontsize=14,fontweight="bold")

axes[1].set_xlabel('Bigrams',fontsize=14,fontweight="bold")

axes[1].set_ylabel('Occurance',fontsize=14,fontweight="bold")

sns.barplot(list(trigram_fake_dict.values()),trigrams_fake,ax=axes[1],palette='rocket')



plt.show()    
from nltk.stem import WordNetLemmatizer     

lemmatizer = WordNetLemmatizer()

#lemmatizer.lemmatize()



def getLemma(row):

    text = row

    text = text.lower().split(" ")

    lemma = [lemmatizer.lemmatize(each) for each in text]

    return lemma



train['text'] = train.text.apply(lambda row: getLemma(row))

train.head()
train['text'] = train.text.apply(lambda row: " ".join(row))
def tokenize(train,test):

    

    train_ = train.text

    test_ = test.text

    full = train_.append(test_)

    tokenizer = Tokenizer() 

    tokenizer.fit_on_texts(full)

    sequences = tokenizer.texts_to_sequences(full)



    word_index = tokenizer.word_index

    both_datasets = pad_sequences(sequences)

    train_data = both_datasets[:len(train)]

    test_data = both_datasets[len(train):]

    labels = train['target']



    return train_data, labels, word_index, test_data







X_train, y_train, word_index, X_test = tokenize(train,test)



X_test.shape
embeddings_index = {}

with open('../input/glove-data/glove.6B.200d.txt') as f:

    for line in (f):

        values = line.split()

        word = values[0]

        coefs = np.asarray(values[1:], dtype='float32')

        embeddings_index[word] = coefs

f.close()

embedding_dim = 200

print('Found %s word vectors in the GloVe library' % len(embeddings_index))
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))



for word, i in word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector



embedding_dim = 200

max_length = 31

trunc_type='post'



train_sentences = train.text.tolist()

train_labels = train.target

tokenizer = Tokenizer()

tokenizer.fit_on_texts(train_sentences)

word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_sentences)

train_padded = pad_sequences(train_sequences,maxlen=max_length, truncating=trunc_type)







model = tf.keras.Sequential()

model.add(tf.keras.layers.Embedding(len(word_index)+1, embedding_dim, input_length=max_length))

model.add(tf.keras.layers.GlobalAveragePooling1D())

model.add(tf.keras.layers.Dense(14, activation='relu'))

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='Nadam',metrics=['accuracy'])

model.summary()

num_epochs = 10

history = model.fit(train_padded, train_labels, epochs=num_epochs,validation_split=0.2)
f, axes = plt.subplots(1, 2,figsize=(25,10))

axes[0].set_title('Progress During Training - Accuracy',fontweight='bold',fontsize=14)

axes[0].set_xlabel('Epochs',fontweight='bold',fontsize=14)

axes[0].set_ylabel('Accuracy',fontweight='bold',fontsize=14)

axes[0].plot(range(1,len(history.history['accuracy'])+1),history.history['accuracy'],label='Training Set')

axes[0].plot(range(1,len(history.history['accuracy'])+1),history.history['val_accuracy'],label='Test Set')

axes[0].legend()



axes[1].set_title('Progress During Training - Accuracy',fontweight='bold',fontsize=14)

axes[1].set_xlabel('Epochs',fontweight='bold',fontsize=14)

axes[1].set_ylabel('Loss',fontweight='bold',fontsize=14)

axes[1].plot(range(1,len(history.history['accuracy'])+1),history.history['loss'],label='Training Set')

axes[1].plot(range(1,len(history.history['accuracy'])+1),history.history['val_loss'],label='Test Set')

axes[1].legend()

plt.show()
train_size = (0.1*X_train.shape[0])

x_train = X_train[:int(train_size)]

y_train_ = y_train[:int(train_size)]

x_validation = X_train[int(train_size):]

y_validation = y_train[int(train_size):]

x_validation.shape
def create_model():



    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Embedding(19633, 200, weights=[embedding_matrix], input_length=31,trainable=False))

    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.GlobalAveragePooling1D())

    model.add(tf.keras.layers.Dense(14, activation='relu'))

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss = 'binary_crossentropy', optimizer = 'Nadam', metrics = ['accuracy'])

    return model



def LSTM():

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Embedding(len(word_index)+1, 200, weights=[embedding_matrix], input_length=31,trainable=False))

    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200, dropout=0.2, recurrent_dropout=0.2)))

    model.add(tf.keras.layers.Dense(14, activation='relu'))

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss = 'binary_crossentropy', optimizer = 'Nadam', metrics = ['accuracy'])

    return model
model2 = create_model()

model2.summary()

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history2 = model2.fit(X_train, y_train, validation_data=(x_validation,y_validation),

         epochs = 30, batch_size = 32, verbose = 1, shuffle = True)
f, axes = plt.subplots(1, 2,figsize=(25,10))

axes[0].set_title('Progress During Training - Accuracy',fontweight='bold',fontsize=14)

axes[0].set_xlabel('Epochs',fontweight='bold',fontsize=14)

axes[0].set_ylabel('Accuracy',fontweight='bold',fontsize=14)

axes[0].plot(range(1,len(history2.history['accuracy'])+1),history2.history['accuracy'],label='Training Set')

axes[0].plot(range(1,len(history2.history['accuracy'])+1),history2.history['val_accuracy'],label='Test Set')

axes[0].legend()



axes[1].set_title('Progress During Training - Accuracy',fontweight='bold',fontsize=14)

axes[1].set_xlabel('Epochs',fontweight='bold',fontsize=14)

axes[1].set_ylabel('Loss',fontweight='bold',fontsize=14)

axes[1].plot(range(1,len(history2.history['accuracy'])+1),history2.history['loss'],label='Training Set')

axes[1].plot(range(1,len(history2.history['accuracy'])+1),history2.history['val_loss'],label='Test Set')

axes[1].legend()

plt.show()


def plot_confusion_matrix(cm,

                          target_names,

                          title='Confusion matrix',

                          cmap=None,

                          normalize=True):



    import matplotlib.pyplot as plt

    import numpy as np

    import itertools



    accuracy = np.trace(cm) / float(np.sum(cm))

    misclass = 1 - accuracy



    if cmap is None:

        cmap = plt.get_cmap('Blues')



    plt.figure(figsize=(8, 6))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title,fontweight='bold',fontsize=14)

    plt.colorbar()



    if target_names is not None:

        tick_marks = np.arange(len(target_names))

        plt.xticks(tick_marks, target_names, rotation=0)

        plt.yticks(tick_marks, target_names)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]





    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        if normalize:

            plt.text(j, i, "{:0.4f}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")

        else:

            plt.text(j, i, "{:,}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")





    plt.tight_layout()

    plt.ylabel('True label',fontweight='bold',fontsize=14)

    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass),fontweight='bold',fontsize=14)

    plt.show()

    

pred = model2.predict(X_train)

Y_pred_classes = np.around(pred.transpose()[0])

Y_true = np.array(y_train)

cm = confusion_matrix(Y_pred_classes,Y_true)     

plot_confusion_matrix(cm,cmap='Blues',normalize=True,target_names=['Disaster','Not Disaster'],title='Confusion Matrix - Training Set')

pred = model2.predict(x_validation)

Y_pred_classes = np.around(pred.transpose()[0])

Y_true = np.array(y_validation)

cm = confusion_matrix(Y_pred_classes,Y_true)     

plot_confusion_matrix(cm,normalize=True,target_names=['Disaster','Not Disaster'],title='Confusion Matrix')

test_id = test.id

submission1 = pd.DataFrame()

submission1['id'] = test_id

submission1['target'] = np.around(model2.predict(X_test).transpose()[0]).astype(int)

submission1

submission1.head(10)

submission1.to_csv('submission.csv', index=False)