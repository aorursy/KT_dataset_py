# Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import re

from bs4 import BeautifulSoup



import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize



# Input data files are available in the "../input/" directory.

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Get train & test data

pd.set_option('display.max_colwidth', -1)

#pd.set_option('display.max_rows', None)

train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

def callback(operation_future):

    result = operation_future.result()
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
def convert_abbrev(word):

    return abbreviations[word.lower()] if word.lower() in abbreviations.keys() else word
stopwords_list = stopwords.words('english')



def clean_text(text):

    # Remove URLs

    text = re.sub(r"http\S+", '', text)

    text = re.sub(r"www\S+", '', text)

    text = re.sub(r"pic.twitter.com\S+", '', text)



    # Remove XML tags

    text = BeautifulSoup(text, "lxml").text

    return text



def get_abbrev(text):    

    tokens = word_tokenize(text)

    tokens = [word for word in tokens if word.lower() in abbreviations.keys()]

    text = ' '.join(tokens)

    return text

    



def convert_abbrev_in_text(text):

    tokens = word_tokenize(text)

    tokens=[convert_abbrev(word) for word in tokens]

    text = ' '.join(tokens)

    return text



train['clean_text'] = train['text'].apply(lambda x: clean_text(x))

train['converted_text'] = train['clean_text'].apply(lambda x: convert_abbrev_in_text(x))

train['abbreviations_in_text'] = train['clean_text'].apply(lambda x: get_abbrev(x))



test['clean_text'] = test['text'].apply(lambda x: clean_text(x))

test['converted_text'] = test['clean_text'].apply(lambda x: convert_abbrev_in_text(x))

test['abbreviations_in_text'] = test['clean_text'].apply(lambda x: get_abbrev(x))
df = pd.concat([train[train['abbreviations_in_text']!= ""], test[test['abbreviations_in_text']!= ""]], sort=False)[['id', 'text', 'converted_text', 'abbreviations_in_text']] ;

print(df.shape)

df
not_in_data = [] ; in_data = []

for i in abbreviations.keys():

    if train[train['abbreviations_in_text'].str.contains(i, na=False, case=False)].empty and test[test['text'].str.contains(i, na=False, case=False)].empty :

        not_in_data.append(i)

    else :

        in_data.append(i)

print("Abbreviations in data : {}\n{}\n\nAbbreviations not in data : {}\n{}\n".format(len(in_data), in_data, len(not_in_data), not_in_data))
# Abbreviations frequency in train and test data

abbrv_in_train = train.abbreviations_in_text.str.get_dummies(sep=' ').sum().sort_values(ascending=False)

abbrv_in_test = test.abbreviations_in_text.str.get_dummies(sep=' ').sum().sort_values(ascending=False)

abbrv_in_data = pd.concat([abbrv_in_train, abbrv_in_test], axis=1, sort=False, keys=['train', 'test']).fillna(0)

abbrv_in_data
# Plotting data

x = np.arange(30)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,14))

ax1.bar(x, abbrv_in_train[:30], color="blue", label='train')

ax2.bar(x, abbrv_in_test[:30], color="orange", label='test')

    

# Add some text for labels, title and custom x-axis tick labels, etc.

ax1.set_xticks(x) ; ax2.set_xticks(x)

ax1.set_xticklabels(abbrv_in_train.index, fontsize=14)

ax2.set_xticklabels(abbrv_in_test.index, fontsize=14)

ax1.legend() ; ax2.legend()

fig.text(0.5, -0.02, 'Abbreviations and/or Slangs', ha='center', fontsize=16)

fig.text(-0.01, 0.5, 'Frenquencies', va='center', rotation='vertical', fontsize=16)

fig.suptitle('30 most common Abbreviations in train and test data', fontsize=20)



for i in x:

    ax1.annotate(abbrv_in_train[i], xy=(i, abbrv_in_train[i]), textcoords="offset points", rotation=0, xytext=(0, 3), ha='center', fontsize=14)

    ax2.annotate(abbrv_in_test[i], xy=(i, abbrv_in_test[i]), textcoords="offset points", rotation=0, xytext=(0, 3), ha='center', fontsize=14)



plt.tight_layout(pad=3.0)

plt.show()