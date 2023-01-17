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
train=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

print(train.head())

print(test.head())
print(train.shape)

print(test.shape)

print(train.columns)

print(test.columns)
print((train.isnull()==True).sum())

print((test.isnull()==True).sum())
train['target'].value_counts()
abbreviations = { 

    "$" : " dollar ",    "€" : " euro ",    "4ao" : "for adults only",    "a.m" : "before midday",    "a3" : "anytime anywhere anyplace",

    "aamof" : "as a matter of fact",    "acct" : "account",    "adih" : "another day in hell",    "afaic" : "as far as i am concerned",

    "afaict" : "as far as i can tell",    "afaik" : "as far as i know",

    "afair" : "as far as i remember",    "afk" : "away from keyboard",

    "app" : "application",    "approx" : "approximately",

    "apps" : "applications",    "asap" : "as soon as possible",    "asl" : "age, sex, location",    "atk" : "at the keyboard",    "ave." : "avenue",    "aymm" : "are you my mother",

    "ayor" : "at your own risk",     "b&b" : "bed and breakfast",

    "b+b" : "bed and breakfast",    "b.c" : "before christ",

    "b2b" : "business to business",    "b2c" : "business to customer",

    "b4" : "before",    "b4n" : "bye for now",    "b@u" : "back at you",

    "bae" : "before anyone else",    "bak" : "back at keyboard",    "bbbg" : "bye bye be good",    "bbc" : "british broadcasting corporation",

    "bbias" : "be back in a second",    "bbl" : "be back later",    "bbs" : "be back soon",    "be4" : "before",

    "bfn" : "bye for now",    "blvd" : "boulevard",    "bout" : "about",    "brb" : "be right back",

    "bros" : "brothers",    "brt" : "be right there",    "bsaaw" : "big smile and a wink",    "btw" : "by the way",

    "bwl" : "bursting with laughter",    "c/o" : "care of",    "cet" : "central european time",    "cf" : "compare",

    "cia" : "central intelligence agency",    "csl" : "can not stop laughing",    "cu" : "see you",    "cul8r" : "see you later",    "cv" : "curriculum vitae",

    "cwot" : "complete waste of time",    "cya" : "see you",    "cyt" : "see you tomorrow",    "dae" : "does anyone else",    "dbmib" : "do not bother me i am busy",

    "diy" : "do it yourself",    "dm" : "direct message",    "dwh" : "during work hours",    "e123" : "easy as one two three",    "eet" : "eastern european time",

    "eg" : "example",    "embm" : "early morning business meeting",    "encl" : "enclosed",    "encl." : "enclosed",    "etc" : "and so on",

    "faq" : "frequently asked questions",    "fawc" : "for anyone who cares",    "fb" : "facebook",    "fc" : "fingers crossed",    "fig" : "figure",

    "fimh" : "forever in my heart",     "ft." : "feet",    "ft" : "featuring",    "ftl" : "for the loss",    "ftw" : "for the win",    "fwiw" : "for what it is worth",

    "fyi" : "for your information",    "g9" : "genius",    "gahoy" : "get a hold of yourself",    "gal" : "get a life", "gcse" : "general certificate of secondary education",

    "gfn" : "gone for now",    "gg" : "good game",    "gl" : "good luck",    "glhf" : "good luck have fun",    "gmt" : "greenwich mean time",

    "gmta" : "great minds think alike",    "gn" : "good night",    "g.o.a.t" : "greatest of all time",    "goat" : "greatest of all time",

    "goi" : "get over it",    "gps" : "global positioning system",    "gr8" : "great",    "gratz" : "congratulations",    "gyal" : "girl",

    "h&c" : "hot and cold",    "hp" : "horsepower",    "hr" : "hour",    "hrh" : "his royal highness",    "ht" : "height",    "ibrb" : "i will be right back",

    "ic" : "i see",    "icq" : "i seek you",    "icymi" : "in case you missed it",    "idc" : "i do not care",    "idgadf" : "i do not give a damn fuck",

    "idgaf" : "i do not give a fuck",    "idk" : "i do not know",    "ie" : "that is",    "i.e" : "that is",    "ifyp" : "i feel your pain",

    "IG" : "instagram",    "iirc" : "if i remember correctly",    "ilu" : "i love you",    "ily" : "i love you",    "imho" : "in my humble opinion",

    "imo" : "in my opinion",    "imu" : "i miss you",    "iow" : "in other words",    "irl" : "in real life",    "j4f" : "just for fun",    "jic" : "just in case",    "jk" : "just kidding",    "jsyk" : "just so you know",

    "l8r" : "later",    "lb" : "pound",    "lbs" : "pounds",    "ldr" : "long distance relationship",    "lmao" : "laugh my ass off",    "lmfao" : "laugh my fucking ass off",

    "lol" : "laughing out loud",    "ltd" : "limited",    "ltns" : "long time no see",    "m8" : "mate",    "mf" : "motherfucker",    "mfs" : "motherfuckers",

    "mfw" : "my face when",    "mofo" : "motherfucker",    "mph" : "miles per hour",    "mr" : "mister",    "mrw" : "my reaction when",    "ms" : "miss",

    "mte" : "my thoughts exactly",    "nagi" : "not a good idea",    "nbc" : "national broadcasting company",    "nbd" : "not big deal",    "nfs" : "not for sale",

    "ngl" : "not going to lie",    "nhs" : "national health service",    "nrn" : "no reply necessary",    "nsfl" : "not safe for life",    "nsfw" : "not safe for work",

    "nth" : "nice to have",    "nvr" : "never",    "nyc" : "new york city",    "oc" : "original content",    "og" : "original",    "ohp" : "overhead projector",

    "oic" : "oh i see",    "omdb" : "over my dead body",   "omg" : "oh my god",    "omw" : "on my way",    "p.a" : "per annum",    "p.m" : "after midday",

    "pm" : "prime minister",    "poc" : "people of color",    "pov" : "point of view",    "pp" : "pages",    "ppl" : "people",    "prw" : "parents are watching",

    "ps" : "postscript",    "pt" : "point",    "ptb" : "please text back",    "pto" : "please turn over",    "qpsa" : "what happens",     "ratchet" : "rude",

    "rbtl" : "read between the lines",    "rlrt" : "real life retweet",     "rofl" : "rolling on the floor laughing",    "roflol" : "rolling on the floor laughing out loud",

    "rotflmao" : "rolling on the floor laughing my ass off",    "rt" : "retweet",    "ruok" : "are you ok",    "sfw" : "safe for work",     "sk8" : "skate",

    "smh" : "shake my head",    "sq" : "square",    "srsly" : "seriously",     "ssdd" : "same stuff different day",    "tbh" : "to be honest",    "tbs" : "tablespooful",

    "tbsp" : "tablespooful",    "tfw" : "that feeling when",    "thks" : "thank you",    "tho" : "though",    "thx" : "thank you",    "tia" : "thanks in advance",

    "til" : "today i learned",    "tl;dr" : "too long i did not read",    "tldr" : "too long i did not read",    "tmb" : "tweet me back",    "tntl" : "trying not to laugh",

    "ttyl" : "talk to you later",    "u" : "you",    "u2" : "you too",    "u4e" : "yours for ever",    "utc" : "coordinated universal time",

    "w/" : "with",    "w/o" : "without",    "w8" : "wait",    "wassup" : "what is up",    "wb" : "welcome back",    "wtf" : "what the fuck",    "wtg" : "way to go",

    "wtpa" : "where the party at",    "wuf" : "where are you from",    "wuzup" : "what is up",    "wywh" : "wish you were here",    "yd" : "yard",

    "ygtr" : "you got that right",    "ynk" : "you never know",    "zzz" : "sleeping bored and tired"

}
Contraction = {

    "isn't": "is not",    "doesn't": "does not",    "don't": "do not",    "can't've": "cannot have",    "aren't": "are not",

    "can't": "cannot",    "cause": "because",    "could've": "could have",    "couldn't": "could not",    "couldn't've": "could not have",

    "didn't": "did not",    "hadn't": "had not",    "hadn't've": "had not have",    "hasn't": "has not",    "haven't": "have not",

    "I'll've": "I will have",    "I'm": "I am",    "I've": "I have",    "i'd": "i would",    "he'd": "he would",    "he'd've": "he would have",

    "he'll": "he will",    "he'll've": "he he will have",    "he's": "he is",    "how'd": "how did",    "how'd'y": "how do you",

    "how'll": "how will",    "how's": "how is",    "I'd": "I would",    "I'd've": "I would have",    "I'll": "I will",    "i'd've": "i would have",

    "i'll": "i will",    "i'll've": "i will have",    "i'm": "i am",    "i've": "i have",    "isn't": "is not",    "it'd": "it would",

    "it'd've": "it would have",    "it'll": "it will",    "it'll've": "it will have",    "it's": "it is",    "let's": "let us",

    "ma'am": "madam",    "mayn't": "may not",    "might've": "might have",    "mightn't": "might not",    "mightn't've": "might not have",

    "must've": "must have",    "mustn't": "must not",    "mustn't've": "must not have",    "needn't": "need not",    "needn't've": "need not have",

    "o'clock": "of the clock",    "oughtn't": "ought not",    "oughtn't've": "ought not have",    "shan't": "shall not",    "sha'n't": "shall not",

    "shan't've": "shall not have",    "she'd": "she would",    "she'd've": "she would have",    "she'll": "she will",    "she'll've": "she will have",

    "she's": "she is",    "should've": "should have",    "shouldn't": "should not",    "shouldn't've": "should not have",    "so've": "so have",

    "so's": "so as",    "that'd": "that would",    "that'd've": "that would have",    "that's": "that is",    "there'd": "there would",

    "there'd've": "there would have",    "there's": "there is",    "they'd": "they would",    "they'd've": "they would have",    "they'll": "they will",

    "they'll've": "they will have",    "they're": "they are",    "they've": "they have",    "to've": "to have",    "wasn't": "was not",

    "we'd": "we would",    "we'd've": "we would have",    "we'll": "we will",    "we'll've": "we will have",    "we're": "we are",

    "we've": "we have",    "weren't": "were not",    "what'll": "what will",    "what'll've": "what will have",    "what're": "what are",

    "what's": "what is",    "what've": "what have",    "when's": "when is",    "when've": "when have",    "where'd": "where did",

    "where's": "where is",    "where've": "where have",    "who'll": "who will",    "who'll've": "who will have",    "who's": "who is",

    "who've": "who have",    "why's": "why is",    "why've": "why have",    "will've": "will have",    "won't": "will not",

    "won't've": "will not have",    "would've": "would have",    "wouldn't": "would not",    "wouldn't've": "would not have",

    "y'all": "you all",    "y'all'd": "you all would",    "y'all'd've": "you all would have",    "y'all're": "you all are",

    "y'all've": "you all have",    "you'd": "you would",    "you'd've": "you would have",    "you'll": "you will",    "you'll've": "you will have",

    "you're": "you are",    "you've": "you have"

}
def no_contraction(text):

    t=[]

    words=text.split()

    t = [Contraction[w.lower()] if w.lower() in Contraction.keys() else w.lower() for w in words]

    return ' '.join(t)    
def convert_abbrev_in_text(text):

    t=[]

    words=text.split()

    t = [abbreviations[w.lower()] if w.lower() in abbreviations.keys() else w for w in words]

    return ' '.join(t)    
train['text']=train['text'].apply(convert_abbrev_in_text)

train['text']  = train['text'].apply(no_contraction)
test['text']=train['text'].apply(convert_abbrev_in_text)

test['text']=test['text'].apply(no_contraction)
print(test['text'].head())

print(train['text'].head())
from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize 

  

stop_words = stopwords.words('english')

punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''





def remove_stopwords(text):

    t=[]

    words=word_tokenize(text)

    words=[w for w in words if w not in punc]

    text = [w.lower() for w in words if not w in stop_words ]

    

    return ' '.join(text) 
train['text']=train['text'].apply(remove_stopwords)

test['text']=test['text'].apply(remove_stopwords)
#from sklearn.model_selection import train_test_split

#x_train,x_validation,y_train,y_validation=train_test_split(train['clean_data'],train['target'],test_size=0.2,random_state=0)



from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer=Tokenizer()

tokenizer.fit_on_texts(train['text'])

print('number of total words:',len(tokenizer.word_index))
length=500

x_train=tokenizer.texts_to_sequences(train['text'])

x_train=pad_sequences(x_train,maxlen=length)

y_train=train['target']





x_test=tokenizer.texts_to_sequences(test['text'])

x_test=pad_sequences(x_test,maxlen=length)
print(train['text'][122])

print(x_train[122])
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,LSTM,Embedding,Dropout,Bidirectional

from tensorflow.keras.activations import sigmoid
model=Sequential()

model.add(Embedding(len(tokenizer.word_index)+1,output_dim=64, trainable=True,input_length=length))

model.add(Bidirectional(LSTM(32,return_sequences=True)))

model.add(Dropout(0.7))



model.add(Bidirectional(LSTM(32,return_sequences=True)))

model.add(Dropout(0.7))



model.add(Bidirectional(LSTM(32)))

model.add(Dropout(0.7))

model.add(Dense(64,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(64,activation='relu'))





model.add(Dense(1,activation='sigmoid'))
model.summary()
from tensorflow.keras.optimizers import Adam

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

accuracy=model.fit(x_train,y_train,epochs=10,batch_size=256,validation_split=0.3)
sample_submission=pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

sample_submission['target'] = model.predict_classes(x_test)

sample_submission.to_csv('/kaggle/working/result.csv', index=False)

print(sample_submission.head())
sample_submission['target'].value_counts()