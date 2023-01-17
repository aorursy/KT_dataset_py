import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import re

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

train.drop(['location','keyword'],axis = 1,inplace = True)
train.target.hist();

print("Negative target (not a disaster) target, % : {}\nPoistive target (it is a disaster), %      : {}".format(train.target.value_counts()[0]/len(train)*100,

                                                                                                                 train.target.value_counts()[1]/len(train)*100))
for i in range(50):

    print(train.text[i])
def to_lowercase(text):

    return text.lower()

    # Alternativly:

    # train.text = train.text.str.lower()

    # train.text = train.text.apply(lambda x: x.lower())

    

text = 'THIS Is a RaNdOm SenTence with CAPITAL LetTers'

print(text,'===>',to_lowercase(text))
# remove urls tags

def remove_url(text):

    return re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", text)



def assign_user(text,remove = True):

    if remove:

        text = re.sub(r"\@[A-Za-z0-9]+", "", text)

    else:

        text = re.sub(r"\@[A-Za-z0-9]+", "USER", text)

    return text



text1 = 'http://t.co/lHYXEOHY6C'

text2 = '@alex reported a disaster'



print(text1,'===>',remove_url(text1))

print(text2,'===>',assign_user(text2))

print(text2,'===>',assign_user(text2,remove = False))
def remove_accented_chars(text):

    # https://www.kdnuggets.com/2019/04/text-preprocessing-nlp-machine-learning.html

    import unicodedata

    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    return text



text = 'Sómě Áccěntěd těxt'



print(text,'===>',remove_accented_chars(text))
# from the notebook from the begining



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



def expand_abrivviation(text,mapping = abbreviations):

    

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    return text



text = 'ynk wtf is going on'



print(text,'===>',expand_abrivviation(text))
# https://www.kaggle.com/theoviel/improve-your-score-with-some-text-preprocessing



contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", 

                       "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", 

                       "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", 

                       "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", 

                       "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",

                       "mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", 

                       "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", 

                       "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have",

                       "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", 

                       "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", 

                       "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", 

                       "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", 

                       "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", 

                       "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", 

                       "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", 

                       "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", 

                       "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }



def expand_contractions(text,mapping = contraction_mapping):

    specials =["’", "‘", "´", "`"]

    for s in specials:

        text = text.replace(s,"'")

    

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    return text



text = "y'all'd've to learn machine learning"

print(text,'===>',expand_contractions(text))
def remove_special_characters(text):

    pat = r'[^a-zA-z0-9.,!?/:;\"\'\s]' 

    return re.sub(pat, '', text)



text = ' # & * are special characters'

print(text,'===>',remove_special_characters(text))
def remove_punctuation(text):

    import string

    text = ''.join([c for c in text if c not in string.punctuation])

    return text



text = ' Sentence, with: different. king of ; punctuations...'

print(text,'===>',remove_punctuation(text))
def remove_numbers(text):

    import re

    pattern = r'[^a-zA-z.,!?/:;\"\'\s]+' 

    return re.sub(pattern, '', text)



text = ' 1984 We are only in the beggining of 2020'

print(text,'===>',remove_numbers(text))
def remove_extra_whitespace_tabs(text):

    import re

    pattern = r'^\s*|\s\s*'

    return re.sub(pattern, ' ', text).strip()



text = ' # & * are special characters'

text1 = remove_punctuation(text)

print(text1, 'containes extra spaces')

print(text,'===>',text1,'===>',remove_extra_whitespace_tabs(text1))
def remove_emojify(text):

    #https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove#Data-Cleaning

    import re

    regrex_pattern = re.compile(pattern = "["

        u"\U0001F600-\U0001F64F"  # emoticons

        u"\U0001F300-\U0001F5FF"  # symbols & pictographs

        u"\U0001F680-\U0001F6FF"  # transport & map symbols

        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           "]+", flags = re.UNICODE)

    return regrex_pattern.sub(r'',text)
from nltk import word_tokenize          

from nltk.stem import WordNetLemmatizer 

from nltk.stem import PorterStemmer

from nltk.corpus import stopwords 



def remove_stop(text):

    return " ".join ([word for word in word_tokenize(text) if not word in stopwords.words('english')])





text = 'Here I am trying to show you a sample sentence with stopwords filtration'

print(text,'===>',remove_stop(text),'----- am to you with are removed')





def stem(text):

    porter = PorterStemmer()

    return " ".join([porter.stem(word) for word in word_tokenize(text)])



print('cat cats dog dogs categories','===>',stem('cat cats dog dogs categories'))





def lemma(text):

    # Difference between stem and lemma is illustrated by 'categories'

    lemma=WordNetLemmatizer()

    return " ".join([lemma.lemmatize(word) for word in word_tokenize(text)])



print('cat cats dog dogs categories','===>',lemma('cat cats dog dogs categories'))



def no_preprocessing(text):

    # just a reference function

    return text
from sklearn.model_selection               import train_test_split,cross_val_score



from sklearn.feature_extraction.text       import CountVectorizer,TfidfVectorizer

from sklearn.naive_bayes                   import MultinomialNB,GaussianNB



from sklearn.linear_model                  import LogisticRegression



from sklearn.metrics                       import f1_score

def ml_modeling(train):

    

    X = train['text']

    y = train['target']



    X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = 0.2,stratify = y, shuffle = True,random_state = 42)



        

    vect    = CountVectorizer(min_df = 5, max_df = 0.9, ngram_range = (1,2))

    X_train = vect.fit_transform(X_train)

    X_val   = vect.transform(X_val)



    clf = MultinomialNB()

    

    score = cross_val_score(clf,X_train,y_train,cv = 5, scoring = 'f1')



    print('Mean f1 score         : {:.5} STD: {:.5}'.format(score.mean(),score.std()))



    print('f1 score on train data: {:.5}'.format(f1_score(y_train,clf.fit(X_train,y_train).predict(X_train))))

    print('f1 score on valid data: {:.5}'.format(f1_score(y_val,clf.fit(X_train,y_train).predict(X_val))))



    return score.mean()

funcs = [no_preprocessing,

         to_lowercase,

         remove_url,

         assign_user,

         remove_stop,

         remove_accented_chars,

         expand_abrivviation,

         expand_contractions,

         remove_special_characters,

         remove_punctuation,

         remove_numbers,

         remove_extra_whitespace_tabs,

         remove_emojify,

         stem,

         lemma]



names = ['no_preprocessing',

         'to_lowercase',

         'remove_url',

         'assign_user',

         'remove_stop',

         'remove_accented_chars',

         'expand_abrivviation',

         'expand_contractions',

         'remove_special_characters',

         'remove_punctuation',

         'remove_numbers',

         'remove_extra_whitespace_tabs',

         'remove_emojify',

         'stem',

         'lemma']





f1 = []

for fun,name in zip(funcs,names):

    df = train.copy()                              # here we use only single function, therefore in every itteration we have fresh dataser

    print('\nFunction ',name.upper(),'\n')

    print()

    df['text'] = df['text'].apply(lambda x: fun(x))

    score = ml_modeling(df)

    

    f1.append(score)
pos = np.arange(len(f1)) 

fig, ax = plt.subplots(figsize = (20,10))

plot = ax.bar(pos,f1, 0.4)

ax.set_ylabel('f1_score')

ax.set_title('Effect of text preprocessing')

ax.set_xticks(pos)

ax.set_xticklabels(names, rotation = 90,fontsize = 20);

df = train.copy()

f1 = []

for fun,name in zip(funcs,names):

    print('Function ',name.upper())

    df['text'] = df['text'].apply(lambda x: fun(x))

    score= ml_modeling(df)

    

    f1.append(score)
pos = np.arange(len(f1)) 

fig, ax = plt.subplots(figsize = (20,10))

plot = ax.plot(f1)

ax.set_ylabel('f1_score')

ax.set_title('Effect of cumulative text preprocessing')

ax.set_xticks(pos)

ax.set_xticklabels(names, rotation = 90,fontsize = 20);
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow as tf





def nn_modeling(train):

    

    # This function was also used for the similar study

    # Results were the same, there is no significant improvment with text preprocessing

    

    X = train['text']

    y = train['target']



    X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = 0.2,stratify = y, shuffle = True,random_state = 42)

    

    text_token = Tokenizer()

    text_token.fit_on_texts(X_train)



    X_train_seq = text_token.texts_to_sequences(X_train)

    X_train_pad = pad_sequences(X_train_seq, maxlen = 100)



    X_val_seq = text_token.texts_to_sequences(X_val)

    X_val_pad = pad_sequences(X_val_seq, maxlen = 100)



    text_vocab_size = len(text_token.word_index)+1







    model_seq = tf.keras.Sequential([

        tf.keras.layers.Embedding(text_vocab_size,12,input_length = X_train.shape[0]),

        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8,dropout=0.25, recurrent_dropout=0.25)),

        tf.keras.layers.Dense(1,activation = 'sigmoid')



    ])



    model_seq.compile(loss = 'binary_crossentropy',

                      optimizer = 'adam',

                      metrics = ['accuracy'])







    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)



    history = model_seq.fit(X_train_pad,y_train,

                            epochs = 2,

                            batch_size=32,                       

                            validation_data=(X_val_pad,y_val),

                            shuffle = True,

                            verbose = 1)

    

    

    pred = model_seq.predict(X_val_pad,verbose = 1)

    pred_temp = np.where(pred.reshape(-1,) > 0.5,1,0)

    scores = []

    f1 = f1_score(y_val,pred_temp)



    print('\nF1 score is: {:.5}'.format(f1))



    return f1,model_seq,text_token
funcs = [no_preprocessing,

         to_lowercase,

         remove_url,

         assign_user,

         remove_stop,

         remove_accented_chars,

         expand_abrivviation,

         expand_contractions,

         remove_special_characters,

         remove_punctuation,

         remove_numbers,

         remove_extra_whitespace_tabs,

         remove_emojify,

         stem,]

         #lemma]



names = ['no_preprocessing',

         'to_lowercase',

         'remove_url',

         'assign_user',

         'remove_stop',

         'remove_accented_chars',

         'expand_abrivviation',

         'expand_contractions',

         'remove_special_characters',

         'remove_punctuation',

         'remove_numbers',

         'remove_extra_whitespace_tabs',

         'remove_emojify',

         'stem',]

         #'lemma']







for fun,name in zip(funcs,names):



    print('(TRAIN) Function ',name.upper(),'\n')

    train['text'] = train['text'].apply(lambda x: fun(x))

        

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

test.drop(['location','keyword'],axis = 1,inplace = True)         



for fun,name in zip(funcs,names):



    print('(TEST) Function ',name.upper(),'\n')

    test['text'] = test['text'].apply(lambda x: fun(x))
_,model,tokenizer = nn_modeling(train)



X_test = tokenizer.texts_to_sequences(test['text'])

X_test = pad_sequences(X_test,maxlen = 100)



pred = model.predict(X_test)

pred = pred.reshape(-1,)





sub = pd.DataFrame({'Id':test.id, 'target':np.where(pred > 0.5,1,0)})

sub.to_csv('submission.csv',index = False)