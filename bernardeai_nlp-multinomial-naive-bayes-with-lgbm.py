import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.model_selection import KFold

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score, confusion_matrix, precision_recall_fscore_support, roc_curve, roc_auc_score

import re

from nltk.tokenize import word_tokenize

from textblob import TextBlob

from lightgbm import LGBMClassifier

import lightgbm as lgb
train_df = pd.read_csv("train.csv")

test_df = pd.read_csv("test.csv")

sample_submission = pd.read_csv("sample_submission.csv")
train_df.head(10)
train_df.info()
train_df["text length"] = train_df["text"].apply(len)
test_df["text length"] = test_df["text"].apply(len)
train_df.hist(column="text length", by="target", bins=50,figsize=(12,4))
import nltk

nltk.download_shell()

import string

from nltk.corpus import stopwords
def text_process(mess): 

    nopunc = [char for char in mess if char not in string.punctuation]

    nopunc = ''.join(nopunc)

    return [word for word in nopunc.split() if word.lower() not in stopwords.words("english")]
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB



nltk.download('stopwords')
from sklearn.pipeline import Pipeline



pipeline = Pipeline([

    ('bow', CountVectorizer(analyzer=text_process)),  

    ('tfidf', TfidfTransformer()),  

    ('classifier', MultinomialNB()),  

])
pipeline.fit(train_df["text"],train_df["target"])
cv = KFold(n_splits=4)

scores = model_selection.cross_val_score(pipeline, train_df["text"], train_df["target"], cv = cv)
scores
scores.mean()
pipeline2 = Pipeline([

    ('bow', CountVectorizer(analyzer=text_process)),  

    ('tfidf', TfidfTransformer()),  

    ('classifier', linear_model.RidgeClassifier()),  

])
pipeline2.fit(train_df["text"],train_df["target"])
scoresRR = model_selection.cross_val_score(pipeline2, train_df["text"], train_df["target"], cv = cv)
scoresRR
scoresRR.mean()
# add a feature of the number of capital letters



train_df[train_df["target"] == 0]["text"].values[0:30]

df=pd.concat([train_df,test_df])

df.shape
df
def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



df['text']=df['text'].apply(lambda x : remove_URL(x))
# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b

def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)



# remove_emoji("Omg another Earthquake 😔😔")
df['text']=df['text'].apply(lambda x: remove_emoji(x))
def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)



example="I am a #king"

print(remove_punct(example))
df['text']=df['text'].apply(lambda x : remove_punct(x))
def remove_punctuations(text):

    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`"

    

    for p in punctuations:

        text = text.replace(p, f' {p} ')



    text = text.replace('...', ' ... ')

    

    if '...' not in text:

        text = text.replace('..', ' ... ')

    

    return text
df['text']=df['text'].apply(lambda x : remove_punctuations(x))
def convert_abbrev(word):

    return abbreviations[word.lower()] if word.lower() in abbreviations.keys() else word
def convert_abbrev_in_text(text):

    tokens = word_tokenize(text)

    tokens = [convert_abbrev(word) for word in tokens]

    text = ' '.join(tokens)

    return text
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
nltk.download('punkt')
df['text']=df['text'].apply(lambda x : convert_abbrev_in_text(x))
df
def clean_tweets(tweet):

    """Removes links and non-ASCII characters"""

    

    tweet = ''.join([x for x in tweet if x in string.printable])

    

    # Removing URLs

    tweet = re.sub(r"http\S+", "", tweet)

    

    return tweet
train_df["text"] = train_df["text"].apply(lambda x: clean_tweets(x))

test_df["text"] = test_df["text"].apply(lambda x: clean_tweets(x))



train_df["text"] = train_df["text"].apply(lambda x: remove_emoji(x))

test_df["text"] = test_df["text"].apply(lambda x: remove_emoji(x))



train_df["text"] = train_df["text"].apply(lambda x: remove_punctuations(x))

test_df["text"] = test_df["text"].apply(lambda x: remove_punctuations(x))



train_df["text"] = train_df["text"].apply(lambda x: convert_abbrev_in_text(x))

test_df["text"] = test_df["text"].apply(lambda x: convert_abbrev_in_text(x))
train_df
pipeline = Pipeline([

    ('bow', CountVectorizer(analyzer=text_process)),  

    ('tfidf', TfidfTransformer()),  

    ('classifier', MultinomialNB()),  

])
pipeline.fit(train_df["text"],train_df["target"])
cv = KFold(n_splits=4)

scores = model_selection.cross_val_score(pipeline, train_df["text"], train_df["target"], cv = cv)

scores
sample_submission["target"] = pipeline.predict(test_df["text"])
test_df["MultinomialNB_predict"] = sample_submission["target"]
train_df["MultinomialNB_predict"] = pipeline.predict(train_df["text"])
train_df
train_df['location_nan_mode'] = train_df['location'].fillna(train_df['location'].mode()[0])
test_df['location_nan_mode'] = test_df['location'].fillna(test_df['location'].mode()[0])
train_df
def extract_nouns(tweet):

    """extract nouns for use as keywords, returns first noun"""

    blob = TextBlob(tweet)

   

    if (len(blob.noun_phrases)) > 0:

        noun = blob.noun_phrases

    else:

        noun = ""



    return noun



print(extract_nouns('Forest fire near La Ronge Sask . Canada'))

nltk.download('brown')
# df['text']=df['text'].apply(lambda x : convert_abbrev_in_text(x))



# df['location'].apply(lambda x: extract_nouns(x) if pd.isnull(x['location']) else x['location'])





# df2 = df.apply(lambda x: x['Col2'] if pd.isnull(x['Col1']) else x['Col1'], axis=1)





# train_df['keyword_ext'] = df.apply(

#     lambda row: extract_nouns(row['text'])

# )



train_df["keyword_ext"] = train_df["text"].apply(lambda x: extract_nouns(x))
test_df["keyword_ext"] = test_df["text"].apply(lambda x: extract_nouns(x))
train_df
train_df['keyword'].fillna(train_df['keyword_ext'], inplace=True)
test_df['keyword'].fillna(test_df['keyword_ext'], inplace=True)
train_df
columns = ['keyword','text length','MultinomialNB_predict','location_nan_mode']
pieces = {'train': train_df, 'test': test_df}

combined_df = pd.concat(pieces)
combined_df
combined_df_dropped = combined_df.drop(['id','keyword_ext','location','location_nan_mode','text'],axis=1)
combined_df_dropped
combined_df_ohe = pd.get_dummies(combined_df_dropped,drop_first=True)
df_train = combined_df_ohe.loc['train']

df_test = combined_df_ohe.loc['test']
y_tr = df_train.pop('target')

x_tr = df_train





# x_te = df_test.drop('target')
x_te = df_test.drop('target',axis=1)
lgbmcl = lgb.LGBMClassifier(boosting_type='gbdt',

                            n_estimators=1000, 

                            objective = 'binary', 

                            learning_rate = 0.01, 

                            reg_alpha = 0.1, 

                            reg_lambda = 0.1, 

                            subsample = 0.8, 

                            n_jobs = -1, 

                            random_state = 50)
cv = KFold(n_splits=4)

scoresLGBM = model_selection.cross_val_score(lgbmcl, x_tr, y_tr, cv = cv, scoring='f1')
scoresLGBM
scoresLGBM.mean()
lgbmcl.fit(x_tr, y_tr, eval_metric='f1')
sample_submission["target"] = lgbmcl.predict(x_te)
sample_submission.head()
sample_submission.to_csv("submission_test3_IO_BE2.csv", index=False)