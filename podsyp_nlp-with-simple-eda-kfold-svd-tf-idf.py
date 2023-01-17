import collections

import unidecode

import re

import string



!pip install pyspellchecker

from spellchecker import SpellChecker



import numpy as np 

import pandas as pd



import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import gc, math

from tqdm import tqdm

import pickle



import warnings

warnings.filterwarnings("ignore")
from sklearn.base import BaseEstimator

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.metrics import roc_auc_score, f1_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn.metrics import roc_auc_score, roc_curve
import nltk

from nltk.tokenize import WordPunctTokenizer

from nltk.corpus import stopwords

from wordcloud import WordCloud
pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

pd.set_option('use_inf_as_na', True)



warnings.simplefilter('ignore')

matplotlib.rcParams['figure.dpi'] = 100

sns.set()

%matplotlib inline
%%time

folder = '../input/nlp-getting-started/'

train_df = pd.read_csv(folder + 'train.csv')

test_df = pd.read_csv(folder + 'test.csv')

sub_df = pd.read_csv(folder + 'sample_submission.csv')
print('train_df')

print('train: ', train_df.shape)

print('test_df')

print('test: ', test_df.shape)

print('sub_df')

print('sub_df: ', sub_df.shape)
train_df.head()
test_df.head()
train_df.drop(['id'], axis=1, inplace=True)

test_df.drop(['id'], axis=1, inplace=True)
train_df.describe(include=['O'])
test_df.describe(include=['O'])
sub_df.head()
def missing_values_table(df, info=True):

        mis_val = df.isnull().sum()

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        if info:

            print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

                "There are " + str(mis_val_table_ren_columns.shape[0]) +

                  " columns that have missing values.")

        return mis_val_table_ren_columns
print('Missing columns in train: ')

missing_values_table(train_df)
print('Missing columns in test: ')

missing_values_table(test_df)
for df_ in (train_df, test_df):

    df_['location'] = df_['location'].fillna('?')

    df_['keyword'] = df_['keyword'].fillna('?')
class MeanEncoding(BaseEstimator):

    """   In Mean Encoding we take the number 

    of labels into account along with the target variable 

    to encode the labels into machine comprehensible values    """

    

    def __init__(self, feature, C=0.1):

        self.C = C

        self.feature = feature

        

    def fit(self, X_train, y_train):

        

        df = pd.DataFrame({'feature': X_train[self.feature], 'target': y_train}).dropna()

        

        self.global_mean = df.target.mean()

        mean = df.groupby('feature').target.mean()

        size = df.groupby('feature').target.size()

        

        self.encoding = (self.global_mean * self.C + mean * size) / (self.C + size)

    

    def transform(self, X_test):

        

        X_test[self.feature] = X_test[self.feature].map(self.encoding).fillna(self.global_mean).values

        

        return X_test

    

    def fit_transform(self, X_train, y_train):

        

        df = pd.DataFrame({'feature': X_train[self.feature], 'target': y_train}).dropna()

        

        self.global_mean = df.target.mean()

        mean = df.groupby('feature').target.mean()

        size = df.groupby('feature').target.size()

        self.encoding = (self.global_mean * self.C + mean * size) / (self.C + size)

        

        X_train[self.feature] = X_train[self.feature].map(self.encoding).fillna(self.global_mean).values

        

        return X_train
for f in ['location', 'keyword']:

    me = MeanEncoding(f, C=0.01*len(train_df[f].unique()))

    me.fit(train_df, train_df['target'])

    train_df = me.transform(train_df)

    test_df = me.transform(test_df)
train_df.tail()
test_df.tail()
def link_flg(string):

    if 'http' in string.lower():

        return 1

    else:

        return 0



for df_ in (train_df, test_df):

    df_['link_flg'] = df_['text'].apply(link_flg)



tmp = train_df.groupby('link_flg').agg('mean')['target'].reset_index()

plt.figure(figsize=(8,5))

fig = sns.barplot(x=tmp['link_flg'], y=tmp['target'], palette="husl")
def remove_accented_chars(text):

    text = unidecode.unidecode(text)

    return text



def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)



def remove_url(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



def text_preprocess(text):

    text = remove_accented_chars(text)

    text = remove_html(text)

    text = remove_url(text)

    return text
train_df['text'] = train_df['text'].apply(text_preprocess)

test_df['text'] = test_df['text'].apply(text_preprocess)
nm = 'cnt_len'

def cnt_len(string) -> int:

    return len(string)



for df_ in (train_df, test_df):

    df_[nm] = df_['text'].apply(cnt_len)



plt.figure(figsize=(14,5))

sns.distplot(train_df[nm], hist=False, rug=True, label="train");

sns.distplot(test_df[nm], hist=False, rug=True, label="test");

plt.legend();
tmp = train_df.groupby(nm).agg('mean')['target'].reset_index()

plt.figure(figsize=(48,15))

fig = sns.barplot(x=tmp[nm], y=tmp['target'], palette="husl")
nm = 'cnt_users'

def cnt_users(string) -> int:

    return sum(list(map(lambda s: 1 if s == '@' else 0, string)))



for df_ in (train_df, test_df):

    df_[nm] = df_['text'].apply(cnt_users)



plt.figure(figsize=(14,5))

sns.distplot(train_df[nm], hist=False, rug=True, label="train");

sns.distplot(test_df[nm], hist=False, rug=True, label="test");

plt.legend();
tmp = train_df.groupby(nm).agg('mean')['target'].reset_index()

plt.figure(figsize=(14,5))

fig = sns.barplot(x=tmp[nm], y=tmp['target'], palette="husl")
nm = 'cnt_hashtags'

def cnt_hashtags(string) -> int:

    return sum(list(map(lambda s: 1 if s == '#' else 0, string)))



for df_ in (train_df, test_df):

    df_[nm] = df_['text'].apply(cnt_hashtags)
tmp = train_df.groupby(nm).agg('mean')['target'].reset_index()

plt.figure(figsize=(14,5))

fig = sns.barplot(x=tmp[nm], y=tmp['target'], palette="husl")
punctuation = ['.',',',':',';','!','?','(',')','/','\\','|','\"','\'','-', '«', '»']

nm = 'cnt_punctuation'



def cnt_punctuation(string) -> int:

    return sum(list(map(lambda s: 1 if s in punctuation else 0, string)))



for df_ in (train_df, test_df):

    df_[nm] = df_['text'].apply(cnt_punctuation)



plt.figure(figsize=(14,5))

sns.distplot(train_df[nm], hist=False, rug=True, label="train");

sns.distplot(test_df[nm], hist=False, rug=True, label="test");

plt.legend();
tmp = train_df.groupby(nm).agg('mean')['target'].reset_index()

plt.figure(figsize=(14,5))

fig = sns.barplot(x=tmp[nm], y=tmp['target'], palette="husl")
nm = 'cnt_whitespace'

def cnt_whitespace(string) -> int:

    return sum(list(map(lambda s: 1 if s == ' ' else 0, string)))



for df_ in (train_df, test_df):

    df_[nm] = df_['text'].apply(cnt_whitespace)



plt.figure(figsize=(14,5))

sns.distplot(train_df[nm], hist=False, rug=True, label="train");

sns.distplot(test_df[nm], hist=False, rug=True, label="test");

plt.legend();
tmp = train_df.groupby(nm).agg('mean')['target'].reset_index()

plt.figure(figsize=(14,5))

fig = sns.barplot(x=tmp[nm], y=tmp['target'], palette="husl")
numeral = '1234567890'

nm = 'cnt_numeral'



def cnt_numeral(string) -> int:

    return sum(list(map(lambda s: 1 if s in numeral else 0, string)))



for df_ in (train_df, test_df):

    df_[nm] = df_['text'].apply(cnt_numeral)



plt.figure(figsize=(14,5))

sns.distplot(train_df[nm], hist=False, rug=True, label="train");

sns.distplot(test_df[nm], hist=False, rug=True, label="test");

plt.legend();
tmp = train_df.groupby(nm).agg('mean')['target'].reset_index()

plt.figure(figsize=(14,5))

fig = sns.barplot(x=tmp[nm], y=tmp['target'], palette="husl")
eng = 'abcdefghijklmnopqrstuvwxyz'

nm = 'cnt_others_chars'

def cnt_others_chars(string) -> int:

    return sum(list(map(lambda s: 1 if s != ' ' and s not in punctuation and s not in numeral and 

                        s != '#' and s != '@' and s not in eng and s not in eng.upper() else 0, string)))



for df_ in (train_df, test_df):

    df_[nm] = df_['text'].apply(cnt_others_chars)

tmp = train_df.groupby(nm).agg('mean').reset_index()

plt.figure(figsize=(18,5))

fig = sns.barplot(x=tmp[nm], y=tmp['target'], palette="husl")
nm = 'cnt_upper'

def cnt_upper(string) -> int:

    return sum(list(map(lambda s: 1 if s.isupper() else 0, string)))



for df_ in (train_df, test_df):

    df_[nm] = df_['text'].apply(cnt_upper)



plt.figure(figsize=(14,5))

sns.distplot(train_df[nm], hist=False, rug=True, label="train");

sns.distplot(test_df[nm], hist=False, rug=True, label="test");

plt.legend();
tmp = train_df.groupby(nm).agg('mean').reset_index()

plt.figure(figsize=(18,5))

fig = sns.barplot(x=tmp[nm], y=tmp['target'], palette="husl")
nm = 'cnt_exclamatory'



def cnt_exclamatory(string) -> int:

    return sum(list(map(lambda s: 1 if s == '!' else 0, string)))



for df_ in (train_df, test_df):

    df_[nm] = df_['text'].apply(cnt_exclamatory)

tmp = train_df.groupby(nm).agg('mean')['target'].reset_index()

plt.figure(figsize=(14,5))

fig = sns.barplot(x=tmp[nm], y=tmp['target'], palette="husl")
def words_len_stats(string) -> list:

    tmp = np.array(list(map(lambda s: len(s), string.split())))

    return pd.Series([np.median(tmp), np.min(tmp), np.max(tmp), np.std(tmp)])

    

for df_ in (train_df, test_df):

    df_[['word_len_median', 'word_len_min', 'word_len_max', 'word_len_std', ]] = df_['text'].apply(words_len_stats)



for c in ['word_len_median', 'word_len_min', 'word_len_max', 'word_len_std', ]:

    plt.figure(figsize=(14,5))

    sns.distplot(train_df[c], hist=False, rug=True, label="train");

    sns.distplot(test_df[c], hist=False, rug=True, label="test");

    plt.legend()

    plt.show();
for c in ['word_len_median', 'word_len_min', 'word_len_max']:

    tmp = train_df.groupby(c).agg('mean')['target'].reset_index()

    plt.figure(figsize=(14,5))

    fig = sns.barplot(x=tmp[c], y=tmp['target'], palette="husl")

    plt.show();
train_df.head()
train_df['cnt_punct_num_div_len'] = (train_df['cnt_punctuation'] + train_df['cnt_numeral'])**2 / train_df['cnt_len']

train_df['cnt_users_hasht_div_len'] = (train_df['cnt_users'] + train_df['cnt_hashtags'])**2 / train_df['cnt_len']

train_df['cnt_upper_div_len'] = train_df['cnt_upper']**2 / train_df['cnt_len']

train_df['cnt_whitespace_div_len'] = train_df['cnt_whitespace']**2 / train_df['cnt_len']

train_df['cnt_len_median_div_len'] = train_df['word_len_median']**2 / train_df['cnt_len']

train_df['cnt_len_min_div_len'] = train_df['word_len_min']**2 / train_df['cnt_len']

train_df['cnt_len_max_div_len'] = train_df['word_len_max']**2 / train_df['cnt_len'] 

train_df['cnt_len_std_div_len'] = train_df['word_len_std']**2 / train_df['cnt_len'] 



test_df['cnt_punct_num_div_len'] = (test_df['cnt_punctuation'] + test_df['cnt_numeral'])**2 / test_df['cnt_len']

test_df['cnt_users_hasht_div_len'] = (test_df['cnt_users'] + test_df['cnt_hashtags'])**2 / test_df['cnt_len']

test_df['cnt_upper_div_len'] = test_df['cnt_upper']**2 / test_df['cnt_len']

test_df['cnt_whitespace_div_len'] = test_df['cnt_whitespace']**2 / test_df['cnt_len']

test_df['cnt_len_median_div_len'] = test_df['word_len_median']**2 / test_df['cnt_len']

test_df['cnt_len_min_div_len'] = test_df['word_len_min']**2 / test_df['cnt_len']

test_df['cnt_len_max_div_len'] = test_df['word_len_max']**2 / test_df['cnt_len']

test_df['cnt_len_std_div_len'] = test_df['word_len_std']**2 / test_df['cnt_len'] 
abbreviations = {

    "$" : " dollar ", "€" : " euro ", "4ao" : "for adults only", "a.m" : "before midday", 

    "a3" : "anytime anywhere anyplace", "aamof" : "as a matter of fact", "acct" : "account", 

    "adih" : "another day in hell", "afaic" : "as far as i am concerned", "ave." : "avenue",

    "afaict" : "as far as i can tell", "afaik" : "as far as i know", 

    "afair" : "as far as i remember", "afk" : "away from keyboard", "app" : "application", 

    "approx" : "approximately", "apps" : "applications","atk" : "at the keyboard", 

    "asap" : "as soon as possible", "asl" : "age, sex, location", "ayor" : "at your own risk", 

    "aymm" : "are you my mother", "b&b" : "bed and breakfast", "b+b" : "bed and breakfast",

    "b.c" : "before christ", "b2b" : "business to business", "b2c" : "business to customer", 

    "b4" : "before", "b4n" : "bye for now", "b@u" : "back at you"

    , "bae" : "before anyone else", "bbc" : "british broadcasting corporation", 

    "bak" : "back at keyboard", "bbbg" : "bye bye be good", "be4" : "before", 

    "bbias" : "be back in a second", "bbl" : "be back later", "bbs" : "be back soon",

    "bfn" : "bye for now", "blvd" : "boulevard", "bout" : "about", "brb" : "be right back", 

    "bros" : "brothers", "brt" : "be right there", "bsaaw" : "big smile and a wink",

    "btw" : "by the way", "bwl" : "bursting with laughter", "c/o" : "care of", 

    "cet" : "central european time", "cf" : "compare", "cia" : "central intelligence agency", 

    "csl" : "can not stop laughing", "cu" : "see you", "cul8r" : "see you later", 

    "cv" : "curriculum vitae", "cwot" : "complete waste of time", "cya" : "see you",

    "cyt" : "see you tomorrow", "dae" : "does anyone else", 

    "dbmib" : "do not bother me i am busy",

    "diy" : "do it yourself", "dm" : "direct message", "dwh" : "during work hours", 

    "e123" : "easy as one two three", "eet" : "eastern european time", "eg" : "example", 

    "embm" : "early morning business meeting", "encl" : "enclosed", "encl." : "enclosed", 

    "etc" : "and so on", "faq" : "frequently asked questions", "fawc" : "for anyone who cares",

    "fb" : "facebook", "fc" : "fingers crossed", "fig" : "figure", 

    "fimh" : "forever in my heart", 

    "ft." : "feet", "ft" : "featuring", "ftl" : "for the loss", "ftw" : "for the win", 

    "fwiw" : "for what it is worth", "fyi" : "for your information", "g9" : "genius", 

    "gahoy" : "get a hold of yourself", "gal" : "get a life", "gfn" : "gone for now", 

    "gg" : "good game", "gl" : "good luck", "glhf" : "good luck have fun", 

    "gmta" : "great minds think alike", "gn" : "good night", 

    "g.o.a.t" : "greatest of all time", 

    "goat" : "greatest of all time", "goi" : "get over it", "gmt" : "greenwich mean time", 

    "gcse" : "general certificate of secondary education", "gps" : "global positioning system", 

    "gr8" : "great", "gratz" : "congratulations", "gyal" : "girl", "h&c" : "hot and cold",

    "hp" : "horsepower", "hr" : "hour", "hrh" : "his royal highness", "ht" : "height", 

    "ibrb" : "i will be right back", "ic" : "i see", "icq" : "i seek you", 

    "icymi" : "in case you missed it", "idc" : "i do not care", 

    "idgadf" : "i do not give a damn fuck", "i.e" : "that is", 

    "idgaf" : "i do not give a fuck", "idk" : "i do not know", "ie" : "that is",

    "ifyp" : "i feel your pain", "IG" : "instagram", "iirc" : "if i remember correctly",

    "ilu" : "i love you", "ily" : "i love you", "imho" : "in my humble opinion", 

    "imo" : "in my opinion", "imu" : "i miss you", "iow" : "in other words", 

    "irl" : "in real life", 

    "j4f" : "just for fun", "jic" : "just in case", "jk" : "just kidding",

    "jsyk" : "just so you know", "l8r" : "later", "lb" : "pound", "lbs" : "pounds", 

    "ldr" : "long distance relationship", "lmao" : "laugh my ass off", 

    "lmfao" : "laugh my fucking ass off", "lol" : "laughing out loud", "ltd" : "limited",

    "ltns" : "long time no see", "m8" : "mate", "mf" : "motherfucker", "mfs" : "motherfuckers", 

    "mfw" : "my face when", "mofo" : "motherfucker", "mph" : "miles per hour", "mr" : "mister", 

    "mrw" : "my reaction when", "ms" : "miss", "mte" : "my thoughts exactly", 

    "nagi" : "not a good idea", 

    "nbc" : "national broadcasting company", "nbd" : "not big deal", "nfs" : "not for sale", 

    "ngl" : "not going to lie", "nhs" : "national health service", 

    "nrn" : "no reply necessary", 

    "nsfl" : "not safe for life", "nsfw" : "not safe for work", "nth" : "nice to have",

    "nvr" : "never", "nyc" : "new york city", "oc" : "original content", "og" : "original", 

    "ohp" : "overhead projector", "oic" : "oh i see", 

    "omdb" : "over my dead body", "omg" : "oh my god", 

    "omw" : "on my way", "p.a" : "per annum", "p.m" : "after midday",  "pm" : "prime minister", 

    "poc" : "people of color", "pov" : "point of view", "pp" : "pages", "ppl" : "people", 

    "prw" : "parents are watching", "ps" : "postscript", 

    "pt" : "point", "ptb" : "please text back",

    "pto" : "please turn over", "qpsa" : "what happens", "ratchet" : "rude", 

    "rbtl" : "read between the lines", "rlrt" : "real life retweet", 

    "rofl" : "rolling on the floor laughing", "ruok" : "are you ok",

    "roflol" : "rolling on the floor laughing out loud", "rt" : "retweet",  

    "rotflmao" : "rolling on the floor laughing my ass off", "sfw" : "safe for work",

    "sk8" : "skate", "smh" : "shake my head", "sq" : "square", "srsly" : "seriously", 

    "ssdd" : "same stuff different day", "tbh" : "to be honest", "tbs" : "tablespooful", 

    "tbsp" : "tablespooful", "tfw" : "that feeling when", "thks" : "thank you",

    "tho" : "though", "thx" : "thank you", "tia" : "thanks in advance", 

    "til" : "today i learned", "tmb" : "tweet me back",

    "tl;dr" : "too long i did not read", "tldr" : "too long i did not read",  

    "tntl" : "trying not to laugh", "ttyl" : "talk to you later", 

    "u" : "you", "u2" : "you too", 

    "u4e" : "yours for ever", "utc" : "coordinated universal time", 

    "w/" : "with", "w/o" : "without", 

    "w8" : "wait", "wassup" : "what is up", "wb" : "welcome back", "wtf" : "what the fuck", 

    "wtg" : "way to go", "wtpa" : "where the party at", "wuf" : "where are you from", 

    "wuzup" : "what is up", "wywh" : "wish you were here", 

    "yd" : "yard", "ygtr" : "you got that right", 

    "ynk" : "you never know", "zzz" : "sleeping bored and tired" }
def convert_abbrev(word):

    return abbreviations[word.lower()] if word.lower() in abbreviations.keys() else word



def convert_abbrev_in_text(text):

    tokenizer = WordPunctTokenizer()

    tokens = tokenizer.tokenize(text)

    tokens = [convert_abbrev(word) for word in tokens]

    text = ' '.join(tokens)

    return text



train_df["text"] = train_df["text"].apply(lambda x: convert_abbrev_in_text(x))

test_df["text"] = test_df["text"].apply(lambda x: convert_abbrev_in_text(x))
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



def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)



spell = SpellChecker()

def correct_spellings(text):

    corrected_text = []

    misspelled_words = spell.unknown(text.split())

    for word in text.split():

        if word in misspelled_words:

            corrected_text.append(spell.correction(word))

        else:

            corrected_text.append(word)

    return " ".join(corrected_text)



# Now compact all the normalization function calls into a single function

def normalization(text):

    text = remove_emoji(text)

    text = remove_punct(text)

    text = correct_spellings(text)

    return text
%%time

train_df['text'] = train_df['text'].apply(normalization)

test_df['text'] = test_df['text'].apply(normalization)
tokenizer = WordPunctTokenizer()

preprocess = lambda text: ' '.join(tokenizer.tokenize(text.lower()))



text = 'How to be a grown-up at work: replace "tweet you" with "Ok, great!!!".'

print("before:", text,)

print("after:", preprocess(text),)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

stop_words_filter = lambda text: ' '.join([t for t in text.split() if t not in stop_words])

print("before:", text,)

print("after:", stop_words_filter(preprocess(text)),)
%%time

texts_train = list(map(lambda text: stop_words_filter(preprocess(text)), train_df['text']))

texts_test = list(map(lambda text: stop_words_filter(preprocess(text)), test_df['text']))

assert len(texts_train) == len(train_df)

assert len(texts_test) == len(test_df)
import collections

k = 3750

c = collections.Counter()



for sentence in texts_train:

    for word in tokenizer.tokenize(sentence):

        c[word] += 1



bow_vocabulary = list([i[0] for i in c.most_common(k)])

print('example features:', sorted(bow_vocabulary)[::500])
def text_to_bow(text) -> np.array:

    """ convert text string to an array of token counts. Use bow_vocabulary. """

    bow = bow_vocabulary

    tmp = []

    for ch_bow in bow:

        val = 0

        for ttk in tokenizer.tokenize(text):

            if ch_bow == ttk:

                val += 1

        tmp.append(val)

    

    return np.array(tmp, 'float32')
%%time

X_train_bow = np.stack(list(map(text_to_bow, texts_train)))

X_test_bow = np.stack(list(map(text_to_bow, texts_test)))
X_train_bow[0:5]
#convert country names to dictionary with values and its occurences

wordcloud = WordCloud(width = 1000, height = 500).generate_from_frequencies(c)



plt.figure(figsize=(15,8))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
def tf_idf_calc(txt, bow: list) -> dict:

    d = {}

    n_txt = len(txt)

    for word in bow:

        cnt = 0

        for sent in txt:

            if word in set(tokenizer.tokenize(sent)):

                cnt += 1

        d[word] = np.log(n_txt / (cnt + 1))

    return d
%%time

tf_idf_dict_train = tf_idf_calc(texts_train, bow_vocabulary)
def tf_idf(df, d) -> list:

    df_ = []

    for bag in df:

        tmp = []

        for n_word in range(len(bag)):

            tmp.append(bag[n_word] * d[bow_vocabulary[n_word]])

        df_.append(tmp)

    return df_
%%time

X_train_tf_idf = pd.DataFrame(tf_idf(X_train_bow, tf_idf_dict_train))

X_test_tf_idf = pd.DataFrame(tf_idf(X_test_bow, tf_idf_dict_train))
y = train_df['target']

train_df.drop(['target'], axis=1, inplace=True)
y.value_counts()
y.value_counts(normalize=True)
y.hist(bins=y.nunique());
X_train_tf_idf.shape
%%time

cols = X_train_tf_idf.shape[1]

# SVD step

step = 10

# different between train and test penalty power:

p = 0.8



skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=777)

w=[]



for n_svd in range(X_train_tf_idf.shape[1]-(step*300), 0, -step):

    tmp_w = []

    for train_index, val_index in skf.split(X_train_tf_idf, y):

        x_train, x_valid = X_train_tf_idf.iloc[train_index, :], X_train_tf_idf.iloc[val_index, :]

        y_train, y_valid = y[train_index], y[val_index]

        

        svd = TruncatedSVD(n_components=n_svd).fit(x_train)

        x_train_svd = svd.transform(x_train)

        x_valid_svd = svd.transform(x_valid)

        

        tf_idf_model = LogisticRegression().fit(x_train_svd, y_train)

        svd_train = f1_score(y_train, tf_idf_model.predict(x_train_svd))

        svd_valid = f1_score(y_valid, tf_idf_model.predict(x_valid_svd))

        tmp_w.append([svd_train, svd_valid])

        

    mn = np.mean(tmp_w, axis=0)

    train = mn[0]

    test = mn[1]

    w.append([n_svd, train, test, np.power(np.square(train-test), p), +test-np.power(np.square(train-test), p)])
svd_valid = pd.DataFrame(w, columns=['n_svd', 'train', 'test', 'diff', 'result'])
svd_valid
svd_valid[svd_valid.index == svd_valid['result'].argmax()]
n_svd = svd_valid[svd_valid.index == svd_valid['result'].argmax()].iloc[0, 0]
%%time

svd = TruncatedSVD(n_components=n_svd).fit(X_train_tf_idf)

X_train_svd = svd.transform(X_train_tf_idf)

X_test_svd = svd.transform(X_test_tf_idf)
train_df.drop('text', inplace=True, axis=1)

test_df.drop('text', inplace=True, axis=1)
train_corr = train_df.corr()

# plot the heatmap and annotation on it

fig, ax = plt.subplots(figsize=(16,16))

sns.heatmap(train_corr, xticklabels=train_corr.columns, yticklabels=train_corr.columns, annot=True, ax=ax);
test_corr = test_df.corr()

# plot the heatmap and annotation on it

fig, ax = plt.subplots(figsize=(16,16))

sns.heatmap(test_corr, xticklabels=test_corr.columns, yticklabels=test_corr.columns, annot=True, ax=ax);
drop_cols = ['cnt_len', 'cnt_hashtags', 'cnt_whitespace', 'cnt_upper', 'word_len_min', 'word_len_max', 'word_len_std',]
train_df[['svd_'+str(c) for c in range(X_train_svd.shape[1])]] = pd.DataFrame(X_train_svd)

test_df[['svd_'+str(c) for c in range(X_test_svd.shape[1])]] = pd.DataFrame(X_test_svd)
train_df.head()
test_df.head()
train_df.shape, test_df.shape
%%time

train_scores=[]

test_scores=[]



skf_1 = StratifiedKFold(n_splits=3, shuffle=True, random_state=777)



lr_grid = {"C": [10, 1, 0.5, 0.1, 0.075, 0.06, 0.05, 0.04, 0.35, 0.25, 0.01, 0.005, 0.001], "penalty":['l1', 'l2']}





for no, (train_index_1, val_index_1) in enumerate(skf_1.split(train_df, y)):

    train_df_ = train_df.drop(drop_cols, axis=1).copy()

    test_df_ = test_df.drop(drop_cols, axis=1).copy()

    x_train_1, x_valid_1 = train_df_.iloc[train_index_1, :], train_df_.iloc[val_index_1, :]

    y_train_1, y_valid_1 = y[train_index_1], y[val_index_1]

    

    logreg=LogisticRegression()

    logreg_cv=GridSearchCV(logreg, lr_grid, cv=3, verbose=False, scoring='f1', n_jobs=-1)

    logreg_cv.fit(x_train_1, y_train_1)

    logreg_model = LogisticRegression(**logreg_cv.best_params_).fit(x_train_1, y_train_1)

    train_pred = logreg_model.predict_proba(train_df_)[:, 1]

    test_pred = logreg_model.predict_proba(test_df_)[:, 1]

    train_scores.append(train_pred)

    test_scores.append(test_pred)

    print('Fold Log: ', no, 'CV F1: ', logreg_cv.best_score_, 'Valid F1: ', f1_score(y_valid_1, logreg_model.predict(x_valid_1)), 

          'Best params: ', logreg_cv.best_params_)
sub_df['target'] = list(map(lambda x: 1 if x>=0.5 else 0, np.mean(test_scores, axis=0)))
sub_df['target'].mean()
sub_df.to_csv('submission.csv', index=False)