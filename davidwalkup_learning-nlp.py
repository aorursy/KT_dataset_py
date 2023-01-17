# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



# plotting

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



# text preprocessing

import re, string, nltk

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

from bs4 import BeautifulSoup 



# Suppress warnings 

import warnings

warnings.filterwarnings('ignore')
# Helper functions



def text_cleaner(text):

    '''Cleans tweet text for modeling.

    

    Changes text to lower case.

    Removes hyperlinks, @users, html tags, punctuation, words with numbers

    inside them, and numbers.

    

    Args:

        text (string): the text to be cleaned

        

    Returns:

        text (string): the cleaned text

    '''

    # change the text to lower case

    text = text.lower()

    # remove hyperlinks from the text

    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # remove @user tagging

    text = re.sub(r'@\S+', '', text)

    # remove html tags

    soup = BeautifulSoup(text, 'lxml')

    text = soup.get_text()

    # replace URL-encoded spaces

    text = re.sub('%20', ' ', text)

    # remove punctuation

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    # remove words with numbers inside them

    text = re.sub('\w*\d\w*', '', text)

    # remove remaining numbers

    text = re.sub('\d', '', text)

    # tokenize the text

    tokenizer = nltk.tokenize.WhitespaceTokenizer()

    text = tokenizer.tokenize(text)

    # rejoin the text using a single space for the seperator

    text = ' '.join(text)

    return text



def text_normalize(text, stem_it = False, lemmatize_it = False):

    '''Normalizes tweet text for modeling.

    

    Tokenizes text, stems or lemmatizes the text if desired, then rejoins

    the text back into a single string.

    

    Args:

        text (string): the text to be normalized

        stem_it (boolean): whether to stem the input text.

            Default = False

        lemmatize_it (boolean): whether to lemmatize the input text

            Default = False

        

    Returns:

        text (string): the normalized text

    '''

    tokenizer = nltk.tokenize.WhitespaceTokenizer()

    stemmer = nltk.PorterStemmer()

    lemmatizer = nltk.WordNetLemmatizer()

    # tokenize the text

    text = tokenizer.tokenize(text)

    # stem or lemmatize the text, if required.

    if stem_it:

        text = [stemmer.stem(word) for word in text]

    elif lemmatize_it:

        text = [lemmatizer.lemmatize(word) for word in text]

    else:

        # if the text wasn't stemmed/lemmatized, rejoin and return it

        return ' '.join(text)

    # rejoin the text and return it

    text = ' '.join(text)

    return text



def model_scoring(score_array):

    '''Prints and returns mean and standard deviation of an array of scores

    

    Args:

        score_array (numpy array): an array of scores to be summarized

    

    Returns:

        mean_score (float): the mean of the array of scores

        stability_score (float): the standard deviation of the array of scores

    '''

    # calculate the mean

    mean_score = np.mean(score_array)

    # calculate the std dev

    stability_score = np.std(score_array)

    # print and return the mean and std dev

    print('Mean score: ', mean_score, '+/-', stability_score * 2)

    return mean_score, stability_score
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
sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")
train.head(5)
train.info()
test.head()
test.info()
train.loc[train.duplicated(subset = 'text') == True]
test.loc[test.duplicated(subset = 'text') == True]
sns.barplot(train['target'].value_counts().index, train['target'].value_counts());
train.sample(5)
test.sample(5)
train.drop_duplicates(subset = 'text', inplace = True)
train.info()
cleaned_train = train.copy()
cleaned_train['text'] = cleaned_train['text'].apply(lambda x: convert_abbrev(x))
cleaned_train['text'] = cleaned_train['text'].apply(lambda x: text_cleaner(x))
len(cleaned_train['location'].unique())
len(cleaned_train['keyword'].unique())
print(cleaned_train['keyword'].unique())
cleaned_train['keyword'].isna().sum()
cleaned_train['keyword'].fillna('nokeyword', inplace = True)
cleaned_train['keyword'] = cleaned_train['keyword'].apply(lambda x: text_cleaner(x))
for row in cleaned_train.index:

    keyword = cleaned_train['keyword'][row]

    text = cleaned_train['text'][row]

    if keyword in text:

        pass

    elif keyword != 'nokeyword':

        print(row, 'MISSING KEYWORD:', keyword, 'IN TEXT:', text)
cleaned_train.drop(columns = ['keyword', 'location'], inplace = True)
cleaned_train.info()
cleaned_train.loc[(cleaned_train['text'].isna()==True) | (cleaned_train['text'] == '')]
cleaned_train.drop(index = 5115, inplace = True)
# lemmatized copy

lemmatized_train = cleaned_train.copy()

lemmatized_train['text'] = lemmatized_train['text'].apply(lambda x: text_normalize(x, lemmatize_it = True))

lemmatized_train.sample(5)
# copy test data

cleaned_test = test.copy()

# drop "keyword" and "location" columns

cleaned_test.drop(columns = ['keyword', 'location'], inplace = True)

# convert abbreviations

cleaned_test['text'] = cleaned_test['text'].apply(lambda x: convert_abbrev(x))

# clean the text

cleaned_test['text'] = cleaned_test['text'].apply(lambda x: text_cleaner(x))

# lemmatize the text

cleaned_test['text'] = cleaned_test['text'].apply(lambda x: text_normalize(x, lemmatize_it = True))

cleaned_test.sample(5)
# check for blank text

cleaned_test.loc[(cleaned_test['text'].isna()==True) | (cleaned_test['text'] == '')]
cleaned_test.info()
# modeling imports



from sklearn.naive_bayes import MultinomialNB

from sklearn import model_selection

from sklearn.model_selection import cross_val_score

from sklearn.feature_extraction.text import CountVectorizer
p_classes = dict(train['target'].value_counts(normalize=True))

naive_approach = p_classes[0]

print('Class probabilities: ', p_classes,

      '\nChance tweet is not about a real disaster: ', np.round(naive_approach, decimals = 4))
count_vectorizer = CountVectorizer(strip_accents = 'unicode',

                                   ngram_range = (1, 3), # consider unigrams, bigrams, and trigrams

                                   binary = True)
mnb_train_vector_lemma = count_vectorizer.fit_transform(lemmatized_train['text'])



clf_mnb = MultinomialNB()



# Weighted F1 score

scores = model_selection.cross_val_score(clf_mnb,

                                         mnb_train_vector_lemma, cleaned_train["target"],

                                         cv=5,

                                         scoring="f1_weighted")



print('Weighted F1:')

clf_score = model_scoring(scores)



# Weighted precision score

scores = model_selection.cross_val_score(clf_mnb,

                                         mnb_train_vector_lemma, cleaned_train["target"],

                                         cv=5,

                                         scoring="precision_weighted")



print('\nWeighted precision:')

clf_score = model_scoring(scores)



# Weighted recall score

scores = model_selection.cross_val_score(clf_mnb,

                                         mnb_train_vector_lemma, cleaned_train["target"],

                                         cv=5,

                                         scoring="recall_weighted")



print('\nWeighted recall:')

clf_score = model_scoring(scores)
count_vectorizer.fit(lemmatized_train['text'])

test_vector_lemma = count_vectorizer.transform(cleaned_test['text'])

clf_mnb.fit(mnb_train_vector_lemma, cleaned_train["target"])

mnb_preds = clf_mnb.predict(test_vector_lemma)
model_sub = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

model_sub['target'] = mnb_preds

model_sub.to_csv('../working/mnb_prediction_submission.csv', index = False)