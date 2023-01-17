%time

import os

import sys

import warnings

if not sys.warnoptions:

    warnings.simplefilter("ignore")

    

import numpy as np

import pandas as pd

import sklearn



# Libraries and packages for text (pre-)processing 

import string

import re

import nltk



print("Python version:", sys.version)

print("Version info.:", sys.version_info)

print("pandas version:", pd.__version__)

print("numpy version:", np.__version__)

print("skearn version:", sklearn.__version__)

print("re version:", re.__version__)

print("nltk version:", nltk.__version__)



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
%time



# read the csv file

train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

display(train_df.shape, train_df.head())
# some early explorations



display(train_df[~train_df["location"].isnull()].head())

display(train_df[train_df["target"] == 0]["text"].values[1])

display(train_df[train_df["target"] == 1]["text"].values[1])
train_df["text_clean"] = train_df["text"].apply(lambda x: x.lower())

display(train_df.head())
# Intall the contractions package - https://github.com/kootenpv/contractions

!pip install contractions
%time

import contractions



# Test

test_text = """

            Y'all can't expand contractions I'd think. I'd like to know how I'd done that! 

            We're going to the zoo and I don't think I'll be home for dinner.

            Theyre going to the zoo and she'll be home for dinner.

            We should've do it in here but we shouldn't've eat it

            """

print("Test: ", contractions.fix(test_text))



train_df["text_clean"] = train_df["text_clean"].apply(lambda x: contractions.fix(x))



# double check

print(train_df["text"][67])

print(train_df["text_clean"][67])

print(train_df["text"][12])

print(train_df["text_clean"][12])
def remove_URL(text):

    """

        Remove URLs from a sample string

    """

    return re.sub(r"https?://\S+|www\.\S+", "", text)
# remove urls from the text

train_df["text_clean"] = train_df["text_clean"].apply(lambda x: remove_URL(x))



# double check

print(train_df["text"][31])

print(train_df["text_clean"][31])

print(train_df["text"][37])

print(train_df["text_clean"][37])

print(train_df["text"][62])

print(train_df["text_clean"][62])
def remove_html(text):

    """

        Remove the html in sample text

    """

    html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")

    return re.sub(html, "", text)
# remove html from the text

train_df["text_clean"] = train_df["text_clean"].apply(lambda x: remove_html(x))



# double check

print(train_df["text"][62])

print(train_df["text_clean"][62])

print(train_df["text"][7385])

print(train_df["text_clean"][7385])
def remove_non_ascii(text):

    """

        Remove non-ASCII characters 

    """

    return re.sub(r'[^\x00-\x7f]',r'', text) # or ''.join([x for x in text if x in string.printable]) 
# remove non-ascii characters from the text

train_df["text_clean"] = train_df["text_clean"].apply(lambda x: remove_non_ascii(x))



# double check

print(train_df["text"][38])

print(train_df["text_clean"][38])

print(train_df["text"][7586])

print(train_df["text_clean"][7586])
train_df_jtcc = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip")

print(train_df_jtcc.shape)

train_df_jtcc.head()
def remove_special_characters(text):

    """

        Remove special special characters, including symbols, emojis, and other graphic characters

    """

    emoji_pattern = re.compile(

        '['

        u'\U0001F600-\U0001F64F'  # emoticons

        u'\U0001F300-\U0001F5FF'  # symbols & pictographs

        u'\U0001F680-\U0001F6FF'  # transport & map symbols

        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)

        u'\U00002702-\U000027B0'

        u'\U000024C2-\U0001F251'

        ']+',

        flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)
%time

# remove non-ascii characters from the text

train_df_jtcc["text_clean"] = train_df_jtcc["comment_text"].apply(lambda x: remove_special_characters(x))

display(train_df_jtcc.head())



# double check

print(train_df_jtcc["comment_text"][143])

print(train_df_jtcc["text_clean"][143])

print(train_df_jtcc["comment_text"][189])

print(train_df_jtcc["text_clean"][189])
# Saving disk space

del train_df_jtcc
def remove_punct(text):

    """

        Remove the punctuation

    """

#     return re.sub(r'[]!"$%&\'()*+,./:;=#@?[\\^_`{|}~-]+', "", text)

    return text.translate(str.maketrans('', '', string.punctuation))
# remove punctuations from the text

train_df["text_clean"] = train_df["text_clean"].apply(lambda x: remove_punct(x))



# double check

print(train_df["text"][5])

print(train_df["text_clean"][5])

print(train_df["text"][7597])

print(train_df["text_clean"][7597])
def other_clean(text):

        """

            Other manual text cleaning techniques

        """

        # Typos, slang and other

        sample_typos_slang = {

                                "w/e": "whatever",

                                "usagov": "usa government",

                                "recentlu": "recently",

                                "ph0tos": "photos",

                                "amirite": "am i right",

                                "exp0sed": "exposed",

                                "<3": "love",

                                "luv": "love",

                                "amageddon": "armageddon",

                                "trfc": "traffic",

                                "16yr": "16 year"

                                }



        # Acronyms

        sample_acronyms =  { 

                            "mh370": "malaysia airlines flight 370",

                            "okwx": "oklahoma city weather",

                            "arwx": "arkansas weather",    

                            "gawx": "georgia weather",  

                            "scwx": "south carolina weather",  

                            "cawx": "california weather",

                            "tnwx": "tennessee weather",

                            "azwx": "arizona weather",  

                            "alwx": "alabama weather",

                            "usnwsgov": "united states national weather service",

                            "2mw": "tomorrow"

                            }



        

        # Some common abbreviations 

        sample_abbr = {

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

            

        sample_typos_slang_pattern = re.compile(r'(?<!\w)(' + '|'.join(re.escape(key) for key in sample_typos_slang.keys()) + r')(?!\w)')

        sample_acronyms_pattern = re.compile(r'(?<!\w)(' + '|'.join(re.escape(key) for key in sample_acronyms.keys()) + r')(?!\w)')

        sample_abbr_pattern = re.compile(r'(?<!\w)(' + '|'.join(re.escape(key) for key in sample_abbr.keys()) + r')(?!\w)')

        

        text = sample_typos_slang_pattern.sub(lambda x: sample_typos_slang[x.group()], text)

        text = sample_acronyms_pattern.sub(lambda x: sample_acronyms[x.group()], text)

        text = sample_abbr_pattern.sub(lambda x: sample_abbr[x.group()], text)

        

        return text
%time



# Test

test_text = """

            brb with some sample ph0tos I lov u. I need some $ for 2mw.

            """

print("Test: ", other_clean(test_text))



# remove punctuations from the text

train_df["text_clean"] = train_df["text_clean"].apply(lambda x: other_clean(x))



# double check

print(train_df["text"][1844])

print(train_df["text_clean"][1844])

print(train_df["text"][4409])

print(train_df["text_clean"][4409])
from textblob import TextBlob

print("Test: ", TextBlob("sleapy and tehre is no plaxe I'm gioong to.").correct())
# Tokenizing the tweet base texts.

from nltk.tokenize import word_tokenize



train_df['tokenized'] = train_df['text_clean'].apply(word_tokenize)

train_df.head()
# Removing stopwords.

nltk.download("stopwords")

from nltk.corpus import stopwords



stop = set(stopwords.words('english'))

train_df['stopwords_removed'] = train_df['tokenized'].apply(lambda x: [word for word in x if word not in stop])

train_df.head()
from nltk.stem import PorterStemmer



def porter_stemmer(text):

    """

        Stem words in list of tokenized words with PorterStemmer

    """

    stemmer = nltk.PorterStemmer()

    stems = [stemmer.stem(i) for i in text]

    return stems
%time 



train_df['porter_stemmer'] = train_df['stopwords_removed'].apply(lambda x: porter_stemmer(x))

train_df.head()
from nltk.stem import SnowballStemmer



def snowball_stemmer(text):

    """

        Stem words in list of tokenized words with SnowballStemmer

    """

    stemmer = nltk.SnowballStemmer("english")

    stems = [stemmer.stem(i) for i in text]

    return stems
%time 



train_df['snowball_stemmer'] = train_df['stopwords_removed'].apply(lambda x: snowball_stemmer(x))

train_df.head()
from nltk.stem import LancasterStemmer



def lancaster_stemmer(text):

    """

        Stem words in list of tokenized words with LancasterStemmer

    """

    stemmer = nltk.LancasterStemmer()

    stems = [stemmer.stem(i) for i in text]

    return stems
%time 



train_df['lancaster_stemmer'] = train_df['stopwords_removed'].apply(lambda x: lancaster_stemmer(x))

train_df.head()
from nltk.corpus import wordnet

from nltk.corpus import brown



wordnet_map = {"N":wordnet.NOUN, 

               "V":wordnet.VERB, 

               "J":wordnet.ADJ, 

               "R":wordnet.ADV

              }

    

train_sents = brown.tagged_sents(categories='news')

t0 = nltk.DefaultTagger('NN')

t1 = nltk.UnigramTagger(train_sents, backoff=t0)

t2 = nltk.BigramTagger(train_sents, backoff=t1)



def pos_tag_wordnet(text, pos_tag_type="pos_tag"):

    """

        Create pos_tag with wordnet format

    """

    pos_tagged_text = t2.tag(text)

    

    # map the pos tagging output with wordnet output 

    pos_tagged_text = [(word, wordnet_map.get(pos_tag[0])) if pos_tag[0] in wordnet_map.keys() else (word, wordnet.NOUN) for (word, pos_tag) in pos_tagged_text ]

    return pos_tagged_text
pos_tag_wordnet(train_df['stopwords_removed'][2])
%time 



train_df['combined_postag_wnet'] = train_df['stopwords_removed'].apply(lambda x: pos_tag_wordnet(x))



train_df.head()
from nltk.stem import WordNetLemmatizer



def lemmatize_word(text):

    """

        Lemmatize the tokenized words

    """



    lemmatizer = WordNetLemmatizer()

    lemma = [lemmatizer.lemmatize(word, tag) for word, tag in text]

    return lemma
%time 



# Test without POS Tagging

lemmatizer = WordNetLemmatizer()



train_df['lemmatize_word_wo_pos'] = train_df['stopwords_removed'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

train_df['lemmatize_word_wo_pos'] = train_df['lemmatize_word_wo_pos'].apply(lambda x: [word for word in x if word not in stop])

train_df.head()
print(train_df["combined_postag_wnet"][8])

print(train_df["lemmatize_word_wo_pos"][8])
%time 



# Test with POS Tagging

lemmatizer = WordNetLemmatizer()



train_df['lemmatize_word_w_pos'] = train_df['combined_postag_wnet'].apply(lambda x: lemmatize_word(x))

train_df['lemmatize_word_w_pos'] = train_df['lemmatize_word_w_pos'].apply(lambda x: [word for word in x if word not in stop]) # double check to remove stop words

train_df['lemmatize_text'] = [' '.join(map(str, l)) for l in train_df['lemmatize_word_w_pos']] # join back to text



train_df.head()
print(train_df["text"][8])

print(train_df["combined_postag_wnet"][8])

print(train_df["lemmatize_word_wo_pos"][8])

print(train_df["lemmatize_word_w_pos"][8])
display(train_df["text"][0], train_df["lemmatize_text"][0])

display(train_df["text"][5], train_df["lemmatize_text"][5])

display(train_df["text"][10], train_df["lemmatize_text"][10])

display(train_df["text"][15], train_df["lemmatize_text"][15])

display(train_df["text"][20], train_df["lemmatize_text"][20])
# Install the main polygot and other neccesary packages

!pip install pyicu

!pip install pycld2

!pip install polyglot
train_df_jmtc = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")

print(train_df_jmtc.shape)

train_df_jmtc.head()
%time 

from polyglot.detect import Detector



def get_language(text):

    return Detector("".join(x for x in text if x.isprintable()), quiet=True).languages[0].name



train_df_jmtc["lang"] = train_df_jmtc["comment_text"].apply(lambda x: get_language(x))



#Test

display(train_df_jmtc[train_df_jmtc["lang"] == "de"].head())

print(train_df_jmtc["comment_text"][823])

print(train_df_jmtc["comment_text"][8130])

print(train_df_jmtc["comment_text"][14511])
# save disk space

del train_df_jmtc
from sklearn.feature_extraction.text import CountVectorizer



def cv(data, ngram = 1, MAX_NB_WORDS = 75000):

    count_vectorizer = CountVectorizer(ngram_range = (ngram, ngram), max_features = MAX_NB_WORDS)

    emb = count_vectorizer.fit_transform(data).toarray()

    print("count vectorize with", str(np.array(emb).shape[1]), "features")

    return emb, count_vectorizer
def print_out(emb, feat, ngram, compared_sentence=0):

    print(ngram,"bag-of-words: ")

    print(feat.get_feature_names(), "\n")

    print(ngram,"bag-of-feature: ")

    print(test_cv_1gram.vocabulary_, "\n")

    print("BoW matrix:")

    print(pd.DataFrame(emb.transpose(), index = feat.get_feature_names()).head(), "\n")

    print(ngram,"vector example:")

    print(train_df["lemmatize_text"][compared_sentence])

    print(emb[compared_sentence], "\n")
test_corpus = train_df["lemmatize_text"][:5].tolist()

print("The test corpus: ", test_corpus, "\n")



test_cv_em_1gram, test_cv_1gram = cv(test_corpus, ngram=1)

print_out(test_cv_em_1gram, test_cv_1gram, ngram="Uni-gram")
test_cv_em_2gram, test_cv_2gram = cv(test_corpus, ngram=2)

print_out(test_cv_em_2gram, test_cv_2gram, ngram="Bi-gram")
test_cv_em_3gram, test_cv_3gram = cv(test_corpus, ngram=3)

print_out(test_cv_em_2gram, test_cv_2gram, ngram="Tri-gram")
%time 



# implement into the whole dataset

train_df_corpus = train_df["lemmatize_text"].tolist()

train_df_em_1gram, vc_1gram = cv(train_df_corpus, 1)

train_df_em_2gram, vc_2gram = cv(train_df_corpus, 2)

train_df_em_3gram, vc_3gram = cv(train_df_corpus, 3)



print(len(train_df_corpus))

print(train_df_em_1gram.shape)

print(train_df_em_2gram.shape)

print(train_df_em_3gram.shape)
del train_df_em_1gram, train_df_em_2gram, train_df_em_3gram
from sklearn.feature_extraction.text import TfidfVectorizer



def TFIDF(data, ngram = 1, MAX_NB_WORDS = 75000):

    tfidf_x = TfidfVectorizer(ngram_range = (ngram, ngram), max_features = MAX_NB_WORDS)

    emb = tfidf_x.fit_transform(data).toarray()

    print("tf-idf with", str(np.array(emb).shape[1]), "features")

    return emb, tfidf_x
test_corpus = train_df["lemmatize_text"][:5].tolist()

print("The test corpus: ", test_corpus, "\n")



test_tfidf_em_1gram, test_tfidf_1gram = TFIDF(test_corpus, ngram=1)

print_out(test_tfidf_em_1gram, test_tfidf_1gram, ngram="Uni-gram")
test_tfidf_em_2gram, test_tfidf_2gram = TFIDF(test_corpus, ngram=2)

print_out(test_tfidf_em_2gram, test_tfidf_2gram, ngram="Bi-gram")
test_tfidf_em_3gram, test_tfidf_3gram = TFIDF(test_corpus, ngram=3)

print_out(test_tfidf_em_3gram, test_tfidf_3gram, ngram="Tri-gram")
%time 



# implement into the whole dataset

train_df_corpus = train_df["lemmatize_text"].tolist()

train_df_tfidf_1gram, tfidf_1gram = TFIDF(train_df_corpus, 1)

train_df_tfidf_2gram, tfidf_2gram = TFIDF(train_df_corpus, 2)

train_df_tfidf_3gram, tfidf_3gram = TFIDF(train_df_corpus, 3)



print(len(train_df_corpus))

print(train_df_tfidf_1gram.shape)

print(train_df_tfidf_1gram.shape)

print(train_df_tfidf_1gram.shape)
del train_df_tfidf_1gram, train_df_tfidf_2gram, train_df_tfidf_3gram
%time 



import gensim

print("gensim version:", gensim.__version__)



word2vec_path = "../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin"



# we only load 200k most common words from Google News corpus 

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True, limit=200000) 
print(word2vec_model.similarity('cat', 'kitten'))

print(word2vec_model.similarity('cat', 'cats'))
def get_average_vec(tokens_list, vector, generate_missing=False, k=300):

    """

        Calculate average embedding value of sentence from each word vector

    """

    

    if len(tokens_list)<1:

        return np.zeros(k)

    

    if generate_missing:

        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]

    else:

        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]

    

    length = len(vectorized)

    summed = np.sum(vectorized, axis=0)

    averaged = np.divide(summed, length)

    return averaged



def get_embeddings(vectors, text, generate_missing=False, k=300):

    """

        create the sentence embedding

    """

    embeddings = text.apply(lambda x: get_average_vec(x, vectors, generate_missing=generate_missing, k=k))

    return list(embeddings)
%time 



embeddings_word2vec = get_embeddings(word2vec_model, train_df["lemmatize_text"], k=300)



print("Embedding matrix size", len(embeddings_word2vec), len(embeddings_word2vec[0]))

print("The sentence: \"%s\" got embedding values: " % train_df["lemmatize_text"][0])

print(embeddings_word2vec[0])
del embeddings_word2vec
%time 



from gensim.scripts.glove2word2vec import glove2word2vec



glove_input_file = "../input/glove6b/glove.6B.300d.txt"

word2vec_output_file = "glove.6B.100d.txt.word2vec"

glove2word2vec(glove_input_file, word2vec_output_file)



# we only load 200k most common words from Google New corpus 

glove_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False, limit=200000) 
print(glove_model.similarity('cat', 'kitten'))

print(glove_model.similarity('cat', 'cats'))
%time 



embeddings_glove = get_embeddings(glove_model, train_df["lemmatize_text"], k=300)



print("Embedding matrix size", len(embeddings_glove), len(embeddings_glove[0]))

print("The sentence: \"%s\" got embedding values: " % train_df["lemmatize_text"][0])

print(embeddings_glove[0])
del embeddings_glove
%time 



from gensim.models.fasttext import FastText



fasttext_path = "../input/fasttext-wikinews/wiki-news-300d-1M.vec"

fasttext_model = gensim.models.KeyedVectors.load_word2vec_format(fasttext_path, binary=False, limit=200000)
print(fasttext_model.similarity('cat', 'kitten'))

print(fasttext_model.similarity('cat', 'cats'))
embeddings_fasttext = get_embeddings(fasttext_model, train_df["lemmatize_text"], k=300)



print("Embedding matrix size", len(embeddings_fasttext), len(embeddings_fasttext[0]))

print("The sentence: \"%s\" got embedding values: " % train_df["lemmatize_text"][0])

print(embeddings_fasttext[0])
del embeddings_fasttext
%time 



import tensorflow_hub as hub



# download the tonkenizer 

!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py

import tokenization
module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"

bert_layer = hub.KerasLayer(module_url, trainable=True)
def bert_encode(texts, tokenizer, max_len=512):

    all_tokens = []

    all_masks = []

    all_segments = []

    

    for text in texts:

        text = tokenizer.tokenize(text)

            

        text = text[:max_len-2]

        input_sequence = ["[CLS]"] + text + ["[SEP]"]

        pad_len = max_len - len(input_sequence)

        

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)

        tokens += [0] * pad_len

        pad_masks = [1] * len(input_sequence) + [0] * pad_len

        segment_ids = [0] * max_len

        

        all_tokens.append(tokens)

        all_masks.append(pad_masks)

        all_segments.append(segment_ids)

    

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
%time 



vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)



bert_input = bert_encode(train_df["text"].values, tokenizer, max_len=300)
print("Embedding tensor size", len(bert_input), len(bert_input[0]), len(bert_input[0][0]))

print("The sentence: \"%s\" got embedding values: " % train_df["lemmatize_text"][0])

print(bert_input[0])