# Don't forget to make sure your Internet connection is turned on

! pip install adaptnlp
import numpy as np

import pandas as pd

import os



target_corrected = True

to_lower = True
MODEL = 'bert'

MODEL_DIR = '/kaggle/' + MODEL + '-working'

try:

    os.mkdir(MODEL_DIR)

except:

    pass
df_train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv', dtype={'id': np.int16, 'target': np.int8})

df_test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv', dtype={'id': np.int16})
# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert

def clean_tweets(tweet):

    """Removes links and non-ASCII characters"""

    

    tweet = ''.join([x for x in tweet if x in string.printable])

    

    # Removing URLs

    tweet = re.sub(r"http\S+", "_url_", tweet)

    

    return tweet
# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert

def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'_emoji_', text)
# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert

def remove_punctuations(text):

    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`"

    

    for p in punctuations:

        text = text.replace(p, f' {p} ')



    text = text.replace('...', ' ... ')

    

    if '...' not in text:

        text = text.replace('..', ' ... ')

    

    return text
# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert

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
# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert

def convert_abbrev(word):

    return abbreviations[word.lower()] if word.lower() in abbreviations.keys() else word
# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert

def convert_abbrev_in_text(text):

    tokens = word_tokenize(text)

    tokens = [convert_abbrev(word) for word in tokens]

    text = ' '.join(tokens)

    return text
import string

import re

from nltk.tokenize import word_tokenize

df_train["text"] = df_train["text"].apply(lambda x: clean_tweets(x))

df_test["text"] = df_test["text"].apply(lambda x: clean_tweets(x))



df_train["text"] = df_train["text"].apply(lambda x: remove_emoji(x))

df_test["text"] = df_test["text"].apply(lambda x: remove_emoji(x))



df_train["text"] = df_train["text"].apply(lambda x: remove_punctuations(x))

df_test["text"] = df_test["text"].apply(lambda x: remove_punctuations(x))



df_train["text"] = df_train["text"].apply(lambda x: convert_abbrev_in_text(x))

df_test["text"] = df_test["text"].apply(lambda x: convert_abbrev_in_text(x))



if to_lower:

    df_train["text"] = df_train["text"].apply(lambda x: x.lower())

    df_test["text"] = df_test["text"].apply(lambda x: x.lower())
# Thanks to https://www.kaggle.com/wrrosa/keras-bert-using-tfhub-modified-train-data - 

# author of this kernel read tweets in training data and figure out that some of them have errors:

if target_corrected:

    ids_with_target_error = [328,443,513,2619,3640,3900,4342,5781,6552,6554,6570,6701,6702,6729,6861,7226]

    df_train.loc[df_train['id'].isin(ids_with_target_error),'target'] = 0

    df_train[df_train['id'].isin(ids_with_target_error)]
from adaptnlp import EasyDocumentEmbeddings, SequenceClassifierTrainer
from sklearn.model_selection import train_test_split

train, tbd_test = train_test_split(df_train, test_size=0.2)

test, dev = train_test_split(tbd_test, test_size=0.5)
label_training_dir = MODEL_DIR + '/label_training'

try:

    os.mkdir(label_training_dir)

except:

    pass
train.to_csv(label_training_dir + "/train.csv", index=False)

test.to_csv(label_training_dir + "/test.csv", index=False)

dev.to_csv(label_training_dir + "/dev.csv", index=False)
configs = {"pool_configs": {"fine_tune_mode": "linear", "pooling": "mean"},

"rnn_configs": {"hidden_size": 512,

                                   "rnn_layers": 1,

                                   "reproject_words": True,

                                   "reproject_words_dimension": 256,

                                   "bidirectional": False,

                                   "dropout": 0.4,

                                   "word_dropout": 0.0,

                                   "locked_dropout": 0.0,

                                   "rnn_type": "GRU",

                                   "fine_tune": True, }}
corpus = label_training_dir # Or path to directory of train.csv, test.csv, dev.csv files at "Path/to/data/directory" 

FINETUNED_MODEL_DIR = MODEL_DIR

OUTPUT_DIR = label_training_dir

doc_embeddings = EasyDocumentEmbeddings("bert-base-uncased", methods = ["rnn"],)
sc_configs = {

              "corpus": corpus,

              "encoder": doc_embeddings,

              "column_name_map": {3: "text", 4: "label"},

              "corpus_in_memory": True,

              "predictive_head": "flair",

             }

sc_trainer = SequenceClassifierTrainer(**sc_configs)
sc_lr_configs = {

        "output_dir": OUTPUT_DIR,

        "file_name": "learning_rate.tsv",

        "start_learning_rate": 1e-8,

        "end_learning_rate": 10,

        "iterations": 100,

        "mini_batch_size": 32,

        "stop_early": True,

        "smoothing_factor": 0.8,

        "plot_learning_rate": True,

}

learning_rate = sc_trainer.find_learning_rate(**sc_lr_configs)
sc_train_configs = {

        "output_dir": OUTPUT_DIR,

        "learning_rate": learning_rate,

        "mini_batch_size": 32,

        "anneal_factor": 0.5,

        "patience": 2, # If the model does not improve after this this many steps the learning rate will decrease

        "max_epochs": 10,

        "plot_weights": False,

        "batch_growth_annealing": False,

}

sc_trainer.train(**sc_train_configs)
from adaptnlp import EasySequenceClassifier

# Set example text and instantiate tagger instance

example_text = ["that was a really bad storm! but I suppose it could have been worse"]

MODEL_PATH = OUTPUT_DIR + '/final-model.pt'



classifier = EasySequenceClassifier()



# Example prediction

sentences = classifier.tag_text(example_text, model_name_or_path=MODEL_PATH)

print("Label output:\n")

for sentence in sentences:

    print(sentence.labels)

    print(sentence.labels[0].value)

    print(sentence.labels[0].score)
df_test['target'] = df_test['text'].apply(lambda x: classifier.tag_text(x, model_name_or_path=MODEL_PATH)[0].labels[0].value)
submission = df_test[['id', 'target']]

submission['target'].value_counts()
submission.to_csv('/kaggle/working/submission.csv', index=False)