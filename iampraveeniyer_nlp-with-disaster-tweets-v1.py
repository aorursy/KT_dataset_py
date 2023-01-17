## Adding imports

import numpy as np
import pandas as pd
import gc
import re
import string
import operator
import matplotlib.pyplot as plt
import seaborn as sns
import operator
from time import time

import nltk
from nltk.corpus import stopwords
from nltk import FreqDist, word_tokenize, sent_tokenize
## Loading necessary datasets

training_set = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test_set = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
output_file = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

print('There are {} rows and {} columns in Training Dataset'.format(training_set.shape[0],training_set.shape[1]))
print('There are {} rows and {} columns in Test Dataset'.format(test_set.shape[0],test_set.shape[1]))
## Basic Data Exploration

training_set.head()
training_set.tail()
print (training_set.shape, test_set.shape, output_file.shape)
training_set.duplicated().sum()
training_set = training_set.drop_duplicates().reset_index(drop=True)
print(training_set.shape)
# Target distribution in Training dataset

bar_plot=training_set.target.value_counts()
sns.barplot(bar_plot.index,bar_plot)
plt.gca().set_ylabel('Training Samples')
## Summary statistics for Training dataset
training_set.describe()
training_set.describe(include=['object'])
training_set.isnull().sum()
## Summary statistics for Test dataset
test_set.describe()
test_set.describe(include=['object'])
test_set.isnull().sum()
# Top keywords for disaster tweets

keywords_disaster = [kw for kw in training_set.loc[training_set.target == 1].keyword]
top_keywords_disaster = training_set[training_set.target==1].keyword.value_counts().head(10)
print ('Top keywords for disaster tweets in Training: ')
print (top_keywords_disaster)
# Top keywords for non-disaster tweets

keywords_non_disaster = [kw for kw in training_set.loc[training_set.target == 0].keyword]
top_keywords_non_disaster = training_set[training_set.target==0].keyword.value_counts().head(10)
print ('Top keywords for non-disaster tweets in Training: ')
print (top_keywords_non_disaster)
# Checking if the same keywords are present in both the positive and negative classes

disaster_KW_cnts = dict(pd.DataFrame(data={'x': keywords_disaster}).x.value_counts())
non_disaster_KW_cnts = dict(pd.DataFrame(data={'x': keywords_non_disaster}).x.value_counts())

all_keywords_counts =  dict(pd.DataFrame(data={'x': training_set.keyword.values}).x.value_counts())

for keyword, _ in sorted(all_keywords_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    print("<Keyword>: {}".format(keyword))
    print("* # in Disaster tweets: {}".format(disaster_KW_cnts.get(keyword, 0)))
    print("* # in Non-Disaster tweets: {}".format(non_disaster_KW_cnts.get(keyword, 0)))
    print('')
# Finding URL, line breaks and extra spaces from series of tweets

def standardize_text(text):
    #Removing available URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Removing line breaks
    text = re.sub(r'\n',' ', text)
    
    # Removing trailing and leading spaces 
    text = re.sub('\s+', ' ', text).strip()
    
    #Removing non-ASCII characters 
    text = ''.join([x for x in text if x in string.printable])
    
    #Removing HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    
    return text


training_set['text_cleaned']=training_set['text'].apply(lambda x : standardize_text(x))
test_set['text_cleaned']=test_set['text'].apply(lambda x : standardize_text(x))

# Removing stopwords

def remove_stopwords(text):
        stop_words=set(stopwords.words('english'))
        if text is not None:
            word_tokens = [x for x in word_tokenize(text) if x not in stop_words]
            return " ".join(word_tokens)
        else:
            return None

training_set['text_cleaned']=training_set['text_cleaned'].apply(lambda x : remove_stopwords(x))
test_set['text_cleaned']=test_set['text_cleaned'].apply(lambda x : remove_stopwords(x))

# Removing Emojis (source: online) and punctuations

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
    trans_table=str.maketrans('','',string.punctuation)
    return text.translate(trans_table)

training_set['text_cleaned']=training_set['text_cleaned'].apply(lambda x: remove_emoji(x))
training_set['text_cleaned']=training_set['text_cleaned'].apply(lambda x: remove_punct(x))

test_set['text_cleaned']=test_set['text_cleaned'].apply(lambda x: remove_emoji(x))
test_set['text_cleaned']=test_set['text_cleaned'].apply(lambda x: remove_punct(x))

training_set.head(10)
# Taking a look at the Top words for disaster tweets post basic pre-processing

# Locating english stopwords
stop_words = set(stopwords.words('english'))

freq_disaster = FreqDist(w for w in word_tokenize(' '.join(training_set.loc[training_set.target==1, 'text_cleaned']).lower()) if 
                     (w not in stop_words) & (w.isalpha()))
disaster_tweets = pd.DataFrame.from_dict(freq_disaster, orient='index', columns=['count'])
Top_disaster_tweets = disaster_tweets.sort_values('count',ascending=False).head(15)
print ('Top words for disaster tweets in Training: ')
print (Top_disaster_tweets)
# Taking a look at the Top words for non-disaster tweets post basic pre-processing

freq_nondisaster = FreqDist(w for w in word_tokenize(' '.join(training_set.loc[training_set.target==0, 'text_cleaned']).lower()) if 
                     (w not in stop_words) & (w.isalpha()))
non_disaster_tweets = pd.DataFrame.from_dict(freq_nondisaster, orient='index', columns=['count'])
Top_non_disaster_tweets = non_disaster_tweets.sort_values('count',ascending=False).head(15)
print ('Top words for non-disaster tweets in Training: ')
print (Top_non_disaster_tweets)
# Getting statistics to find the effect of stop words

training_set['text_length'] = training_set['text_cleaned'].apply(len)
training_set['word_count'] = training_set["text_cleaned"].apply(lambda x: len(str(x).split()))
training_set['stop_word_count'] = training_set['text_cleaned'].apply(lambda x: len([w for w in str(x).lower().split() if w in stop_words]))
training_set['caps_count'] = training_set['text_cleaned'].apply(lambda x: sum(1 for c in str(x) if c.isupper()))
training_set['caps_ratio'] = training_set['caps_count'] / training_set['text_length']

print(training_set.shape, test_set.shape)
#Checking for correlation of statistics features with the target variable
training_set.corr()['target'].drop('target').sort_values()
def additional_cleaning(tweet): 

    # Correcting character based references
    tweet = re.sub(r"&gt;", ">", tweet)
    tweet = re.sub(r"&lt;", "<", tweet)
    tweet = re.sub(r"&amp;", "&", tweet)
    
    # Correcting informal abbreviations
    tweet = re.sub(r"w/e", "whatever", tweet)
    tweet = re.sub(r"w/", "with", tweet)
    tweet = re.sub(r"USAgov", "USA government", tweet)
    tweet = re.sub(r"recentlu", "recently", tweet)
    tweet = re.sub(r"Ph0tos", "Photos", tweet)
    tweet = re.sub(r"amirite", "am I right", tweet)
    tweet = re.sub(r"exp0sed", "exposed", tweet)
    tweet = re.sub(r"<3", "love", tweet)
    tweet = re.sub(r"amageddon", "armageddon", tweet)
    tweet = re.sub(r"Trfc", "Traffic", tweet)
    tweet = re.sub(r"8/5/2015", "2015-08-05", tweet)
    tweet = re.sub(r"WindStorm", "Wind Storm", tweet)
    tweet = re.sub(r"8/6/2015", "2015-08-06", tweet)
    tweet = re.sub(r"10:38PM", "10:38 PM", tweet)
    tweet = re.sub(r"10:30pm", "10:30 PM", tweet)
    tweet = re.sub(r"16yr", "16 year", tweet)
    tweet = re.sub(r"lmao", "laughing my ass off", tweet)   
    tweet = re.sub(r"TRAUMATISED", "traumatized", tweet)
    
    # Correcting URL's
    tweet = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", tweet)
    
    # Words with punctuations and special characters (source - online)
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`"
    for p in punctuations:
        tweet = tweet.replace(p, f' {p} ')
    
    # Correcting acronymns that we could find
    tweet = re.sub(r"MH370", "Malaysia Airlines Flight 370", tweet)
    tweet = re.sub(r"mÌ¼sica", "music", tweet)
    tweet = re.sub(r"okwx", "Oklahoma City Weather", tweet)
    tweet = re.sub(r"arwx", "Arkansas Weather", tweet)    
    tweet = re.sub(r"gawx", "Georgia Weather", tweet)  
    tweet = re.sub(r"scwx", "South Carolina Weather", tweet)  
    tweet = re.sub(r"cawx", "California Weather", tweet)
    tweet = re.sub(r"tnwx", "Tennessee Weather", tweet)
    tweet = re.sub(r"azwx", "Arizona Weather", tweet)  
    tweet = re.sub(r"alwx", "Alabama Weather", tweet)
    tweet = re.sub(r"wordpressdotcom", "wordpress", tweet)    
    tweet = re.sub(r"usNWSgov", "United States National Weather Service", tweet)
    
    # Removing line breaks
    tweet = re.sub(r'\n',' ', tweet) 
    
    # Removing leading, trailing and extra spaces
    tweet = re.sub('\s+', ' ', tweet).strip() 
    
    return tweet


# Building cleaner versions of both the training and test datasets

training_set['text_cleaned'] = training_set['text_cleaned'].apply(lambda s : additional_cleaning(s))
test_set['text_cleaned'] = test_set['text_cleaned'].apply(lambda s : additional_cleaning(s))
# Set of other abbreviations found online

import nltk
from nltk.corpus import stopwords
from nltk import FreqDist, word_tokenize

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
    "qpsa" : "what happens",
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

def convert_abbrev_in_text(text):
    tokens = word_tokenize(text)
    tokens = [convert_abbrev(word) for word in tokens]
    text = ' '.join(tokens)
    return text

training_set["text_cleaned"] = training_set["text_cleaned"].apply(lambda x: convert_abbrev_in_text(x))
test_set["text_cleaned"] = test_set["text_cleaned"].apply(lambda x: convert_abbrev_in_text(x))
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer

porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")

def stemTweet_Porter(tweet):
    token_words=word_tokenize(tweet)
    clean_tweet=[]
    for word in token_words:
        clean_tweet.append(porter.stem(word))
        clean_tweet.append(" ")
    return "".join(clean_tweet)

def stemTweet_Lancaster(tweet):
    token_words=word_tokenize(tweet)
    clean_tweet=[]
    for word in token_words:
        clean_tweet.append(lancaster.stem(word))
        clean_tweet.append(" ")
    return "".join(clean_tweet)

def stemTweet_Snowball(tweet):
    token_words=word_tokenize(tweet)
    clean_tweet=[]
    for word in token_words:
        clean_tweet.append(snowball.stem(word))
        clean_tweet.append(" ")
    return "".join(clean_tweet)

training_set["text_cleaned_Port"] = training_set["text_cleaned"].apply(lambda x: stemTweet_Porter(x))
test_set["text_cleaned_Port"] = test_set["text_cleaned"].apply(lambda x: stemTweet_Porter(x))

training_set["text_cleaned_Lanc"] = training_set["text_cleaned"].apply(lambda x: stemTweet_Lancaster(x))
test_set["text_cleaned_Lanc"] = test_set["text_cleaned"].apply(lambda x: stemTweet_Lancaster(x))

training_set["text_cleaned_Snow"] = training_set["text_cleaned"].apply(lambda x: stemTweet_Lancaster(x))
test_set["text_cleaned_Snow"] = test_set["text_cleaned"].apply(lambda x: stemTweet_Lancaster(x))
from nltk.stem import WordNetLemmatizer

lemmmatizer=WordNetLemmatizer()

def lemmatization_wordnet(tweet):
    token_words=word_tokenize(tweet)
    tokens = [lemmmatizer.lemmatize(word.lower(), pos = "v") for word in token_words]
    clean_tweet = ' '.join(tokens)
    return clean_tweet

training_set["text_cleaned_Wordnet"] = training_set["text_cleaned"].apply(lambda x: lemmatization_wordnet(x))
test_set["text_cleaned_Wordnet"] = test_set["text_cleaned"].apply(lambda x: lemmatization_wordnet(x))
training_set["text_cleaned"] = training_set["text_cleaned_Wordnet"]
test_set["text_cleaned"] = test_set["text_cleaned_Wordnet"]

mislabeled_tweets = training_set.groupby(['text']).nunique().sort_values(by='target', ascending=False)
mislabeled_tweets = mislabeled_tweets[mislabeled_tweets['target'] > 1]['target']
mislabeled_tweets.index.tolist()
## Treating mislabled samples in training set

training_set['target_updated'] = training_set['target'].copy() 

training_set.loc[training_set['text'] == 'like for the music video I want some real action shit like burning buildings and police chases not some weak ben winston shit', 'target_updated'] = 0
training_set.loc[training_set['text'] == 'Hellfire is surrounded by desires so be careful and donÛªt let your desires control you! #Afterlife', 'target_updated'] = 0
training_set.loc[training_set['text'] == 'To fight bioterrorism sir.', 'target_updated'] = 0
training_set.loc[training_set['text'] == '.POTUS #StrategicPatience is a strategy for #Genocide; refugees; IDP Internally displaced people; horror; etc. https://t.co/rqWuoy1fm4', 'target_updated'] = 1
training_set.loc[training_set['text'] == 'CLEARED:incident with injury:I-495  inner loop Exit 31 - MD 97/Georgia Ave Silver Spring', 'target_updated'] = 1
training_set.loc[training_set['text'] == '#foodscare #offers2go #NestleIndia slips into loss after #Magginoodle #ban unsafe and hazardous for #humanconsumption', 'target_updated'] = 0
training_set.loc[training_set['text'] == 'In #islam saving a person is equal in reward to saving all humans! Islam is the opposite of terrorism!', 'target_updated'] = 0
training_set.loc[training_set['text'] == 'Who is bringing the tornadoes and floods. Who is bringing the climate change. God is after America He is plaguing her\n \n#FARRAKHAN #QUOTE', 'target_updated'] = 1
training_set.loc[training_set['text'] == 'RT NotExplained: The only known image of infamous hijacker D.B. Cooper. http://t.co/JlzK2HdeTG', 'target_updated'] = 1
training_set.loc[training_set['text'] == "Mmmmmm I'm burning.... I'm burning buildings I'm building.... Oooooohhhh oooh ooh...", 'target_updated'] = 0
training_set.loc[training_set['text'] == "wowo--=== 12000 Nigerian refugees repatriated from Cameroon", 'target_updated'] = 0
training_set.loc[training_set['text'] == "He came to a land which was engulfed in tribal war and turned it into a land of peace i.e. Madinah. #ProphetMuhammad #islam", 'target_updated'] = 0
training_set.loc[training_set['text'] == "Hellfire! We donÛªt even want to think about it or mention it so letÛªs not do anything that leads to it #islam!", 'target_updated'] = 0
training_set.loc[training_set['text'] == "The Prophet (peace be upon him) said 'Save yourself from Hellfire even if it is by giving half a date in charity.'", 'target_updated'] = 0
training_set.loc[training_set['text'] == "Caution: breathing may be hazardous to your health.", 'target_updated'] = 1
training_set.loc[training_set['text'] == "I Pledge Allegiance To The P.O.P.E. And The Burning Buildings of Epic City. ??????", 'target_updated'] = 0
training_set.loc[training_set['text'] == "#Allah describes piling up #wealth thinking it would last #forever as the description of the people of #Hellfire in Surah Humaza. #Reflect", 'target_updated'] = 0
training_set.loc[training_set['text'] == "that horrible sinking feeling when youÛªve been at home on your phone for a while and you realise its been on 3G this whole time", 'target_updated'] = 0
# Using Bag of words count

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def bag_of_words_count(data):
    count_vectorizer = CountVectorizer()
    embed = count_vectorizer.fit_transform(data)
    return embed, count_vectorizer

list_corpus = training_set["text_cleaned"].tolist()
list_labels = training_set["target_updated"].tolist()

X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.3, random_state=40)

X_train_counts, count_vectorizer = bag_of_words_count(X_train)

X_test_counts = count_vectorizer.transform(X_test)
# Fitting a Classifier using Logistic Regression - Baseline model

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(solver='liblinear', random_state=777)

classifier.fit(X_train_counts, y_train)

y_predicted_counts = classifier.predict(X_test_counts)
# Function to get model evaluation metrics

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

def get_metrics(y_test, y_predicted):  
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                    average='weighted')             
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                              average='weighted')
    
    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')
    
    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1

accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
from sklearn import metrics
from sklearn.metrics import confusion_matrix

print ('Confusion Matrix for Logistic Model')
print(metrics.confusion_matrix(y_test, y_predicted_counts))
# Creating submission file for the basic bag-of-words model

preds_corpus = test_set["text_cleaned"].tolist()
preds_counts = count_vectorizer.transform(preds_corpus)
preds_1 = classifier.predict(preds_counts)

output_file_1 = output_file.copy()
output_file_1.target = preds_1
output_file_1.to_csv('submission_LR_1.csv',index=False)

# Using Tf-iDF Vectorizer for Text

def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer()
    
    train = tfidf_vectorizer.fit_transform(data)

    return train, tfidf_vectorizer

X_train_tfidf, tfidf_vectorizer = tfidf(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
# Simple Logistic Regression Model post TFIDF

classifier_tfidf = LogisticRegression(solver='liblinear', random_state=777)
classifier_tfidf.fit(X_train_tfidf, y_train)

y_predicted_tfidf = classifier_tfidf.predict(X_test_tfidf)

accuracy_tfidf, precision_tfidf, recall_tfidf, f1_tfidf = get_metrics(y_test, y_predicted_tfidf)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_tfidf, precision_tfidf, 
                                                                       recall_tfidf, f1_tfidf))
# Confusion Matrix for the LR model post TFIDF

print ('Confusion Matrix for Logistic Model')
print(metrics.confusion_matrix(y_test, y_predicted_tfidf))
# Creating submission file for TFIDF Model

preds_corpus = test_set["text_cleaned"].tolist()
preds_tfidf = tfidf_vectorizer.transform(preds_corpus)
preds_2 = classifier_tfidf.predict(preds_tfidf)

output_file_2 = output_file.copy()
output_file_2.target = preds_2
output_file_2.to_csv('submission_TFIDF_2.csv',index=False)
## Loading necessary datasets back again

training_set_w2v = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test_set_w2v = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
output_file_w2v = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

print('There are {} rows and {} columns in Training Dataset'.format(training_set_w2v.shape[0],training_set_w2v.shape[1]))
print('There are {} rows and {} columns in Test Dataset'.format(test_set_w2v.shape[0],test_set_w2v.shape[1]))
def additional_cleaning(tweet): 

    # Correcting character based references
    tweet = re.sub(r"&gt;", ">", tweet)
    tweet = re.sub(r"&lt;", "<", tweet)
    tweet = re.sub(r"&amp;", "&", tweet)
    
    # Correcting informal abbreviations
    tweet = re.sub(r"w/e", "whatever", tweet)
    tweet = re.sub(r"w/", "with", tweet)
    tweet = re.sub(r"USAgov", "USA government", tweet)
    tweet = re.sub(r"recentlu", "recently", tweet)
    tweet = re.sub(r"Ph0tos", "Photos", tweet)
    tweet = re.sub(r"amirite", "am I right", tweet)
    tweet = re.sub(r"exp0sed", "exposed", tweet)
    tweet = re.sub(r"<3", "love", tweet)
    tweet = re.sub(r"amageddon", "armageddon", tweet)
    tweet = re.sub(r"Trfc", "Traffic", tweet)
    tweet = re.sub(r"8/5/2015", "2015-08-05", tweet)
    tweet = re.sub(r"WindStorm", "Wind Storm", tweet)
    tweet = re.sub(r"8/6/2015", "2015-08-06", tweet)
    tweet = re.sub(r"10:38PM", "10:38 PM", tweet)
    tweet = re.sub(r"10:30pm", "10:30 PM", tweet)
    tweet = re.sub(r"16yr", "16 year", tweet)
    tweet = re.sub(r"lmao", "laughing my ass off", tweet)   
    tweet = re.sub(r"TRAUMATISED", "traumatized", tweet)
    
    # Correcting URL's
    tweet = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", tweet)
    
    # Words with punctuations and special characters
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`"
    for p in punctuations:
        tweet = tweet.replace(p, f' {p} ')
    
    # Correcting acronymns that we could find
    tweet = re.sub(r"MH370", "Malaysia Airlines Flight 370", tweet)
    tweet = re.sub(r"mÌ¼sica", "music", tweet)
    tweet = re.sub(r"okwx", "Oklahoma City Weather", tweet)
    tweet = re.sub(r"arwx", "Arkansas Weather", tweet)    
    tweet = re.sub(r"gawx", "Georgia Weather", tweet)  
    tweet = re.sub(r"scwx", "South Carolina Weather", tweet)  
    tweet = re.sub(r"cawx", "California Weather", tweet)
    tweet = re.sub(r"tnwx", "Tennessee Weather", tweet)
    tweet = re.sub(r"azwx", "Arizona Weather", tweet)  
    tweet = re.sub(r"alwx", "Alabama Weather", tweet)
    tweet = re.sub(r"wordpressdotcom", "wordpress", tweet)    
    tweet = re.sub(r"usNWSgov", "United States National Weather Service", tweet)
    
    # Removing line breaks
    tweet = re.sub(r'\n',' ', tweet) 
    
    # Removing leading, trailing and extra spaces
    tweet = re.sub('\s+', ' ', tweet).strip() 
    
    # Removing non-ASCII characters
    tweet = ''.join([x for x in tweet if x in string.printable])
    
    #Removing HTML tags
    tweet = re.sub(r'<.*?>', ' ', tweet)
    
    return tweet

# Building cleaner versions of both the training and test datasets

training_set_w2v['text_cleaned'] = training_set_w2v['text'].apply(lambda s : additional_cleaning(s))
test_set_w2v['text_cleaned'] = test_set_w2v['text'].apply(lambda s : additional_cleaning(s))
# Set of other abbreviations found online

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
    "qpsa" : "what happens",
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

def convert_abbrev_in_text(text):
    tokens = word_tokenize(text)
    tokens = [convert_abbrev(word) for word in tokens]
    text = ' '.join(tokens)
    return text

training_set_w2v["text_cleaned"] = training_set_w2v["text_cleaned"].apply(lambda x: convert_abbrev_in_text(x))
test_set_w2v["text_cleaned"] = test_set_w2v["text_cleaned"].apply(lambda x: convert_abbrev_in_text(x))
# Lemmatization using WordNet Lemmatizer

from nltk.stem import WordNetLemmatizer

lemmmatizer=WordNetLemmatizer()

def lemmatization_wordnet(tweet):
    token_words=word_tokenize(tweet)
    tokens = [lemmmatizer.lemmatize(word.lower(), pos = "v") for word in token_words]
    clean_tweet = ' '.join(tokens)
    return clean_tweet

training_set_w2v["text_cleaned"] = training_set_w2v["text_cleaned"].apply(lambda x: lemmatization_wordnet(x))
test_set_w2v["text_cleaned"] = test_set_w2v["text_cleaned"].apply(lambda x: lemmatization_wordnet(x))
## Treating mislabled samples in training set

training_set_w2v['target_updated'] = training_set_w2v['target'].copy() 

training_set_w2v.loc[training_set_w2v['text'] == 'like for the music video I want some real action shit like burning buildings and police chases not some weak ben winston shit', 'target_updated'] = 0
training_set_w2v.loc[training_set_w2v['text'] == 'Hellfire is surrounded by desires so be careful and donÛªt let your desires control you! #Afterlife', 'target_updated'] = 0
training_set_w2v.loc[training_set_w2v['text'] == 'To fight bioterrorism sir.', 'target_updated'] = 0
training_set_w2v.loc[training_set_w2v['text'] == '.POTUS #StrategicPatience is a strategy for #Genocide; refugees; IDP Internally displaced people; horror; etc. https://t.co/rqWuoy1fm4', 'target_updated'] = 1
training_set_w2v.loc[training_set_w2v['text'] == 'CLEARED:incident with injury:I-495  inner loop Exit 31 - MD 97/Georgia Ave Silver Spring', 'target_updated'] = 1
training_set_w2v.loc[training_set_w2v['text'] == '#foodscare #offers2go #NestleIndia slips into loss after #Magginoodle #ban unsafe and hazardous for #humanconsumption', 'target_updated'] = 0
training_set_w2v.loc[training_set_w2v['text'] == 'In #islam saving a person is equal in reward to saving all humans! Islam is the opposite of terrorism!', 'target_updated'] = 0
training_set_w2v.loc[training_set_w2v['text'] == 'Who is bringing the tornadoes and floods. Who is bringing the climate change. God is after America He is plaguing her\n \n#FARRAKHAN #QUOTE', 'target_updated'] = 1
training_set_w2v.loc[training_set_w2v['text'] == 'RT NotExplained: The only known image of infamous hijacker D.B. Cooper. http://t.co/JlzK2HdeTG', 'target_updated'] = 1
training_set_w2v.loc[training_set_w2v['text'] == "Mmmmmm I'm burning.... I'm burning buildings I'm building.... Oooooohhhh oooh ooh...", 'target_updated'] = 0
training_set_w2v.loc[training_set_w2v['text'] == "wowo--=== 12000 Nigerian refugees repatriated from Cameroon", 'target_updated'] = 0
training_set_w2v.loc[training_set_w2v['text'] == "He came to a land which was engulfed in tribal war and turned it into a land of peace i.e. Madinah. #ProphetMuhammad #islam", 'target_updated'] = 0
training_set_w2v.loc[training_set_w2v['text'] == "Hellfire! We donÛªt even want to think about it or mention it so letÛªs not do anything that leads to it #islam!", 'target_updated'] = 0
training_set_w2v.loc[training_set_w2v['text'] == "The Prophet (peace be upon him) said 'Save yourself from Hellfire even if it is by giving half a date in charity.'", 'target_updated'] = 0
training_set_w2v.loc[training_set_w2v['text'] == "Caution: breathing may be hazardous to your health.", 'target_updated'] = 1
training_set_w2v.loc[training_set_w2v['text'] == "I Pledge Allegiance To The P.O.P.E. And The Burning Buildings of Epic City. ??????", 'target_updated'] = 0
training_set_w2v.loc[training_set_w2v['text'] == "#Allah describes piling up #wealth thinking it would last #forever as the description of the people of #Hellfire in Surah Humaza. #Reflect", 'target_updated'] = 0
training_set_w2v.loc[training_set_w2v['text'] == "that horrible sinking feeling when youÛªve been at home on your phone for a while and you realise its been on 3G this whole time", 'target_updated'] = 0
# Using word2vec

import gensim

word2vec_path = "../input/nlpword2vecembeddingspretrained/GoogleNews-vectors-negative300.bin"
word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
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

def get_word2vec_embeddings(vectors, clean_questions, generate_missing=False):
    embeddings = training_set_w2v['text_cleaned'].apply(lambda x: get_average_word2vec(x, vectors, 
                                                                                generate_missing=generate_missing))
    return list(embeddings)
list_labels = training_set_w2v["target_updated"].tolist()
embeddings = get_word2vec_embeddings(word2vec, training_set_w2v)
X_train_word2vec, X_test_word2vec, y_train_word2vec, y_test_word2vec = train_test_split(embeddings, list_labels, 
                                                                                        test_size=0.3, random_state=40)
classifier_w2v = LogisticRegression(solver='liblinear', random_state=777)

classifier_w2v.fit(X_train_word2vec, y_train_word2vec)

y_predicted_word2vec = classifier_w2v.predict(X_test_word2vec)
accuracy_word2vec, precision_word2vec, recall_word2vec, f1_word2vec = get_metrics(y_test_word2vec, y_predicted_word2vec)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_word2vec, precision_word2vec, 
                                                                       recall_word2vec, f1_word2vec))
# Confusion Matrix for the LR model Word2Vec

print ('Confusion Matrix for Logistic Model')
print(metrics.confusion_matrix(y_test_word2vec, y_predicted_word2vec))
import re
import string
import operator
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk import FreqDist, word_tokenize, sent_tokenize

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub

# Using official tokenization script used by Google Team

!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py

import tokenization
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
def build_model(bert_layer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(clf_output)
    
    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(lr=3e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
%%time
module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
bert_layer = hub.KerasLayer(module_url, trainable=True)
## Loading necessary datasets back again

training_set_bert = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test_set_bert = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
output_file_bert = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

print('There are {} rows and {} columns in Training Dataset'.format(training_set_bert.shape[0],training_set_bert.shape[1]))
print('There are {} rows and {} columns in Test Dataset'.format(test_set_bert.shape[0],test_set_bert.shape[1]))
# Loading tokenizer from BERT layer

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
def additional_cleaning(tweet): 

    # Correcting character based references
    tweet = re.sub(r"&gt;", ">", tweet)
    tweet = re.sub(r"&lt;", "<", tweet)
    tweet = re.sub(r"&amp;", "&", tweet)
    
    # Correcting informal abbreviations
    tweet = re.sub(r"w/e", "whatever", tweet)
    tweet = re.sub(r"w/", "with", tweet)
    tweet = re.sub(r"USAgov", "USA government", tweet)
    tweet = re.sub(r"recentlu", "recently", tweet)
    tweet = re.sub(r"Ph0tos", "Photos", tweet)
    tweet = re.sub(r"amirite", "am I right", tweet)
    tweet = re.sub(r"exp0sed", "exposed", tweet)
    tweet = re.sub(r"<3", "love", tweet)
    tweet = re.sub(r"amageddon", "armageddon", tweet)
    tweet = re.sub(r"Trfc", "Traffic", tweet)
    tweet = re.sub(r"8/5/2015", "2015-08-05", tweet)
    tweet = re.sub(r"WindStorm", "Wind Storm", tweet)
    tweet = re.sub(r"8/6/2015", "2015-08-06", tweet)
    tweet = re.sub(r"10:38PM", "10:38 PM", tweet)
    tweet = re.sub(r"10:30pm", "10:30 PM", tweet)
    tweet = re.sub(r"16yr", "16 year", tweet)
    tweet = re.sub(r"lmao", "laughing my ass off", tweet)   
    tweet = re.sub(r"TRAUMATISED", "traumatized", tweet)
    
    # Correcting URL's
    tweet = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", tweet)
    
    # Words with punctuations and special characters
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`"
    for p in punctuations:
        tweet = tweet.replace(p, f' {p} ')
    
    # Correcting acronymns that we could find
    tweet = re.sub(r"MH370", "Malaysia Airlines Flight 370", tweet)
    tweet = re.sub(r"mÌ¼sica", "music", tweet)
    tweet = re.sub(r"okwx", "Oklahoma City Weather", tweet)
    tweet = re.sub(r"arwx", "Arkansas Weather", tweet)    
    tweet = re.sub(r"gawx", "Georgia Weather", tweet)  
    tweet = re.sub(r"scwx", "South Carolina Weather", tweet)  
    tweet = re.sub(r"cawx", "California Weather", tweet)
    tweet = re.sub(r"tnwx", "Tennessee Weather", tweet)
    tweet = re.sub(r"azwx", "Arizona Weather", tweet)  
    tweet = re.sub(r"alwx", "Alabama Weather", tweet)
    tweet = re.sub(r"wordpressdotcom", "wordpress", tweet)    
    tweet = re.sub(r"usNWSgov", "United States National Weather Service", tweet)
    
    # Removing line breaks
    tweet = re.sub(r'\n',' ', tweet) 
    
    # Removing leading, trailing and extra spaces
    tweet = re.sub('\s+', ' ', tweet).strip() 
    
    # Removing HTML tags
    tweet = re.sub(r'<.*?>', ' ', tweet)
    
    # Removing non-ASCII characters
    tweet = ''.join([x for x in tweet if x in string.printable])
    
    return tweet

# Building cleaner versions of both the training and test datasets

training_set_bert['text_cleaned'] = training_set_bert['text'].apply(lambda s : additional_cleaning(s))
test_set_bert['text_cleaned'] = test_set_bert['text'].apply(lambda s : additional_cleaning(s))
# Set of other abbreviations found online

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
    "qpsa" : "what happens",
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

def convert_abbrev_in_text(text):
    tokens = word_tokenize(text)
    tokens = [convert_abbrev(word) for word in tokens]
    text = ' '.join(tokens)
    return text

training_set_bert["text_cleaned"] = training_set_bert["text_cleaned"].apply(lambda x: convert_abbrev_in_text(x))
test_set_bert["text_cleaned"] = test_set_bert["text_cleaned"].apply(lambda x: convert_abbrev_in_text(x))
# Removing Emojis (source: online) and punctuations

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
    trans_table=str.maketrans('','',string.punctuation)
    return text.translate(trans_table)

training_set_bert['text_cleaned']=training_set_bert['text_cleaned'].apply(lambda x: remove_emoji(x))
training_set_bert['text_cleaned']=training_set_bert['text_cleaned'].apply(lambda x: remove_punct(x))

test_set_bert['text_cleaned']=test_set_bert['text_cleaned'].apply(lambda x: remove_emoji(x))
test_set_bert['text_cleaned']=test_set_bert['text_cleaned'].apply(lambda x: remove_punct(x))
## Treating mislabled samples in training set

training_set_bert['target_updated'] = training_set_bert['target'].copy() 

training_set_bert.loc[training_set_bert['text'] == 'like for the music video I want some real action shit like burning buildings and police chases not some weak ben winston shit', 'target_updated'] = 0
training_set_bert.loc[training_set_bert['text'] == 'Hellfire is surrounded by desires so be careful and donÛªt let your desires control you! #Afterlife', 'target_updated'] = 0
training_set_bert.loc[training_set_bert['text'] == 'To fight bioterrorism sir.', 'target_updated'] = 0
training_set_bert.loc[training_set_bert['text'] == '.POTUS #StrategicPatience is a strategy for #Genocide; refugees; IDP Internally displaced people; horror; etc. https://t.co/rqWuoy1fm4', 'target_updated'] = 1
training_set_bert.loc[training_set_bert['text'] == 'CLEARED:incident with injury:I-495  inner loop Exit 31 - MD 97/Georgia Ave Silver Spring', 'target_updated'] = 1
training_set_bert.loc[training_set_bert['text'] == '#foodscare #offers2go #NestleIndia slips into loss after #Magginoodle #ban unsafe and hazardous for #humanconsumption', 'target_updated'] = 0
training_set_bert.loc[training_set_bert['text'] == 'In #islam saving a person is equal in reward to saving all humans! Islam is the opposite of terrorism!', 'target_updated'] = 0
training_set_bert.loc[training_set_bert['text'] == 'Who is bringing the tornadoes and floods. Who is bringing the climate change. God is after America He is plaguing her\n \n#FARRAKHAN #QUOTE', 'target_updated'] = 1
training_set_bert.loc[training_set_bert['text'] == 'RT NotExplained: The only known image of infamous hijacker D.B. Cooper. http://t.co/JlzK2HdeTG', 'target_updated'] = 1
training_set_bert.loc[training_set_bert['text'] == "Mmmmmm I'm burning.... I'm burning buildings I'm building.... Oooooohhhh oooh ooh...", 'target_updated'] = 0
training_set_bert.loc[training_set_bert['text'] == "wowo--=== 12000 Nigerian refugees repatriated from Cameroon", 'target_updated'] = 0
training_set_bert.loc[training_set_bert['text'] == "He came to a land which was engulfed in tribal war and turned it into a land of peace i.e. Madinah. #ProphetMuhammad #islam", 'target_updated'] = 0
training_set_bert.loc[training_set_bert['text'] == "Hellfire! We donÛªt even want to think about it or mention it so letÛªs not do anything that leads to it #islam!", 'target_updated'] = 0
training_set_bert.loc[training_set_bert['text'] == "The Prophet (peace be upon him) said 'Save yourself from Hellfire even if it is by giving half a date in charity.'", 'target_updated'] = 0
training_set_bert.loc[training_set_bert['text'] == "Caution: breathing may be hazardous to your health.", 'target_updated'] = 1
training_set_bert.loc[training_set_bert['text'] == "I Pledge Allegiance To The P.O.P.E. And The Burning Buildings of Epic City. ??????", 'target_updated'] = 0
training_set_bert.loc[training_set_bert['text'] == "#Allah describes piling up #wealth thinking it would last #forever as the description of the people of #Hellfire in Surah Humaza. #Reflect", 'target_updated'] = 0
training_set_bert.loc[training_set_bert['text'] == "that horrible sinking feeling when youÛªve been at home on your phone for a while and you realise its been on 3G this whole time", 'target_updated'] = 0
# Encoding tweets into tokens, masks and segment flags

train_input = bert_encode(training_set_bert.text_cleaned.values, tokenizer, max_len=160)
test_input = bert_encode(test_set_bert.text_cleaned.values, tokenizer, max_len=160)
train_labels = training_set_bert.target_updated.values

print (training_set_bert.text_cleaned.values)
print (training_set_bert.target_updated.values)

# Model build and statistics

model = build_model(bert_layer, max_len=160)
model.summary()
# Model Train and save

checkpoint = ModelCheckpoint('model3.h5', monitor='val_loss', save_best_only=True)

train_history = model.fit(
    train_input, train_labels,
    validation_split=0.2,
    epochs=3,
    callbacks=[checkpoint],
    batch_size=16
)
# BERT Predictions and Submission

model.load_weights('model2.h5')
test_pred = model.predict(test_input)

output_file_bert['target'] = test_pred.round().astype(int)
output_file_bert.to_csv('BERT_submission_3.csv', index=False)
## Imports, Loading necessary data sets and tokenization

import pandas as pd
import numpy as np
from numpy import array
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from collections import  Counter
stop=set(stopwords.words('english'))
import re
from nltk.tokenize import word_tokenize
import gensim
import string
from keras.preprocessing.text import one_hot,Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from keras.models import Sequential,Model
from keras.layers.embeddings import Embedding
from keras.layers import LSTM,Dense,SpatialDropout1D,Flatten,Input
from keras.initializers import Constant
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


## Loading necessary datasets back again

training_set_glove = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test_set_glove = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
output_file_glove = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

print('There are {} rows and {} columns in Training Dataset'.format(training_set_glove.shape[0],training_set_glove.shape[1]))
print('There are {} rows and {} columns in Test Dataset'.format(test_set_glove.shape[0],test_set_glove.shape[1]))

dataset_glove = training_set_glove.append(test_set_glove,ignore_index=True)
# Loading glove

from numpy import array
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()
glove_file = open('../input/glove-common-crawl-42b-tokens/glove.42B.300d.txt','r')

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions

glove_file.close()
def build_vocab(X):
    
    tweets = X.apply(lambda s: s.split()).values      
    vocab = {}
    
    for tweet in tweets:
        for word in tweet:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1                
    return vocab


def check_embeddings_coverage(X, embeddings):
    vocab = build_vocab(X)    
    covered = {}
    oov = {}    
    n_covered = 0
    n_oov = 0
    
    for word in vocab:
        try:
            covered[word] = embeddings[word]
            n_covered += vocab[word]
        except:
            oov[word] = vocab[word]
            n_oov += vocab[word]
    return covered, oov, n_covered, n_oov
covered, oov, n_covered, n_oov = check_embeddings_coverage(dataset_glove["text"], embeddings_dictionary)
print(f"Number of words covered by Glove embeddings :: {n_covered}")
print(f"Number of words not covered by Glove embeddings :: {n_oov}")
print(f"Percentage of words covered by Glove embeddings :: {(n_covered/(n_covered + n_oov)) * 100}%")
dataset_glove["text"] = dataset_glove["text"].apply(lambda x : x.lower())
dataset_glove["keyword"].fillna("keyword", inplace = True)

dataset_glove["text"] = dataset_glove["text"] + " " + dataset_glove["keyword"]
dataset_glove.drop(["keyword", "location"], axis = 1, inplace = True)

words_list = " ".join(dataset_glove["text"])
not_english = [word for word in words_list.split() if word.isalpha() == False]
def additional_cleaning(tweet): 

    # Correcting character based references
    tweet = re.sub(r"&gt;", ">", tweet)
    tweet = re.sub(r"&lt;", "<", tweet)
    tweet = re.sub(r"&amp;", "&", tweet)
    
    # Correcting informal abbreviations
    tweet = re.sub(r"w/e", "whatever", tweet)
    tweet = re.sub(r"w/", "with", tweet)
    tweet = re.sub(r"USAgov", "USA government", tweet)
    tweet = re.sub(r"recentlu", "recently", tweet)
    tweet = re.sub(r"Ph0tos", "Photos", tweet)
    tweet = re.sub(r"amirite", "am I right", tweet)
    tweet = re.sub(r"exp0sed", "exposed", tweet)
    tweet = re.sub(r"<3", "love", tweet)
    tweet = re.sub(r"amageddon", "armageddon", tweet)
    tweet = re.sub(r"Trfc", "Traffic", tweet)
    tweet = re.sub(r"8/5/2015", "2015-08-05", tweet)
    tweet = re.sub(r"WindStorm", "Wind Storm", tweet)
    tweet = re.sub(r"8/6/2015", "2015-08-06", tweet)
    tweet = re.sub(r"10:38PM", "10:38 PM", tweet)
    tweet = re.sub(r"10:30pm", "10:30 PM", tweet)
    tweet = re.sub(r"16yr", "16 year", tweet)
    tweet = re.sub(r"lmao", "laughing my ass off", tweet)   
    tweet = re.sub(r"TRAUMATISED", "traumatized", tweet)
    
    # Correcting URL's
    tweet = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", tweet)
    
    # Words with punctuations and special characters
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`"
    for p in punctuations:
        tweet = tweet.replace(p, f' {p} ')
    
    # Correcting acronymns that we could find
    tweet = re.sub(r"MH370", "Malaysia Airlines Flight 370", tweet)
    tweet = re.sub(r"mÌ¼sica", "music", tweet)
    tweet = re.sub(r"okwx", "Oklahoma City Weather", tweet)
    tweet = re.sub(r"arwx", "Arkansas Weather", tweet)    
    tweet = re.sub(r"gawx", "Georgia Weather", tweet)  
    tweet = re.sub(r"scwx", "South Carolina Weather", tweet)  
    tweet = re.sub(r"cawx", "California Weather", tweet)
    tweet = re.sub(r"tnwx", "Tennessee Weather", tweet)
    tweet = re.sub(r"azwx", "Arizona Weather", tweet)  
    tweet = re.sub(r"alwx", "Alabama Weather", tweet)
    tweet = re.sub(r"wordpressdotcom", "wordpress", tweet)    
    tweet = re.sub(r"usNWSgov", "United States National Weather Service", tweet)
    
    # Removing line breaks
    tweet = re.sub(r'\n',' ', tweet) 
    
    # Removing leading, trailing and extra spaces
    tweet = re.sub('\s+', ' ', tweet).strip() 
    
    # Removing HTML tags
    tweet = re.sub(r'<.*?>', ' ', tweet)
    
    # Removing non-ASCII characters
    tweet = ''.join([x for x in tweet if x in string.printable])
    
    # Removing words that are not alphabets
    t = [w for w in tweet.split() if w not in not_english]
    data = " ".join(t)
    
    return tweet

# Building cleaner version of the dataset

dataset_glove['text_cleaned'] = dataset_glove['text'].apply(lambda s : additional_cleaning(s))
# Set of other abbreviations found online

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
    "qpsa" : "what happens",
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

def convert_abbrev_in_text(text):
    tokens = word_tokenize(text)
    tokens = [convert_abbrev(word) for word in tokens]
    text = ' '.join(tokens)
    return text

dataset_glove["text_cleaned"] = dataset_glove["text_cleaned"].apply(lambda x: convert_abbrev_in_text(x))

# Removing Emojis (source: online) and punctuations

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
    trans_table=str.maketrans('','',string.punctuation)
    return text.translate(trans_table)

dataset_glove['text_cleaned']=dataset_glove['text_cleaned'].apply(lambda x: remove_emoji(x))
dataset_glove['text_cleaned']=dataset_glove['text_cleaned'].apply(lambda x: remove_punct(x))
## Treating mislabled samples in training set

training_set_glove['target_updated'] = training_set_glove['target'].copy() 

training_set_glove.loc[training_set_glove['text'] == 'like for the music video I want some real action shit like burning buildings and police chases not some weak ben winston shit', 'target_updated'] = 0
training_set_glove.loc[training_set_glove['text'] == 'Hellfire is surrounded by desires so be careful and donÛªt let your desires control you! #Afterlife', 'target_updated'] = 0
training_set_glove.loc[training_set_glove['text'] == 'To fight bioterrorism sir.', 'target_updated'] = 0
training_set_glove.loc[training_set_glove['text'] == '.POTUS #StrategicPatience is a strategy for #Genocide; refugees; IDP Internally displaced people; horror; etc. https://t.co/rqWuoy1fm4', 'target_updated'] = 1
training_set_glove.loc[training_set_glove['text'] == 'CLEARED:incident with injury:I-495  inner loop Exit 31 - MD 97/Georgia Ave Silver Spring', 'target_updated'] = 1
training_set_glove.loc[training_set_glove['text'] == '#foodscare #offers2go #NestleIndia slips into loss after #Magginoodle #ban unsafe and hazardous for #humanconsumption', 'target_updated'] = 0
training_set_glove.loc[training_set_glove['text'] == 'In #islam saving a person is equal in reward to saving all humans! Islam is the opposite of terrorism!', 'target_updated'] = 0
training_set_glove.loc[training_set_glove['text'] == 'Who is bringing the tornadoes and floods. Who is bringing the climate change. God is after America He is plaguing her\n \n#FARRAKHAN #QUOTE', 'target_updated'] = 1
training_set_glove.loc[training_set_glove['text'] == 'RT NotExplained: The only known image of infamous hijacker D.B. Cooper. http://t.co/JlzK2HdeTG', 'target_updated'] = 1
training_set_glove.loc[training_set_glove['text'] == "Mmmmmm I'm burning.... I'm burning buildings I'm building.... Oooooohhhh oooh ooh...", 'target_updated'] = 0
training_set_glove.loc[training_set_glove['text'] == "wowo--=== 12000 Nigerian refugees repatriated from Cameroon", 'target_updated'] = 0
training_set_glove.loc[training_set_glove['text'] == "He came to a land which was engulfed in tribal war and turned it into a land of peace i.e. Madinah. #ProphetMuhammad #islam", 'target_updated'] = 0
training_set_glove.loc[training_set_glove['text'] == "Hellfire! We donÛªt even want to think about it or mention it so letÛªs not do anything that leads to it #islam!", 'target_updated'] = 0
training_set_glove.loc[training_set_glove['text'] == "The Prophet (peace be upon him) said 'Save yourself from Hellfire even if it is by giving half a date in charity.'", 'target_updated'] = 0
training_set_glove.loc[training_set_glove['text'] == "Caution: breathing may be hazardous to your health.", 'target_updated'] = 1
training_set_glove.loc[training_set_glove['text'] == "I Pledge Allegiance To The P.O.P.E. And The Burning Buildings of Epic City. ??????", 'target_updated'] = 0
training_set_glove.loc[training_set_glove['text'] == "#Allah describes piling up #wealth thinking it would last #forever as the description of the people of #Hellfire in Surah Humaza. #Reflect", 'target_updated'] = 0
training_set_glove.loc[training_set_glove['text'] == "that horrible sinking feeling when youÛªve been at home on your phone for a while and you realise its been on 3G this whole time", 'target_updated'] = 0
# Checking again after some data cleaning
covered, oov, n_covered, n_oov = check_embeddings_coverage(dataset_glove["text_cleaned"], embeddings_dictionary)
print(f"Number of words covered by Glove embeddings --> {n_covered}")
print(f"Number of words not covered by Glove embeddings --> {n_oov}")
print(f"Percentage of words covered by Glove embeddings --> {(n_covered/(n_covered + n_oov)) * 100}%")
# Tokenization

embed_size = 300 
maxlen = 20
max_features = 20000

tokenizer = Tokenizer(oov_token = "<OOV>", num_words = max_features)
tokenizer.fit_on_texts(dataset_glove["text_cleaned"])
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(dataset_glove["text_cleaned"])
padded = pad_sequences(sequences, padding = "post", maxlen = maxlen)

training_glove = padded[:7613, :]
test_glove = padded[7613:, :]
train_y = dataset_glove[dataset_glove["target"].isnull() == False]["target"].apply(int).values.reshape(-1, 1)
num_words = min(max_features, len(word_index)) + 1

embedding_dim = 300

# first create a matrix of zeros, this is our embedding matrix
embedding_matrix = np.zeros((num_words, embedding_dim))

for word, i in word_index.items():
    if i > max_features:
        continue
    embedding_vector = covered.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
# Model build

import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(num_words,
                    embedding_dim,
                    embeddings_initializer=Constant(embedding_matrix),
                    input_length=maxlen,
                    trainable=False),
    tf.keras.layers.SpatialDropout1D(0.2),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dropout(0.10),
    tf.keras.layers.Dense(units=32, activation="relu"),
    tf.keras.layers.Dense(units=8, activation="relu"),
    tf.keras.layers.Dense(units=1, activation="sigmoid")
])

model.compile(loss = "binary_crossentropy", optimizer='adam', metrics = ["accuracy"])
model.summary()
# Model compilation and summary

batch_size = 128
num_epochs = 20

history = model.fit(training_glove, train_y, batch_size = batch_size, epochs = num_epochs)
# Creating submission file for Glove

y_predict=model.predict(test_glove)
y_predict = np.round(y_predict).astype(int)
#print (y_predict)
output_file_glove['target'] = y_predict.round().astype(int)
output_file_glove.to_csv('Glove_submission_2.csv', index=False)