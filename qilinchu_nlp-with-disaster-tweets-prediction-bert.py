import pandas as pd

import numpy as np

import seaborn as sns

import os

import re

import string

import warnings

import operator

import matplotlib

import matplotlib.pyplot as plt

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer

from nltk.stem.porter import PorterStemmer

from nltk.tokenize import word_tokenize

from nltk.util import ngrams

from collections import defaultdict, Counter

from tqdm import tqdm, tqdm_notebook

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import confusion_matrix

from keras.preprocessing.text import Tokenizer
# Setting some options for general use.

warnings.filterwarnings('ignore')

stop_words = set(stopwords.words("english"))

stemmer_snowball = SnowballStemmer("english")

stemmer_porter = PorterStemmer()

plt.style.use('ggplot')

sns.set(font_scale=1.5)

pd.options.display.max_columns = 250

pd.options.display.max_rows = 250
# parameters

Max_length = 42

Dropout_num = 0  

learning_rate = 6e-6 

valid = 0.2

epochs_num = 3

batch_size_num = 16

ids_error_corrected = True
# Load CSV files containing training data

train_path = "/kaggle/input/nlp-getting-started/train.csv"

test_path = "/kaggle/input/nlp-getting-started/test.csv"

train_df = pd.read_csv(train_path, dtype={'id': np.int16, 'target': np.int8})

test_df = pd.read_csv(test_path, dtype={'id': np.int16})

submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')



# Checking observation and feature numbers for train and test data.

print(f'train: {train_df.shape}')

print(f'test: {test_df.shape}')
train_df.isnull().sum()
# target distribution

fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(12, 4), dpi=100)

sns.countplot(train_df['target'], ax=axes[0])

axes[1].pie(train_df['target'].value_counts(),

            labels=['Not Disaster', 'Disaster'],

            autopct='%1.2f%%',

            shadow=True,

            explode=(0.05, 0),

            startangle=60)

fig.suptitle('Distribution of the Tweets', fontsize=24)

plt.show()
# keyword & location

missing_cols = ['keyword', 'location']

fig, axes = plt.subplots(ncols=2, figsize=(17, 4), dpi=100)

sns.barplot(x=train_df[missing_cols].isnull().sum().index, y=train_df[missing_cols].isnull().sum().values, ax=axes[0])

sns.barplot(x=test_df[missing_cols].isnull().sum().index, y=test_df[missing_cols].isnull().sum().values, ax=axes[1])

axes[0].set_ylabel('Missing Value Count', size=15, labelpad=20)

axes[0].tick_params(axis='x', labelsize=15)

axes[0].tick_params(axis='y', labelsize=15)

axes[1].tick_params(axis='x', labelsize=15)

axes[1].tick_params(axis='y', labelsize=15)

axes[0].set_title('Training Set', fontsize=13)

axes[1].set_title('Test Set', fontsize=13)

plt.show()
# dropping unwanted column

train_df = train_df.drop(['location', 'keyword'], axis=1)

test_df = test_df.drop(['location', 'keyword'], axis=1)
def remove_url(tweet):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',tweet)



def remove_html(tweet):

    html=re.compile(r'<.*?>')

    return html.sub(r'',tweet)



def remove_emoji(tweet):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', tweet)
def special_characters(tweet):

    

    tweet = re.sub(r"SuruÌ¤", "Suruc", tweet)

    tweet = re.sub(r"JapÌ_n", "Japan", tweet)  

    tweet = re.sub(r"\x89ÛÏWhen", "When", tweet)

    tweet = re.sub(r"å£3million", "3 million", tweet)

    tweet = re.sub(r"fromåÊwounds", "from wounds", tweet)

    tweet = re.sub(r"mÌ¼sica", "music", tweet)

    tweet = re.sub(r"donå«t", "do not", tweet)

    tweet = re.sub(r"didn`t", "did not", tweet)

    tweet = re.sub(r"i\x89Ûªm", "I am", tweet)

    tweet = re.sub(r"I\x89Ûªm", "I am", tweet)

    tweet = re.sub(r"it\x89Ûªs", "it is", tweet)

    tweet = re.sub(r"It\x89Ûªs", "It is", tweet)

    tweet = re.sub(r"i\x89Ûªd", "I would", tweet)

    tweet = re.sub(r"I\x89Ûªd", "I would", tweet)

    tweet = re.sub(r"i\x89Ûªve", "I have", tweet)

    tweet = re.sub(r"I\x89Ûªve", "I have", tweet)

    tweet = re.sub(r"let\x89Ûªs", "let us", tweet)

    tweet = re.sub(r"don\x89Ûªt", "do not", tweet)

    tweet = re.sub(r"Don\x89Ûªt", "Do not", tweet)

    tweet = re.sub(r"can\x89Ûªt", "cannot", tweet)

    tweet = re.sub(r"Can\x89Ûªt", "Cannot", tweet)

    tweet = re.sub(r"that\x89Ûªs", "that is", tweet)

    tweet = re.sub(r"That\x89Ûªs", "That is", tweet)

    tweet = re.sub(r"here\x89Ûªs", "here is", tweet)

    tweet = re.sub(r"Here\x89Ûªs", "Here is", tweet)

    tweet = re.sub(r"you\x89Ûªre", "you are", tweet)

    tweet = re.sub(r"You\x89Ûªre", "You are", tweet)

    tweet = re.sub(r"you\x89Ûªve", "you have", tweet)

    tweet = re.sub(r"You\x89Ûªve", "You have", tweet)

    tweet = re.sub(r"you\x89Ûªll", "you will", tweet)

    tweet = re.sub(r"You\x89Ûªll", "You will", tweet)

    tweet = re.sub(r"China\x89Ûªs", "China's", tweet)

    tweet = re.sub(r"doesn\x89Ûªt", "does not", tweet)

    tweet = re.sub(r"wouldn\x89Ûªt", "would not", tweet)

    tweet = re.sub(r"\x89Û_", "", tweet)

    tweet = re.sub(r"\x89Û¢", "", tweet)

    tweet = re.sub(r"\x89ÛÒ", "", tweet)

    tweet = re.sub(r"\x89ÛÓ", "", tweet)

    tweet = re.sub(r"\x89ÛÏ", "", tweet)

    tweet = re.sub(r"\x89Û÷", "", tweet)

    tweet = re.sub(r"\x89Ûª", "", tweet)

    tweet = re.sub(r"\x89Û¢åÊ", "", tweet)

    tweet = re.sub(r"\x89Û\x9d", "", tweet)

    tweet = re.sub(r"å_", "", tweet)

    tweet = re.sub(r"å¨", "", tweet)

    tweet = re.sub(r"åÀ", "", tweet)

    tweet = re.sub(r"åÇ", "", tweet)

    tweet = re.sub(r"åÊ", "", tweet)

    tweet = re.sub(r"åÈ", "", tweet)  

    tweet = re.sub(r"Ì©", "", tweet)

    

    # Character entity references

    tweet = re.sub(r"&lt;", "<", tweet)

    tweet = re.sub(r"&gt;", ">", tweet)

    tweet = re.sub(r"&amp;", "&", tweet)

    return tweet



# Removes non-ASCII characters

def remove_nonASCII(tweet):

    tweet = ''.join([x for x in tweet if x in string.printable])

    return tweet
def expand_contractions(tweet):

    

    tweet = re.sub(r"I'm", "I am", tweet)

    tweet = re.sub(r"I'M", "I am", tweet)

    tweet = re.sub(r"i'm", "I am", tweet)

    tweet = re.sub(r"i'M", "I am", tweet)

    tweet = re.sub(r"i'd", "I would", tweet)

    tweet = re.sub(r"I'd", "I would", tweet)

    tweet = re.sub(r"i'll", "I will", tweet)

    tweet = re.sub(r"I'll", "I will", tweet)

    tweet = re.sub(r"i've", "I have", tweet)

    tweet = re.sub(r"I've", "I have", tweet)

    tweet = re.sub(r"you're", "you are", tweet)

    tweet = re.sub(r"You're", "You are", tweet)

    tweet = re.sub(r"you'd", "you would", tweet)

    tweet = re.sub(r"You'd", "You would", tweet)

    tweet = re.sub(r"you've", "you have", tweet)

    tweet = re.sub(r"You've", "You have", tweet)

    tweet = re.sub(r"you'll", "you will", tweet)

    tweet = re.sub(r"You'll", "You will", tweet)  

    tweet = re.sub(r"y'know", "you know", tweet)  

    tweet = re.sub(r"Y'know", "You know", tweet)  

    tweet = re.sub(r"y'all", "you all", tweet)

    tweet = re.sub(r"Y'all", "You all", tweet)

    tweet = re.sub(r"we're", "we are", tweet)

    tweet = re.sub(r"We're", "We are", tweet)

    tweet = re.sub(r"we've", "we have", tweet)

    tweet = re.sub(r"We've", "We have", tweet) 

    tweet = re.sub(r"we'd", "we would", tweet)

    tweet = re.sub(r"We'd", "We would", tweet)

    tweet = re.sub(r"WE'VE", "We have", tweet)

    tweet = re.sub(r"we'll", "we will", tweet)

    tweet = re.sub(r"We'll", "We will", tweet)

    tweet = re.sub(r"they're", "they are", tweet)

    tweet = re.sub(r"They're", "They are", tweet)

    tweet = re.sub(r"they'd", "they would", tweet)

    tweet = re.sub(r"They'd", "They would", tweet)  

    tweet = re.sub(r"they've", "they have", tweet)

    tweet = re.sub(r"They've", "They have", tweet)

    tweet = re.sub(r"they'll", "they will", tweet)

    tweet = re.sub(r"They'll", "They will", tweet)

    tweet = re.sub(r"he's", "he is", tweet)

    tweet = re.sub(r"He's", "He is", tweet)

    tweet = re.sub(r"he'll", "he will", tweet)

    tweet = re.sub(r"He'll", "He will", tweet)

    tweet = re.sub(r"she's", "she is", tweet)

    tweet = re.sub(r"She's", "She is", tweet)

    tweet = re.sub(r"she'll", "she will", tweet)

    tweet = re.sub(r"She'll", "She will", tweet)

    tweet = re.sub(r"it's", "it is", tweet)

    tweet = re.sub(r"It's", "It is", tweet)

    tweet = re.sub(r"it'll", "it will", tweet)

    tweet = re.sub(r"It'll", "It will", tweet)

    tweet = re.sub(r"isn't", "is not", tweet)

    tweet = re.sub(r"Isn't", "Is not", tweet)

    tweet = re.sub(r"who's", "who is", tweet)

    tweet = re.sub(r"Who's", "Who is", tweet)

    tweet = re.sub(r"what's", "what is", tweet)

    tweet = re.sub(r"What's", "What is", tweet)

    tweet = re.sub(r"that's", "that is", tweet)

    tweet = re.sub(r"That's", "That is", tweet)

    tweet = re.sub(r"here's", "here is", tweet)

    tweet = re.sub(r"Here's", "Here is", tweet)

    tweet = re.sub(r"there's", "there is", tweet)

    tweet = re.sub(r"There's", "There is", tweet)

    tweet = re.sub(r"where's", "where is", tweet)

    tweet = re.sub(r"Where's", "Where is", tweet)  

    tweet = re.sub(r"wHeRE's", "where is", tweet)  

    tweet = re.sub(r"how's", "how is", tweet)  

    tweet = re.sub(r"How's", "How is", tweet)  

    tweet = re.sub(r"how're", "how are", tweet)  

    tweet = re.sub(r"How're", "How are", tweet) 

    tweet = re.sub(r"let's", "let us", tweet)

    tweet = re.sub(r"Let's", "Let us", tweet)

    tweet = re.sub(r"won't", "will not", tweet)

    tweet = re.sub(r"wasn't", "was not", tweet)

    tweet = re.sub(r"aren't", "are not", tweet)

    tweet = re.sub(r"couldn't", "could not", tweet)

    tweet = re.sub(r"shouldn't", "should not", tweet)

    tweet = re.sub(r"haven't", "have not", tweet)

    tweet = re.sub(r"Haven't", "Have not", tweet)

    tweet = re.sub(r"hasn't", "has not", tweet)

    tweet = re.sub(r"wouldn't", "would not", tweet)

    tweet = re.sub(r"weren't", "were not", tweet)

    tweet = re.sub(r"Weren't", "Were not", tweet)

    tweet = re.sub(r"ain't", "am not", tweet)

    tweet = re.sub(r"Ain't", "am not", tweet)

    tweet = re.sub(r"don't", "do not", tweet)

    tweet = re.sub(r"Don't", "do not", tweet)

    tweet = re.sub(r"DON'T", "Do not", tweet)

    tweet = re.sub(r"didn't", "did not", tweet)

    tweet = re.sub(r"Didn't", "Did not", tweet)

    tweet = re.sub(r"DIDN'T", "Did not", tweet)

    tweet = re.sub(r"doesn't", "does not", tweet)

    tweet = re.sub(r"can't", "cannot", tweet)

    tweet = re.sub(r"Can't", "Cannot", tweet)

    tweet = re.sub(r"Could've", "Could have", tweet)

    tweet = re.sub(r"should've", "should have", tweet)

    tweet = re.sub(r"would've", "would have", tweet)

    

    return tweet
def specific_corrections(tweet):

    

    '''Typos, slang and informal abbreviations'''

    

    tweet = re.sub(r"b/c", "because", tweet)

    tweet = re.sub(r"w/e", "whatever", tweet)

    tweet = re.sub(r"w/out", "without", tweet)

    tweet = re.sub(r"w/o", "without", tweet)

    tweet = re.sub(r"w/", "with ", tweet)   

    tweet = re.sub(r"<3", "love", tweet)

    tweet = re.sub(r"c/o", "care of", tweet)

    tweet = re.sub(r"p/u", "pick up", tweet)

    tweet = re.sub(r"\n", " ", tweet)

   

    # Typos

    tweet = re.sub(r"Trfc", "Traffic", tweet)

    tweet = re.sub(r"recentlu", "recently", tweet)

    tweet = re.sub(r"Ph0tos", "Photos", tweet)

    tweet = re.sub(r"exp0sed", "exposed", tweet)

    tweet = re.sub(r"amageddon", "armageddon", tweet)

    tweet = re.sub(r"TRAUMATISED", "traumatized", tweet)

    tweet = re.sub(r"Newss", "News", tweet)

    tweet = re.sub(r"remedyyyy", "remedy", tweet)

    tweet = re.sub(r"Bstrd", "bastard", tweet)

    tweet = re.sub(r"bldy", "bloody", tweet)

    tweet = re.sub(r"epicenterr", "epicenter", tweet)

    tweet = re.sub(r"approachng", "approaching", tweet)

    tweet = re.sub(r"evng", "evening", tweet)

    tweet = re.sub(r"Sumthng", "something", tweet)

    tweet = re.sub(r"kostumes", "costumes", tweet)

    tweet = re.sub(r"glowng", "glowing", tweet)

    tweet = re.sub(r"kindlng", "kindling", tweet)

    tweet = re.sub(r"riggd", "rigged", tweet)

    tweet = re.sub(r"HLPS", "helps", tweet)

    tweet = re.sub(r"SNCTIONS", "sanctions", tweet)

    tweet = re.sub(r"Politifiact", "PolitiFact", tweet)

    tweet = re.sub(r"Kowing", "Knowing", tweet)

    tweet = re.sub(r"wrld", "world", tweet)   

    tweet = re.sub(r"shld", "should", tweet)    

    tweet = re.sub(r"thruuu", "through", tweet)

    tweet = re.sub(r"probaly", "probably", tweet)

    tweet = re.sub(r"whatevs", "whatever", tweet)

    tweet = re.sub(r"colomr", "colour", tweet)

    tweet = re.sub(r"pileq", "pile", tweet)

    tweet = re.sub(r"firefightr", "firefighter", tweet)

    tweet = re.sub(r"LAIGHIGN", "laughing", tweet)

    tweet = re.sub(r"EXCLUSIV", "Exclusive", tweet) 

    tweet = re.sub(r"belo-ooow", "below", tweet)  

    tweet = re.sub(r"who-ooo-ole", "whole", tweet)  

    tweet = re.sub(r"brother-n-law", "father-in-law", tweet)  

    tweet = re.sub(r"referencereference", "reference", tweet)

    

    # Hashtags and usernames

    tweet = re.sub(r"IranDeal", "Iran Deal", tweet)

    tweet = re.sub(r"ProphetMuhammad", "Prophet Muhammad", tweet)

    tweet = re.sub(r"StrategicPatience", "Strategic Patience", tweet)

    tweet = re.sub(r"NASAHurricane", "NASA Hurricane", tweet)

    tweet = re.sub(r"onlinecommunities", "online communities", tweet)

    tweet = re.sub(r"LakeCounty", "Lake County", tweet)

    tweet = re.sub(r"thankU", "thank you", tweet)

    tweet = re.sub(r"iTunesMusic", "iTunes Music", tweet)

    tweet = re.sub(r"OffensiveContent", "Offensive Content", tweet)

    tweet = re.sub(r"WorstSummerJob", "Worst Summer Job", tweet)

    tweet = re.sub(r"NASASolarSystem", "NASA Solar System", tweet)

    tweet = re.sub(r"animalrescue", "animal rescue", tweet)

    tweet = re.sub(r"Ptbo", "Peterborough", tweet)

    tweet = re.sub(r"Throwingknifes", "Throwing knives", tweet)

    tweet = re.sub(r"NestleIndia", "Nestle India", tweet)

    tweet = re.sub(r"weathernetwork", "weather network", tweet)

    tweet = re.sub(r"GOPDebate", "GOP Debate", tweet)

    tweet = re.sub(r"volcanoinRussia", "volcano in Russia", tweet)

    tweet = re.sub(r"53inch", "53 inch", tweet)

    tweet = re.sub(r"FaroeIslands", "Faroe Islands", tweet)

    tweet = re.sub(r"UTC2015", "UTC 2015", tweet)

    tweet = re.sub(r"Time2015", "Time 2015", tweet)

    tweet = re.sub(r"LivingSafely", "Living Safely", tweet)

    tweet = re.sub(r"FIFA16", "Fifa 2016", tweet)

    tweet = re.sub(r"bbcnews", "bbc news", tweet)

    tweet = re.sub(r"UndergroundRailraod", "Underground Railraod", tweet)

    tweet = re.sub(r"NoSurrender", "No Surrender", tweet)

    tweet = re.sub(r"greatbritishbakeoff", "great british bake off", tweet)

    tweet = re.sub(r"LondonFire", "London Fire", tweet)

    tweet = re.sub(r"KOTAWeather", "KOTA Weather", tweet)

    tweet = re.sub(r"LuchaUnderground", "Lucha Underground", tweet)

    tweet = re.sub(r"KOIN6News", "KOIN 6 News", tweet)

    tweet = re.sub(r"9NewsGoldCoast", "9 News Gold Coast", tweet)

    tweet = re.sub(r"BlackLivesMatter", "Black Lives Matter", tweet)

    tweet = re.sub(r"ENGvAUS", "England vs Australia", tweet)

    tweet = re.sub(r"PlannedParenthood", "Planned Parenthood", tweet)

    tweet = re.sub(r"calgaryweather", "Calgary Weather", tweet)

    tweet = re.sub(r"renew911health", "renew 911 health", tweet)

    tweet = re.sub(r"pdx911", "Portland Police", tweet)

    tweet = re.sub(r"NJTurnpike", "New Jersey Turnpike", tweet)

    tweet = re.sub(r"HannaPH", "Typhoon Hanna", tweet)

    tweet = re.sub(r"cnnbrk", "CNN Breaking News", tweet)

    tweet = re.sub(r"IndianNews", "Indian News", tweet)

    tweet = re.sub(r"Daesh", "ISIS", tweet)

    tweet = re.sub(r"FoxNew", "Fox News", tweet)

    tweet = re.sub(r"RohnertParkDPS", "Rohnert Park DPS", tweet)

    tweet = re.sub(r"FantasticFour", "Fantastic Four", tweet)

    tweet = re.sub(r"BathAndNorthEastSomerset", "Bath and North East Somerset", tweet)

    tweet = re.sub(r"residualincome", "residual income", tweet)

    tweet = re.sub(r"YahooNewsDigest", "Yahoo News Digest", tweet)

    tweet = re.sub(r"MalaysiaAirlines", "Malaysia Airlines", tweet)

    tweet = re.sub(r"AmazonDeals", "Amazon Deals", tweet)

    tweet = re.sub(r"EndConflict", "End Conflict", tweet)

    tweet = re.sub(r"EndOccupation", "End Occupation", tweet)

    tweet = re.sub(r"KindleCountdown", "Kindle Countdown", tweet)

    tweet = re.sub(r"NoMoreHandouts", "No More Handouts", tweet)

    tweet = re.sub(r"WindstormInsurer", "Windstorm Insurer", tweet)

    tweet = re.sub(r"USAgov", "USA government", tweet)

    tweet = re.sub(r"US govt", "USA government", tweet)  

    tweet = re.sub(r"WAwildfire", "WA Wildfire", tweet)

    tweet = re.sub(r"fingerrockfire", "Finger Rock Fire", tweet)

    tweet = re.sub(r"newnewnew", "new new new", tweet)

    tweet = re.sub(r"freshoutofthebox", "fresh out of the box", tweet)

    tweet = re.sub(r"yycweather", "Calgary Weather", tweet)

    tweet = re.sub(r"calgarysun", "Calgary Sun", tweet)

    tweet = re.sub(r"shondarhimes", "Shonda Rhimes", tweet)

    tweet = re.sub(r"SushmaSwaraj", "Sushma Swaraj", tweet)

    tweet = re.sub(r"pray4japan", "Pray for Japan", tweet)

    tweet = re.sub(r"hope4japan", "Hope for Japan", tweet)

    tweet = re.sub(r"Illusionimagess", "Illusion images", tweet)

    tweet = re.sub(r"ShallWeDance", "Shall We Dance", tweet)

    tweet = re.sub(r"TCMParty", "TCM Party", tweet)

    tweet = re.sub(r"marijuananews", "marijuana news", tweet)

    tweet = re.sub(r"HeadlinesApp", "Headlines App", tweet)

    tweet = re.sub(r"BBCNewsAsia", "BBC News Asia", tweet)

    tweet = re.sub(r"BombEffects", "Bomb Effects", tweet)

    tweet = re.sub(r"idkidk", "idk idk", tweet)

    tweet = re.sub(r"BBCLive", "BBC Live", tweet)

    tweet = re.sub(r"NaturalBirth", "Natural Birth", tweet)

    tweet = re.sub(r"FusionFestival", "Fusion Festival", tweet)

    tweet = re.sub(r"50Mixed", "50 Mixed", tweet)

    tweet = re.sub(r"NoAgenda", "No Agenda", tweet)

    tweet = re.sub(r"WhiteGenocide", "White Genocide", tweet)

    tweet = re.sub(r"dirtylying", "dirty lying", tweet)

    tweet = re.sub(r"SyrianRefugees", "Syrian Refugees", tweet)

    tweet = re.sub(r"Auspol", "Australia Politics", tweet)

    tweet = re.sub(r"WhiteTerrorism", "White Terrorism", tweet)

    tweet = re.sub(r"truthfrequencyradio", "Truth Frequency Radio", tweet)

    tweet = re.sub(r"ErasureIsNotEquality", "Erasure is not equality", tweet)

    tweet = re.sub(r"toopainful", "too painful", tweet)

    tweet = re.sub(r"melindahaunton", "Melinda Haunton", tweet)

    tweet = re.sub(r"NoNukes", "No Nukes", tweet)

    tweet = re.sub(r"curryspcworld", "Currys PC World", tweet)

    tweet = re.sub(r"blackforestgateau", "black forest gateau", tweet)

    tweet = re.sub(r"BBCOne", "BBC One", tweet)

    tweet = re.sub(r"sebastianstanisaliveandwell", "Sebastian Stan is alive and well", tweet)

    tweet = re.sub(r"concertphotography", "concert photography", tweet)

    tweet = re.sub(r"TheaterTrial", "Theater Trial", tweet)

    tweet = re.sub(r"TheBrooklynLife", "The Brooklyn Life", tweet)

    tweet = re.sub(r"jokethey", "joke they", tweet)

    tweet = re.sub(r"nflweek1picks", "NFL week 1 picks", tweet)

    tweet = re.sub(r"nflnetwork", "NFL Network", tweet)

    tweet = re.sub(r"NYDNSports", "NY Daily News Sports", tweet)

    tweet = re.sub(r"crunchysensible", "crunchy sensible", tweet)

    tweet = re.sub(r"RandomActsOfRomance", "Random acts of romance", tweet)

    tweet = re.sub(r"MomentsAtHill", "Moments at hill", tweet)

    tweet = re.sub(r"liveleakfun", "live leak fun", tweet)

    tweet = re.sub(r"SahelNews", "Sahel News", tweet)

    tweet = re.sub(r"abc7newsbayarea", "ABC 7 News Bay Area", tweet)

    tweet = re.sub(r"CampLogistics", "Camp logistics", tweet)

    tweet = re.sub(r"alaskapublic", "Alaska public", tweet)

    tweet = re.sub(r"MarketResearch", "Market Research", tweet)

    tweet = re.sub(r"AccuracyEsports", "Accuracy Esports", tweet)

    tweet = re.sub(r"yychail", "Calgary hail", tweet)

    tweet = re.sub(r"yyctraffic", "Calgary traffic", tweet)

    tweet = re.sub(r"eliotschool", "eliot school", tweet)

    tweet = re.sub(r"TheBrokenCity", "The Broken City", tweet)

    tweet = re.sub(r"fieldworksmells", "field work smells", tweet)

    tweet = re.sub(r"IranElection", "Iran Election", tweet)

    tweet = re.sub(r"MyanmarFlood", "Myanmar Flood", tweet)

    tweet = re.sub(r"abc7chicago", "ABC 7 Chicago", tweet)

    tweet = re.sub(r"copolitics", "Colorado Politics", tweet)

    tweet = re.sub(r"massiveflooding", "massive flooding", tweet)

    tweet = re.sub(r"greektheatrela", "Greek Theatre Los Angeles", tweet)

    tweet = re.sub(r"publicsafetyfirst", "public safety first", tweet)

    tweet = re.sub(r"myhometown", "my hometown", tweet)

    tweet = re.sub(r"tankerfire", "tanker fire", tweet)

    tweet = re.sub(r"MEMORIALDAY", "memorial day", tweet)

    tweet = re.sub(r"MEMORIAL_DAY", "memorial day", tweet)

    tweet = re.sub(r"VirtualReality", "Virtual Reality", tweet)

    tweet = re.sub(r"mortalkombatx", "Mortal Kombat X", tweet)

    tweet = re.sub(r"mortalkombat", "Mortal Kombat", tweet)

    tweet = re.sub(r"ToshikazuKatayama", "Toshikazu Katayama", tweet)

    tweet = re.sub(r"ExtremeWeather", "Extreme Weather", tweet)

    tweet = re.sub(r"WereNotGruberVoters", "We are not gruber voters", tweet)

    tweet = re.sub(r"PhiladelphiaMuseu", "Philadelphia Museum", tweet)

    tweet = re.sub(r"NorthIowa", "North Iowa", tweet)

    tweet = re.sub(r"WillowFire", "Willow Fire", tweet)

    tweet = re.sub(r"P_EOPLE", "PEOPLE", tweet)

    tweet = re.sub(r"ThisIsAfrica", "This is Africa", tweet)

    tweet = re.sub(r"viaYouTube", "via YouTube", tweet)

    

    return tweet
def clean_others(tweet):  

    

    tweet = re.sub(r"2007he", "2007 he", tweet)  

    tweet = re.sub(r"Hwy27", "Hwy 27", tweet) 

    tweet = re.sub(r"jokethey", "joke they", tweet)  

    tweet = re.sub(r"40%money", "40% money", tweet)  

    tweet = re.sub(r"hegot", "he got", tweet)

    tweet = re.sub(r"wannabe", "wanna be", tweet) 

    tweet = re.sub(r"dadwho", "dad who", tweet)  

    tweet = re.sub(r"fundwhen", "fund when", tweet)

    tweet = re.sub(r"next chp", "next chapter", tweet)

    tweet = re.sub(r"UR sons", "your sons", tweet)  

    tweet = re.sub(r"Yr voice ws", "Your voice was", tweet) 

    tweet = re.sub(r"U're not", "You are not", tweet)  

    tweet = re.sub(r"u'd win", "you had win", tweet)  

    tweet = re.sub(r"Jus Kame", "Just came", tweet)  

    tweet = re.sub(r"b4federal", "B-4, Federal", tweet) 

    tweet = re.sub(r"ppor child", "poor child", tweet)  

    tweet = re.sub(r"stand ogt", "stand out", tweet)

    tweet = re.sub(r"stand oup", "stand out", tweet) 

    tweet = re.sub(r"IS claims", "ISIS claims", tweet)

    tweet = re.sub(r"2slow2report", "too slow to report", tweet)

    tweet = re.sub(r"@ft", "@Financial Times", tweet)

    tweet = re.sub(r"50ft", "50 ft", tweet)

    tweet = re.sub(r"Ft ABH Shadow", "featuring ABH Shadow", tweet)

    tweet = re.sub(r"Since1970the", "Since 1970 the", tweet) 

    tweet = re.sub(r"whats cracking cuz", "what is cracking cause", tweet) 

    tweet = re.sub(r"mentally ill", "mental illness", tweet)

    tweet = re.sub(r"RIPRIPRIP", "RIP RIP RIP", tweet)

    tweet = re.sub(r"RIPROSS", "RIP ROSS", tweet)  

    tweet = re.sub(r"ABQ NM", "Albuquerque New Mexico", tweet)

    tweet = re.sub(r"#BC", "#British Columbia", tweet)

    tweet = re.sub(r"in BC", "in British Columbia", tweet)

    tweet = re.sub(r"BC DROUGHT", "British Columbia Drought", tweet)

    tweet = re.sub(r"in OK", "in Oklahoma", tweet)

    tweet = re.sub(r"City OK", "City Oklahoma", tweet)

    tweet = re.sub(r"Hinton OK", "Hinton Oklahoma", tweet)

    tweet = re.sub(r"Guthrie OK", "Guthrie Oklahoma", tweet)

    tweet = re.sub(r"Choctaw OK", "Choctaw Oklahoma", tweet)

    tweet = re.sub(r"Oklahoma-OK", "Oklahoma City", tweet)

    tweet = re.sub(r"Oklahoma [OK]", "Oklahoma City", tweet)

    tweet = re.sub(r"JADE FL", "JADE Florida", tweet) 

    tweet = re.sub(r"Jacksonville FL", "Jacksonville Florida", tweet)

    tweet = re.sub(r"Saint Petersburg FL", "Saint Petersburg Florida", tweet)

    tweet = re.sub(r"Wahpeton ND", "Wahpeton, North Dakota", tweet)

    tweet = re.sub(r"Northern Marians", "Northern Mariana Islands", tweet)

    tweet = re.sub(r"Northern Ma", "Northern Mariana Islands", tweet)

    

    # Abbreviation point

    tweet = re.sub(r"Dr\.", "Doctor", tweet)

    tweet = re.sub(r"f\. M\.O\.P\.", "featuring Mash Out Posse", tweet)

    tweet = re.sub(r"M\.O\.P\.", "Mash Out Posse", tweet)

    tweet = re.sub(r"M\.O\.P", "Mash Out Posse", tweet)

    tweet = re.sub(r"P\.O\.P\.E\.", "Pope", tweet)

    tweet = re.sub(r"S\.O\.S\.", "SOS", tweet)

    tweet = re.sub(r"s\.o\.s\.", "SOS", tweet)  

    tweet = re.sub(r"Fire Co\.", "Fire Company", tweet)

    tweet = re.sub(r"Holt and Co\.", "Holt and Company", tweet)

    tweet = re.sub(r"roofing co\.", "roofing company", tweet)

    tweet = re.sub(r"Costa Co\.", "Costa County", tweet)

    tweet = re.sub(r"York Co\.", "York County", tweet)

    tweet = re.sub(r"Fairfax Co\.", "Fairfax County", tweet)

    tweet = re.sub(r"I\.S\.I\.S\.", "ISIS", tweet)

    tweet = re.sub(r"U\.N\.", "United Nations", tweet)

    tweet = re.sub(r"U\.S\.", "United States", tweet)

    tweet = re.sub(r"U\.S", "United States", tweet)

    tweet = re.sub(r"U\.s\.", "United States", tweet)

    tweet = re.sub(r"U\.s", "United States", tweet)

    tweet = re.sub(r"U-S\.", "United States", tweet)

    tweet = re.sub(r"U\.S National", "United States National", tweet)

    tweet = re.sub(r"LANCASTER N\.H\.", "Lancaster New Hampshire", tweet)

    tweet = re.sub(r"Manchester N\.H\.", "Manchester New Hampshire", tweet)

   

    # Normalization

    tweet = re.sub(r"\:33333", "smile", tweet)    # :33333

    tweet = re.sub(r"\:\)\)\)\)", "smile", tweet) # :))))

    tweet = re.sub(r"\:\)\)\)", "smile", tweet) # :)))

    tweet = re.sub(r"\:\)\)", "smile", tweet)   # :))

    tweet = re.sub(r"\:-\)",  "smile", tweet)   # :-)

    tweet = re.sub(r"\;-\)",  "smile", tweet)   # ;-)

    tweet = re.sub(r"3\-D", "smile", tweet)  # 3-D

    tweet = re.sub(r"\:O", "smile", tweet)   # :O

    tweet = re.sub(r"\:D", "smile", tweet)   # :D

    tweet = re.sub(r"\:P", "smile", tweet)   # :P

    tweet = re.sub(r"\:p", "smile", tweet)   # :p

    tweet = re.sub(r"\;\)", "smile", tweet)  # ;)

    tweet = re.sub(r"\:\)", "smile", tweet)  # :)

    tweet = re.sub(r"\=\)", "smile", tweet)  # =)

    tweet = re.sub(r"\^\^", "smile", tweet)  # ^^

    tweet = re.sub(r"\:-\(", "sad", tweet)   # :-(

    tweet = re.sub(r"\:\(", "sad", tweet)    # :(

    tweet = re.sub(r"\=\(", "sad", tweet)    # =(

    tweet = re.sub(r"\-\_\_\-", "", tweet)   # -__-

    tweet = re.sub(r"\.\_\.", "", tweet)     # ._.

    tweet = re.sub(r"T\_T", "", tweet)       # T_T

    

    return tweet
# Normalization

abbreviations = {

    

    "i.e":"that is", "mofo":"mother fucker", "til":"till",

    "ft.":"featuring", "mf":"mother fucker", "bout":"about",

    "ft":"featuring", "mfs":"mother fucker", "nd":"and", 

    "feat.":"featuring", "ltd":"limited", "nvr":"never",

    "feat":"featuring", "pls":"please", "ppl":"people",

    "tbs":"tablespoons", "tho":"though", "fav":"favorite",

    "bc":"because", "cuz":"because", "bcuz":"because",

    "btwn":"between", "fwy":"Freeway", "hwy":"Highway",

    "diff":"different", "appx":"approximately", 

    "im":"I am", "ive":"I have", "uve":"you have", 

    "youd":"you had", "hadnt":"had not", "isnt":"is not",

    "dont":"do not", "didnt":"did not", "cant":"cannot",

    "urself":"yourself", "wont":"would not", 

    "heres":"Here is", "lets":"Let us", "2day":"today", 

    "s2g":"swear to god", "be4":"before", "b4":"before", 

    "4the":"for the", "1st":"first",

   

    # location

    "okwx":"Oklahoma Weather", "arwx":"Arkansas Weather",    

    "gawx":"Georgia Weather", "cawx":"California Weather",

    "tnwx":"Tennessee Weather", "azwx":"Arizona Weather",  

    "alwx":"Alabama Weather", "scwx":"South Carolina Weather",

    "isis":"Islamic State", "okc":"Oklahoma","oun":"Oklahoma",

    "isil":"Islamic State", "suruc":"Urfa", "pdx":"Portland", 

    "nm":"New Mexico", "newyork":"New York", "alska":"Alaska",

    "nh":"New Hampshire", "nyc":"New York City",

    "cnmi":"Northern Mariana Islands", "calif":"California",

    "sarabia":"Saudi Arabia", "saudiarabia":"Saudi Arabia", 

    "mh370":"Malaysia Airlines Flight 370", 

    

    # units

    "12hr":"12 hr","16yr":"16 year", "hrs":"hour","hr":"hour",

    "19yrs":"19 year", "yrs":"year", "min":"minute", 

    "20yrs":"20 year", "yr":"year", "mins":"minute", 

    

    # Typos

    "tren":"trend", "kno":"know", "swea":"swear", "stil":"still",

    "fab":"fabulous", "srsly":"seriously", "epicente":"epicenter", 

    "jumpin":"jumping", "burnin":"burning", "throwin":"throwing",

    "killin":"killing", "nothin":"nothing", "thinkin":"thinking",  

    "tryin":"trying", "lookg":"looking", "fforecast":"Forecast",

    "comin":"Coming", "newss":"news", "memez":"meme", "oli":"oil",

}



def convert_abbrev(word):

    return abbreviations[word.lower()] if word.lower() in abbreviations.keys() else word



def convert_abbrev_in_text(text):

    tokens = word_tokenize(text)

    tokens = [convert_abbrev(word) for word in tokens]

    text = ' '.join(tokens)

    return text
# Remove unwanted words

def remove_non_alnum(tweet):

    punctuation = re.compile('[^A-Za-z0-9]+')

    return punctuation.sub(r' ',tweet)



# Remove punctuations.

def remove_punct(text):

    table = str.maketrans('', '', string.punctuation)

    return text.translate(table)



# Remove leading, trailing, and extra spaces

def remove_extra_spaces(text):

    text = re.sub('\s+', ' ', text).strip() 

    return text
preprocessed_tweets_train = [] 

for tweet in tqdm_notebook(train_df['text'].values):

    tweet = remove_url(tweet)

    tweet = remove_html(tweet)

    tweet = remove_emoji(tweet)

    tweet = special_characters(tweet)

    tweet = remove_nonASCII(tweet)

    tweet = expand_contractions(tweet)

    tweet = specific_corrections(tweet)

    tweet = remove_html(tweet)

    tweet = clean_others(tweet)

    tweet = convert_abbrev_in_text(tweet)

    tweet = remove_punct(tweet)

    tweet = remove_non_alnum(tweet)

    tweet = remove_extra_spaces(tweet)

    preprocessed_tweets_train.append(tweet.strip())

    

train_df['text'] = preprocessed_tweets_train
preprocessed_tweets_test = []  

for tweet in tqdm_notebook(test_df['text'].values):

    tweet = remove_url(tweet)

    tweet = remove_html(tweet)

    tweet = remove_emoji(tweet)

    tweet = special_characters(tweet)

    tweet = remove_nonASCII(tweet)

    tweet = expand_contractions(tweet)

    tweet = specific_corrections(tweet)

    tweet = remove_html(tweet)

    tweet = clean_others(tweet)

    tweet = convert_abbrev_in_text(tweet)

    tweet = remove_punct(tweet)

    tweet = remove_non_alnum(tweet)

    tweet = remove_extra_spaces(tweet)

    preprocessed_tweets_test.append(tweet.strip())

    

test_df['text'] = preprocessed_tweets_test
df_mislabeled = train_df.groupby(['text']).nunique().sort_values(by='target', ascending=False)

df_mislabeled = df_mislabeled[df_mislabeled['target'] > 1]['target']

df_mislabeled_all = df_mislabeled.index.tolist()

print(f'Number of repeated tweets(after preprocessing): {len(df_mislabeled_all)}')

df_mislabeled_all
train_df['target_new'] = train_df['target'].copy()  



target_1_list = [      

    "POTUS Strategic Patience is a strategy for Genocide refugees IDP Internally displaced people horror etc",

    "CLEARED incident with injury I 495 inner loop Exit 31 MD 97 Georgia Ave Silver Spring",

    "RT NotExplained The only known image of infamous hijacker D B Cooper",

    "wowo 12000 Nigerian refugees repatriated from Cameroon", 

    "Bayelsa poll Tension in Bayelsa as Patience Jonathan plans to hijack APC PDP Plans by former First Lady and",

    "hot C 130 specially modified to land in a stadium and rescue hostages in Iran in 1980 prebreak best",

    "world FedEx no longer to transport bioterror germs in wake of anthrax lab mishaps",

    "FedEx no longer to transport bioterror germs in wake of anthrax lab mishaps",

    "FedEx no longer to transport bioterror germs in wake of anthrax lab mishaps via usatoday",

    "Governor weighs parole for California school bus hijacker",

    "Kosciusko police investigating pedestrian fatality hit by a train Thursday", 

    "A look at state actions a year after Ferguson s upheaval", 

    "Here is how media in Pakistan covered the capture of terrorist Mohammed Naved" ]



for mislabeled_sample in df_mislabeled_all:

    if mislabeled_sample in target_1_list:

        train_df.loc[train_df['text'] == mislabeled_sample, 'target_new'] = 1

    else:

        train_df.loc[train_df['text'] == mislabeled_sample, 'target_new'] = 0



filter_mislabel = (train_df['target'] != train_df['target_new'])

print(f'Number of relabeled: {len(train_df[filter_mislabel])}')

train_df[filter_mislabel][:20]
# Remove stopwords.

stop_words = set(stopwords.words("english"))

def remove_stopwords(tweet):

    sentance = ' '.join(e.lower() for e in tweet.split() if e.lower() not in stop_words)

    return sentance



# Stemming words

stemmer = SnowballStemmer("english")

def stemming(text):    

    text = [stemmer.stem(word) for word in text.split()]

    return " ".join(text)



# Lemmatizing

wn = nltk.WordNetLemmatizer()

def lemmatizing(text):    

    text = [wn.lemmatize(word.lower()) for word in text.split()]

    return " ".join(text)



train_df['text_pure'] = train_df['text'].apply(lambda x: remove_stopwords(x))

train_df['text_pure'] = train_df['text_pure'].apply(lambda x: stemming(x))
df_mislabeled = train_df.groupby(['text_pure']).nunique().sort_values(by='target_new', ascending=False)

df_mislabeled = df_mislabeled[df_mislabeled['target_new'] > 1]['target_new']

df_mislabeled_all = df_mislabeled.index.tolist()

print(f'Number of repeated tweets (after Stemming): {len(df_mislabeled_all)}')

df_mislabeled_all
# The texts "look state action year ferguson upheav" should be marked as 1

train_df.loc[train_df['text_pure'] == "look state action year ferguson upheav", 'target_new'] = 1

df_mislabeled = train_df[train_df['text_pure'].isin(df_mislabeled_all)]

filter_mislabel = (df_mislabeled['target'] != df_mislabeled['target_new'])

print(f'Number of relabeled: {len(df_mislabeled[filter_mislabel])}')

df_mislabeled
train_df = train_df.drop('text_pure', axis=1)
ids_target1_error = [

    328,443,513,791,794,882,883,886,890,893,894,896,923,926,928,1688,1709,2033,

    2063,2619,2885,3097,3640,3802,3837,3842,3900,4026,4342,4530,4533,4575,4773,

    4778,4790,5223,5781,6552,6554,6570,6701,6702,6729,6731,6745,6861,6945,6965,

    7201,7226,7231,7264,7494,7797,8309,8317,8329,8330,8905,8908,8913,8916,8926,

    8934,8939,8972,9337,9446,9775,9791,9808,10127,10543,10552 ]



print(f'Number of ids with target1 error: {len(ids_target1_error)}')
if ids_error_corrected:

    train_df.at[train_df['id'].isin(ids_target1_error),'target_new'] = 0

train_df[train_df['id'].isin(ids_target1_error)][:20]
ids_target0_error = [

    832,833,836,841,851,859,860,864,868,874,878,903,5990,6002,6188,6192,6211]



print(f'Length of ids_error_target0: {len(ids_target0_error)}')
if ids_error_corrected:

    train_df.at[train_df['id'].isin(ids_target0_error),'target_new'] = 1

train_df[train_df['id'].isin(ids_target0_error)]
# Displaying target distribution.

fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(12, 4), dpi=100)

sns.countplot(train_df['target_new'], ax=axes[0])

axes[1].pie(train_df['target_new'].value_counts(),

            labels=['Not Disaster', 'Disaster'],

            autopct='%1.2f%%',

            shadow=True,

            explode=(0.05, 0),

            startangle=60)

fig.suptitle('Distribution of the Tweets', fontsize=24)

plt.show()
# Comparing word counts Word Counts

def plot_word_number_histogram(textno, textyes):



    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(13, 5), sharey=True)

    sns.distplot(textno.str.split().map(lambda x: len(x)), ax=axes[0], color='#e74c3c')

    sns.distplot(textyes.str.split().map(lambda x: len(x)), ax=axes[1], color='#e74c3c')

    

    axes[0].set_xlabel('Word Count')

    axes[0].set_ylabel('Frequency')

    axes[0].set_title('Non Disaster Tweets')

    axes[1].set_xlabel('Word Count')

    axes[1].set_title('Disaster Tweets')

    

    fig.suptitle('Words Per Tweet', fontsize=24, va='baseline')

    fig.tight_layout()

    

# number of words per tweet

textno = train_df[train_df['target_new'] == 0]

textyes = train_df[train_df['target_new'] == 1]

plot_word_number_histogram(textno['text'], textyes['text'])
# Comparing average Word Length

def plot_word_len_histogram(textno, textyes):



    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(13, 5), sharey=True)

    sns.distplot(textno.str.split().apply(

                 lambda x: [len(i) for i in x]).map(

                 lambda x: np.mean(x)), ax=axes[0], color='#e74c3c')

    sns.distplot(textyes.str.split().apply(

                 lambda x: [len(i) for i in x]).map(

                 lambda x: np.mean(x)), ax=axes[1], color='#e74c3c')

    axes[0].set_xlabel('Word Length')

    axes[0].set_ylabel('Frequency')

    axes[0].set_title('Non Disaster Tweets')

    axes[1].set_xlabel('Word Length')

    axes[1].set_title('Disaster Tweets')

    fig.suptitle('Mean Word Lengths', fontsize=24, va='baseline')

    fig.tight_layout()

    

plot_word_len_histogram(textno['text'], textyes['text'])
# Displaying most common words.

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

axes = axes.flatten()



train_df['text_lemma'] = train_df['text'].apply(lambda x: remove_stopwords(x))

train_df['text_lemma'] = train_df['text_lemma'].apply(lambda x: lemmatizing(x))



lis = [train_df[train_df['target_new'] == 0]['text_lemma'],

       train_df[train_df['target_new'] == 1]['text_lemma']]



for i, j in zip(lis, axes):

    new = i.str.split()

    new = new.values.tolist()

    corpus = [word for i in new for word in i]

    counter = Counter(corpus)

    most = counter.most_common()

    x, y = [], []

    for word, count in most[:30]:

        if (word not in stop_words):

            x.append(word)

            y.append(count)

    sns.barplot(x=y, y=x, palette='plasma', ax=j)

    

axes[0].set_title('Non Disaster Tweets')

axes[1].set_title('Disaster Tweets')

axes[0].set_xlabel('Count')

axes[0].set_ylabel('Word')

axes[1].set_xlabel('Count')

axes[1].set_ylabel('Word')



fig.suptitle('Most Common Unigrams', fontsize=24, va='baseline')

plt.tight_layout()
# plot most common ngrams

def ngrams(n, title):

    

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    axes = axes.flatten()

    for i, j in zip(lis, axes):

        new = i.str.split()

        new = new.values.tolist()

        corpus = [word for i in new for word in i]



        def _get_top_ngram(corpus, n=None):

            #getting top ngrams

            vec = CountVectorizer(ngram_range=(n, n), max_df=0.9,

                                  stop_words='english').fit(corpus)

            bag_of_words = vec.transform(corpus)

            sum_words = bag_of_words.sum(axis=0)

            words_freq = [(word, sum_words[0, idx])

                          for word, idx in vec.vocabulary_.items()]

            words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

            return words_freq[:15]



        top_n_bigrams = _get_top_ngram(i, n)[:15]

        x, y = map(list, zip(*top_n_bigrams))

        sns.barplot(x=y, y=x, palette='plasma', ax=j)

        axes[0].set_title('Non Disaster Tweets')

        axes[1].set_title('Disaster Tweets')

        axes[0].set_xlabel('Count')

        axes[0].set_ylabel('Words')

        axes[1].set_xlabel('Count')

        axes[1].set_ylabel('Words')

        fig.suptitle(title, fontsize=24, va='baseline')

        plt.tight_layout()
# Bigrams

ngrams(2, 'Most Common Bigrams')
# Trigrams

ngrams(3, 'Most Common Trigrams')
# We will use the official tokenization script created by the Google team

!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py

import tokenization

import tensorflow as tf

import tensorflow_hub as hub

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint
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

    

    if Dropout_num == 0:

        # Without Dropout

        out = Dense(1, activation='sigmoid')(clf_output)

    else:

        # With Dropout(Dropout_num), Dropout_num > 0

        x = Dropout(Dropout_num)(clf_output)

        out = Dense(1, activation='sigmoid')(x)



    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

    model.compile(Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    

    return model
# Load BERT from the Tensorflow Hub

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"

bert_layer = hub.KerasLayer(module_url, trainable=True)



# Load tokenizer from the bert layer

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
# Encode the text into tokens, masks, and segment flags  

train_input = bert_encode(train_df['text'].values, tokenizer, max_len=Max_length)

test_input = bert_encode(test_df['text'].values, tokenizer, max_len=Max_length)

train_labels = train_df['target_new'].values
# Build BERT model with my tuning

model = build_model(bert_layer, max_len=Max_length)  

model.summary()
# Train BERT model with my tuning

checkpoint = ModelCheckpoint('model_BERT.h5', monitor='val_loss', save_best_only=True)



train_history = model.fit(

    train_input, train_labels,

    validation_split = valid,

    epochs = epochs_num, # recomended 3-5 epochs

    callbacks=[checkpoint],

    batch_size = batch_size_num

)
model.load_weights('model_BERT.h5')



# for the testing data

test_pred = model.predict(test_input)

test_pred_int = test_pred.round().astype('int')



# for the training data - for the Confusion Matrix

train_pred = model.predict(train_input)

train_pred_int = train_pred.round().astype('int')
# Showing Confusion Matrix

def plot_cm(y_true, y_pred, title, figsize=(5,5)):

    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))

    cm_sum = np.sum(cm, axis=1, keepdims=True)

    cm_perc = cm / cm_sum.astype(float) * 100

    annot = np.empty_like(cm).astype(str)

    nrows, ncols = cm.shape

    for i in range(nrows):

        for j in range(ncols):

            c = cm[i, j]

            p = cm_perc[i, j]

            if i == j:

                s = cm_sum[i]

                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

            elif c == 0: annot[i, j] = ''

            else: annot[i, j] = '%.1f%%\n%d' % (p, c)

    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))

    cm.index.name = 'Actual'

    cm.columns.name = 'Predicted'

    fig, ax = plt.subplots(figsize=figsize)

    plt.title(title)

    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)

    

# Confusion Matrix (Original target)

plot_cm(train_df.target.values, train_pred_int, 'Confusion matrix(Original target)', figsize=(6,6))
# Confusion Matrix (target relabeled)

plot_cm(train_df.target_new.values, train_pred_int, 'Confusion matrix(target relabeled)', figsize=(6,6))
plt.style.use('ggplot') 

pred = pd.DataFrame(test_pred, columns=['preds'])

pred.plot.hist()
# Submission by BERT

submission['target'] = test_pred_int

submission.to_csv("submission_final.csv", index=False)

submission.head(5)
def encode(text, tokenizer, max_len=512):

    

    all_tokens = []

    all_masks = []

    all_segments = []

    

    text = tokenizer.tokenize(str(text))  

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
ResultDict={1:'Disaster', 0:'Non Disaster'}   

def predict_review(input_text):

    input_seq = encode(input_text, tokenizer, max_len=Max_length)

    predict_result = model.predict(input_seq)

    i = predict_result[0][0].round().astype('int')

    print('Input:', input_text) 

    pre_score = round(float(predict_result[0][0])*100, 4)

    print(f'Output: {ResultDict[i]} ({pre_score}%)\n')
userInput1 = '''Fire shuts down part of NJ Turnpike 96'''

userInput2 = '''600 passengers abandoned at LRT station during Tuesday's hailstorm # yyc # Calgary Storm # Alberta Storm'''

userInput3 = '''How did I know as soon as I walked out of class that Calgary would flood again today'''

userInput4 = '''for sixth year in a row premium costs for windstorm insurance to climb . this time by 5 percent'''

userInput5 = '''Truth... #News #BBC #CNN #Islam #Truth #god #ISIS #terrorism #Quran #Lies'''

userInput6 = '''Here is how media in Pakistan covered the capture of terrorist Mohammed Naved'''

userInput7 = '''Hellfire is surrounded by desires so be careful and donÛªt let your desires control you! #Afterlife'''

userInput8 = '''Who is bringing the tornadoes and floods. Who is bringing the climate change. #FARRAKHAN #QUOTE'''    

userInput9 = '''RT NotExplained: The only known image of infamous hijacker D.B. Cooper.'''  

userInput10 = '''Hollywood Movie About Trapped Miners Released in Chile'''

userInput11 = '''Texas Seeks Comment on Rules for Changes to Windstorm Insurer'''

userInput12 = '''TWIA board approves 5 percent rate hike : The TWIA Board of Directors...'''

userInput13 = '''Bayelsa poll : Tension in Bayelsa as Patience Jonathan plans to hijack APC PDP..'''

userInput14 = '''A look at state actions a year after Ferguson ' s upheaval'''
predict_review(userInput1)

predict_review(userInput2)

predict_review(userInput3)

predict_review(userInput4)

predict_review(userInput5)

predict_review(userInput6)

predict_review(userInput7)
predict_review(userInput8) 

predict_review(userInput9)  

predict_review(userInput10) 

predict_review(userInput11) 

predict_review(userInput12) 

predict_review(userInput13) 

predict_review(userInput14) 