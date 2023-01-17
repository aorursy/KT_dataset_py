#from transformers import pipeline
#text_generator = pipeline("text-generation")
#print(text_generator("My name is Ryan.", max_length=20))
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
import math
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, TFBertModel, BertConfig, TFBertForSequenceClassification
import matplotlib.pyplot as plt
import re

# Remove the useless url tag 
def remove_url(raw_str):
    clean_str = re.sub(r'http\S+', '', raw_str)
    return clean_str
sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
train_df = pd.read_csv("../input/nlp-getting-started/train.csv")
test_df = pd.read_csv("../input/nlp-getting-started/test.csv")
#Randomization
state = 1
train_df = train_df.sample(frac=1,random_state=state)
test_df = test_df.sample(frac=1,random_state=state)
train_df.reset_index(inplace=True, drop=True) 
test_df.reset_index(inplace=True, drop=True) 
train_df.head()
train_df.fillna('nan', inplace = True)
test_df.fillna('nan', inplace = True)
#Credit: Gunes Evitan
#https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-full-cleaning#4.-Embeddings-&-Text-Cleaning
def clean(tweet):
    
    # Punctuations at the start or end of words    
    #for punctuation in "#@!?()[]*%":
    #    tweet = tweet.replace(punctuation, f' {punctuation} ').strip()
        
    #tweet = tweet.replace('...', ' ... ').strip()
    #tweet = tweet.replace("'", " ' ").strip()        
    
    # Special characters
    tweet = re.sub(r"\x89Û_", "", tweet)
    tweet = re.sub(r"\x89ÛÒ", "", tweet)
    tweet = re.sub(r"\x89ÛÓ", "", tweet)
    tweet = re.sub(r"\x89ÛÏWhen", "When", tweet)
    tweet = re.sub(r"\x89ÛÏ", "", tweet)
    tweet = re.sub(r"China\x89Ûªs", "China's", tweet)
    tweet = re.sub(r"let\x89Ûªs", "let's", tweet)
    tweet = re.sub(r"\x89Û÷", "", tweet)
    tweet = re.sub(r"\x89Ûª", "", tweet)
    tweet = re.sub(r"\x89Û\x9d", "", tweet)
    tweet = re.sub(r"å_", "", tweet)
    tweet = re.sub(r"\x89Û¢", "", tweet)
    tweet = re.sub(r"\x89Û¢åÊ", "", tweet)
    tweet = re.sub(r"fromåÊwounds", "from wounds", tweet)
    tweet = re.sub(r"åÊ", "", tweet)
    tweet = re.sub(r"åÈ", "", tweet)
    tweet = re.sub(r"JapÌ_n", "Japan", tweet)    
    tweet = re.sub(r"Ì©", "e", tweet)
    tweet = re.sub(r"å¨", "", tweet)
    tweet = re.sub(r"SuruÌ¤", "Suruc", tweet)
    
    # Contractions
    tweet = re.sub(r"he's", "he is", tweet)
    tweet = re.sub(r"there's", "there is", tweet)
    tweet = re.sub(r"We're", "We are", tweet)
    tweet = re.sub(r"That's", "That is", tweet)
    tweet = re.sub(r"won't", "will not", tweet)
    tweet = re.sub(r"they're", "they are", tweet)
    tweet = re.sub(r"Can't", "Cannot", tweet)
    tweet = re.sub(r"wasn't", "was not", tweet)
    tweet = re.sub(r"don\x89Ûªt", "do not", tweet)
    tweet = re.sub(r"aren't", "are not", tweet)
    tweet = re.sub(r"isn't", "is not", tweet)
    tweet = re.sub(r"What's", "What is", tweet)
    tweet = re.sub(r"haven't", "have not", tweet)
    tweet = re.sub(r"hasn't", "has not", tweet)
    tweet = re.sub(r"There's", "There is", tweet)
    tweet = re.sub(r"He's", "He is", tweet)
    tweet = re.sub(r"It's", "It is", tweet)
    tweet = re.sub(r"You're", "You are", tweet)
    tweet = re.sub(r"I'M", "I am", tweet)
    tweet = re.sub(r"shouldn't", "should not", tweet)
    tweet = re.sub(r"wouldn't", "would not", tweet)
    tweet = re.sub(r"i'm", "I am", tweet)
    tweet = re.sub(r"I\x89Ûªm", "I am", tweet)
    tweet = re.sub(r"I'm", "I am", tweet)
    tweet = re.sub(r"Isn't", "is not", tweet)
    tweet = re.sub(r"Here's", "Here is", tweet)
    tweet = re.sub(r"you've", "you have", tweet)
    tweet = re.sub(r"you\x89Ûªve", "you have", tweet)
    tweet = re.sub(r"we're", "we are", tweet)
    tweet = re.sub(r"what's", "what is", tweet)
    tweet = re.sub(r"couldn't", "could not", tweet)
    tweet = re.sub(r"we've", "we have", tweet)
    tweet = re.sub(r"it\x89Ûªs", "it is", tweet)
    tweet = re.sub(r"doesn\x89Ûªt", "does not", tweet)
    tweet = re.sub(r"It\x89Ûªs", "It is", tweet)
    tweet = re.sub(r"Here\x89Ûªs", "Here is", tweet)
    tweet = re.sub(r"who's", "who is", tweet)
    tweet = re.sub(r"I\x89Ûªve", "I have", tweet)
    tweet = re.sub(r"y'all", "you all", tweet)
    tweet = re.sub(r"can\x89Ûªt", "cannot", tweet)
    tweet = re.sub(r"would've", "would have", tweet)
    tweet = re.sub(r"it'll", "it will", tweet)
    tweet = re.sub(r"we'll", "we will", tweet)
    tweet = re.sub(r"wouldn\x89Ûªt", "would not", tweet)
    tweet = re.sub(r"We've", "We have", tweet)
    tweet = re.sub(r"he'll", "he will", tweet)
    tweet = re.sub(r"Y'all", "You all", tweet)
    tweet = re.sub(r"Weren't", "Were not", tweet)
    tweet = re.sub(r"Didn't", "Did not", tweet)
    tweet = re.sub(r"they'll", "they will", tweet)
    tweet = re.sub(r"they'd", "they would", tweet)
    tweet = re.sub(r"DON'T", "DO NOT", tweet)
    tweet = re.sub(r"That\x89Ûªs", "That is", tweet)
    tweet = re.sub(r"they've", "they have", tweet)
    tweet = re.sub(r"i'd", "I would", tweet)
    tweet = re.sub(r"should've", "should have", tweet)
    tweet = re.sub(r"You\x89Ûªre", "You are", tweet)
    tweet = re.sub(r"where's", "where is", tweet)
    tweet = re.sub(r"Don\x89Ûªt", "Do not", tweet)
    tweet = re.sub(r"we'd", "we would", tweet)
    tweet = re.sub(r"i'll", "I will", tweet)
    tweet = re.sub(r"weren't", "were not", tweet)
    tweet = re.sub(r"They're", "They are", tweet)
    tweet = re.sub(r"Can\x89Ûªt", "Cannot", tweet)
    tweet = re.sub(r"you\x89Ûªll", "you will", tweet)
    tweet = re.sub(r"I\x89Ûªd", "I would", tweet)
    tweet = re.sub(r"let's", "let us", tweet)
    
    # Character entity references
    tweet = re.sub(r"&gt;", ">", tweet)
    tweet = re.sub(r"&lt;", "<", tweet)
    tweet = re.sub(r"&amp;", "&", tweet)
        
    # Typos, slang and informal abbreviations
    tweet = re.sub(r"w/e", "whatever", tweet)
    tweet = re.sub(r"w/", "with", tweet)
    tweet = re.sub(r"USAgov", "USA government", tweet)
    tweet = re.sub(r"recentlu", "recently", tweet)
    tweet = re.sub(r"Ph0tos", "Photos", tweet)
    tweet = re.sub(r"exp0sed", "exposed", tweet)
    tweet = re.sub(r"<3", "love", tweet)
    tweet = re.sub(r"amageddon", "armageddon", tweet)
    tweet = re.sub(r"Trfc", "Traffic", tweet)
    tweet = re.sub(r"8/5/2015", "2015-08-05", tweet)
    tweet = re.sub(r"chest/torso", "chest / torso", tweet)
    tweet = re.sub(r"WindStorm", "Wind Storm", tweet)
    tweet = re.sub(r"8/6/2015", "2015-08-06", tweet)
    tweet = re.sub(r"10:38PM", "10:38 PM", tweet)
    tweet = re.sub(r"10:30pm", "10:30 PM", tweet)
    
    # Separating other punctuations
    tweet = re.sub(r"MH370:", "MH370 :", tweet)
    tweet = re.sub(r"PM:", "Prime Minister :", tweet)
    tweet = re.sub(r"Legionnaires:", "Legionnaires :", tweet)
    tweet = re.sub(r"Latest:", "Latest :", tweet)
    tweet = re.sub(r"Crash:", "Crash :", tweet)
    tweet = re.sub(r"News:", "News :", tweet)
    tweet = re.sub(r"derailment:", "derailment :", tweet)
    tweet = re.sub(r"attack:", "attack :", tweet)
    tweet = re.sub(r"Saipan:", "Saipan :", tweet)
    tweet = re.sub(r"Photo:", "Photo :", tweet)
    tweet = re.sub(r"Funtenna:", "Funtenna :", tweet)
    tweet = re.sub(r"quiz:", "quiz :", tweet)
    tweet = re.sub(r"VIDEO:", "VIDEO :", tweet)
    tweet = re.sub(r"MP:", "MP :", tweet)
    tweet = re.sub(r"UTC2015-08-05", "UTC 2015-08-05", tweet)
    tweet = re.sub(r"California:", "California :", tweet)
    tweet = re.sub(r"horror:", "horror :", tweet)
    tweet = re.sub(r"Past:", "Past :", tweet)
    tweet = re.sub(r"Time2015-08-06", "Time 2015-08-06", tweet)
    tweet = re.sub(r"here:", "here :", tweet)
    tweet = re.sub(r"fires.", "fires .", tweet)
    tweet = re.sub(r"Forest:", "Forest :", tweet)
    tweet = re.sub(r"Cramer:", "Cramer :", tweet)
    tweet = re.sub(r"Chile:", "Chile :", tweet)
    tweet = re.sub(r"link:", "link :", tweet)
    tweet = re.sub(r"crash:", "crash :", tweet)
    tweet = re.sub(r"Video:", "Video :", tweet)
    tweet = re.sub(r"Bestnaijamade:", "bestnaijamade :", tweet)
    tweet = re.sub(r"NWS:", "National Weather Service :", tweet)
    tweet = re.sub(r".caught", ". caught", tweet)
    tweet = re.sub(r"Hobbit:", "Hobbit :", tweet)
    tweet = re.sub(r"2015:", "2015 :", tweet)
    tweet = re.sub(r"post:", "post :", tweet)
    tweet = re.sub(r"BREAKING:", "BREAKING :", tweet)
    tweet = re.sub(r"Island:", "Island :", tweet)
    tweet = re.sub(r"Med:", "Med :", tweet)
    tweet = re.sub(r"97/Georgia", "97 / Georgia", tweet)
    tweet = re.sub(r"Here:", "Here :", tweet)
    tweet = re.sub(r"horror;", "horror ;", tweet)
    tweet = re.sub(r"people;", "people ;", tweet)
    tweet = re.sub(r"refugees;", "refugees ;", tweet)
    tweet = re.sub(r"Genocide;", "Genocide ;", tweet)
    tweet = re.sub(r".POTUS", ". POTUS", tweet)
    tweet = re.sub(r"Collision-No", "Collision - No", tweet)
    tweet = re.sub(r"Rear-", "Rear -", tweet)
    tweet = re.sub(r"Broadway:", "Broadway :", tweet)
    tweet = re.sub(r"Correction:", "Correction :", tweet)
    tweet = re.sub(r"UPDATE:", "UPDATE :", tweet)
    tweet = re.sub(r"Times:", "Times :", tweet)
    tweet = re.sub(r"RT:", "RT :", tweet)
    tweet = re.sub(r"Police:", "Police :", tweet)
    tweet = re.sub(r"Training:", "Training :", tweet)
    tweet = re.sub(r"Hawaii:", "Hawaii :", tweet)
    tweet = re.sub(r"Selfies:", "Selfies :", tweet)
    tweet = re.sub(r"Content:", "Content :", tweet)
    tweet = re.sub(r"101:", "101 :", tweet)
    tweet = re.sub(r"story:", "story :", tweet)
    tweet = re.sub(r"injured:", "injured :", tweet)
    tweet = re.sub(r"poll:", "poll :", tweet)
    tweet = re.sub(r"Guide:", "Guide :", tweet)
    tweet = re.sub(r"Update:", "Update :", tweet)
    tweet = re.sub(r"alarm:", "alarm :", tweet)
    tweet = re.sub(r"floods:", "floods :", tweet)
    tweet = re.sub(r"Flood:", "Flood :", tweet)
    tweet = re.sub(r"MH370;", "MH370 ;", tweet)
    tweet = re.sub(r"life:", "life :", tweet)
    tweet = re.sub(r"crush:", "crush :", tweet)
    tweet = re.sub(r"now:", "now :", tweet)
    tweet = re.sub(r"Vote:", "Vote :", tweet)
    tweet = re.sub(r"Catastrophe.", "Catastrophe .", tweet)
    tweet = re.sub(r"library:", "library :", tweet)
    tweet = re.sub(r"Bush:", "Bush :", tweet)
    tweet = re.sub(r";ACCIDENT", "; ACCIDENT", tweet)
    tweet = re.sub(r"accident:", "accident :", tweet)
    tweet = re.sub(r"Taiwan;", "Taiwan ;", tweet)
    tweet = re.sub(r"Map:", "Map :", tweet)
    tweet = re.sub(r"failure:", "failure :", tweet)
    tweet = re.sub(r"150-Foot", "150 - Foot", tweet)
    tweet = re.sub(r"failure:", "failure :", tweet)
    tweet = re.sub(r"prefer:", "prefer :", tweet)
    tweet = re.sub(r"CNN:", "CNN :", tweet)
    tweet = re.sub(r"Oops:", "Oops :", tweet)
    tweet = re.sub(r"Disco:", "Disco :", tweet)
    tweet = re.sub(r"Disease:", "Disease :", tweet)
    tweet = re.sub(r"Grows:", "Grows :", tweet)
    tweet = re.sub(r"projected:", "projected :", tweet)
    tweet = re.sub(r"Pakistan.", "Pakistan .", tweet)
    tweet = re.sub(r"ministers:", "ministers :", tweet)
    tweet = re.sub(r"Photos:", "Photos :", tweet)
    tweet = re.sub(r"Disease:", "Disease :", tweet)
    tweet = re.sub(r"pres:", "press :", tweet)
    tweet = re.sub(r"winds.", "winds .", tweet)
    tweet = re.sub(r"MPH.", "MPH .", tweet)
    tweet = re.sub(r"PHOTOS:", "PHOTOS :", tweet)
    tweet = re.sub(r"Time2015-08-05", "Time 2015-08-05", tweet)
    tweet = re.sub(r"Denmark:", "Denmark :", tweet)
    tweet = re.sub(r"Articles:", "Articles :", tweet)
    tweet = re.sub(r"Crash:", "Crash :", tweet)
    tweet = re.sub(r"casualties.:", "casualties .:", tweet)
    tweet = re.sub(r"Afghanistan:", "Afghanistan :", tweet)
    tweet = re.sub(r"Day:", "Day :", tweet)
    tweet = re.sub(r"AVERTED:", "AVERTED :", tweet)
    tweet = re.sub(r"sitting:", "sitting :", tweet)
    tweet = re.sub(r"Multiplayer:", "Multiplayer :", tweet)
    tweet = re.sub(r"Kaduna:", "Kaduna :", tweet)
    tweet = re.sub(r"favorite:", "favorite :", tweet)
    tweet = re.sub(r"home:", "home :", tweet)
    tweet = re.sub(r"just:", "just :", tweet)
    tweet = re.sub(r"Collision-1141", "Collision - 1141", tweet)
    tweet = re.sub(r"County:", "County :", tweet)
    tweet = re.sub(r"Duty:", "Duty :", tweet)
    tweet = re.sub(r"page:", "page :", tweet)
    tweet = re.sub(r"Attack:", "Attack :", tweet)
    tweet = re.sub(r"Minecraft:", "Minecraft :", tweet)
    tweet = re.sub(r"wounds;", "wounds ;", tweet)
    tweet = re.sub(r"Shots:", "Shots :", tweet)
    tweet = re.sub(r"shots:", "shots :", tweet)
    tweet = re.sub(r"Gunfire:", "Gunfire :", tweet)
    tweet = re.sub(r"hike:", "hike :", tweet)
    tweet = re.sub(r"Email:", "Email :", tweet)
    tweet = re.sub(r"System:", "System :", tweet)
    tweet = re.sub(r"Radio:", "Radio :", tweet)
    tweet = re.sub(r"King:", "King :", tweet)
    tweet = re.sub(r"upheaval:", "upheaval :", tweet)
    tweet = re.sub(r"tragedy;", "tragedy ;", tweet)
    tweet = re.sub(r"HERE:", "HERE :", tweet)
    tweet = re.sub(r"terrorism:", "terrorism :", tweet)
    tweet = re.sub(r"police:", "police :", tweet)
    tweet = re.sub(r"Mosque:", "Mosque :", tweet)
    tweet = re.sub(r"Rightways:", "Rightways :", tweet)
    tweet = re.sub(r"Brooklyn:", "Brooklyn :", tweet)
    tweet = re.sub(r"Arrived:", "Arrived :", tweet)
    tweet = re.sub(r"Home:", "Home :", tweet)
    tweet = re.sub(r"Earth:", "Earth :", tweet)
    tweet = re.sub(r"three:", "three :", tweet)
    
    # Hashtags and usernames
    tweet = re.sub(r"IranDeal", "Iran Deal", tweet)
    tweet = re.sub(r"ArianaGrande", "Ariana Grande", tweet)
    tweet = re.sub(r"camilacabello97", "camila cabello", tweet) 
    tweet = re.sub(r"RondaRousey", "Ronda Rousey", tweet)     
    tweet = re.sub(r"MTVHottest", "MTV Hottest", tweet)
    tweet = re.sub(r"TrapMusic", "Trap Music", tweet)
    tweet = re.sub(r"ProphetMuhammad", "Prophet Muhammad", tweet)
    tweet = re.sub(r"PantherAttack", "Panther Attack", tweet)
    tweet = re.sub(r"StrategicPatience", "Strategic Patience", tweet)
    tweet = re.sub(r"socialnews", "social news", tweet)
    tweet = re.sub(r"NASAHurricane", "NASA Hurricane", tweet)
    tweet = re.sub(r"onlinecommunities", "online communities", tweet)
    tweet = re.sub(r"humanconsumption", "human consumption", tweet)
    tweet = re.sub(r"Typhoon-Devastated", "Typhoon Devastated", tweet)
    tweet = re.sub(r"Meat-Loving", "Meat Loving", tweet)
    tweet = re.sub(r"facialabuse", "facial abuse", tweet)
    tweet = re.sub(r"LakeCounty", "Lake County", tweet)
    tweet = re.sub(r"BeingAuthor", "Being Author", tweet)
    tweet = re.sub(r"withheavenly", "with heavenly", tweet)
    tweet = re.sub(r"thankU", "thank you", tweet)
    tweet = re.sub(r"iTunesMusic", "iTunes Music", tweet)
    tweet = re.sub(r"OffensiveContent", "Offensive Content", tweet)
    tweet = re.sub(r"WorstSummerJob", "Worst Summer Job", tweet)
    tweet = re.sub(r"HarryBeCareful", "Harry Be Careful", tweet)
    tweet = re.sub(r"NASASolarSystem", "NASA Solar System", tweet)
    tweet = re.sub(r"animalrescue", "animal rescue", tweet)
    tweet = re.sub(r"KurtSchlichter", "Kurt Schlichter", tweet)
    tweet = re.sub(r"aRmageddon", "armageddon", tweet)
    tweet = re.sub(r"Throwingknifes", "Throwing knives", tweet)
    tweet = re.sub(r"GodsLove", "God's Love", tweet)
    tweet = re.sub(r"bookboost", "book boost", tweet)
    tweet = re.sub(r"ibooklove", "I book love", tweet)
    tweet = re.sub(r"NestleIndia", "Nestle India", tweet)
    tweet = re.sub(r"realDonaldTrump", "Donald Trump", tweet)
    tweet = re.sub(r"DavidVonderhaar", "David Vonderhaar", tweet)
    tweet = re.sub(r"CecilTheLion", "Cecil The Lion", tweet)
    tweet = re.sub(r"weathernetwork", "weather network", tweet)
    tweet = re.sub(r"withBioterrorism&use", "with Bioterrorism & use", tweet)
    tweet = re.sub(r"Hostage&2", "Hostage & 2", tweet)
    tweet = re.sub(r"GOPDebate", "GOP Debate", tweet)
    tweet = re.sub(r"RickPerry", "Rick Perry", tweet)
    tweet = re.sub(r"frontpage", "front page", tweet)
    tweet = re.sub(r"NewsInTweets", "News In Tweets", tweet)
    tweet = re.sub(r"ViralSpell", "Viral Spell", tweet)
    tweet = re.sub(r"til_now", "until now", tweet)
    tweet = re.sub(r"volcanoinRussia", "volcano in Russia", tweet)
    tweet = re.sub(r"ZippedNews", "Zipped News", tweet)
    tweet = re.sub(r"MicheleBachman", "Michele Bachman", tweet)
    tweet = re.sub(r"53inch", "53 inch", tweet)
    tweet = re.sub(r"KerrickTrial", "Kerrick Trial", tweet)
    tweet = re.sub(r"abstorm", "Alberta Storm", tweet)
    tweet = re.sub(r"Beyhive", "Beyonce hive", tweet)
    tweet = re.sub(r"IDFire", "Idaho Fire", tweet)
    tweet = re.sub(r"DETECTADO", "Detected", tweet)
    tweet = re.sub(r"RockyFire", "Rocky Fire", tweet)
    tweet = re.sub(r"Listen/Buy", "Listen / Buy", tweet)
    tweet = re.sub(r"NickCannon", "Nick Cannon", tweet)
    tweet = re.sub(r"FaroeIslands", "Faroe Islands", tweet)
    tweet = re.sub(r"yycstorm", "Calgary Storm", tweet)
    tweet = re.sub(r"IDPs:", "Internally Displaced People :", tweet)
    tweet = re.sub(r"ArtistsUnited", "Artists United", tweet)
    tweet = re.sub(r"ClaytonBryant", "Clayton Bryant", tweet)
    tweet = re.sub(r"jimmyfallon", "jimmy fallon", tweet)
    
    return tweet
stopwords = ["they've", 'k', 'whom', "he'll", 'could', 'itself', 'hence',  "when's", 'where', 'through',
             'was', 'its', 'into', 'however', "she'd", "we'd", 'they', 'below', 'again', "she'll", "he's",
             'did', 'my', 'are', 'our', "where's", 'above', 'ever', 'yourself', "i'd", 'just', 'we', "i'll",
             'it', 'from', "we'll", 'that', 'he', 'cannot', "you're", 'her', 'this', "why's", 'once',
             'am', 'ourselves', 'out', 'get', 'would', 'up', "it's", 'same', 'these',  "you've", 'such', 'between',
             'himself', 'also', "that's", 'r', 'www', 'all', "what's", 'if', 'http', 'herself',   'after', 'had',
             'has', 'your', 'while', 'other', 'their', 'shall', 'more', 'off', 'as', 'hers', 'with',   'over',
             'by',  'there', "we're", "they'll", 'any', 'to',  'no',  'about',  'both', "he'd", 'only', 'here', 
             'than', 'what', 'been', 'does', "we've", 'theirs', 'being', 'ought', "they'd", 'few', 'you', 'under', 
             'since',  'can', 'them', 'at', 'else', 'each', 'ours', 'therefore', 'most', 'before', 'then', 'his', 'me',
             'a', 'further', "how's", 'during', 'of', 'like', 'on',  'themselves', 'why', 'those', 'in', 'too', 'she',
             'because', "i've", "let's", 'how', 'very', "you'd", 'own', 'but', "she's", 'i', 'yourselves', 'down', 'should',
             'and', 'do', 'or', 'were', 'some', 'an', "who's", 'otherwise', 'be', 'him', 'myself', 'have', 'which', "there's",
             "i'm", 'when', 'doing', "you'll", 'com', 'for', 'who', 'yours', 'until', 'the', 'is', 'so', "they're", "here's", 
             'nor', 'having', 'will', 'may', 'one', 'now']
def preprocessor2(text): 

    # Remove punctuations and numbers
    text = re.sub('[^a-zA-Z]', ' ', text)

    # Single character removal
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)

    # Removing multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    text = text.lower()

    return text
train_df['text'] = train_df['text'].apply(lambda x:remove_url(x))
train_df['merged'] = train_df['text'].astype(str)+' '+train_df['keyword']+' '+train_df['location']
test_df['merged'] = test_df['text'].astype(str)+' '+test_df['keyword']+' '+test_df['location']
train_df['merged_cleaned'] = train_df['merged'].copy().apply(lambda x : clean(str(x))).apply(lambda x : preprocessor2(x)).apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
test_df['merged_cleaned'] = test_df['merged'].copy().apply(lambda x : clean(str(x))).apply(lambda x : preprocessor2(x)).apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
train_df['totalwords'] = train_df['merged_cleaned'].str.count(' ') + 1
#mapper = {1:"Positive", 0:"Negative"}
#train_df['target'] = train_df['target'].map(mapper)
train_df.head()
train_df.describe()
max_len = 50
train_df = train_df[['merged_cleaned', 'target']]
test_df = test_df[['id','merged_cleaned']]
test_df.shape
train_df.head()
label_L = list(train_df.target)
configuration = BertConfig.from_pretrained(f'/kaggle/input/bert-tensorflow/bert-base-uncased-config.json')
# Save the slow pretrained tokenizer
slow_tokenizer = BertTokenizer.from_pretrained(f'/kaggle/input/bertbaseuncased/vocab.txt')
save_path = "bert-base-uncased/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
slow_tokenizer.save_pretrained(save_path)

# Load the fast tokenizer from saved file
tokenizer = BertWordPieceTokenizer(f'/kaggle/input/bertbaseuncased/vocab.txt', lowercase=True)
#tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
class SquadExample:
    def __init__(self, text):
        self.text = text
        self.skip = False

        
    def preprocess(self):
        
        text = self.text

        text = str(text)
                   
        # Tokenize context
        tokenized_context = tokenizer.encode(text)
        #print(tokenized_context.tokens)
        
        # Create inputs
        input_ids = tokenized_context.ids
        #print(input_ids)       
        token_type_ids = [0] * len(tokenized_context.ids)
        attention_mask = [1] * len(input_ids)

        # Pad and create attention masks.
        # Skip if truncation is needed
        padding_length = max_len - len(input_ids)
        if padding_length > 0:  # pad
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
        elif padding_length < 0:  # skip
            self.skip = True
            #return


        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask

def create_examples(raw_data):
    squad_examples = []
    for i in range(raw_data.shape[0]):
        text = raw_data["merged_cleaned"][i]
        squad_eg = SquadExample(text)
        squad_eg.preprocess()
        squad_examples.append(squad_eg)
    return squad_examples
def create_inputs_targets(squad_examples,label):
    dataset_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],

    }
    
    for item in squad_examples:
        if item.skip == False:
            for key in dataset_dict:
                dataset_dict[key].append(getattr(item, key))
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [
        dataset_dict["input_ids"],
        dataset_dict["token_type_ids"],
        dataset_dict["attention_mask"],
    ]
    
    
    y = [np.array(label)]
    
    return x, y
train_examples = create_examples(train_df)
test_examples = create_examples(test_df)
len(train_examples)
len(label_L)
x_train, y_train = create_inputs_targets(train_examples,label_L)
len(x_train[0])
len(y_train[0])
x_test, y_test = create_inputs_targets(test_examples,None)
print(f"{len(train_examples)} training points created.")
"""
base_path = '/kaggle/input/bert-tensorflow/bert-base-uncased-tf_model.h5'

def create_model(path = base_path):
    ## BERT encoder
    encoder = TFBertForSequenceClassification.from_pretrained(path, config=configuration)

    ## QA Model
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32)
    embedding = encoder(
        input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
    )[0]
    
    label_class_0 = layers.Dense(128, name="process", use_bias=False)(embedding)
    label_class = layers.Dense(2, name="label_class", use_bias=False)(label_class_0)
    label_class = layers.Flatten()(label_class)


    label_probs = layers.Activation(keras.activations.softmax)(label_class)

    model = keras.Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[label_probs],
    )
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = keras.optimizers.Adam(lr=2e-5)
    model.compile(optimizer=optimizer, loss=[loss, loss])
    return model
"""
base_path = '/kaggle/input/bert-tensorflow/bert-base-uncased-tf_model.h5'

def create_model(path = base_path):
    ## BERT encoder
    encoder = TFBertForSequenceClassification.from_pretrained(path, config=configuration)

    ## QA Model
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32)
    embedding = encoder(
        input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
    )[0]
    reshape_L = layers.Reshape((2,1))(embedding)
    lstm = layers.LSTM(64)(reshape_L)
    label_class_0 = layers.Dense(128, name="process-1", use_bias=False)(lstm)
    label_class_1 = layers.Dense(128, name="process-2", use_bias=False)(label_class_0)
    label_class = layers.Dense(2, name="label_class", use_bias=False)(label_class_1)
    label_class = layers.Flatten()(label_class)


    label_probs = layers.Activation(keras.activations.softmax)(label_class)

    model = keras.Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[label_probs],
    )
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = keras.optimizers.Adam(lr=2e-5)
    model.compile(optimizer=optimizer, loss=[loss, loss])
    return model
model = create_model(base_path)

model.summary()
model.fit(
    x_train,
    y_train,
    epochs=3, 
    verbose=2,
    batch_size=16
)
Y_pred = model.predict(x_test)
Y_pred = np.argmax(Y_pred,axis=1)
Y_pred
pred_df = pd.DataFrame(Y_pred, columns=['target'])
result = pd.concat([test_df,pred_df], axis=1, join='outer', ignore_index=False, keys=None, sort = False)
result = result[['id','target']]
print(Y_pred)
result
result.to_csv('sample_submission2.csv',index=False)
