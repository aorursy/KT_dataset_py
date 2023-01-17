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
test_df.head()
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
from wordcloud import STOPWORDS

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
    text = text.replace('%20',' ')
    text = text.lower()
    text = text.replace("n't","nt")
    text = re.sub(r"\s\w\s", "", text)
    text = re.sub(r"\s\d\s", "", text)
    #emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    #text = (re.sub('[^a-zA-Z0-9_]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    return text
a= "i don't like you 2 a r!"
a=preprocessor2(a)
print(a)
train_df['text'] = train_df['text'].apply(lambda x:remove_url(x))
train_df['keyword_cleaned'] = train_df['keyword'].copy().apply(lambda x : clean(str(x))).apply(lambda x : preprocessor2(x)).apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
train_df['location_cleaned'] = train_df['location'].copy().apply(lambda x : clean(str(x))).apply(lambda x : preprocessor2(x)).apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
train_df['text_cleaned'] = train_df['text'].copy().apply(lambda x : clean(str(x))).apply(lambda x : preprocessor2(x)).apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
test_df['text'] = test_df['text'].apply(lambda x:remove_url(x))
test_df['keyword_cleaned'] = test_df['keyword'].copy().apply(lambda x : clean(str(x))).apply(lambda x : preprocessor2(x)).apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
test_df['location_cleaned'] = test_df['location'].copy().apply(lambda x : clean(str(x))).apply(lambda x : preprocessor2(x)).apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
test_df['text_cleaned'] = test_df['text'].copy().apply(lambda x : clean(str(x))).apply(lambda x : preprocessor2(x)).apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
train_df.head()
test_df.head()
# Target count
fig, ax = plt.subplots(figsize = (8,5))
pd.value_counts(train_df['target']).plot(kind="bar")
ax.set_title('Target Count')
ax.set_ylabel('Frequency')
ax.grid(True)
plt.show()
#Top 20 Locations with most target occurance
loc_pos = train_df[(train_df.location_cleaned != 'nan') & (train_df.location_cleaned != '') & (train_df['target'] == 1)]['location_cleaned'].value_counts()
loc_neg = train_df[(train_df.location_cleaned != 'nan') & (train_df.location_cleaned != '') & (train_df['target'] == 0)]['location_cleaned'].value_counts()

loc_pos_dict = loc_pos[:20].to_dict()
loc_neg_dict = loc_neg[:20].to_dict()

names0 = list(loc_neg_dict.keys())
values0 = list(loc_neg_dict.values())
names1 = list(loc_pos_dict.keys())
values1 = list(loc_pos_dict.values())

#Graph
fig, (ax1, ax2) = plt.subplots(figsize = (20,5), nrows=1, ncols=2)

ax1.bar(range(len(loc_pos_dict)),values1,tick_label=names1)
ax1.set_xticklabels(names1, rotation="vertical")
ax1.set_ylim(0, 100)
ax1.grid(True)
ax1.set_title('Location with most Pos target')
ax1.set_ylabel('Frequency')

ax2.bar(range(len(loc_neg_dict)),values0,tick_label=names0)
ax2.set_xticklabels(names0, rotation="vertical")
ax2.set_ylim(0, 100)
ax2.grid(True)
ax2.set_title('Location with most Neg target')
ax2.set_ylabel('Frequency')
#As we can see some same meaning words using different abbreviation, so that we try to make a function to align these words
def preprocessor3(text):
    text = re.sub(r'^washington d c ', "washington dc", text)
    text = re.sub(r'^washington +[\w]*', "washington dc", text)
    text = re.sub(r'^new york +[\w]*', "new york", text)
    text = re.sub(r'^nyc$', "new york", text)
    text = re.sub(r'^chicago +[\w]*', "chicago", text)
    text = re.sub(r'^california +[\w]*', "california", text)
    text = re.sub(r'^los angeles +[\w]*', "los angeles", text)
    text = re.sub(r'^san francisco +[\w]*', "san francisco", text)
    text = re.sub(r'^london +[\w]*', "london", text)
    text = re.sub(r'^usa$', "united states", text)
    text = re.sub(r'^us$', "united states", text)
    text = re.sub(r'^uk$', "united kingdom", text)
    
    return text

def preprocessor4(text):
    abb = ['ak', 'al', 'az', 'ar', 'ca', 'co',
           'ct', 'de', 'dc', 'fl', 'ga', 'hi',
           'id', 'il', 'in', 'ia', 'ks', 'ky',
           'la', 'me', 'mt', 'ne', 'nv', 'nh',
           'nj', 'nm', 'ny', 'nc', 'nd', 'oh',
           'ok', 'or', 'md', 'ma', 'mi', 'mn',
           'ms', 'mo', 'pa', 'ri', 'sc', 'sd',
           'tn', 'tx', 'ut', 'vt', 'va', 'wa',
           'wv', 'wi', 'wy']
    
    for i in abb:
        text = re.sub(r'^{0}$'.format(i), '', text)
        
    return text 
train_df['location_cleaned'] = train_df['location_cleaned'].copy().apply(lambda x : preprocessor3(x)).apply(lambda x : preprocessor4(x))
train_df['text_cleaned'] = train_df['text_cleaned'].copy().apply(lambda x : preprocessor3(x)).apply(lambda x : preprocessor4(x))

test_df['location_cleaned'] = test_df['location_cleaned'].copy().apply(lambda x : preprocessor3(x)).apply(lambda x : preprocessor4(x))
test_df['text_cleaned'] = test_df['text_cleaned'].copy().apply(lambda x : preprocessor3(x)).apply(lambda x : preprocessor4(x))
loc_pos_dict = loc_pos[:].to_dict()
loc_neg_dict = loc_neg[:].to_dict()

loc_list = list(loc_pos_dict.keys()) + list(loc_neg_dict.keys())
unique_loc = []
lower_loc = []
for x in loc_list:
    if x not in unique_loc:
        unique_loc.append(x)
        
for x in unique_loc:
    lower_loc.append(x.lower().replace(',','').replace('.',''))
from wordcloud import WordCloud,ImageColorGenerator
import matplotlib.pyplot as plt
%matplotlib inline

#stopwords = list(STOPWORDS)+['will','may','one','now','nan','don'] #+ lower_loc

class wc_base2:
    def __init__(self, data):
        self.temp = data.apply(lambda x: ' '.join([word for word in x.split()]))
        self.text = " ".join(word for word in data)
        self.wordlist = []
        
    def plot_wc(self, mask=None, max_words=200, figure_size=(20,10), title=None, stopwords=stopwords):

        print ("There are {} words in the combination of all review.".format(len(self.text)))

        wordcloud = WordCloud(background_color='black',
                        stopwords=stopwords,
                        max_words = max_words,
                        collocations=False,
                        random_state = 10,
                        width = 800,
                        height =400)

        wordcloud.generate(self.text)
         
        self.wordlist = list(wordcloud.words_.keys())

        plt.figure(figsize=figure_size)
        plt.imshow(wordcloud)
        plt.title(title)
        plt.axis("off")
print(stopwords)
wc = wc_base2(train_df[train_df.target == 0].text_cleaned)
wc.plot_wc(title="Word Cloud of tweets with Negative target")
wc_s = set(wc.wordlist)
wc2 = wc_base2(train_df[train_df.target == 1].text_cleaned)
wc2.plot_wc(title="Word Cloud of tweets with Positive target")
wc2_s = set(wc2.wordlist)
train_df['text_cleaned'] = train_df['text_cleaned'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
test_df['text_cleaned'] = test_df['text_cleaned'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
train_df.text_cleaned.head()
train_df['index_no'] = train_df.index
train_df['sent_w_index'] = train_df['text_cleaned'] + ' ' + train_df['index_no'].astype('str')
train_df.head()
train_df.sent_w_index[1]
def random_swap(text):
    text_list = text.split()
    seed = int(text_list[-1])

    text_list=text_list[:-1]
    text_length = len(text_list)
   
    np.random.seed(seed)
    try:
        a = np.random.randint(0, text_length,size=2)
    except:
        return
    #print(a)

    temp_a = text_list[a[0]]
    temp_b = text_list[a[1]]
    
    text_list[a[0]] = temp_b
    text_list[a[1]] = temp_a
    
    redo = ' '.join([str(i) for i in text_list])
   
    return redo    

def random_del(text):
    text_list = text.split()
    seed = int(text_list[-1])

    text_list=text_list[:-1]
    text_length = len(text_list)
   
    np.random.seed(seed)
    try:
        a = np.random.randint(0, text_length,size=1)
    except:
        return
    
    text_list.pop((a[0]))
    
    redo = ' '.join([str(i) for i in text_list])
      
    return redo    
train_df['da_text_cleaned'] = train_df['sent_w_index'].apply(lambda x:random_swap(x))
train_df['da_text_cleaned2'] = train_df['sent_w_index'].apply(lambda x:random_del(x))
train_df.drop(['index_no','sent_w_index'], axis=1,inplace=True)
train_df.head()
#temp_df1 = list(zip(train_df.target,train_df.keyword_cleaned,train_df.location_cleaned,train_df.da_text_cleaned))
#temp_df2 = list(zip(train_df.target,train_df.keyword_cleaned,train_df.location_cleaned,train_df.da_text_cleaned2))
#x = pd.DataFrame(temp_df1, columns =['target','keyword_cleaned','location_cleaned','text_cleaned'])
#y = pd.DataFrame(temp_df2, columns =['target','keyword_cleaned','location_cleaned','text_cleaned'])

#train_df.drop(['da_text_cleaned','da_text_cleaned2'], axis=1,inplace=True)

#z = pd.concat([train_df,x,y], axis=0, join='outer', ignore_index=False, keys=None, sort = False)
#z = z[['id','keyword_cleaned','location_cleaned','text_cleaned','target']].copy()
#z.reset_index(inplace=True, drop=True) 
#train_df = z 
test_df = test_df[['id','keyword_cleaned','location_cleaned','text_cleaned']].copy()
#Randomization
state = 1
train_df = train_df.sample(frac=1,random_state=state)
train_df.reset_index(inplace=True, drop=True) 
train_df['text_cleaned'] = train_df['text_cleaned'].apply(lambda x : str(x))
train_df.info()
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

top_word = 35000

text_lengths = [len(x.split()) for x in (train_df.text_cleaned)]
#text_lengths = [x for x in text_lengths if x < 50]
plt.hist(text_lengths, bins=25)
plt.title('Histogram of # of Words in Texts')

tok = Tokenizer(num_words=top_word)
tok.fit_on_texts((train_df['text_cleaned']+train_df['keyword_cleaned']+train_df['location_cleaned']))

max_words = max(text_lengths) + 1
max_words_ky = max([len(x.split()) for x in (train_df.keyword_cleaned)]) + 1
max_words_lc = max([len(x.split()) for x in (train_df.location_cleaned)]) + 1
print("top_word: ", str(top_word))
print("max_words: ", str(max_words))
print("max_words_ky: ", str(max_words_ky))
print("max_words_lc: ", str(max_words_lc))
#Training set

X_train_tx = tok.texts_to_sequences(train_df['text_cleaned'])
X_train_ky = tok.texts_to_sequences(train_df['keyword_cleaned'])
X_train_lc = tok.texts_to_sequences(train_df['location_cleaned'])

X_test_tx = tok.texts_to_sequences(test_df['text_cleaned'])
X_test_ky = tok.texts_to_sequences(test_df['keyword_cleaned'])
X_test_lc = tok.texts_to_sequences(test_df['location_cleaned'])


Y_train = train_df['target']

print('Found %s unique tokens.' % len(tok.word_index))
from keras.utils import to_categorical
# One-hot category
Y_train = to_categorical(Y_train)
print("Y_train.shape: ", Y_train.shape)
X_train_tx = sequence.pad_sequences(X_train_tx, maxlen=max_words)
X_train_ky = sequence.pad_sequences(X_train_ky, maxlen=max_words_ky)
X_train_lc = sequence.pad_sequences(X_train_lc, maxlen=max_words_lc)

X_test_tx = sequence.pad_sequences(X_test_tx, maxlen=max_words)
X_test_ky = sequence.pad_sequences(X_test_ky, maxlen=max_words_ky)
X_test_lc = sequence.pad_sequences(X_test_lc, maxlen=max_words_lc)

print("X_train_tx.shape: ", X_train_tx.shape)
print("X_train_ky.shape: ", X_train_ky.shape)
print("X_train_lc.shape: ", X_train_lc.shape)
from numpy import array
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()
glove_file = open('/kaggle/input/glove6b/glove.6B.300d.txt', encoding="utf8")
for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    
    embeddings_dictionary [word] = vector_dimensions

glove_file.close()
embedding_dim = 300
embedding_matrix = zeros((top_word, embedding_dim))
for word, index in tok.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, Conv2D, MaxPooling2D,Conv1D,MaxPooling1D
from keras.layers import Bidirectional,  Reshape, Flatten, GRU
from keras.layers.merge import concatenate
input1 = Input(shape=(max_words,))
embedding_layer1 = Embedding(top_word, embedding_dim, weights=[embedding_matrix], input_length=max_words, trainable=False)(input1)
dropout1 = Dropout(0.2)(embedding_layer1)
lstm1_1 = LSTM(128,return_sequences = True)(dropout1)
lstm1_2 = LSTM(128,return_sequences = True)(lstm1_1)
lstm1_2a = LSTM(128,return_sequences = True)(lstm1_2)
lstm1_3 = LSTM(128)(lstm1_2a)

input2 = Input(shape=(max_words_ky,))
embedding_layer2 = Embedding(top_word, embedding_dim, weights=[embedding_matrix], input_length=max_words_ky, trainable=False)(input2)
dropout2 = Dropout(0.2)(embedding_layer2)
lstm2_1 = LSTM(64,return_sequences = True)(dropout2)
lstm2_2 = LSTM(64,return_sequences = True)(lstm2_1)
lstm2_3 = LSTM(64)(lstm2_2)

input3 = Input(shape=(max_words_lc,))
embedding_layer3 = Embedding(top_word, embedding_dim, weights=[embedding_matrix], input_length=max_words_lc, trainable=False)(input3)
dropout3 = Dropout(0.2)(embedding_layer3)
lstm3_1 = LSTM(32,return_sequences = True)(dropout3)
lstm3_2 = LSTM(32,return_sequences = True)(lstm3_1)
lstm3_3 = LSTM(32)(lstm3_2)

merge = concatenate([lstm1_3, lstm2_3,lstm3_3])

dropout = Dropout(0.8)(merge)
dense1 = Dense(256, activation='relu')(dropout)
dense2 = Dense(128, activation='relu')(dense1)
output = Dense(2, activation='softmax')(dense2)
model1 = Model(inputs=[input1,input2,input3], outputs=output)
model1.summary()
input1 = Input(shape=(max_words,))
embedding_layer1 = Embedding(top_word, embedding_dim, weights=[embedding_matrix], input_length=max_words, trainable=False)(input1)
lstm1_1 = Bidirectional(LSTM(100, return_sequences=True,dropout = 0.2))(embedding_layer1)
lstm1_1a = Bidirectional(LSTM(100, return_sequences=True,dropout = 0.2))(lstm1_1)
lstm1_1b = Bidirectional(LSTM(100, return_sequences=True,dropout = 0.2))(lstm1_1a)
res = Reshape((-1, X_train_tx.shape[1], 100))(lstm1_1b)
conv1 = Conv2D(100, (3,3), padding='same',activation="relu")(res)
pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
flat1 = Flatten()(pool1)

input2 = Input(shape=(max_words_ky,))
embedding_layer2 = Embedding(top_word, embedding_dim, weights=[embedding_matrix], input_length=max_words_ky, trainable=False)(input2)
lstm2_1 = Bidirectional(LSTM(100, return_sequences=True,dropout = 0.2))(embedding_layer2)
lstm2_1a = Bidirectional(LSTM(100, return_sequences=True,dropout = 0.2))(lstm2_1)
lstm2_1b = Bidirectional(LSTM(100, return_sequences=True,dropout = 0.2))(lstm2_1a)
res2 = Reshape((-1, X_train_ky.shape[1], 100))(lstm2_1b)
conv2 = Conv2D(100, (3,3), padding='same',activation="relu")(res2)
pool2 = MaxPooling2D(pool_size=(2,2))(conv2)
flat2 = Flatten()(pool2)

input3 = Input(shape=(max_words_lc,))
embedding_layer3 = Embedding(top_word, embedding_dim, weights=[embedding_matrix], input_length=max_words_lc, trainable=False)(input3)
lstm3_1 = Bidirectional(LSTM(100, return_sequences=True,dropout = 0.2))(embedding_layer3)
lstm3_1a = Bidirectional(LSTM(100, return_sequences=True,dropout = 0.2))(lstm3_1)
lstm3_1b = Bidirectional(LSTM(100, return_sequences=True,dropout = 0.2))(lstm3_1a)
res3 = Reshape((-1, X_train_lc.shape[1], 100))(lstm3_1b)
conv3 = Conv2D(100, (3,3), padding='same',activation="relu")(res3)
pool3 = MaxPooling2D(pool_size=(2,2))(conv3)
flat3 = Flatten()(pool3)

merge = concatenate([flat1, flat2, flat3])

dropout = Dropout(0.4)(merge)
dense1 = Dense(256, activation='relu')(dropout)
dense2 = Dense(128, activation='relu')(dense1)
output = Dense(2, activation='softmax')(dense2)
model2 = Model(inputs=[input1,input2,input3], outputs=output)
model2.summary()
input1 = Input(shape=(max_words,))
input2 = Input(shape=(max_words_ky,))
input3 = Input(shape=(max_words_lc,))

merge = concatenate([input1, input2, input3])

embedding_layer1 = Embedding(top_word, embedding_dim, weights=[embedding_matrix], input_length=42, trainable=False)(merge)
lstm1_1 = Bidirectional(LSTM(128, return_sequences=True,dropout = 0.2))(embedding_layer1)
lstm1_1a = Bidirectional(LSTM(128, return_sequences=True,dropout = 0.2))(lstm1_1)
#lstm1_1b = Bidirectional(LSTM(128, return_sequences=True,dropout = 0.2))(lstm1_1a)
lstm1_1b = Bidirectional(LSTM(128, return_sequences=True,dropout = 0.2))(lstm1_1a)

#res2 = Reshape((-1, 40, 256))(lstm1_1b)
conv2 = Conv1D(64, 3, padding='same',activation="relu")(lstm1_1b)
pool2 = MaxPooling1D(pool_size=2)(conv2)
conv3 = Conv1D(64, 3, padding='same',activation="relu")(pool2)
pool3 = MaxPooling1D(pool_size=2)(conv3)
flat2 = Flatten()(pool3)

dense1 = Dense(256, activation='relu')(flat2)
dropout = Dropout(0.8)(dense1)
dense2 = Dense(128, activation='relu')(dropout)
output = Dense(2, activation='softmax')(dense2)
model3 = Model(inputs=[input1,input2,input3], outputs=output)
model3.summary()
from keras.optimizers import Adam
optimizer1 = Adam(lr = .0001, beta_1 = .9, beta_2 = .999, epsilon = 1e-10, decay = .0, amsgrad = False)
optimizer2 = Adam(lr = .0001, beta_1 = .9, beta_2 = .999, epsilon = 1e-10, decay = .0, amsgrad = False)
model1.compile(loss="binary_crossentropy", optimizer=optimizer1,
              metrics=["accuracy"])
model2.compile(loss="binary_crossentropy", optimizer=optimizer2,
              metrics=["accuracy"])
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', mode='min',verbose=1, patience = 4)
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_accuracy', patience = 2, verbose = 1, 
                                           factor = 0.5, min_lr = 1e-8, cooldown=1)
#history = model.fit([X_train_tx,X_train_ky], Y_train, validation_split=0.2, epochs=30, batch_size=64, verbose=2, callbacks=[es])
history = model2.fit([X_train_tx,X_train_ky,X_train_lc], Y_train, validation_split=0.2, epochs=20, batch_size=16, verbose=2, callbacks=[es, learning_rate_reduction])
history2 = model1.fit([X_train_tx,X_train_ky,X_train_lc], Y_train, validation_split=0.2, epochs=20, batch_size=16, verbose=2, callbacks=[es, learning_rate_reduction])
def result_eva (loss,val_loss,acc,val_acc):
    import matplotlib.pyplot as plt
    %matplotlib inline
    
    epochs = range(1,len(loss)+1)
    plt.plot(epochs, loss,'b-o', label ='Training Loss')
    plt.plot(epochs, val_loss,'r-o', label ='Validation Loss')
    plt.title("Training and Validation Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    epochs = range(1, len(acc)+1)
    plt.plot(epochs, acc, "b-o", label="Training Acc")
    plt.plot(epochs, val_acc, "r-o", label="Validation Acc")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
result_eva(history.history['loss'], history.history['val_loss'], history.history['accuracy'], history.history['val_accuracy'])
result_eva(history2.history['loss'], history2.history['val_loss'], history2.history['accuracy'], history2.history['val_accuracy'])
model2.save('nlp_disaster.h5')
model1.save('nlp_disaster2.h5')
from keras.models import load_model

model = Model()
model = load_model('nlp_disaster.h5')
#model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])
#Y_pred = model.predict([X_test_tx,X_test_ky], batch_size=64, verbose=2)
Y_pred = model.predict([X_test_tx,X_test_ky,X_test_lc], batch_size=16, verbose=2)
Y_pred = np.argmax(Y_pred,axis=1)

pred_df = pd.DataFrame(Y_pred, columns=['target'])
result = pd.concat([test_df,pred_df], axis=1, join='outer', ignore_index=False, keys=None, sort = False)
result = result[['id','target']]
print(Y_pred)
result.to_csv('sample_submission.csv',index=False)
Y_pred = model1.predict([X_test_tx,X_test_ky,X_test_lc], batch_size=16, verbose=2)
Y_pred = np.argmax(Y_pred,axis=1)

pred_df = pd.DataFrame(Y_pred, columns=['target'])
result = pd.concat([test_df,pred_df], axis=1, join='outer', ignore_index=False, keys=None, sort = False)
result = result[['id','target']]
print(Y_pred)
result.to_csv('sample_submission2.csv',index=False)
