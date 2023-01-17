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
import torch

from torchtext import data

import torch.nn as nn



SEED = 42



torch.manual_seed(SEED)

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False
train_data = pd.read_csv("../input/nlp-getting-started/train.csv")

train_data.info()

train_data.sample(10)
test_data = pd.read_csv("../input/nlp-getting-started/test.csv")

test_data.info()

test_data.sample(10)
print('Training Set Shape = {}'.format(train_data.shape))

print('Test Set Shape = {}'.format(test_data.shape))
import matplotlib.pyplot as plt

import seaborn as sns



target_count = train_data.target.value_counts()

print('Class 0:', target_count[0])

print('Class 1:', target_count[1])

print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')



target_count.plot(kind='bar', title='Count (target)');
mislabeled_df = train_data.groupby(['text']).nunique().sort_values(by='target', ascending=False)

mislabeled_df = mislabeled_df[mislabeled_df['target'] > 1]['target']

mislabeled_list = mislabeled_df.index.tolist()

mislabeled_list
# Relabel mislabeled samples

train_data['target_relabeled'] = train_data['target'].copy() 



train_data.loc[train_data['text'] == 'like for the music video I want some real action shit like burning buildings and police chases not some weak ben winston shit', 'target_relabeled'] = 0

train_data.loc[train_data['text'] == 'Hellfire is surrounded by desires so be careful and donÛªt let your desires control you! #Afterlife', 'target_relabeled'] = 0

train_data.loc[train_data['text'] == 'To fight bioterrorism sir.', 'target_relabeled'] = 0

train_data.loc[train_data['text'] == '.POTUS #StrategicPatience is a strategy for #Genocide; refugees; IDP Internally displaced people; horror; etc. https://t.co/rqWuoy1fm4', 'target_relabeled'] = 1

train_data.loc[train_data['text'] == 'CLEARED:incident with injury:I-495  inner loop Exit 31 - MD 97/Georgia Ave Silver Spring', 'target_relabeled'] = 1

train_data.loc[train_data['text'] == '#foodscare #offers2go #NestleIndia slips into loss after #Magginoodle #ban unsafe and hazardous for #humanconsumption', 'target_relabeled'] = 0

train_data.loc[train_data['text'] == 'In #islam saving a person is equal in reward to saving all humans! Islam is the opposite of terrorism!', 'target_relabeled'] = 0

train_data.loc[train_data['text'] == 'Who is bringing the tornadoes and floods. Who is bringing the climate change. God is after America He is plaguing her\n \n#FARRAKHAN #QUOTE', 'target_relabeled'] = 1

train_data.loc[train_data['text'] == 'RT NotExplained: The only known image of infamous hijacker D.B. Cooper. http://t.co/JlzK2HdeTG', 'target_relabeled'] = 1

train_data.loc[train_data['text'] == "Mmmmmm I'm burning.... I'm burning buildings I'm building.... Oooooohhhh oooh ooh...", 'target_relabeled'] = 0

train_data.loc[train_data['text'] == "wowo--=== 12000 Nigerian refugees repatriated from Cameroon", 'target_relabeled'] = 0

train_data.loc[train_data['text'] == "He came to a land which was engulfed in tribal war and turned it into a land of peace i.e. Madinah. #ProphetMuhammad #islam", 'target_relabeled'] = 0

train_data.loc[train_data['text'] == "Hellfire! We donÛªt even want to think about it or mention it so letÛªs not do anything that leads to it #islam!", 'target_relabeled'] = 0

train_data.loc[train_data['text'] == "The Prophet (peace be upon him) said 'Save yourself from Hellfire even if it is by giving half a date in charity.'", 'target_relabeled'] = 0

train_data.loc[train_data['text'] == "Caution: breathing may be hazardous to your health.", 'target_relabeled'] = 1

train_data.loc[train_data['text'] == "I Pledge Allegiance To The P.O.P.E. And The Burning Buildings of Epic City. ??????", 'target_relabeled'] = 0

train_data.loc[train_data['text'] == "#Allah describes piling up #wealth thinking it would last #forever as the description of the people of #Hellfire in Surah Humaza. #Reflect", 'target_relabeled'] = 0

train_data.loc[train_data['text'] == "that horrible sinking feeling when youÛªve been at home on your phone for a while and you realise its been on 3G this whole time", 'target_relabeled'] = 0
train_data
import re

import string



def clean(tweet): 

            

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

    tweet = re.sub(r"åÇ", "", tweet)

    tweet = re.sub(r"å£3million", "3 million", tweet)

    tweet = re.sub(r"åÀ", "", tweet)

    

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

    tweet = re.sub(r"it's", "it is", tweet)

    tweet = re.sub(r"can't", "cannot", tweet)

    tweet = re.sub(r"don't", "do not", tweet)

    tweet = re.sub(r"you're", "you are", tweet)

    tweet = re.sub(r"i've", "I have", tweet)

    tweet = re.sub(r"that's", "that is", tweet)

    tweet = re.sub(r"i'll", "I will", tweet)

    tweet = re.sub(r"doesn't", "does not", tweet)

    tweet = re.sub(r"i'd", "I would", tweet)

    tweet = re.sub(r"didn't", "did not", tweet)

    tweet = re.sub(r"ain't", "am not", tweet)

    tweet = re.sub(r"you'll", "you will", tweet)

    tweet = re.sub(r"I've", "I have", tweet)

    tweet = re.sub(r"Don't", "do not", tweet)

    tweet = re.sub(r"I'll", "I will", tweet)

    tweet = re.sub(r"I'd", "I would", tweet)

    tweet = re.sub(r"Let's", "Let us", tweet)

    tweet = re.sub(r"you'd", "You would", tweet)

    tweet = re.sub(r"It's", "It is", tweet)

    tweet = re.sub(r"Ain't", "am not", tweet)

    tweet = re.sub(r"Haven't", "Have not", tweet)

    tweet = re.sub(r"Could've", "Could have", tweet)

    tweet = re.sub(r"youve", "you have", tweet)  

    tweet = re.sub(r"donå«t", "do not", tweet)   

            

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

    tweet = re.sub(r"justinbieber", "justin bieber", tweet)  

    tweet = re.sub(r"UTC2015", "UTC 2015", tweet)

    tweet = re.sub(r"Time2015", "Time 2015", tweet)

    tweet = re.sub(r"djicemoon", "dj icemoon", tweet)

    tweet = re.sub(r"LivingSafely", "Living Safely", tweet)

    tweet = re.sub(r"FIFA16", "Fifa 2016", tweet)

    tweet = re.sub(r"thisiswhywecanthavenicethings", "this is why we cannot have nice things", tweet)

    tweet = re.sub(r"bbcnews", "bbc news", tweet)

    tweet = re.sub(r"UndergroundRailraod", "Underground Railraod", tweet)

    tweet = re.sub(r"c4news", "c4 news", tweet)

    tweet = re.sub(r"OBLITERATION", "obliteration", tweet)

    tweet = re.sub(r"MUDSLIDE", "mudslide", tweet)

    tweet = re.sub(r"NoSurrender", "No Surrender", tweet)

    tweet = re.sub(r"NotExplained", "Not Explained", tweet)

    tweet = re.sub(r"greatbritishbakeoff", "great british bake off", tweet)

    tweet = re.sub(r"LondonFire", "London Fire", tweet)

    tweet = re.sub(r"KOTAWeather", "KOTA Weather", tweet)

    tweet = re.sub(r"LuchaUnderground", "Lucha Underground", tweet)

    tweet = re.sub(r"KOIN6News", "KOIN 6 News", tweet)

    tweet = re.sub(r"LiveOnK2", "Live On K2", tweet)

    tweet = re.sub(r"9NewsGoldCoast", "9 News Gold Coast", tweet)

    tweet = re.sub(r"nikeplus", "nike plus", tweet)

    tweet = re.sub(r"david_cameron", "David Cameron", tweet)

    tweet = re.sub(r"peterjukes", "Peter Jukes", tweet)

    tweet = re.sub(r"JamesMelville", "James Melville", tweet)

    tweet = re.sub(r"megynkelly", "Megyn Kelly", tweet)

    tweet = re.sub(r"cnewslive", "C News Live", tweet)

    tweet = re.sub(r"JamaicaObserver", "Jamaica Observer", tweet)

    tweet = re.sub(r"TweetLikeItsSeptember11th2001", "Tweet like it is september 11th 2001", tweet)

    tweet = re.sub(r"cbplawyers", "cbp lawyers", tweet)

    tweet = re.sub(r"fewmoretweets", "few more tweets", tweet)

    tweet = re.sub(r"BlackLivesMatter", "Black Lives Matter", tweet)

    tweet = re.sub(r"cjoyner", "Chris Joyner", tweet)

    tweet = re.sub(r"ENGvAUS", "England vs Australia", tweet)

    tweet = re.sub(r"ScottWalker", "Scott Walker", tweet)

    tweet = re.sub(r"MikeParrActor", "Michael Parr", tweet)

    tweet = re.sub(r"4PlayThursdays", "Foreplay Thursdays", tweet)

    tweet = re.sub(r"TGF2015", "Tontitown Grape Festival", tweet)

    tweet = re.sub(r"realmandyrain", "Mandy Rain", tweet)

    tweet = re.sub(r"GraysonDolan", "Grayson Dolan", tweet)

    tweet = re.sub(r"ApolloBrown", "Apollo Brown", tweet)

    tweet = re.sub(r"saddlebrooke", "Saddlebrooke", tweet)

    tweet = re.sub(r"TontitownGrape", "Tontitown Grape", tweet)

    tweet = re.sub(r"AbbsWinston", "Abbs Winston", tweet)

    tweet = re.sub(r"ShaunKing", "Shaun King", tweet)

    tweet = re.sub(r"MeekMill", "Meek Mill", tweet)

    tweet = re.sub(r"TornadoGiveaway", "Tornado Giveaway", tweet)

    tweet = re.sub(r"GRupdates", "GR updates", tweet)

    tweet = re.sub(r"SouthDowns", "South Downs", tweet)

    tweet = re.sub(r"braininjury", "brain injury", tweet)

    tweet = re.sub(r"auspol", "Australian politics", tweet)

    tweet = re.sub(r"PlannedParenthood", "Planned Parenthood", tweet)

    tweet = re.sub(r"calgaryweather", "Calgary Weather", tweet)

    tweet = re.sub(r"weallheartonedirection", "we all heart one direction", tweet)

    tweet = re.sub(r"edsheeran", "Ed Sheeran", tweet)

    tweet = re.sub(r"TrueHeroes", "True Heroes", tweet)

    tweet = re.sub(r"S3XLEAK", "sex leak", tweet)

    tweet = re.sub(r"ComplexMag", "Complex Magazine", tweet)

    tweet = re.sub(r"TheAdvocateMag", "The Advocate Magazine", tweet)

    tweet = re.sub(r"CityofCalgary", "City of Calgary", tweet)

    tweet = re.sub(r"EbolaOutbreak", "Ebola Outbreak", tweet)

    tweet = re.sub(r"SummerFate", "Summer Fate", tweet)

    tweet = re.sub(r"RAmag", "Royal Academy Magazine", tweet)

    tweet = re.sub(r"offers2go", "offers to go", tweet)

    tweet = re.sub(r"foodscare", "food scare", tweet)

    tweet = re.sub(r"MNPDNashville", "Metropolitan Nashville Police Department", tweet)

    tweet = re.sub(r"TfLBusAlerts", "TfL Bus Alerts", tweet)

    tweet = re.sub(r"GamerGate", "Gamer Gate", tweet)

    tweet = re.sub(r"IHHen", "Humanitarian Relief", tweet)

    tweet = re.sub(r"spinningbot", "spinning bot", tweet)

    tweet = re.sub(r"ModiMinistry", "Modi Ministry", tweet)

    tweet = re.sub(r"TAXIWAYS", "taxi ways", tweet)

    tweet = re.sub(r"Calum5SOS", "Calum Hood", tweet)

    tweet = re.sub(r"po_st", "po.st", tweet)

    tweet = re.sub(r"scoopit", "scoop.it", tweet)

    tweet = re.sub(r"UltimaLucha", "Ultima Lucha", tweet)

    tweet = re.sub(r"JonathanFerrell", "Jonathan Ferrell", tweet)

    tweet = re.sub(r"aria_ahrary", "Aria Ahrary", tweet)

    tweet = re.sub(r"rapidcity", "Rapid City", tweet)

    tweet = re.sub(r"OutBid", "outbid", tweet)

    tweet = re.sub(r"lavenderpoetrycafe", "lavender poetry cafe", tweet)

    tweet = re.sub(r"EudryLantiqua", "Eudry Lantiqua", tweet)

    tweet = re.sub(r"15PM", "15 PM", tweet)

    tweet = re.sub(r"OriginalFunko", "Funko", tweet)

    tweet = re.sub(r"rightwaystan", "Richard Tan", tweet)

    tweet = re.sub(r"CindyNoonan", "Cindy Noonan", tweet)

    tweet = re.sub(r"RT_America", "RT America", tweet)

    tweet = re.sub(r"narendramodi", "Narendra Modi", tweet)

    tweet = re.sub(r"BakeOffFriends", "Bake Off Friends", tweet)

    tweet = re.sub(r"TeamHendrick", "Hendrick Motorsports", tweet)

    tweet = re.sub(r"alexbelloli", "Alex Belloli", tweet)

    tweet = re.sub(r"itsjustinstuart", "Justin Stuart", tweet)

    tweet = re.sub(r"gunsense", "gun sense", tweet)

    tweet = re.sub(r"DebateQuestionsWeWantToHear", "debate questions we want to hear", tweet)

    tweet = re.sub(r"RoyalCarribean", "Royal Carribean", tweet)

    tweet = re.sub(r"samanthaturne19", "Samantha Turner", tweet)

    tweet = re.sub(r"JonVoyage", "Jon Stewart", tweet)

    tweet = re.sub(r"renew911health", "renew 911 health", tweet)

    tweet = re.sub(r"SuryaRay", "Surya Ray", tweet)

    tweet = re.sub(r"pattonoswalt", "Patton Oswalt", tweet)

    tweet = re.sub(r"minhazmerchant", "Minhaz Merchant", tweet)

    tweet = re.sub(r"TLVFaces", "Israel Diaspora Coalition", tweet)

    tweet = re.sub(r"pmarca", "Marc Andreessen", tweet)

    tweet = re.sub(r"pdx911", "Portland Police", tweet)

    tweet = re.sub(r"jamaicaplain", "Jamaica Plain", tweet)

    tweet = re.sub(r"Japton", "Arkansas", tweet)

    tweet = re.sub(r"RouteComplex", "Route Complex", tweet)

    tweet = re.sub(r"INSubcontinent", "Indian Subcontinent", tweet)

    tweet = re.sub(r"NJTurnpike", "New Jersey Turnpike", tweet)

    tweet = re.sub(r"Politifiact", "PolitiFact", tweet)

    tweet = re.sub(r"Hiroshima70", "Hiroshima", tweet)

    tweet = re.sub(r"GMMBC", "Greater Mt Moriah Baptist Church", tweet)

    tweet = re.sub(r"versethe", "verse the", tweet)

    tweet = re.sub(r"TubeStrike", "Tube Strike", tweet)

    tweet = re.sub(r"MissionHills", "Mission Hills", tweet)

    tweet = re.sub(r"ProtectDenaliWolves", "Protect Denali Wolves", tweet)

    tweet = re.sub(r"NANKANA", "Nankana", tweet)

    tweet = re.sub(r"SAHIB", "Sahib", tweet)

    tweet = re.sub(r"PAKPATTAN", "Pakpattan", tweet)

    tweet = re.sub(r"Newz_Sacramento", "News Sacramento", tweet)

    tweet = re.sub(r"gofundme", "go fund me", tweet)

    tweet = re.sub(r"pmharper", "Stephen Harper", tweet)

    tweet = re.sub(r"IvanBerroa", "Ivan Berroa", tweet)

    tweet = re.sub(r"LosDelSonido", "Los Del Sonido", tweet)

    tweet = re.sub(r"bancodeseries", "banco de series", tweet)

    tweet = re.sub(r"timkaine", "Tim Kaine", tweet)

    tweet = re.sub(r"IdentityTheft", "Identity Theft", tweet)

    tweet = re.sub(r"AllLivesMatter", "All Lives Matter", tweet)

    tweet = re.sub(r"mishacollins", "Misha Collins", tweet)

    tweet = re.sub(r"BillNeelyNBC", "Bill Neely", tweet)

    tweet = re.sub(r"BeClearOnCancer", "be clear on cancer", tweet)

    tweet = re.sub(r"Kowing", "Knowing", tweet)

    tweet = re.sub(r"ScreamQueens", "Scream Queens", tweet)

    tweet = re.sub(r"AskCharley", "Ask Charley", tweet)

    tweet = re.sub(r"BlizzHeroes", "Heroes of the Storm", tweet)

    tweet = re.sub(r"BradleyBrad47", "Bradley Brad", tweet)

    tweet = re.sub(r"HannaPH", "Typhoon Hanna", tweet)

    tweet = re.sub(r"meinlcymbals", "MEINL Cymbals", tweet)

    tweet = re.sub(r"Ptbo", "Peterborough", tweet)

    tweet = re.sub(r"cnnbrk", "CNN Breaking News", tweet)

    tweet = re.sub(r"IndianNews", "Indian News", tweet)

    tweet = re.sub(r"savebees", "save bees", tweet)

    tweet = re.sub(r"GreenHarvard", "Green Harvard", tweet)

    tweet = re.sub(r"StandwithPP", "Stand with planned parenthood", tweet)

    tweet = re.sub(r"hermancranston", "Herman Cranston", tweet)

    tweet = re.sub(r"WMUR9", "WMUR-TV", tweet)

    tweet = re.sub(r"RockBottomRadFM", "Rock Bottom Radio", tweet)

    tweet = re.sub(r"ameenshaikh3", "Ameen Shaikh", tweet)

    tweet = re.sub(r"ProSyn", "Project Syndicate", tweet)

    tweet = re.sub(r"Daesh", "ISIS", tweet)

    tweet = re.sub(r"s2g", "swear to god", tweet)

    tweet = re.sub(r"listenlive", "listen live", tweet)

    tweet = re.sub(r"CDCgov", "Centers for Disease Control and Prevention", tweet)

    tweet = re.sub(r"FoxNew", "Fox News", tweet)

    tweet = re.sub(r"CBSBigBrother", "Big Brother", tweet)

    tweet = re.sub(r"JulieDiCaro", "Julie DiCaro", tweet)

    tweet = re.sub(r"theadvocatemag", "The Advocate Magazine", tweet)

    tweet = re.sub(r"RohnertParkDPS", "Rohnert Park Police Department", tweet)

    tweet = re.sub(r"THISIZBWRIGHT", "Bonnie Wright", tweet)

    tweet = re.sub(r"Popularmmos", "Popular MMOs", tweet)

    tweet = re.sub(r"WildHorses", "Wild Horses", tweet)

    tweet = re.sub(r"FantasticFour", "Fantastic Four", tweet)

    tweet = re.sub(r"HORNDALE", "Horndale", tweet)

    tweet = re.sub(r"PINER", "Piner", tweet)

    tweet = re.sub(r"BathAndNorthEastSomerset", "Bath and North East Somerset", tweet)

    tweet = re.sub(r"thatswhatfriendsarefor", "that is what friends are for", tweet)

    tweet = re.sub(r"residualincome", "residual income", tweet)

    tweet = re.sub(r"YahooNewsDigest", "Yahoo News Digest", tweet)

    tweet = re.sub(r"MalaysiaAirlines", "Malaysia Airlines", tweet)

    tweet = re.sub(r"AmazonDeals", "Amazon Deals", tweet)

    tweet = re.sub(r"MissCharleyWebb", "Charley Webb", tweet)

    tweet = re.sub(r"shoalstraffic", "shoals traffic", tweet)

    tweet = re.sub(r"GeorgeFoster72", "George Foster", tweet)

    tweet = re.sub(r"pop2015", "pop 2015", tweet)

    tweet = re.sub(r"_PokemonCards_", "Pokemon Cards", tweet)

    tweet = re.sub(r"DianneG", "Dianne Gallagher", tweet)

    tweet = re.sub(r"KashmirConflict", "Kashmir Conflict", tweet)

    tweet = re.sub(r"BritishBakeOff", "British Bake Off", tweet)

    tweet = re.sub(r"FreeKashmir", "Free Kashmir", tweet)

    tweet = re.sub(r"mattmosley", "Matt Mosley", tweet)

    tweet = re.sub(r"BishopFred", "Bishop Fred", tweet)

    tweet = re.sub(r"EndConflict", "End Conflict", tweet)

    tweet = re.sub(r"EndOccupation", "End Occupation", tweet)

    tweet = re.sub(r"UNHEALED", "unhealed", tweet)

    tweet = re.sub(r"CharlesDagnall", "Charles Dagnall", tweet)

    tweet = re.sub(r"Latestnews", "Latest news", tweet)

    tweet = re.sub(r"KindleCountdown", "Kindle Countdown", tweet)

    tweet = re.sub(r"NoMoreHandouts", "No More Handouts", tweet)

    tweet = re.sub(r"datingtips", "dating tips", tweet)

    tweet = re.sub(r"charlesadler", "Charles Adler", tweet)

    tweet = re.sub(r"twia", "Texas Windstorm Insurance Association", tweet)

    tweet = re.sub(r"txlege", "Texas Legislature", tweet)

    tweet = re.sub(r"WindstormInsurer", "Windstorm Insurer", tweet)

    tweet = re.sub(r"Newss", "News", tweet)

    tweet = re.sub(r"hempoil", "hemp oil", tweet)

    tweet = re.sub(r"CommoditiesAre", "Commodities are", tweet)

    tweet = re.sub(r"tubestrike", "tube strike", tweet)

    tweet = re.sub(r"JoeNBC", "Joe Scarborough", tweet)

    tweet = re.sub(r"LiteraryCakes", "Literary Cakes", tweet)

    tweet = re.sub(r"TI5", "The International 5", tweet)

    tweet = re.sub(r"thehill", "the hill", tweet)

    tweet = re.sub(r"3others", "3 others", tweet)

    tweet = re.sub(r"stighefootball", "Sam Tighe", tweet)

    tweet = re.sub(r"whatstheimportantvideo", "what is the important video", tweet)

    tweet = re.sub(r"ClaudioMeloni", "Claudio Meloni", tweet)

    tweet = re.sub(r"DukeSkywalker", "Duke Skywalker", tweet)

    tweet = re.sub(r"carsonmwr", "Fort Carson", tweet)

    tweet = re.sub(r"offdishduty", "off dish duty", tweet)

    tweet = re.sub(r"andword", "and word", tweet)

    tweet = re.sub(r"rhodeisland", "Rhode Island", tweet)

    tweet = re.sub(r"easternoregon", "Eastern Oregon", tweet)

    tweet = re.sub(r"WAwildfire", "Washington Wildfire", tweet)

    tweet = re.sub(r"fingerrockfire", "Finger Rock Fire", tweet)

    tweet = re.sub(r"57am", "57 am", tweet)

    tweet = re.sub(r"fingerrockfire", "Finger Rock Fire", tweet)

    tweet = re.sub(r"JacobHoggard", "Jacob Hoggard", tweet)

    tweet = re.sub(r"newnewnew", "new new new", tweet)

    tweet = re.sub(r"under50", "under 50", tweet)

    tweet = re.sub(r"getitbeforeitsgone", "get it before it is gone", tweet)

    tweet = re.sub(r"freshoutofthebox", "fresh out of the box", tweet)

    tweet = re.sub(r"amwriting", "am writing", tweet)

    tweet = re.sub(r"Bokoharm", "Boko Haram", tweet)

    tweet = re.sub(r"Nowlike", "Now like", tweet)

    tweet = re.sub(r"seasonfrom", "season from", tweet)

    tweet = re.sub(r"epicente", "epicenter", tweet)

    tweet = re.sub(r"epicenterr", "epicenter", tweet)

    tweet = re.sub(r"sicklife", "sick life", tweet)

    tweet = re.sub(r"yycweather", "Calgary Weather", tweet)

    tweet = re.sub(r"calgarysun", "Calgary Sun", tweet)

    tweet = re.sub(r"approachng", "approaching", tweet)

    tweet = re.sub(r"evng", "evening", tweet)

    tweet = re.sub(r"Sumthng", "something", tweet)

    tweet = re.sub(r"EllenPompeo", "Ellen Pompeo", tweet)

    tweet = re.sub(r"shondarhimes", "Shonda Rhimes", tweet)

    tweet = re.sub(r"ABCNetwork", "ABC Network", tweet)

    tweet = re.sub(r"SushmaSwaraj", "Sushma Swaraj", tweet)

    tweet = re.sub(r"pray4japan", "Pray for Japan", tweet)

    tweet = re.sub(r"hope4japan", "Hope for Japan", tweet)

    tweet = re.sub(r"Illusionimagess", "Illusion images", tweet)

    tweet = re.sub(r"SummerUnderTheStars", "Summer Under The Stars", tweet)

    tweet = re.sub(r"ShallWeDance", "Shall We Dance", tweet)

    tweet = re.sub(r"TCMParty", "TCM Party", tweet)

    tweet = re.sub(r"marijuananews", "marijuana news", tweet)

    tweet = re.sub(r"onbeingwithKristaTippett", "on being with Krista Tippett", tweet)

    tweet = re.sub(r"Beingtweets", "Being tweets", tweet)

    tweet = re.sub(r"newauthors", "new authors", tweet)

    tweet = re.sub(r"remedyyyy", "remedy", tweet)

    tweet = re.sub(r"44PM", "44 PM", tweet)

    tweet = re.sub(r"HeadlinesApp", "Headlines App", tweet)

    tweet = re.sub(r"40PM", "40 PM", tweet)

    tweet = re.sub(r"myswc", "Severe Weather Center", tweet)

    tweet = re.sub(r"ithats", "that is", tweet)

    tweet = re.sub(r"icouldsitinthismomentforever", "I could sit in this moment forever", tweet)

    tweet = re.sub(r"FatLoss", "Fat Loss", tweet)

    tweet = re.sub(r"02PM", "02 PM", tweet)

    tweet = re.sub(r"MetroFmTalk", "Metro Fm Talk", tweet)

    tweet = re.sub(r"Bstrd", "bastard", tweet)

    tweet = re.sub(r"bldy", "bloody", tweet)

    tweet = re.sub(r"MetrofmTalk", "Metro Fm Talk", tweet)

    tweet = re.sub(r"terrorismturn", "terrorism turn", tweet)

    tweet = re.sub(r"BBCNewsAsia", "BBC News Asia", tweet)

    tweet = re.sub(r"BehindTheScenes", "Behind The Scenes", tweet)

    tweet = re.sub(r"GeorgeTakei", "George Takei", tweet)

    tweet = re.sub(r"WomensWeeklyMag", "Womens Weekly Magazine", tweet)

    tweet = re.sub(r"SurvivorsGuidetoEarth", "Survivors Guide to Earth", tweet)

    tweet = re.sub(r"incubusband", "incubus band", tweet)

    tweet = re.sub(r"Babypicturethis", "Baby picture this", tweet)

    tweet = re.sub(r"BombEffects", "Bomb Effects", tweet)

    tweet = re.sub(r"win10", "Windows 10", tweet)

    tweet = re.sub(r"idkidk", "I do not know I do not know", tweet)

    tweet = re.sub(r"TheWalkingDead", "The Walking Dead", tweet)

    tweet = re.sub(r"amyschumer", "Amy Schumer", tweet)

    tweet = re.sub(r"crewlist", "crew list", tweet)

    tweet = re.sub(r"Erdogans", "Erdogan", tweet)

    tweet = re.sub(r"BBCLive", "BBC Live", tweet)

    tweet = re.sub(r"TonyAbbottMHR", "Tony Abbott", tweet)

    tweet = re.sub(r"paulmyerscough", "Paul Myerscough", tweet)

    tweet = re.sub(r"georgegallagher", "George Gallagher", tweet)

    tweet = re.sub(r"JimmieJohnson", "Jimmie Johnson", tweet)

    tweet = re.sub(r"pctool", "pc tool", tweet)

    tweet = re.sub(r"DoingHashtagsRight", "Doing Hashtags Right", tweet)

    tweet = re.sub(r"ThrowbackThursday", "Throwback Thursday", tweet)

    tweet = re.sub(r"SnowBackSunday", "Snowback Sunday", tweet)

    tweet = re.sub(r"LakeEffect", "Lake Effect", tweet)

    tweet = re.sub(r"RTphotographyUK", "Richard Thomas Photography UK", tweet)

    tweet = re.sub(r"BigBang_CBS", "Big Bang CBS", tweet)

    tweet = re.sub(r"writerslife", "writers life", tweet)

    tweet = re.sub(r"NaturalBirth", "Natural Birth", tweet)

    tweet = re.sub(r"UnusualWords", "Unusual Words", tweet)

    tweet = re.sub(r"wizkhalifa", "Wiz Khalifa", tweet)

    tweet = re.sub(r"acreativedc", "a creative DC", tweet)

    tweet = re.sub(r"vscodc", "vsco DC", tweet)

    tweet = re.sub(r"VSCOcam", "vsco camera", tweet)

    tweet = re.sub(r"TheBEACHDC", "The beach DC", tweet)

    tweet = re.sub(r"buildingmuseum", "building museum", tweet)

    tweet = re.sub(r"WorldOil", "World Oil", tweet)

    tweet = re.sub(r"redwedding", "red wedding", tweet)

    tweet = re.sub(r"AmazingRaceCanada", "Amazing Race Canada", tweet)

    tweet = re.sub(r"WakeUpAmerica", "Wake Up America", tweet)

    tweet = re.sub(r"\\Allahuakbar\\", "Allahu Akbar", tweet)

    tweet = re.sub(r"bleased", "blessed", tweet)

    tweet = re.sub(r"nigeriantribune", "Nigerian Tribune", tweet)

    tweet = re.sub(r"HIDEO_KOJIMA_EN", "Hideo Kojima", tweet)

    tweet = re.sub(r"FusionFestival", "Fusion Festival", tweet)

    tweet = re.sub(r"50Mixed", "50 Mixed", tweet)

    tweet = re.sub(r"NoAgenda", "No Agenda", tweet)

    tweet = re.sub(r"WhiteGenocide", "White Genocide", tweet)

    tweet = re.sub(r"dirtylying", "dirty lying", tweet)

    tweet = re.sub(r"SyrianRefugees", "Syrian Refugees", tweet)

    tweet = re.sub(r"changetheworld", "change the world", tweet)

    tweet = re.sub(r"Ebolacase", "Ebola case", tweet)

    tweet = re.sub(r"mcgtech", "mcg technologies", tweet)

    tweet = re.sub(r"withweapons", "with weapons", tweet)

    tweet = re.sub(r"advancedwarfare", "advanced warfare", tweet)

    tweet = re.sub(r"letsFootball", "let us Football", tweet)

    tweet = re.sub(r"LateNiteMix", "late night mix", tweet)

    tweet = re.sub(r"PhilCollinsFeed", "Phil Collins", tweet)

    tweet = re.sub(r"RudyHavenstein", "Rudy Havenstein", tweet)

    tweet = re.sub(r"22PM", "22 PM", tweet)

    tweet = re.sub(r"54am", "54 AM", tweet)

    tweet = re.sub(r"38am", "38 AM", tweet)

    tweet = re.sub(r"OldFolkExplainStuff", "Old Folk Explain Stuff", tweet)

    tweet = re.sub(r"BlacklivesMatter", "Black Lives Matter", tweet)

    tweet = re.sub(r"InsaneLimits", "Insane Limits", tweet)

    tweet = re.sub(r"youcantsitwithus", "you cannot sit with us", tweet)

    tweet = re.sub(r"2k15", "2015", tweet)

    tweet = re.sub(r"TheIran", "Iran", tweet)

    tweet = re.sub(r"JimmyFallon", "Jimmy Fallon", tweet)

    tweet = re.sub(r"AlbertBrooks", "Albert Brooks", tweet)

    tweet = re.sub(r"defense_news", "defense news", tweet)

    tweet = re.sub(r"nuclearrcSA", "Nuclear Risk Control Self Assessment", tweet)

    tweet = re.sub(r"Auspol", "Australia Politics", tweet)

    tweet = re.sub(r"NuclearPower", "Nuclear Power", tweet)

    tweet = re.sub(r"WhiteTerrorism", "White Terrorism", tweet)

    tweet = re.sub(r"truthfrequencyradio", "Truth Frequency Radio", tweet)

    tweet = re.sub(r"ErasureIsNotEquality", "Erasure is not equality", tweet)

    tweet = re.sub(r"ProBonoNews", "Pro Bono News", tweet)

    tweet = re.sub(r"JakartaPost", "Jakarta Post", tweet)

    tweet = re.sub(r"toopainful", "too painful", tweet)

    tweet = re.sub(r"melindahaunton", "Melinda Haunton", tweet)

    tweet = re.sub(r"NoNukes", "No Nukes", tweet)

    tweet = re.sub(r"curryspcworld", "Currys PC World", tweet)

    tweet = re.sub(r"ineedcake", "I need cake", tweet)

    tweet = re.sub(r"blackforestgateau", "black forest gateau", tweet)

    tweet = re.sub(r"BBCOne", "BBC One", tweet)

    tweet = re.sub(r"AlexxPage", "Alex Page", tweet)

    tweet = re.sub(r"jonathanserrie", "Jonathan Serrie", tweet)

    tweet = re.sub(r"SocialJerkBlog", "Social Jerk Blog", tweet)

    tweet = re.sub(r"ChelseaVPeretti", "Chelsea Peretti", tweet)

    tweet = re.sub(r"irongiant", "iron giant", tweet)

    tweet = re.sub(r"RonFunches", "Ron Funches", tweet)

    tweet = re.sub(r"TimCook", "Tim Cook", tweet)

    tweet = re.sub(r"sebastianstanisaliveandwell", "Sebastian Stan is alive and well", tweet)

    tweet = re.sub(r"Madsummer", "Mad summer", tweet)

    tweet = re.sub(r"NowYouKnow", "Now you know", tweet)

    tweet = re.sub(r"concertphotography", "concert photography", tweet)

    tweet = re.sub(r"TomLandry", "Tom Landry", tweet)

    tweet = re.sub(r"showgirldayoff", "show girl day off", tweet)

    tweet = re.sub(r"Yougslavia", "Yugoslavia", tweet)

    tweet = re.sub(r"QuantumDataInformatics", "Quantum Data Informatics", tweet)

    tweet = re.sub(r"FromTheDesk", "From The Desk", tweet)

    tweet = re.sub(r"TheaterTrial", "Theater Trial", tweet)

    tweet = re.sub(r"CatoInstitute", "Cato Institute", tweet)

    tweet = re.sub(r"EmekaGift", "Emeka Gift", tweet)

    tweet = re.sub(r"LetsBe_Rational", "Let us be rational", tweet)

    tweet = re.sub(r"Cynicalreality", "Cynical reality", tweet)

    tweet = re.sub(r"FredOlsenCruise", "Fred Olsen Cruise", tweet)

    tweet = re.sub(r"NotSorry", "not sorry", tweet)

    tweet = re.sub(r"UseYourWords", "use your words", tweet)

    tweet = re.sub(r"WordoftheDay", "word of the day", tweet)

    tweet = re.sub(r"Dictionarycom", "Dictionary.com", tweet)

    tweet = re.sub(r"TheBrooklynLife", "The Brooklyn Life", tweet)

    tweet = re.sub(r"jokethey", "joke they", tweet)

    tweet = re.sub(r"nflweek1picks", "NFL week 1 picks", tweet)

    tweet = re.sub(r"uiseful", "useful", tweet)

    tweet = re.sub(r"JusticeDotOrg", "The American Association for Justice", tweet)

    tweet = re.sub(r"autoaccidents", "auto accidents", tweet)

    tweet = re.sub(r"SteveGursten", "Steve Gursten", tweet)

    tweet = re.sub(r"MichiganAutoLaw", "Michigan Auto Law", tweet)

    tweet = re.sub(r"birdgang", "bird gang", tweet)

    tweet = re.sub(r"nflnetwork", "NFL Network", tweet)

    tweet = re.sub(r"NYDNSports", "NY Daily News Sports", tweet)

    tweet = re.sub(r"RVacchianoNYDN", "Ralph Vacchiano NY Daily News", tweet)

    tweet = re.sub(r"EdmontonEsks", "Edmonton Eskimos", tweet)

    tweet = re.sub(r"david_brelsford", "David Brelsford", tweet)

    tweet = re.sub(r"TOI_India", "The Times of India", tweet)

    tweet = re.sub(r"hegot", "he got", tweet)

    tweet = re.sub(r"SkinsOn9", "Skins on 9", tweet)

    tweet = re.sub(r"sothathappened", "so that happened", tweet)

    tweet = re.sub(r"LCOutOfDoors", "LC Out Of Doors", tweet)

    tweet = re.sub(r"NationFirst", "Nation First", tweet)

    tweet = re.sub(r"IndiaToday", "India Today", tweet)

    tweet = re.sub(r"HLPS", "helps", tweet)

    tweet = re.sub(r"HOSTAGESTHROSW", "hostages throw", tweet)

    tweet = re.sub(r"SNCTIONS", "sanctions", tweet)

    tweet = re.sub(r"BidTime", "Bid Time", tweet)

    tweet = re.sub(r"crunchysensible", "crunchy sensible", tweet)

    tweet = re.sub(r"RandomActsOfRomance", "Random acts of romance", tweet)

    tweet = re.sub(r"MomentsAtHill", "Moments at hill", tweet)

    tweet = re.sub(r"eatshit", "eat shit", tweet)

    tweet = re.sub(r"liveleakfun", "live leak fun", tweet)

    tweet = re.sub(r"SahelNews", "Sahel News", tweet)

    tweet = re.sub(r"abc7newsbayarea", "ABC 7 News Bay Area", tweet)

    tweet = re.sub(r"facilitiesmanagement", "facilities management", tweet)

    tweet = re.sub(r"facilitydude", "facility dude", tweet)

    tweet = re.sub(r"CampLogistics", "Camp logistics", tweet)

    tweet = re.sub(r"alaskapublic", "Alaska public", tweet)

    tweet = re.sub(r"MarketResearch", "Market Research", tweet)

    tweet = re.sub(r"AccuracyEsports", "Accuracy Esports", tweet)

    tweet = re.sub(r"TheBodyShopAust", "The Body Shop Australia", tweet)

    tweet = re.sub(r"yychail", "Calgary hail", tweet)

    tweet = re.sub(r"yyctraffic", "Calgary traffic", tweet)

    tweet = re.sub(r"eliotschool", "eliot school", tweet)

    tweet = re.sub(r"TheBrokenCity", "The Broken City", tweet)

    tweet = re.sub(r"OldsFireDept", "Olds Fire Department", tweet)

    tweet = re.sub(r"RiverComplex", "River Complex", tweet)

    tweet = re.sub(r"fieldworksmells", "field work smells", tweet)

    tweet = re.sub(r"IranElection", "Iran Election", tweet)

    tweet = re.sub(r"glowng", "glowing", tweet)

    tweet = re.sub(r"kindlng", "kindling", tweet)

    tweet = re.sub(r"riggd", "rigged", tweet)

    tweet = re.sub(r"slownewsday", "slow news day", tweet)

    tweet = re.sub(r"MyanmarFlood", "Myanmar Flood", tweet)

    tweet = re.sub(r"abc7chicago", "ABC 7 Chicago", tweet)

    tweet = re.sub(r"copolitics", "Colorado Politics", tweet)

    tweet = re.sub(r"AdilGhumro", "Adil Ghumro", tweet)

    tweet = re.sub(r"netbots", "net bots", tweet)

    tweet = re.sub(r"byebyeroad", "bye bye road", tweet)

    tweet = re.sub(r"massiveflooding", "massive flooding", tweet)

    tweet = re.sub(r"EndofUS", "End of United States", tweet)

    tweet = re.sub(r"35PM", "35 PM", tweet)

    tweet = re.sub(r"greektheatrela", "Greek Theatre Los Angeles", tweet)

    tweet = re.sub(r"76mins", "76 minutes", tweet)

    tweet = re.sub(r"publicsafetyfirst", "public safety first", tweet)

    tweet = re.sub(r"livesmatter", "lives matter", tweet)

    tweet = re.sub(r"myhometown", "my hometown", tweet)

    tweet = re.sub(r"tankerfire", "tanker fire", tweet)

    tweet = re.sub(r"MEMORIALDAY", "memorial day", tweet)

    tweet = re.sub(r"MEMORIAL_DAY", "memorial day", tweet)

    tweet = re.sub(r"instaxbooty", "instagram booty", tweet)

    tweet = re.sub(r"Jerusalem_Post", "Jerusalem Post", tweet)

    tweet = re.sub(r"WayneRooney_INA", "Wayne Rooney", tweet)

    tweet = re.sub(r"VirtualReality", "Virtual Reality", tweet)

    tweet = re.sub(r"OculusRift", "Oculus Rift", tweet)

    tweet = re.sub(r"OwenJones84", "Owen Jones", tweet)

    tweet = re.sub(r"jeremycorbyn", "Jeremy Corbyn", tweet)

    tweet = re.sub(r"paulrogers002", "Paul Rogers", tweet)

    tweet = re.sub(r"mortalkombatx", "Mortal Kombat X", tweet)

    tweet = re.sub(r"mortalkombat", "Mortal Kombat", tweet)

    tweet = re.sub(r"FilipeCoelho92", "Filipe Coelho", tweet)

    tweet = re.sub(r"OnlyQuakeNews", "Only Quake News", tweet)

    tweet = re.sub(r"kostumes", "costumes", tweet)

    tweet = re.sub(r"YEEESSSS", "yes", tweet)

    tweet = re.sub(r"ToshikazuKatayama", "Toshikazu Katayama", tweet)

    tweet = re.sub(r"IntlDevelopment", "Intl Development", tweet)

    tweet = re.sub(r"ExtremeWeather", "Extreme Weather", tweet)

    tweet = re.sub(r"WereNotGruberVoters", "We are not gruber voters", tweet)

    tweet = re.sub(r"NewsThousands", "News Thousands", tweet)

    tweet = re.sub(r"EdmundAdamus", "Edmund Adamus", tweet)

    tweet = re.sub(r"EyewitnessWV", "Eye witness WV", tweet)

    tweet = re.sub(r"PhiladelphiaMuseu", "Philadelphia Museum", tweet)

    tweet = re.sub(r"DublinComicCon", "Dublin Comic Con", tweet)

    tweet = re.sub(r"NicholasBrendon", "Nicholas Brendon", tweet)

    tweet = re.sub(r"Alltheway80s", "All the way 80s", tweet)

    tweet = re.sub(r"FromTheField", "From the field", tweet)

    tweet = re.sub(r"NorthIowa", "North Iowa", tweet)

    tweet = re.sub(r"WillowFire", "Willow Fire", tweet)

    tweet = re.sub(r"MadRiverComplex", "Mad River Complex", tweet)

    tweet = re.sub(r"feelingmanly", "feeling manly", tweet)

    tweet = re.sub(r"stillnotoverit", "still not over it", tweet)

    tweet = re.sub(r"FortitudeValley", "Fortitude Valley", tweet)

    tweet = re.sub(r"CoastpowerlineTramTr", "Coast powerline", tweet)

    tweet = re.sub(r"ServicesGold", "Services Gold", tweet)

    tweet = re.sub(r"NewsbrokenEmergency", "News broken emergency", tweet)

    tweet = re.sub(r"Evaucation", "evacuation", tweet)

    tweet = re.sub(r"leaveevacuateexitbe", "leave evacuate exit be", tweet)

    tweet = re.sub(r"P_EOPLE", "PEOPLE", tweet)

    tweet = re.sub(r"Tubestrike", "tube strike", tweet)

    tweet = re.sub(r"CLASS_SICK", "CLASS SICK", tweet)

    tweet = re.sub(r"localplumber", "local plumber", tweet)

    tweet = re.sub(r"awesomejobsiri", "awesome job siri", tweet)

    tweet = re.sub(r"PayForItHow", "Pay for it how", tweet)

    tweet = re.sub(r"ThisIsAfrica", "This is Africa", tweet)

    tweet = re.sub(r"crimeairnetwork", "crime air network", tweet)

    tweet = re.sub(r"KimAcheson", "Kim Acheson", tweet)

    tweet = re.sub(r"cityofcalgary", "City of Calgary", tweet)

    tweet = re.sub(r"prosyndicate", "pro syndicate", tweet)

    tweet = re.sub(r"660NEWS", "660 NEWS", tweet)

    tweet = re.sub(r"BusInsMagazine", "Business Insurance Magazine", tweet)

    tweet = re.sub(r"wfocus", "focus", tweet)

    tweet = re.sub(r"ShastaDam", "Shasta Dam", tweet)

    tweet = re.sub(r"go2MarkFranco", "Mark Franco", tweet)

    tweet = re.sub(r"StephGHinojosa", "Steph Hinojosa", tweet)

    tweet = re.sub(r"Nashgrier", "Nash Grier", tweet)

    tweet = re.sub(r"NashNewVideo", "Nash new video", tweet)

    tweet = re.sub(r"IWouldntGetElectedBecause", "I would not get elected because", tweet)

    tweet = re.sub(r"SHGames", "Sledgehammer Games", tweet)

    tweet = re.sub(r"bedhair", "bed hair", tweet)

    tweet = re.sub(r"JoelHeyman", "Joel Heyman", tweet)

    tweet = re.sub(r"viaYouTube", "via YouTube", tweet)

    

    return tweet



def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)



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



def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)





def clean_data(df):

    df['text']=df['text'].apply(lambda x : remove_URL(x))

    df['text']=df['text'].apply(lambda x : remove_html(x))

    df['text']=df['text'].apply(lambda x: remove_emoji(x))

    df['text']=df['text'].apply(lambda x : remove_punct(x))

    df['text']=df['text'].apply(lambda x : clean(x))

    

    return df



train_data = clean_data(train_data)

test_data = clean_data(test_data)
from sklearn.model_selection import train_test_split

import random



# split data into train and validation 

train_df, valid_df = train_test_split(train_data, test_size=0.20, random_state= random.seed(SEED))
TEXT = data.Field(tokenize = 'spacy', batch_first=True, include_lengths = True)

LABEL = data.LabelField(dtype = torch.float, batch_first=True)
class DataFrameDataset(data.Dataset):



    def __init__(self, df, fields, is_test=False, **kwargs):

        examples = []

        for i, row in df.iterrows():

            label = row.target_relabeled if not is_test else None

            text = row.text

            examples.append(data.Example.fromlist([text, label], fields))



        super().__init__(examples, fields, **kwargs)



    @staticmethod

    def sort_key(ex):

        return len(ex.text)



    @classmethod

    def splits(cls, fields, train_df, val_df=None, test_df=None, **kwargs):

        train_data, val_data, test_data = (None, None, None)

        data_field = fields



        if train_df is not None:

            train_data = cls(train_df.copy(), data_field, **kwargs)

        if val_df is not None:

            val_data = cls(val_df.copy(), data_field, **kwargs)

        if test_df is not None:

            test_data = cls(test_df.copy(), data_field, False, **kwargs)



        return tuple(d for d in (train_data, val_data, test_data) if d is not None)
fields = [('text',TEXT), ('label',LABEL)]



train_ds, val_ds = DataFrameDataset.splits(fields, train_df=train_df, val_df=valid_df)
# Lets look at a random example

print(vars(train_ds[15]))



# Check the type 

print(type(train_ds[15]))
from torchtext.vocab import Vectors



vectors = Vectors(name='../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec', cache='./')



MAX_VOCAB_SIZE = 100000



TEXT.build_vocab(train_ds, 

                 max_size = MAX_VOCAB_SIZE, 

                 vectors = vectors,

                 unk_init = torch.Tensor.zero_)

LABEL.build_vocab(train_ds)
#No. of unique tokens in text

print("Size of TEXT vocabulary:",len(TEXT.vocab))



#No. of unique tokens in label

print("Size of LABEL vocabulary:",len(LABEL.vocab))



#Commonly used words

print(TEXT.vocab.freqs.most_common(10))  



#Word dictionary

# print(TEXT.vocab.stoi)   
BATCH_SIZE = 64



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



train_iterator, valid_iterator = data.BucketIterator.splits(

    (train_ds, val_ds), 

    batch_size = BATCH_SIZE,

    sort_within_batch = True,

    device = device)
# Hyperparameters

num_epochs = 25

learning_rate = 0.001



INPUT_DIM = len(TEXT.vocab)

EMBEDDING_DIM = 300

HIDDEN_DIM = 256

OUTPUT_DIM = 1

N_LAYERS = 2

BIDIRECTIONAL = True

DROPOUT = 0.2

PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token] # padding
class LSTM_net(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 

                 bidirectional, dropout, pad_idx):

        

        super().__init__()

        

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)

        

        self.rnn = nn.LSTM(embedding_dim, 

                           hidden_dim, 

                           num_layers=n_layers, 

                           bidirectional=bidirectional, 

                           dropout=dropout,

                           batch_first=True)

        

        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)

        

        self.fc2 = nn.Linear(hidden_dim, 1)

        

        self.dropout = nn.Dropout(dropout)



        

    def forward(self, text, text_lengths):

        

        # text = [batch size, sent len]

        

        embedded = self.embedding(text)

        # embedded = [batch size, sent len, emb dim]

        

        

        #pack sequence

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True)

        

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        

        #unpack sequence

        # output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)



        # output = [sent len, batch size, hid dim * num directions]

        # output over padding tokens are zero tensors

        

        # hidden = [num layers * num directions, batch size, hid dim]

        # cell = [num layers * num directions, batch size, hid dim]

        

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers

        # and apply dropout

        

        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))

        output = self.fc1(hidden)

        output = self.dropout(self.fc2(output))

                

        #hidden = [batch size, hid dim * num directions]

            

        return output
#creating instance of our LSTM_net class



model = LSTM_net(INPUT_DIM, 

            EMBEDDING_DIM, 

            HIDDEN_DIM, 

            OUTPUT_DIM, 

            N_LAYERS, 

            BIDIRECTIONAL, 

            DROPOUT, 

            PAD_IDX)
print(model)



#No. of trianable parameters

def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

    

print(f'The model has {count_parameters(model):,} trainable parameters')
pretrained_embeddings = TEXT.vocab.vectors

model.embedding.weight.data.copy_(pretrained_embeddings)



#  to initiaise padded to zeros

model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
def binary_accuracy(preds, y):

    """

    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8

    """



    #round predictions to the closest integer

    rounded_preds = torch.round(torch.sigmoid(preds))

    correct = (rounded_preds == y).float() #convert into float for division 

    acc = correct.sum() / len(correct)

    return acc
# training function 

def train(model, iterator, optimizer, criterion):

    

    epoch_loss = 0

    epoch_acc = 0

    

    model.train()

    

    for batch in iterator:

        text, text_lengths = batch.text

        

        optimizer.zero_grad()

        predictions = model(text, text_lengths).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)



        loss.backward()

        optimizer.step()

        

        epoch_loss += loss.item()

        epoch_acc += acc.item()

        



    return epoch_loss / len(iterator), epoch_acc / len(iterator)
def evaluate(model, iterator, criterion):

    

    #initialize every epoch

    epoch_loss = 0

    epoch_acc = 0

    

    model.eval()

    

    with torch.no_grad():

        for batch in iterator:

            text, text_lengths = batch.text

            predictions = model(text, text_lengths).squeeze(1)

            

            #compute loss and accuracy

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            

            #keep track of loss and accuracy

            epoch_loss += loss.item()

            epoch_acc += acc.item()

        

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
import time

t = time.time()

# loss=[]

# acc=[]

# val_acc=[]



best_valid_loss = float('inf')



model.to(device) #CNN to GPU



# Loss and optimizer

criterion = nn.BCEWithLogitsLoss()



optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



for epoch in range(num_epochs):

    

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)

    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)



    print(f'\tTrain Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.4f}%')

    print(f'\t Val. Loss: {valid_loss:.4f} | Val. Acc: {valid_acc*100:.4f}%')

    

#     loss.append(train_loss)

#     acc.append(train_acc)

#     val_acc.append(valid_acc)



    #save the best model

    if valid_loss < best_valid_loss:

        best_valid_loss = valid_loss

        torch.save(model.state_dict(), 'best_model.pt')

    

print(f'time:{time.time()-t:.3f}')
#inference 

import spacy

nlp = spacy.load('en')



def predict(model, sentence):

    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]  #tokenize the sentence 

    indexed = [TEXT.vocab.stoi[t] for t in tokenized]          #convert to integer sequence

    length = [len(indexed)]                                    #compute no. of words

    tensor = torch.LongTensor(indexed).to(device)              #convert to tensor

    tensor = tensor.unsqueeze(1).T                             #reshape in form of batch,no. of words

    length_tensor = torch.LongTensor(length)                   #convert to tensor

    

    prediction = model(tensor, length_tensor).squeeze(1)       #prediction 



    rounded_preds = torch.round(torch.sigmoid(prediction))

    predict_class = rounded_preds.tolist()[0]

    return predict_class                                
PATH = "../working/best_model.pt"

model.load_state_dict(torch.load(PATH))



predicts = []



for i in range(len(test_data.text)):

    predict_class = predict(model, test_data.text[i])

    predicts.append(int(predict_class))
submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

submission['target'] = predicts

submission
submission.to_csv('submission.csv',index=False)
gt_df = pd.read_csv("../input/disasters-on-social-media/socialmedia-disaster-tweets-DFE.csv", encoding='latin_1')

gt_df = gt_df[['choose_one', 'text']]

gt_df['target'] = (gt_df['choose_one']=='Relevant').astype(int)

gt_df['id'] = gt_df.index



merged_df = pd.merge(test_data, gt_df, on='id')

merged_df
target_df = merged_df[['id', 'target']]

target_df
target_df.to_csv('perfect_submission.csv', index=False)
target_df["predict"] = list(submission.target)



from sklearn import metrics



print('\t\tCLASSIFICATIION METRICS\n')

print(metrics.classification_report(target_df.target, target_df.predict))