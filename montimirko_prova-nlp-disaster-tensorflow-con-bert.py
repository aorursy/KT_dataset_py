#https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub



!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py

    

import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, BatchNormalization, Activation, Dropout

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow_hub as hub



import tokenization
import re

import string





# Text Lowercase ---------------------------------------------------------------

# It is a very common practise. Lowercasing the text is used to reduce the size of the vocabulary of our text data.

# With it, 'Roma' and 'roma' and 'rOma' and 'ROma' became a single word to analyze. It reduces text entropy. 

def text_lowercase(text):

    return text.lower() 





# Remove punctuation -----------------------------------------------------------

# You should remove punctuations so that you don’t have different forms of the same word. If you don’t remove the punctuation, then been. been, been! will be treated separately.

# String punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

def remove_punctuation(text): 

    no_punct_text = "".join([char for char in text if char not in string.punctuation])

    return no_punct_text





# Remove or convert numbers into words -----------------------------------------

# Another operation you can do for decreasing text entropy is to convert all the digits into words, so that '25' and 'twenty-five' are treated as the same entity.

!pip install inflect

import inflect #  generate plurals, singular nouns, ordinals, cardinals, indefinite articles; convert numbers to words

p = inflect.engine() 



# convert number into words 

def convert_number(text): 

    # split string into list of words 

    temp_str = text.split() 

    # initialise empty list 

    new_string = [] 



    for word in temp_str: 

        # if word is a digit, convert the digit 

        # to numbers and append into the new_string list 

        if word.isdigit(): 

            temp = p.number_to_words(word) 

            new_string.append(temp) 



        # append the word as it is 

        else: 

            new_string.append(word) 



    # join the words of new_string to form a string 

    temp_str = ' '.join(new_string) 

    return temp_str 



  

# Remove whitespaces -----------------------------------------------------------

# Use the join and split function to remove all the white spaces in a string

def remove_whitespace(text): 

    return  " ".join(text.split()) 



# Final cleaning : ALWAYS LOOK INSIDE DATA!!

def remove_unwanted_text(text):

    new_text = str(text)  

  #new_text = re.sub('3+[0-9]{9}', '<mobilephone>', new_text) # adding <mobilphone> tag

  #new_text = re.sub('(\:(-)?\)|\:(-)?\(|<3|\:(-)?\/|\:-\/|\:(-)?\||\:(-)?[pP]|\s\:+(-)?([0-9])?\s|\^\^|\s\:+(-)?(\D)?\s)', '', new_text)  # removing smile with :

  #new_text = new_text.replace('1st', 'first')

  #new_text = new_text.replace('2nd', 'second')  

  #new_text = re.sub('xké|xkè|xchè|xke|xche|perche|perché', 'perchè',new_text, flags=re.IGNORECASE)

  #new_text = re.sub('xo|xò', 'però',new_text, flags=re.IGNORECASE) 

    return new_text





def clean_tweet(tweet): 

            

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

    tweet = re.sub(r"SOUDELOR", "Soudelor", tweet)    

           

    # Urls

    tweet = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", tweet)

    

    #  Words with punctuations and special characters

    punctuations = '@#!?+&*[]-%.:/();$=><|{}' + "'"

    for p in punctuations:

        tweet = tweet.replace(p, f' {p} ')

        

    # ... and ..

    tweet = tweet.replace('...', ' ... ')

    if '...' not in tweet:

        tweet = tweet.replace('..', ' ... ')      

        

    # Acronyms

    tweet = re.sub(r"MH370", "Malaysia Airlines Flight 370", tweet)

    tweet = re.sub(r"mÌ¼sica", "music", tweet)

    tweet = re.sub(r"okwx", "Oklahoma City Weather", tweet)

    tweet = re.sub(r"arwx", "Arkansas Weather", tweet)    

    tweet = re.sub(r"gawx", "Georgia Weather", tweet)  

    tweet = re.sub(r"scwx", "South Carolina Weather", tweet)  

    tweet = re.sub(r"cawx", "California Weather", tweet)

    tweet = re.sub(r"tnwx", "Tennessee Weather", tweet)

    tweet = re.sub(r"azwx", "Arizona Weather", tweet)    

    tweet = re.sub(r"wordpressdotcom", "wordpress", tweet)    

    tweet = re.sub(r"usNWSgov", "United States National Weather Service", tweet)

    

    return tweet



import nltk

from nltk.corpus import stopwords

nltk.download('stopwords')





from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer('\w+')

stopwords.words("english")

#stopwords.words("italian")



def remove_stopwords(text, lang = "english"): 

    stop_words = set(stopwords.words(lang)) 

    word_tokens = tokenizer.tokenize(text)

    filtered_text = [word for word in word_tokens if word not in stop_words] 

    return " ".join(filtered_text) 







#We put all togheter 

def data_cleaning_and_preprocessing(text, remove_stop_words = True):

    text_cleaned = str(text)  

    text_cleaned = clean_tweet(text_cleaned)

    text_cleaned = text_lowercase(text_cleaned)

    text_cleaned = remove_punctuation(text_cleaned)

    text_cleaned = convert_number(text_cleaned)

    text_cleaned = remove_whitespace(text_cleaned)

    #text_cleaned = remove_unwanted_text(text_cleaned)  

    #if remove_stop_words:

    #    text_cleaned = remove_stopwords(text_cleaned)

    return text_cleaned

#max_length = 160

max_length = 160



def bert_encode(texts, tokenizer, max_len=max_length):

    all_tokens = []

    all_masks = []

    all_segments = []

    

    for text in texts:

        text = tokenizer.tokenize(text)

            

        text = text[:max_len-2]

        input_sequence = ["[CLS]"] + text + ["[SEP]"]

        pad_len = max_len - len(input_sequence)

        

        tokens = tokenizer.convert_tokens_to_ids(input_sequence) #tockenizzo con vocabolario 

        tokens += [0] * pad_len #paddo

        pad_masks = [1] * len(input_sequence) + [0] * pad_len # 1 per tutte le parole poi 0 per padding

        segment_ids = [0] * max_len #tutti 0...

        

        all_tokens.append(tokens)

        all_masks.append(pad_masks)

        all_segments.append(segment_ids)

    

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)




def build_model(bert_layer, max_len=max_length):

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")



    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    clf_output = sequence_output[:, 0, :]

    clf_output = Dense(64)(clf_output) #era 64

    clf_output = BatchNormalization()(clf_output)

    clf_output = Dropout(0.29)(clf_output)#era 0.29 il best

    clf_output = Activation('relu')(clf_output)



    out = Dense(1, activation='sigmoid')(clf_output)

    

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

    model.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy'])

    

    return model
#module_url = "https://tfhub.dev/tensorflow/bert_en_wwm_uncased_L-24_H-1024_A-16/1"

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"

#module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"

bert_layer = hub.KerasLayer(module_url, trainable=True)
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
train_df = pd.DataFrame(columns=['text'])



for row in range(0,len(train)):

    text = train.iloc[row,:]['text']

    text = data_cleaning_and_preprocessing(text)

    train_df.loc[row] = text

    

train_df.iloc[0,:]  



test_df = pd.DataFrame(columns=['text'])

for row in range(0,len(test)):

    text = test.iloc[row,:]['text']

    text = data_cleaning_and_preprocessing(text)

    test_df.loc[row] = text

train_input = bert_encode(train_df.text.values, tokenizer, max_len=max_length)

test_input = bert_encode(test_df.text.values, tokenizer, max_len=max_length)

#train_input = bert_encode(train_list_cleaned, tokenizer, max_len=max_length)

#test_input = bert_encode(test_list_cleaned, tokenizer, max_len=max_length)

train_labels = train.target.values
model = build_model(bert_layer, max_len=max_length)

model.summary()
train_history = model.fit(

    train_input, train_labels,

    validation_split=0.0,

    epochs=6,

    batch_size=16

)



model.save('model.h5')
test_pred = model.predict(test_input)
submission['target'] = test_pred.round().astype(int)

submission.to_csv('submission.csv', index=False)