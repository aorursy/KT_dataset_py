import numpy as np

import pandas as pd 

import re

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
train = pd.read_csv("../input/nlp-getting-started/train.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")

train.head()
x_train = train.text

y_train = train.target

x_train[:5]
x_test = test.text

x_test[:5]
contraction_mapping ={

    "ain't": "am not / are not / is not / has not / have not", "aren't": "are not / am not", "can't": "cannot",

    "can't've": "cannot have", "'cause": "because", "could've": "could have", "couldn't": "could not",

    "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not", "don't": "do not",

    "hadn't": "had not", "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not",

    "he'd": "he had / he would", "he'd've": "he would have",

    "he'll": "he shall / he will", "he'll've": "he shall have / he will have", "he's": "he has / he is",

    "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how has / how is / how does",

    "I'd": "I had / I would", "I'd've": "I would have", "I'll": "I shall / I will", "I'll've": "I shall have / I will have",

    "I'm": "I am", "I've": "I have", "isn't": "is not", "it'd": "it had / it would", "it'd've": "it would have",

    "it'll": "it shall / it will", "it'll've": "it shall have / it will have",

    "it's": "it has / it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",

    "mightn't": "might not", "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",

    "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock",

    "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",

    "shan't've": "shall not have", "she'd": "she had / she would", "she'd've": "she would have",

    "she'll": "she shall / she will", "she'll've": "she shall have / she will have",

    "she's": "she has / she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",

    "so've": "so have", "so's": "so as / so is", "that'd": "that would / that had", "that'd've": "that would have",

    "that's": "that has / that is", "there'd": "there had / there would", "there'd've": "there would have",

    "there's": "there has / there is", "they'd": "they had / they would", "they'd've": "they would have",

    "they'll": "they shall / they will", "they'll've": "they shall have / they will have",

    "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",

    "we'd": "we had / we would", "we'd've": "we would have", "we'll": "we will",

    "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not",

    "what'll": "what shall / what will", "what'll've": "what shall have / what will have",

    "what're": "what are", "what's": "what has / what is", "what've": "what have",

    "when's": "when has / when is", "when've": "when have",

    "where'd": "where did", "where's": "where has / where is", "where've": "where have", "who'll": "who shall / who will",

    "who'll've": "who shall have / who will have", "who's": "who has / who is", "who've": "who have",

    "why's": "why has / why is", "why've": "why have", "will've": "will have", "won't": "will not",

    "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",

    "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are",

    "y'all've": "you all have", "you'd": "you had / you would", "you'd've": "you would have",

    "you'll": "you shall / you will", "you'll've": "you shall have / you will have",

    "you're": "you are", "you've": "you have",

    "he's" : "he is", "there's" : "there is","We're" : "We are","That's" : "That is","won't" : "will not","they're" : "they are",

    "Can't" : "Cannot","wasn't" : "was not","aren't" : "are not","isn't" : "is not","What's" : "What is","i'd" : "I would",

    "should've" : "should have","where's" : "where is","we'd" : "we would","i'll" : "I will","weren't" : "were not",

    "They're" : "They are","let's" : "let us","it's" : "it is","can't" : "cannot","don't" : "do not","you're" : "you are",

    "i've" : "I have","that's" : "that is","i'll" : "I will","doesn't" : "does not","i'd" : "I would","didn't" : "did not",

    "ain't" : "am not","you'll" : "you will","I've" : "I have","Don't" : "do not","I'll" : "I will","I'd" : "I would",

    "Let's" : "Let us","you'd" : "You would","It's" : "It is","Ain't" : "am not","Haven't" : "Have not","Could've" : "Could have",

    "youve" : "you have","haven't" : "have not","hasn't" : "has not","There's" : "There is","He's" : "He is","It's" : "It is",

    "You're" : "You are","I'M" : "I am","shouldn't" : "should not","wouldn't" : "would not","i'm" : "I am","I'm" : "I am",

    "Isn't" : "is not","Here's" : "Here is","you've" : "you have","we're" : "we are","what's" : "what is","couldn't" : "could not",

    "we've" : "we have","who's" : "who is","y'all" : "you all","would've" : "would have","it'll" : "it will","we'll" : "we will",

    "We've" : "We have","he'll" : "he will","Y'all" : "You all","Weren't" : "Were not","Didn't" : "Did not","they'll" : "they will",

    "they'd" : "they would","DON'T" : "DO NOT","they've" : "they have",

    

    #correct some acronyms while we are at it

    "tnwx" : "Tennessee Weather", "azwx" : "Arizona Weather", "alwx" : "Alabama Weather", "wordpressdotcom" : "wordpress",

    "gawx" : "Georgia Weather", "scwx" : "South Carolina Weather", "cawx" : "California Weather",

    "usNWSgov" : "United States National Weather Service", "MH370" : "Malaysia Airlines Flight 370",

    "okwx" : "Oklahoma City Weather", "arwx" : "Arkansas Weather",  "lmao" : "laughing my ass off",  

    "amirite" : "am I right",

    

    #and some typos/abbreviations

    "w/e" : "whatever", "w/" : "with", "USAgov" : "USA government", "recentlu" : "recently", "Ph0tos" : "Photos", 

    "exp0sed" : "exposed", "<3" : "love", "amageddon" : "armageddon", "Trfc" : "Traffic", "WindStorm" : "Wind Storm",

    "16yr" : "16 year", "TRAUMATISED" : "traumatized",

    

    #hashtags and usernames

    "IranDeal" : "Iran Deal", "ArianaGrande" : "Ariana Grande", "camilacabello97" : "camila cabello", 

    "RondaRousey" : "Ronda Rousey", "MTVHottest" : "MTV Hottest", "TrapMusic" : "Trap Music",

    "ProphetMuhammad" : "Prophet Muhammad", "PantherAttack" : "Panther Attack", "StrategicPatience" : "Strategic Patience",

    "socialnews" : "social news", "IDPs:" : "Internally Displaced People :", "ArtistsUnited" : "Artists United",

    "ClaytonBryant" : "Clayton Bryant", "jimmyfallon" : "jimmy fallon", "justinbieber" : "justin bieber", "Time2015" : "Time 2015",

    "djicemoon" : "dj icemoon", "LivingSafely" : "Living Safely", "FIFA16" : "Fifa 2016",

    "thisiswhywecanthavenicethings" : "this is why we cannot have nice things", "bbcnews" : "bbc news",

    "UndergroundRailraod" : "Underground Railraod", "c4news" : "c4 news", "MUDSLIDE" : "mudslide", "NoSurrender" : "No Surrender",

    "NotExplained" : "Not Explained", "greatbritishbakeoff" : "great british bake off", "LondonFire" : "London Fire",

    "KOTAWeather" : "KOTA Weather", "LuchaUnderground" : "Lucha Underground", "KOIN6News" : "KOIN 6 News",

    "LiveOnK2" : "Live On K2", "9NewsGoldCoast" : "9 News Gold Coast", "nikeplus" : "nike plus", "david_cameron" : "David Cameron",

    "peterjukes" : "Peter Jukes", "MikeParrActor" : "Michael Parr", "4PlayThursdays" : "Foreplay Thursdays",

    "TGF2015" : "Tontitown Grape Festival", "realmandyrain" : "Mandy Rain", "GraysonDolan" : "Grayson Dolan", 

    "ApolloBrown" : "Apollo Brown", "saddlebrooke" : "Saddlebrooke", "TontitownGrape" : "Tontitown Grape", "AbbsWinston" : "Abbs Winston",

    "ShaunKing" : "Shaun King", "MeekMill" : "Meek Mill", "TornadoGiveaway" : "Tornado Giveaway", "GRupdates" : "GR updates",

    "SouthDowns" : "South Downs", "braininjury" : "brain injury", "auspol" : "Australian politics", "PlannedParenthood" : "Planned Parenthood",

    "calgaryweather" : "Calgary Weather", "weallheartonedirection" : "we all heart one direction", "edsheeran" : "Ed Sheeran",

    "TrueHeroes" : "True Heroes", "ComplexMag" : "Complex Magazine", "TheAdvocateMag" : "The Advocate Magazine",

    "CityofCalgary" : "City of Calgary", "EbolaOutbreak" : "Ebola Outbreak", "SummerFate" : "Summer Fate",

    "RAmag" : "Royal Academy Magazine", "offers2go" : "offers to go", "ModiMinistry" : "Modi Ministry", "TAXIWAYS" : "taxi ways",

    "Calum5SOS" : "Calum Hood", "JamesMelville" : "James Melville", "JamaicaObserver" : "Jamaica Observer",

    "TweetLikeItsSeptember11th2001" : "Tweet like it is september 11th 2001", "cbplawyers" : "cbp lawyers",

    "fewmoretweets" : "few more tweets", "BlackLivesMatter" : "Black Lives Matter", "NASAHurricane" : "NASA Hurricane",

    "onlinecommunities" : "online communities", "humanconsumption" : "human consumption", "Typhoon-Devastated" : "Typhoon Devastated",

    "Meat-Loving" : "Meat Loving", "facialabuse" : "facial abuse", "LakeCounty" : "Lake County", "BeingAuthor" : "Being Author",

    "withheavenly" : "with heavenly", "thankU" : "thank you", "iTunesMusic" : "iTunes Music",

    "OffensiveContent" : "Offensive Content", "WorstSummerJob" : "Worst Summer Job", "HarryBeCareful" : "Harry Be Careful",

    "NASASolarSystem" : "NASA Solar System", "animalrescue" : "animal rescue", "KurtSchlichter" : "Kurt Schlichter",

    "Throwingknifes" : "Throwing knives", "GodsLove" : "God's Love", "bookboost" : "book boost", "ibooklove" : "I book love",

    "NestleIndia" : "Nestle India", "realDonaldTrump" : "Donald Trump", "DavidVonderhaar" : "David Vonderhaar", "CecilTheLion" : "Cecil The Lion",

    "weathernetwork" : "weather network", "GOPDebate" : "GOP Debate",

    "RickPerry" : "Rick Perry", "frontpage" : "front page", "NewsInTweets" : "News In Tweets",

    "ViralSpell" : "Viral Spell", "til_now" : "until now",

    "volcanoinRussia" : "volcano in Russia", "ZippedNews" : "Zipped News", "MicheleBachman" : "Michele Bachman",

    "53inch" : "53 inch", "KerrickTrial" : "Kerrick Trial", "abstorm" : "Alberta Storm", "Beyhive" : "Beyonce hive",

    "RockyFire" : "Rocky Fire","Listen/Buy" : "Listen / Buy","ArtistsUnited" : "Artists United",

    "ENGvAUS" : "England vs Australia", "ScottWalker" : "Scott Walker",

}
contractions_re = re.compile('(%s)' % '|'.join(contraction_mapping.keys()))

def expand_contractions(s, contraction_mapping=contraction_mapping):

    def replace(match):

        return contraction_mapping[match.group(0)]

    return contractions_re.sub(replace, s)
def clean(t):

    ids = []

    index = 0

    for i in range(len(t.split())):

        if "http" in t.split()[i] or "@" in t.split()[i]:

            index = i

            ids.append(index)

    tex = []

    for i in range(len(t.split())):

        if i not in ids:

            tex.append(t.split()[i])

    return (" ".join(tex)).strip()
stop_words = set(stopwords.words('english'))
def text_cleaner(text):

    text = clean(text)

    newString = expand_contractions(text)

    newString = re.sub("[^a-zA-Z]", " ", newString) 

    newString = re.sub("\W", " ", newString)

    tokens = [w for w in newString.split() if w not in stop_words]

    long_words = []

    for i in tokens:

        if len(i)>1:                                                 #removing short word

            long_words.append(i)   

    return (" ".join(long_words)).strip()
#call the function

train_text = []

for t in x_train:

    train_text.append(text_cleaner(t))

train_text[:10]
#call the function

test_text = []

for t in x_test:

    test_text.append(text_cleaner(t))

test_text[:10]
df = train_text+test_text
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

df = vectorizer.fit_transform(df)

df.shape
x_train = df[:7613]

x_test = df[7613:]

x_train.shape, x_test.shape
x_tr, x_val, y_tr, y_val = train_test_split(x_train,y_train, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(x_tr, y_tr)
y_pred = clf.predict(x_val)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_pred, y_val)

cm
accuracy_score = clf.score(x_val, y_val)

print("Accuracy of the model is " + str(accuracy_score*100)+"%")
submission = pd.DataFrame()

submission['id'] = test.id

submission['target'] = clf.predict(x_test)

submission
submission.to_csv('submission.csv', index = False)

print('Submission saved')