!pip install tweepy

import tweepy
from bs4 import BeautifulSoup

import requests

import pandas as pd

import numpy as np

import datetime
CONSUMER_KEY = 'uWgt6k1Mzl6IN1IiEKvHAHaNm'

CONSUMER_SECRET = 'PNsb7eZpt8AUdN0TzUNZOsvAaxajMmw094nvcTIVlRd2ne64yx'

ACCESS_KEY = '1061225034407206912-v5VBoAVQddfuZdSABSDZH8azyQXmLm'

ACCESS_SECRET = 'eiwbyRCFz1TeKXVFyIVAJ3RGqoJ5HLoMeqX4NaFs9lfrU'
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)

auth.set_access_token(ACCESS_KEY, ACCESS_SECRET)

api = tweepy.API(auth)
# Lucknow, Patna, Ranchi, Amritsar, Ahmedabad, Mumbai, Hyderabad, Bangalore, Chennai

# Chicago, Detroit, NYC, Washington

# London, Manchester, Newcastle, Liverpool

# Paris

# Berlin



woeid = {"Worldwide": [1],

        "India": [2295377, 2295381, 2295383, 2295388, 2295402, 2295411, 2295414, 2295420, 2295424],

         "US": [2379574, 2391585, 2459115, 2514815],

         "UK": [44418, 28218, 30079, 26734],

         "France": [615702],

         "Germany": [638242]

        }
twitter_trends = []



for country in woeid:

    for state_code in woeid[country]:

        trends1 = api.trends_place(state_code)

        trends = [trend['name'][1:] for trend in trends1[0]['trends']]

        twitter_trends.extend(trends)
twitter_tags = []



for trend in twitter_trends:

    if trend not in twitter_tags:

        twitter_tags.append(trend)
print(len(twitter_tags), twitter_tags)
def all_hashtag(keyword):

    try:

        url = "https://www.all-hashtag.com/library/contents/ajax_generator.php"

        payload = {'keyword': keyword,

                   'filter': 'top'}

        files = []

        headers = {}

        response = requests.request("POST", url, headers=headers, data=payload, files=files)

        text = str(response.text)

        soup = BeautifulSoup(text, features="html.parser")

        tags_str = ""

        for val in soup.find_all("div", class_="copy-hashtags"):

            tags_str = val.get_text()

        tags_str = tags_str.split(" ")

        tags = list()

        for tag in tags_str:

            if tag:

                tags.append(tag.replace("\n", ""))

        return tags

    except Exception as ex:

        print(ex)

        return False
def best_hashtag_handler(keyword):

    try:

        url = "http://best-hashtags.com/hashtag/{}/".format(keyword)

        response = requests.request("POST", url)

        soup = BeautifulSoup(response.text, features="html.parser")

        hashtags = list()

        all_text = ''

        for val in soup.find_all("table", class_="table"):

            all_text = val.get_text()

        all_text = all_text.split("\n")

        for val in all_text:

            if val.startswith("#") and val != '#':

                hashtags.append(val)

        return hashtags

    except Exception as ex:

        print(ex)

        return False
def top_hashtags(keyword):

    tags_with_popularity = {}

    try:

        pages = 4

        for i in range(1, pages + 1):

            url = 'https://top-hashtags.com/search/?q={}&opt=top&sp={}'.format(keyword, i)

            req = requests.get(url)

            html_doc = req.text

            soup = BeautifulSoup(html_doc, "html5lib")

            for litag in soup.find_all('li', {'class': 'i-row'}):

                currentKey = None

                for div in litag.find_all('div', {'class': 'i-tag'}):

                    currentKey = div.text

                for div in litag.find_all('div', {'class': 'i-total'}):

                    tags_with_popularity[currentKey] = div.text

        return tags_with_popularity

    except Exception as ex:

        print(ex)

        return False
import string

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from nltk.stem import WordNetLemmatizer

import re
lemmatizer = WordNetLemmatizer()

stemmer = PorterStemmer()



stop_words = set(stopwords.words('english')) 



punctuation_to_remove = string.punctuation
EMOTICONS = {

    u":‑\)":"Happy face or smiley",

    u":\)":"Happy face or smiley",

    u":-\]":"Happy face or smiley",

    u":\]":"Happy face or smiley",

    u":-3":"Happy face smiley",

    u":3":"Happy face smiley",

    u":->":"Happy face smiley",

    u":>":"Happy face smiley",

    u"8-\)":"Happy face smiley",

    u":o\)":"Happy face smiley",

    u":-\}":"Happy face smiley",

    u":\}":"Happy face smiley",

    u":-\)":"Happy face smiley",

    u":c\)":"Happy face smiley",

    u":\^\)":"Happy face smiley",

    u"=\]":"Happy face smiley",

    u"=\)":"Happy face smiley",

    u":‑D":"Laughing, big grin or laugh with glasses",

    u":D":"Laughing, big grin or laugh with glasses",

    u"8‑D":"Laughing, big grin or laugh with glasses",

    u"8D":"Laughing, big grin or laugh with glasses",

    u"X‑D":"Laughing, big grin or laugh with glasses",

    u"XD":"Laughing, big grin or laugh with glasses",

    u"=D":"Laughing, big grin or laugh with glasses",

    u"=3":"Laughing, big grin or laugh with glasses",

    u"B\^D":"Laughing, big grin or laugh with glasses",

    u":-\)\)":"Very happy",

    u":‑\(":"Frown, sad, andry or pouting",

    u":-\(":"Frown, sad, andry or pouting",

    u":\(":"Frown, sad, andry or pouting",

    u":‑c":"Frown, sad, andry or pouting",

    u":c":"Frown, sad, andry or pouting",

    u":‑<":"Frown, sad, andry or pouting",

    u":<":"Frown, sad, andry or pouting",

    u":‑\[":"Frown, sad, andry or pouting",

    u":\[":"Frown, sad, andry or pouting",

    u":-\|\|":"Frown, sad, andry or pouting",

    u">:\[":"Frown, sad, andry or pouting",

    u":\{":"Frown, sad, andry or pouting",

    u":@":"Frown, sad, andry or pouting",

    u">:\(":"Frown, sad, andry or pouting",

    u":'‑\(":"Crying",

    u":'\(":"Crying",

    u":'‑\)":"Tears of happiness",

    u":'\)":"Tears of happiness",

    u"D‑':":"Horror",

    u"D:<":"Disgust",

    u"D:":"Sadness",

    u"D8":"Great dismay",

    u"D;":"Great dismay",

    u"D=":"Great dismay",

    u"DX":"Great dismay",

    u":‑O":"Surprise",

    u":O":"Surprise",

    u":‑o":"Surprise",

    u":o":"Surprise",

    u":-0":"Shock",

    u"8‑0":"Yawn",

    u">:O":"Yawn",

    u":-\*":"Kiss",

    u":\*":"Kiss",

    u":X":"Kiss",

    u";‑\)":"Wink or smirk",

    u";\)":"Wink or smirk",

    u"\*-\)":"Wink or smirk",

    u"\*\)":"Wink or smirk",

    u";‑\]":"Wink or smirk",

    u";\]":"Wink or smirk",

    u";\^\)":"Wink or smirk",

    u":‑,":"Wink or smirk",

    u";D":"Wink or smirk",

    u":‑P":"Tongue sticking out, cheeky, playful or blowing a raspberry",

    u":P":"Tongue sticking out, cheeky, playful or blowing a raspberry",

    u"X‑P":"Tongue sticking out, cheeky, playful or blowing a raspberry",

    u"XP":"Tongue sticking out, cheeky, playful or blowing a raspberry",

    u":‑Þ":"Tongue sticking out, cheeky, playful or blowing a raspberry",

    u":Þ":"Tongue sticking out, cheeky, playful or blowing a raspberry",

    u":b":"Tongue sticking out, cheeky, playful or blowing a raspberry",

    u"d:":"Tongue sticking out, cheeky, playful or blowing a raspberry",

    u"=p":"Tongue sticking out, cheeky, playful or blowing a raspberry",

    u">:P":"Tongue sticking out, cheeky, playful or blowing a raspberry",

    u":‑/":"Skeptical, annoyed, undecided, uneasy or hesitant",

    u":/":"Skeptical, annoyed, undecided, uneasy or hesitant",

    u":-[.]":"Skeptical, annoyed, undecided, uneasy or hesitant",

    u">:[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",

    u">:/":"Skeptical, annoyed, undecided, uneasy or hesitant",

    u":[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",

    u"=/":"Skeptical, annoyed, undecided, uneasy or hesitant",

    u"=[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",

    u":L":"Skeptical, annoyed, undecided, uneasy or hesitant",

    u"=L":"Skeptical, annoyed, undecided, uneasy or hesitant",

    u":S":"Skeptical, annoyed, undecided, uneasy or hesitant",

    u":‑\|":"Straight face",

    u":\|":"Straight face",

    u":$":"Embarrassed or blushing",

    u":‑x":"Sealed lips or wearing braces or tongue-tied",

    u":x":"Sealed lips or wearing braces or tongue-tied",

    u":‑#":"Sealed lips or wearing braces or tongue-tied",

    u":#":"Sealed lips or wearing braces or tongue-tied",

    u":‑&":"Sealed lips or wearing braces or tongue-tied",

    u":&":"Sealed lips or wearing braces or tongue-tied",

    u"O:‑\)":"Angel, saint or innocent",

    u"O:\)":"Angel, saint or innocent",

    u"0:‑3":"Angel, saint or innocent",

    u"0:3":"Angel, saint or innocent",

    u"0:‑\)":"Angel, saint or innocent",

    u"0:\)":"Angel, saint or innocent",

    u":‑b":"Tongue sticking out, cheeky, playful or blowing a raspberry",

    u"0;\^\)":"Angel, saint or innocent",

    u">:‑\)":"Evil or devilish",

    u">:\)":"Evil or devilish",

    u"\}:‑\)":"Evil or devilish",

    u"\}:\)":"Evil or devilish",

    u"3:‑\)":"Evil or devilish",

    u"3:\)":"Evil or devilish",

    u">;\)":"Evil or devilish",

    u"\|;‑\)":"Cool",

    u"\|‑O":"Bored",

    u":‑J":"Tongue-in-cheek",

    u"#‑\)":"Party all night",

    u"%‑\)":"Drunk or confused",

    u"%\)":"Drunk or confused",

    u":-###..":"Being sick",

    u":###..":"Being sick",

    u"<:‑\|":"Dump",

    u"\(>_<\)":"Troubled",

    u"\(>_<\)>":"Troubled",

    u"\(';'\)":"Baby",

    u"\(\^\^>``":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",

    u"\(\^_\^;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",

    u"\(-_-;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",

    u"\(~_~;\) \(・\.・;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",

    u"\(-_-\)zzz":"Sleeping",

    u"\(\^_-\)":"Wink",

    u"\(\(\+_\+\)\)":"Confused",

    u"\(\+o\+\)":"Confused",

    u"\(o\|o\)":"Ultraman",

    u"\^_\^":"Joyful",

    u"\(\^_\^\)/":"Joyful",

    u"\(\^O\^\)／":"Joyful",

    u"\(\^o\^\)／":"Joyful",

    u"\(__\)":"Kowtow as a sign of respect, or dogeza for apology",

    u"_\(\._\.\)_":"Kowtow as a sign of respect, or dogeza for apology",

    u"<\(_ _\)>":"Kowtow as a sign of respect, or dogeza for apology",

    u"<m\(__\)m>":"Kowtow as a sign of respect, or dogeza for apology",

    u"m\(__\)m":"Kowtow as a sign of respect, or dogeza for apology",

    u"m\(_ _\)m":"Kowtow as a sign of respect, or dogeza for apology",

    u"\('_'\)":"Sad or Crying",

    u"\(/_;\)":"Sad or Crying",

    u"\(T_T\) \(;_;\)":"Sad or Crying",

    u"\(;_;":"Sad of Crying",

    u"\(;_:\)":"Sad or Crying",

    u"\(;O;\)":"Sad or Crying",

    u"\(:_;\)":"Sad or Crying",

    u"\(ToT\)":"Sad or Crying",

    u";_;":"Sad or Crying",

    u";-;":"Sad or Crying",

    u";n;":"Sad or Crying",

    u";;":"Sad or Crying",

    u"Q\.Q":"Sad or Crying",

    u"T\.T":"Sad or Crying",

    u"QQ":"Sad or Crying",

    u"Q_Q":"Sad or Crying",

    u"\(-\.-\)":"Shame",

    u"\(-_-\)":"Shame",

    u"\(一一\)":"Shame",

    u"\(；一_一\)":"Shame",

    u"\(=_=\)":"Tired",

    u"\(=\^\·\^=\)":"cat",

    u"\(=\^\·\·\^=\)":"cat",

    u"=_\^=	":"cat",

    u"\(\.\.\)":"Looking down",

    u"\(\._\.\)":"Looking down",

    u"\^m\^":"Giggling with hand covering mouth",

    u"\(\・\・?":"Confusion",

    u"\(?_?\)":"Confusion",

    u">\^_\^<":"Normal Laugh",

    u"<\^!\^>":"Normal Laugh",

    u"\^/\^":"Normal Laugh",

    u"\（\*\^_\^\*）" :"Normal Laugh",

    u"\(\^<\^\) \(\^\.\^\)":"Normal Laugh",

    u"\(^\^\)":"Normal Laugh",

    u"\(\^\.\^\)":"Normal Laugh",

    u"\(\^_\^\.\)":"Normal Laugh",

    u"\(\^_\^\)":"Normal Laugh",

    u"\(\^\^\)":"Normal Laugh",

    u"\(\^J\^\)":"Normal Laugh",

    u"\(\*\^\.\^\*\)":"Normal Laugh",

    u"\(\^—\^\）":"Normal Laugh",

    u"\(#\^\.\^#\)":"Normal Laugh",

    u"\（\^—\^\）":"Waving",

    u"\(;_;\)/~~~":"Waving",

    u"\(\^\.\^\)/~~~":"Waving",

    u"\(-_-\)/~~~ \($\·\·\)/~~~":"Waving",

    u"\(T_T\)/~~~":"Waving",

    u"\(ToT\)/~~~":"Waving",

    u"\(\*\^0\^\*\)":"Excited",

    u"\(\*_\*\)":"Amazed",

    u"\(\*_\*;":"Amazed",

    u"\(\+_\+\) \(@_@\)":"Amazed",

    u"\(\*\^\^\)v":"Laughing,Cheerful",

    u"\(\^_\^\)v":"Laughing,Cheerful",

    u"\(\(d[-_-]b\)\)":"Headphones,Listening to music",

    u'\(-"-\)':"Worried",

    u"\(ーー;\)":"Worried",

    u"\(\^0_0\^\)":"Eyeglasses",

    u"\(\＾ｖ\＾\)":"Happy",

    u"\(\＾ｕ\＾\)":"Happy",

    u"\(\^\)o\(\^\)":"Happy",

    u"\(\^O\^\)":"Happy",

    u"\(\^o\^\)":"Happy",

    u"\)\^o\^\(":"Happy",

    u":O o_O":"Surprised",

    u"o_0":"Surprised",

    u"o\.O":"Surpised",

    u"\(o\.o\)":"Surprised",

    u"oO":"Surprised",

    u"\(\*￣m￣\)":"Dissatisfied",

    u"\(‘A`\)":"Snubbed or Deflated"

}

emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')
emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)
def preprocess_text(text):

    text = text.lower()

    text = " ".join([stemmer.stem(word) for word in text.split()])

    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])

    text = emoji_pattern.sub(r'', text)

    text = emoticon_pattern.sub(r'', text)

    return text
text = "We want love and support ❤️❤️❤️"
text = " ".join([word for word in str(text).split() if '@' not in word])

text = " ".join([word for word in str(text).split() if word not in stop_words])

text = text.translate(str.maketrans('', '', punctuation_to_remove))
twitter_df = pd.DataFrame({"hashtag": twitter_tags})

twitter_df['hashTagLower'] = twitter_df['hashtag'].apply(lambda x: x.lower())
trending_on_twitter = []



for word in text.split():

    word = word.lower()

    if word[0] == '#':

        word = word[1:]

    for i in range(twitter_df.shape[0]):

        if word in twitter_df.iloc[i]["hashTagLower"]:

            hashtag_to_add = '#' + twitter_df.iloc[i]["hashtag"]

            if hashtag_to_add not in trending_on_twitter:

                trending_on_twitter.append(hashtag_to_add)

            

print(f'Trending on twitter: {trending_on_twitter}')
preprocess_text = preprocess_text(text)

unique_text = []



for txt in preprocess_text.split():

    if txt not in unique_text:

        unique_text.append(txt)

        

unique_text = " ".join(unique_text)
unique_text
pos = nltk.pos_tag(unique_text.split()) 



tags = []

filter_pos = ["VB", "VBN", "VBP", "VBZ", "RB", "RBS", "RBR", "JJ", "JJR", "JJS", "NN", "NNS"]



for word in pos:

    if word[1] in filter_pos:

        tags.append(word[0])
tags
hashTags = []

hashTagFor = []

popularity = []

common = []
for tag in tags:

    ht = []

    ht.extend(all_hashtag(tag))

    ht.extend(best_hashtag_handler(tag))

    hashTags.extend(ht)

    hashTagFor.extend([tag] * len(ht))

    popularity.extend([0] * len(ht))

    common.extend([1] * len(ht))
for tag in tags:

    tags_with_popularity = top_hashtags(tag)

    for key in tags_with_popularity:

        hashTags.append(key)

        hashTagFor.append(tag) 

        common.append(1)

        val = tags_with_popularity[key]

        if 'K' in val:

            popularity.append(round(float(val[:-1]) * 1000, 2))

        elif 'M' in val:

            popularity.append(round(float(val[:-1]) * 1000000, 2))

        else:

             popularity.append(0)
df = pd.DataFrame({"word": hashTagFor, "hashTag": hashTags, "popularity": popularity, "common": common})

df['hashTagLower'] = df['hashTag'].apply(lambda x: x.lower()[1:])

df["popularity"] = pd.to_numeric(df["popularity"], errors='coerce')

df = df.drop_duplicates(subset='hashTag', keep='first')

df = df.sort_values(by="popularity", ascending=False)
df.head()
# convert common to %

df["common"] = df["common"] / 3 
df.head()
# This part of code is unused





# unique_hashtags = list(df["word"].unique())

# # suggest_hashtags = {"hashtag": [], "popularity": [], "trending": []}

# suggest_hashtags = {"hashtag": [], "popularity": []}



# # add unique most popular hashtags of each word

# for i in range(df.shape[0]):

#     current_word = df.iloc[i]["word"]

#     if current_word in unique_hashtags:

#         suggest_hashtags["hashtag"].append(df.iloc[i]["hashTag"])

#         suggest_hashtags["popularity"].append(df.iloc[i]["popularity"])

# #         if df.iloc[i]["hashTag"] in trending:

# #              suggest_hashtags["trending"].append("Trending")

# #         else:

# #             suggest_hashtags["trending"].append("Not Trending")

#         unique_hashtags.remove(current_word)



# #  add insta/gram hashtags

# for i in range(df.shape[0]):

#     current_hashtag = df.iloc[i]["hashTagLower"]

#     if "insta" in current_hashtag or "gram" in current_hashtag:

#         if df.iloc[i]["popularity"] > 0:

#             suggest_hashtags["hashtag"].append(df.iloc[i]["hashTag"])

#             suggest_hashtags["popularity"].append(df.iloc[i]["popularity"])

# #             if df.iloc[i]["hashTag"] in trending:

# #                  suggest_hashtags["trending"].append("Trending")

# #             else:

# #                 suggest_hashtags["trending"].append("Not Trending")
insta_hashtag_suggest = {"hashtag": [], "score": [], "popularity": []}



count = 0

targetCount = 26



while count < targetCount and count < df.shape[0]:

    score = 0

    current_hashtag = df.iloc[count]["hashTag"]

    current_popularity = df.iloc[count]["popularity"]

    if current_popularity >= 100000 and current_popularity <= 1000000:

        score = 7

    if current_popularity >= 1000000 and current_popularity <= 5000000:

        score = 8

    if current_popularity >= 5000000:  

        score = 9

    insta_hashtag_suggest["score"].append(score)

    insta_hashtag_suggest["hashtag"].append(current_hashtag)

    insta_hashtag_suggest["popularity"].append(current_popularity)

    count += 1
print("Trending hashtags on Instagram with popularity scores: ")

print("_________________________________________________________")

print(insta_hashtag_suggest)
print(f'Trending hashtags on twitter: {trending_on_twitter}')
# Banned hashtags

banned_tags = [

        "alone", "always", "armparty", "adulting", "assday", "ass", "abdl", "assworship", "addmysc", "asiangirl",

        "beautyblogger", "brain", "boho", "besties", "bikinibody", "costumes", "curvygirls",

        "date", "dating", "desk", "dm", "direct", "elevator", "eggplant", "edm", 

        "fuck", "girlsonly", "gloves", "graffitiigers", "happythanksgiving", 

        "hawks", "hotweather", "humpday", "hustler", "ilovemyinstagram", "instababy", "instasport", "iphonegraphy", "italiano", "ice", 

        "killingit", "kansas", "kissing", "kickoff", "leaves", "like", "lulu", "lean", 

        "master", "milf", "mileycyrus", "models", "mustfollow", "nasty", "newyearsday", "nude", "nudism", "nudity", 

        "overnight", "orderweedonline", "parties", "petite", "pornfood", "pushups", "prettygirl", 

        "rate", "ravens", "samelove", "selfharm", "skateboarding", "skype", "snap", "snapchat", "single", "singlelife", "stranger",

        "saltwater", "shower", "shit", "sopretty", "sunbathing", "streetphoto", "swole", "snowstorm", "sun", "sexy", 

        "tanlines", "todayimwearing", "teens", "teen", "thought", "tag4like", "tagsforlikes", "thighs", "undies", "valentinesday",

        "workflow", "wtf", "xanax", "youngmodel"

]