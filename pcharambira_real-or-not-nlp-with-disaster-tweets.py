import warnings

warnings.filterwarnings("ignore")
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from nltk.corpus import stopwords

from wordcloud import STOPWORDS, WordCloud

from sklearn.feature_extraction.text import CountVectorizer

plt.style.use('ggplot')

stop=set(stopwords.words('english'))

import re

import string

from keras.models import Sequential

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn import svm

from sklearn import model_selection

from sklearn import metrics

from keras.optimizers import Adam



%matplotlib inline

# Set how big the canvas is for the plot

sns.set(rc = {'figure.figsize':(13,8)})
# Read in the data

train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")



# Have a look at the data

train.head()
# What is the state of the training dataset 

print('There are {} rows and {} columns in the training dataset'.format(train.shape[0], train.shape[1]))

print('There are {} rows and {} columns in the testing dataset'.format(test.shape[0], test.shape[1]))
# Check to see how many missings are in each column

print(train.isnull().sum())
# Drop duplicate records

train = train.drop_duplicates().reset_index(drop=True)
# Plot the count of the targets



# Set theme

sns.set_style('whitegrid')



# Get the graph

graph = sns.countplot(x = 'target', data = train)



# Set titles

plt.title('Count of Targets', fontsize = 45)

plt.xlabel('Target', fontsize = 25)

plt.ylabel('Count', fontsize = 25)



# Access the bar attributes to place the text in the appropriate location

for p in graph.patches:

    graph.annotate('{:.0f}'.format(p.get_height()), (p.get_x() + 0.4, p.get_height()), ha='center', va='bottom',

color= 'black', fontsize = 15)
# How many unique words are in the training dataset 

print ('There are {} unique words are in the training dataset'.format(train.keyword.nunique()))
# Get unique words counts 

ax = sns.countplot(y = 'keyword', data = train, order = train.keyword.value_counts().iloc[:10].index)

ax.set_title('Top 10 Unique Word Count', fontsize = 30)

ax.set_xlabel('Count', fontsize = 20)

ax.set_ylabel('Unique Word', fontsize = 20)
# Get unique words counts by response



real_keyword = train[train.target==1].keyword.value_counts().head(10)

fake_keyword = train[train.target==0].keyword.value_counts().head(10)



# Set how big the canvas is for the plot

sns.set(rc = {'figure.figsize':(15,8)})



plt.subplot(121)

sns.barplot(real_keyword, real_keyword.index, color = 'blue')



# Set titles

plt.title('Top keywords for real disaster tweets', fontsize = 20)

plt.xlabel('Count', fontsize = 15)

plt.ylabel('Word', fontsize = 15)



plt.subplot(122)

sns.barplot(fake_keyword, fake_keyword.index, color = 'red')



# Set titles

plt.title('Top keywords for fake disaster tweets', fontsize = 20)

plt.xlabel('Count', fontsize = 15)

plt.ylabel('Word', fontsize = 15)



plt.show()
# How many unique locations are in the training dataset 

print ('There are {} locations in the training dataset'.format(train.location.nunique()))
# Get location counts 

ax = sns.countplot(y = 'location', data = train, order = train.location.value_counts().iloc[:10].index)

ax.set_title('Top 10 Location Count', fontsize = 30)

ax.set_xlabel('Count', fontsize = 20)

ax.set_ylabel('Location', fontsize = 20)
# Get location counts by response



real_location = train[train.target==1].location.value_counts().head(10)

fake_location = train[train.target==0].location.value_counts().head(10)



# Set how big the canvas is for the plot

sns.set(rc = {'figure.figsize':(15,8)})



plt.subplot(121)

sns.barplot(real_location, real_location.index, color = 'blue')



# Set titles

plt.title('Top locations for real disaster tweets', fontsize = 20)

plt.xlabel('Count', fontsize = 15)

plt.ylabel('Word', fontsize = 15)



plt.subplot(122)

sns.barplot(fake_location, fake_location.index, color = 'red')



# Set titles

plt.title('Top locations for fake disaster tweets', fontsize = 20)

plt.xlabel('Count', fontsize = 15)

plt.ylabel('Word', fontsize = 15)



plt.show()
# Cleanup the location variable



raw_location = train.location.value_counts()

top_location = list(raw_location[raw_location >= 10].index)

top_only = train[train.location.isin(top_location)]



for col in ['keyword', 'location']:

    train[col] = train[col].fillna('None')

    test[col] = test[col].fillna('None')



def clean_loc(x):

    if x == 'None':

        return 'None'

    elif x == 'Earth' or x =='Worldwide' or x == 'Everywhere':

        return 'World'

    elif 'New York' in x or 'NYC' in x or 'New York, NY' in x:

        return 'New York'    

    elif 'London' in x or 'London, UK' in x:

        return 'London'

    elif 'Mumbai' in x or 'Mumbai, India' in x:

        return 'Mumbai'

    elif 'Washington' in x and 'D' in x and 'C' in x:

        return 'Washington DC'

    elif 'San Francisco' in x or 'San Francisco, CA' in x:

        return 'San Francisco'

    elif 'Los Angeles' in x or 'Los Angeles, CA' in x:

        return 'Los Angeles'

    elif 'Seattle' in x or 'Seattle, WA' in x:

        return 'Seattle'

    elif 'Chicago' in x or 'Chicago, IL' in x:

        return 'Chicago'

    elif 'Toronto' in x:

        return 'Toronto'

    elif 'Sacramento' in x or 'Sacramento, CA' in x:

        return 'Sacramento'

    elif 'Atlanta' in x or 'Atalanta, GA' in x:

        return 'Atlanta'

    elif 'California' in x:

        return 'California'

    elif 'Florida' in x:

        return 'Florida'

    elif 'Texas' in x:

        return 'Texas'

    elif 'United States' in x or 'USA' in x:

        return 'USA'

    elif 'United Kingdom' in x or 'UK' in x or 'Britain' in x:

        return 'UK'

    elif 'Canada' in x:

        return 'Canada'

    elif 'India' in x:

        return 'India'

    elif 'Kenya' in x:

        return 'Kenya'

    elif 'Nigeria' in x:

        return 'Nigeria'

    elif 'Australia' in x:

        return 'Australia'

    elif 'Indonesia' in x:

        return 'Indonesia'

    elif x in top_location:

        return x

    else: return 'Others'

    

train['location_clean'] = train['location'].apply(lambda x: clean_loc(str(x)))

test['location_clean'] = test['location'].apply(lambda x: clean_loc(str(x)))
# Get location counts 

ax = sns.countplot(y = 'location_clean', data = train, order = train.location_clean.value_counts().iloc[:10].index)

ax.set_title('Top 10 Location Count', fontsize = 30)

ax.set_xlabel('Count', fontsize = 20)

ax.set_ylabel('Location', fontsize = 20)
# Get location counts by response



real_location = train[train.target==1].location_clean.value_counts().head(10)

fake_location = train[train.target==0].location_clean.value_counts().head(10)



# Set how big the canvas is for the plot

sns.set(rc = {'figure.figsize':(15,8)})



plt.subplot(121)

sns.barplot(real_location, real_location.index, color = 'blue')



# Set titles

plt.title('Top locations for real disaster tweets', fontsize = 20)

plt.xlabel('Count', fontsize = 15)

plt.ylabel('Location', fontsize = 15)



plt.subplot(122)

sns.barplot(fake_location, fake_location.index, color = 'red')



# Set titles

plt.title('Top locations for fake disaster tweets', fontsize = 20)

plt.xlabel('Count', fontsize = 15)

plt.ylabel('Location', fontsize = 15)



plt.show()
# word_count

train['word_count'] = train['text'].apply(lambda x: len(str(x).split()))

test['word_count'] = test['text'].apply(lambda x: len(str(x).split()))



# unique_word_count

train['unique_word_count'] = train['text'].apply(lambda x: len(set(str(x).split())))

test['unique_word_count'] = test['text'].apply(lambda x: len(set(str(x).split())))



# stop_word_count

train['stop_word_count'] = train['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))

test['stop_word_count'] = test['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))



# url_count

train['url_count'] = train['text'].apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))

test['url_count'] = test['text'].apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))



# mean_word_length

train['mean_word_length'] = train['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

test['mean_word_length'] = test['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))



# Tweet length

train['text_len'] = train['text'].apply(len)

test['text_len'] = test['text'].apply(len)



# char_count

train['char_count'] = train['text'].apply(lambda x: len(str(x)))

test['char_count'] = test['text'].apply(lambda x: len(str(x)))



# punctuation_count

train['punctuation_count'] = train['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

test['punctuation_count'] = test['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
meta_features = ['word_count', 'unique_word_count', 'stop_word_count', 'url_count', 'mean_word_length', 'char_count', 

                 'punctuation_count']



real_disaster_tweets = train.target == 1



fig, axes = plt.subplots(ncols = 2, nrows = len(meta_features), figsize = (20, 50), dpi=100)



for i, feature in enumerate(meta_features):

    

    sns.distplot(train.loc[~real_disaster_tweets][feature], label = 'Fake Disaster', ax = axes[i][0], color = 'red')

    sns.distplot(train.loc[real_disaster_tweets][feature], label = 'Real Disaster', ax = axes[i][0], color = 'blue')



    sns.distplot(train[feature], label = 'Training', ax = axes[i][1])

    sns.distplot(test[feature], label = 'Test', ax = axes[i][1])

    

    for j in range(2):

        

        axes[i][j].set_xlabel('')

        axes[i][j].tick_params(axis = 'x', labelsize = 15)

        axes[i][j].tick_params(axis = 'y', labelsize = 15)

        axes[i][j].legend()

    

    axes[i][0].set_title(f'{feature} Target Distribution in Training Set', fontsize = 15)

    axes[i][1].set_title(f'{feature} Training & Test Set Distribution', fontsize = 15)



plt.show()
def find_hashtags(tweet):

    return " ".join([match.group(0)[1:] for match in re.finditer(r"#\w+", tweet)]) or 'no'



def find_mentions(tweet):

    return " ".join([match.group(0)[1:] for match in re.finditer(r"@\w+", tweet)]) or 'no'



def find_links(tweet):

    return " ".join([match.group(0)[:] for match in re.finditer(r"https?://\S+", tweet)]) or 'no'



def process_text(df):

    

    df['hashtags'] = df['text'].apply(lambda x: find_hashtags(x))

    df['mentions'] = df['text'].apply(lambda x: find_mentions(x))

    df['links'] = df['text'].apply(lambda x: find_links(x))

    

    return df

    

train = process_text(train)

test = process_text(test)
def create_stat(df):

    # Count of hashtags (#)

    df['hashtag_count'] = df['hashtags'].apply(lambda x: len(str(x).split()))

    # Count of mentions (@)

    df['mention_count'] = df['mentions'].apply(lambda x: len(str(x).split()))

    # Count of links

    df['link_count'] = df['links'].apply(lambda x: len(str(x).split()))

    return df



train = create_stat(train)

test = create_stat(test)
def clean(tweet): 

            

    # Special characters

    tweet = re.sub(r"\x89??_", "", tweet)

    tweet = re.sub(r"\x89????", "", tweet)

    tweet = re.sub(r"\x89????", "", tweet)

    tweet = re.sub(r"\x89????When", "When", tweet)

    tweet = re.sub(r"\x89????", "", tweet)

    tweet = re.sub(r"China\x89????s", "China's", tweet)

    tweet = re.sub(r"let\x89????s", "let's", tweet)

    tweet = re.sub(r"\x89????", "", tweet)

    tweet = re.sub(r"\x89????", "", tweet)

    tweet = re.sub(r"\x89??\x9d", "", tweet)

    tweet = re.sub(r"??_", "", tweet)

    tweet = re.sub(r"\x89????", "", tweet)

    tweet = re.sub(r"\x89????????", "", tweet)

    tweet = re.sub(r"from????wounds", "from wounds", tweet)

    tweet = re.sub(r"????", "", tweet)

    tweet = re.sub(r"????", "", tweet)

    tweet = re.sub(r"Jap??_n", "Japan", tweet)    

    tweet = re.sub(r"????", "e", tweet)

    tweet = re.sub(r"????", "", tweet)

    tweet = re.sub(r"Suru????", "Suruc", tweet)

    tweet = re.sub(r"????", "", tweet)

    tweet = re.sub(r"????3million", "3 million", tweet)

    tweet = re.sub(r"????", "", tweet)

    

    # Contractions

    tweet = re.sub(r"he's", "he is", tweet)

    tweet = re.sub(r"there's", "there is", tweet)

    tweet = re.sub(r"We're", "We are", tweet)

    tweet = re.sub(r"That's", "That is", tweet)

    tweet = re.sub(r"won't", "will not", tweet)

    tweet = re.sub(r"they're", "they are", tweet)

    tweet = re.sub(r"Can't", "Cannot", tweet)

    tweet = re.sub(r"wasn't", "was not", tweet)

    tweet = re.sub(r"don\x89????t", "do not", tweet)

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

    tweet = re.sub(r"I\x89????m", "I am", tweet)

    tweet = re.sub(r"I'm", "I am", tweet)

    tweet = re.sub(r"Isn't", "is not", tweet)

    tweet = re.sub(r"Here's", "Here is", tweet)

    tweet = re.sub(r"you've", "you have", tweet)

    tweet = re.sub(r"you\x89????ve", "you have", tweet)

    tweet = re.sub(r"we're", "we are", tweet)

    tweet = re.sub(r"what's", "what is", tweet)

    tweet = re.sub(r"couldn't", "could not", tweet)

    tweet = re.sub(r"we've", "we have", tweet)

    tweet = re.sub(r"it\x89????s", "it is", tweet)

    tweet = re.sub(r"doesn\x89????t", "does not", tweet)

    tweet = re.sub(r"It\x89????s", "It is", tweet)

    tweet = re.sub(r"Here\x89????s", "Here is", tweet)

    tweet = re.sub(r"who's", "who is", tweet)

    tweet = re.sub(r"I\x89????ve", "I have", tweet)

    tweet = re.sub(r"y'all", "you all", tweet)

    tweet = re.sub(r"can\x89????t", "cannot", tweet)

    tweet = re.sub(r"would've", "would have", tweet)

    tweet = re.sub(r"it'll", "it will", tweet)

    tweet = re.sub(r"we'll", "we will", tweet)

    tweet = re.sub(r"wouldn\x89????t", "would not", tweet)

    tweet = re.sub(r"We've", "We have", tweet)

    tweet = re.sub(r"he'll", "he will", tweet)

    tweet = re.sub(r"Y'all", "You all", tweet)

    tweet = re.sub(r"Weren't", "Were not", tweet)

    tweet = re.sub(r"Didn't", "Did not", tweet)

    tweet = re.sub(r"they'll", "they will", tweet)

    tweet = re.sub(r"they'd", "they would", tweet)

    tweet = re.sub(r"DON'T", "DO NOT", tweet)

    tweet = re.sub(r"That\x89????s", "That is", tweet)

    tweet = re.sub(r"they've", "they have", tweet)

    tweet = re.sub(r"i'd", "I would", tweet)

    tweet = re.sub(r"should've", "should have", tweet)

    tweet = re.sub(r"You\x89????re", "You are", tweet)

    tweet = re.sub(r"where's", "where is", tweet)

    tweet = re.sub(r"Don\x89????t", "Do not", tweet)

    tweet = re.sub(r"we'd", "we would", tweet)

    tweet = re.sub(r"i'll", "I will", tweet)

    tweet = re.sub(r"weren't", "were not", tweet)

    tweet = re.sub(r"They're", "They are", tweet)

    tweet = re.sub(r"Can\x89????t", "Cannot", tweet)

    tweet = re.sub(r"you\x89????ll", "you will", tweet)

    tweet = re.sub(r"I\x89????d", "I would", tweet)

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

    tweet = re.sub(r"don????t", "do not", tweet)   

            

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

           

    # Urls

    tweet = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", tweet)

        

    # Words with punctuations and special characters

    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`"

    for p in punctuations:

        tweet = tweet.replace(p, f' {p} ')

        

    # ... and ..

    tweet = tweet.replace('...', ' ... ')

    if '...' not in tweet:

        tweet = tweet.replace('..', ' ... ')      

        

    # Acronyms

    tweet = re.sub(r"MH370", "Malaysia Airlines Flight 370", tweet)

    tweet = re.sub(r"m????sica", "music", tweet)

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

    tweet = re.sub(r"Suruc", "Sanliurfa", tweet)   

    

    # Grouping same words without embeddings

    tweet = re.sub(r"Bestnaijamade", "bestnaijamade", tweet)

    tweet = re.sub(r"SOUDELOR", "Soudelor", tweet)

    

    return tweet



train['text_cleaned'] = train['text'].apply(lambda tweet : clean(tweet))

test['text_cleaned'] = test['text'].apply(lambda tweet : clean(tweet))
# Get the top words for the Fake Disasters

fake_tweets = train['text_cleaned'].dropna().loc[train['target'] == 0].tolist()

fake_tweets = ''.join(fake_tweets).lower()



# Get the top words for the Real Disasters

real_tweets = train['text_cleaned'].dropna().loc[train['target'] == 1].tolist()

real_tweets = ''.join(real_tweets).lower()



# Get the overall top words 

overall = train['text_cleaned'].dropna().tolist()

overall = ''.join(overall).lower()
# Get the word cloud

wordcloud = WordCloud(stopwords=STOPWORDS,

                      background_color='white',

                      width=1200,

                      height=1000).generate(fake_tweets)

plt.imshow(wordcloud)

plt.title('Frequent words amongst Fake Disasters', fontsize = 45)

plt.axis('off')

plt.show()
# Get the word cloud

wordcloud = WordCloud(stopwords=STOPWORDS,

                      background_color='white',

                      width=1200,

                      height=1000).generate(real_tweets)

plt.imshow(wordcloud)

plt.title('Frequent words amongst Real Disasters', fontsize = 45)

plt.axis('off')

plt.show()
# Get the word cloud

wordcloud = WordCloud(stopwords=STOPWORDS,

                      background_color='white',

                      width=1200,

                      height=1000).generate(overall)

plt.imshow(wordcloud)

plt.title('Frequent words Overall', fontsize = 45)

plt.axis('off')

plt.show()
# Just wanted to move the column around to make it easier to split the data for myself later



train = train.set_index('target').reset_index()



# Do not need these two variables in the model building process



train = train.drop(['id', 'text'], axis=1)

test = test.drop(['text'], axis=1)



#  What is most highly correlated with the target variables?



train.corr()['target'].drop('target').sort_values()
#  Split the target and the features

features = train.iloc[:, 1:].values # The features 

labels = train.iloc[:, 0].values # The target



test_features = test.iloc[:, :].values # The features 
#  Need to do more cleaning to get the data ready for analysis



processed_features = []



for sentence in range(0, len(features)):

    # Remove all the special characters

    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))



    # remove all single characters

    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)



    # Remove single characters from the start

    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 



    # Substituting multiple spaces with single space

    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)



    # Removing prefixed 'b'

    processed_feature = re.sub(r'^b\s+', '', processed_feature)



    # Converting to Lowercase

    processed_feature = processed_feature.lower()



    processed_features.append(processed_feature)

    

test_processed_features = []



for sentence in range(0, len(test_features)):

    # Remove all the special characters

    test_processed_feature = re.sub(r'\W', ' ', str(test_features[sentence]))



    # remove all single characters

    test_processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', test_processed_feature)



    # Remove single characters from the start

    test_processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', test_processed_feature) 



    # Substituting multiple spaces with single space

    test_processed_feature = re.sub(r'\s+', ' ', test_processed_feature, flags=re.I)



    # Removing prefixed 'b'

    test_processed_feature = re.sub(r'^b\s+', '', test_processed_feature)



    # Converting to Lowercase

    test_processed_feature = test_processed_feature.lower()



    test_processed_features.append(test_processed_feature)
# Vectorize the words and transform them for analysis

vectorizer = TfidfVectorizer (max_features=1300, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))

processed_features = vectorizer.fit_transform(processed_features).toarray()



vectorizer_test = TfidfVectorizer (max_features=1300, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))

test_processed_features = vectorizer_test.fit_transform(test_processed_features).toarray()
# Split the data into a training and testing set.  The algorithm is to be built on the training set and the algorithm is evaluated on the testing set



X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)
from sklearn.ensemble import ExtraTreesClassifier



# Create ETC Classifier

np.random.seed(1729)

etc = ExtraTreesClassifier()



# Train the model using the training sets



etc_model = etc.fit(X_train, y_train)



# Predict the response for test dataset



predictions = etc_model.predict(X_test)



# Model Accuracy, how often is the classifier correct?



print("The algorithm accuracy is:",metrics.accuracy_score(y_test, predictions)*100)
y_pred = etc_model.predict(test_processed_features)
etc_pred = pd.DataFrame(y_pred, columns=['target'])



etc_pred.head()
etc_sub = test.merge(etc_pred, how='outer', left_index=True, right_index=True)

etc_sub.head()
Final_sub = etc_sub[['id','target']]

Final_sub.head()
Final_sub.to_csv('submission.csv', index=False)