import pandas as pd

import numpy as np



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, classification_report, log_loss

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.naive_bayes import MultinomialNB, GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import PowerTransformer, MinMaxScaler, MaxAbsScaler

import nltk 

#nltk.download('stopwords')

#nltk.download()

from nltk.corpus import stopwords 
#pip install emoji
# use Pandas to read in the csv files. The pd.read_csv() method creates a DataFrame from a csv file

dataset = pd.read_csv('../input/mbti-1/mbti_1.csv')
#print(dataset.head())

dataset.head()
# Let's take a look at our data

next(iter(dataset.keys()))
# Notice that our dictionary is currently in key: type, value: list of posts

next(iter(dataset.values))
datasets = dataset

datasets.set_index('type',inplace=True)

datasets.head()
# Let's take a look at a post for INFJ

dataset.posts.loc['INFJ']
# Apply a first round of text cleaning techniques

import re

import string



def cleaning_data(text):

    '''Remove web url'''

    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', str(text), flags=re.MULTILINE)

    '''Make text lowercase'''

    text = text.lower()

    '''remove text in square brackets'''

    text = re.sub('\[.*?\]', '', text)

    '''remove punctuations'''

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    '''remove digits'''

    text = re.sub('\w*\d\w*', '', text)

    '''remove stop words'''

    STOPWORDS = set(stopwords.words('english'))

    text = ' '.join(word for word in text.split() if word not in STOPWORDS)

    return text



data_round1 = lambda x: cleaning_data(x)
# Lets take a look at the updated text

data_cleaning = pd.DataFrame(dataset.posts.apply(data_round1))

data_cleaning
def cleaning_data2(text):

    '''Get rid of some additional punctuations '''

    text = re.sub('\[''""...]', '', text)

    '''Get rid of non-sensical'''

    text = re.sub('\n', '', text)

    '''Remove single characters from the start'''

    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)

    '''Removing prefixed 'b'''

    text = re.sub(r'^b\s+', '', text)

    '''Correcting typos'''

    text = text.correct()

    '''Remove rare words'''

    freq = pd.Series(' '.join(data_cleaning['posts']).split()).value_counts()[-500:]

    # let's remove these words as their presence will be of any use

    freq = list(freq.index)

    text = data_cleaning['posts'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

    return text



data_round2 = lambda x: cleaning_data(x)
# Lets take a look at the updated text

data_cleaning = pd.DataFrame(data_cleaning.posts.apply(data_round2))

data_cleaning
def cleaning_data3(text):

    '''Get rid of all single characters'''

    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', str(text))

    '''Substituting multiple spaces with single space'''

    text = re.sub(r'\s+', ' ', text, flags=re.I)

    '''Remove all the special characters'''

    text = re.sub(r'\W', ' ', str(text))

    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text))

    '''Remove Frequent words'''

    freq = pd.Series(' '.join(data_cleaning['posts']).split()).value_counts()[:500]

    # let's remove these words as their presence will be of any use

    freq = list(freq.index)

    text = data_cleaning['posts'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

    return text

                  

data_round3 = lambda x: cleaning_data(x) 
# Lets take a look at the updated text

data_cleaning = pd.DataFrame(data_cleaning.posts.apply(data_round3))

data_cleaning
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()



def stem_tokens(tokens, stemmer):

    ''' Stemming - removing and replacing suffixes'''

    stemmed = []

    for item in tokens:

        stemmed.append(stemmer.stem(item))

        stems =ste

    return stems

data_round4 = lambda x: cleaning_data(x)
# Lets take a look at the updated text

data_cleaning = pd.DataFrame(data_cleaning.posts.apply(data_round4))

data_cleaning
def Lemmatization(text):

    ''' Lemmatization - returns the dictionary form of a word '''

    text = text.split()



    text = [stemmer.lemmatize(word) for word in text]

    text = ' '.join(text)



    texts.append(text)

    return text



data_round5 = lambda x: cleaning_data(x) 
# Lets take a look at the updated text

data_cleaning = pd.DataFrame(data_cleaning.posts.apply(data_round5))

data_cleaning
def remove_emoji(text):

    '''Remove all sorts of emojis'''

    emoji_pattern = emoji.get_emoji_regexp().sub(u'', text)

    return emoji_pattern

data_round6 = lambda x: cleaning_data(x) 
# Lets take a look at the updated text

data_cleaning = pd.DataFrame(data_cleaning.posts.apply(data_round6))

data_cleaning
dataset.head()
# Let's pickle it for later use

dataset.to_pickle("corpus.pkl")
# Sentence Tokenizatin

'''Sentence tokenizer breaks text paragraph into sentences'''

from nltk import sent_tokenize

tokenized_sent=sent_tokenize(str(data_cleaning.posts))



'''Word tokenizer breaks text paragraph into words'''

from nltk import word_tokenize

tokens = word_tokenize(str(data_cleaning.posts))

tokens
# Frequency Distribution

from nltk.probability import FreqDist

fdist = FreqDist(tokens)

print(fdist)

fdist.most_common(2)
# Frequency Distribution Plot

import matplotlib.pyplot as plt

fdist.plot(30,cumulative=False)

plt.show()
'''Stopwords - Stopwords considered as noise in the text. such as is, am, are, this, a, an, the, etc.'''

from nltk.corpus import stopwords

stop_words=set(stopwords.words("english"))

print(stop_words)
# We are going to create a document-term matrix using CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer  

filtered_sent = ['being', "haven't", 'they', 'but', 'my', 'through', 'up', 'once', "wasn't", 'over', 'his', 'all', 'the', 'further', 'doing', 

 'am', 'd', 'until', 'when', 'it', 'shan', 'on', 'him', 'she', 'yourselves', 'themselves', 'theirs', 'as', 'while', 'more', 's', 

 'have', 'been', 'just', "doesn't", 'aren', "hasn't", 'will', 'were', 'your', 'ain', 'doesn', 'this', 'these', 'with', 'o', 'here', 

 're', 'same', 'isn', 'had', 'above', 'whom', 'nor', 'by', 'herself', 'such', 'ourselves', 'where', 'any', 'mightn', 'what', 'because', 

 'are', 'you', 'its', 'won', 'yourself', 'needn', 'why', "didn't", 'ma', 'no', 'against', 'don', "she's", 'has', 'be', 'ours', 'only', 

 'yours', 'm', 'hadn', 'those', 'during', 'into', 'and', "that'll", 'is', "should've", "mustn't", 'under', 'mustn', 'them', 'in', 'some', 

 'a', 'was', 'off', 'me', 'wasn', 'after', 'i', 'who', 'than', 'both', "you're", 'to', 'not', 'himself', 'he', 'again', 'now', 'how', 'so', 

 'if', 'that', "hadn't", 'which', 'too', "you'll", "aren't", "it's", 'below', 'y', 'or', 'then', 'their', 'wouldn', 'should', 've', 'can', 

 "you've", "couldn't", 'there', 'hasn', 'having', 'most', "won't", 'each', 'hers', 'did', "shouldn't", 'an', 't', 'very', "weren't", 'between', 

 'out', 'down', 'own', 'do', 'itself', 'from', "don't", 'll', 'haven', 'her', "needn't", 'couldn', "you'd", 'myself', "mightn't", 'about', 'didn', 

 'for', 'few', 'other', 'does', 'before', "wouldn't", 'we', "isn't", 'shouldn', "shan't", 'of', 'at', 'our', 'weren'

 'im','like', 'think', 'people', 'dont', 'know', 'really', 'would', 'one', 'get',

 'feel', 'love', 'time', 'ive', 'much', 'say', 'something', 'good',

 'things', 'want', 'see', 'way', 'someone', 'also', 'well', 'friends',

 'always', 'type', 'lot', 'could', 'make', 'go', 'thing', 'even', 'person', 'need',

 'find', 'right', 'never', 'youre', 'thats', 'going', 'life', 'friend',

 'pretty', 'though', 'sure', 'said', 'cant', 'first', 'actually', 'still',

 'best', 'many', 'take', 'others', 'work', 'read', 'sometimes', 'got',

 'around', 'thought', 'try', 'back', 'makes', 'better', 'trying', 'didnt',

 'agree', 'kind', 'mean', 'tell', 'post', 'two', 'probably', 'talk',

 'anything', 'since', 'maybe', 'understand', 'seems', 'ill', 'id', 'little',

 'doesnt', 'thread', 'new', 'long', 'ever', 'years', 'hard', 'might',

 'types', 'us', 'everyone','different', 'look', 'usually', 'may', 'day', 'give',

 'come', 'personality', 'guess', 'mind', 'relationship', 'bit', 'quite',

 'great', 'made', 'thinking', 'everything', 'school', 'seem', 'bad', 'every',

 'help', 'yes', 'definitely', 'believe', 'point', 'used', 'infp', 'guys', 'tend','hes', 'use', 'intj', 

 'often', 'getting', 'interesting', 'last', 'talking', 'infj', 'times',

 'another', 'mbti', 'enfp', 'world','question','part', 'theres',

 'feeling', 'fun', 'intp', 'enough', 'isnt', 'else', 'hate', 'lol', 'keep',

 'anyone', 'nice', 'idea', 'sense','least','enfj', 'entj', 'entp', 'esfj', 'esfp', 'estj', 'estp',

 'isfj', 'isfp', 'istj', 'istp','sound','thank']

vectorizer = CountVectorizer(max_features=1500, min_df=1, max_df=1.0, stop_words=filtered_sent)  

X = vectorizer.fit_transform(data_cleaning.posts)

data_x = pd.DataFrame(X.toarray(), columns = vectorizer.get_feature_names()) 

data_x.index = data_cleaning.index

del data_x.index.name

data_x
''' Parts of Speech tagging - identify the grammatical group of a given word '''

nltk.pos_tag(data_x.columns)
# Let's pickle it for later use

data_x.to_pickle("xdata.pkl")
# Let's also pickle the cleaned data (before we put it in documnet-term matrix)

import pickle



data_cleaning.to_pickle('data_cleaning.pkl')

pickle.dump(X, open("vectorizer.pkl", "wb"))
# Read in the document-term matrix

data = pd.read_pickle('xdata.pkl')

data = data.transpose()

data = data.groupby(level=0, axis=1).sum()

data.head()
data.index.name = 'tokens'

data
# Find the top 500 words said by each personality type

top_dict = {}

for c in data.columns:

    top = data[c].sort_values(ascending=False).head(500)

    top_dict[c] = list(zip(top.index, top.values))

    

top_dict
del data.index.name
# print the top 200 words said by each personality type

for Personality_type, top_words in top_dict.items():

    print(Personality_type)

    print(', '.join([str(word) for word, count in top_words[0:200]]))

    print('_ _ _')
Personality_type
# Look at the most commom top words __> add them to the stop word list

from collections import Counter



# Let's first pull out the top 500 words for each Personality type

words = []

for Personality_type in data.columns:

    top = [word for(word, count) in top_dict[Personality_type][:500]]

    for t in top:

        words.append(t)

words
# Let's aggregate this list and identify the most common words along with how many times they occur in

Counter(words).most_common()
# if more than half of the personality types have it as a top word, exclude it from the list

add_stop_words = [word for word, count in Counter(words).most_common() if count > 7]

add_stop_words
# Let's update our documnet-term matrix with the new list of stop words

from sklearn.feature_extraction import text

from sklearn.feature_extraction.text import CountVectorizer



# Read in cleaned data

data_clean = pd.read_pickle('data_cleaning.pkl')



# Add new stop words

#stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

add_stop_words = extra_stop_words = ['im','like', 'think', 'people', 'dont', 'know', 'really', 'would', 'one', 'get',

 'feel', 'love', 'time', 'ive', 'much', 'say', 'something', 'good', 'start', 'girl',

 'things', 'want', 'see', 'way', 'someone', 'also', 'well', 'friends',

 'always', 'type', 'lot', 'could', 'make', 'go', 'thing', 'even', 'person', 'need',

 'find', 'right', 'never', 'youre', 'thats', 'going', 'life', 'friend',

 'pretty', 'though', 'sure', 'said', 'cant', 'first', 'actually', 'still', 'best', 'many', 'take', 'others', 'work', 'read', 'sometimes', 'got',

 'around', 'thought', 'try', 'back', 'makes', 'better', 'trying', 'didnt', 'agree', 'kind', 'mean', 'tell', 'post', 'two', 'probably', 'talk',

 'anything', 'since', 'maybe', 'understand', 'seems', 'ill', 'id', 'little', 'doesnt', 'thread', 'new', 'long', 'ever', 'years', 'hard', 'might',

 'types', 'us', 'everyone','different', 'look', 'usually', 'may', 'day', 'give', 'come', 'personality', 'guess', 'mind', 'relationship', 'bit', 'quite',

 'great', 'made', 'thinking', 'everything', 'school', 'seem', 'bad', 'every',

 'help', 'yes', 'definitely', 'believe', 'point', 'used', 'infp', 'guys', 'tend','hes', 'use', 'intj', 

 'often', 'getting', 'interesting', 'last', 'talking', 'infj', 'times', 'another', 'mbti', 'enfp', 'world','question','part', 'theres',

 'feeling', 'fun', 'intp', 'enough', 'isnt', 'else', 'hate', 'lol', 'keep',

 'anyone', 'nice', 'idea', 'sense','least','enfj', 'entj', 'entp', 'esfj', 'esfp', 'estj', 'estp',

 'isfj', 'isfp', 'istj', 'istp','sound','thank','enfjs', 'entjs', 'entps', 'esfjs', 'esfps', 'estjs', 'estps',

 'isfjs', 'isfps', 'istjs', 'istps','infjs','infps','intjs','enfps','intps','isnts', 'let', 'problem','im', 'fe',

 'feelings', 'happy', 'thanks', 'sorry', 'care', 'especially', 'guy', 'true', 'experience', 'almost',

 'hope', 'wrong', 'sounds', 'ask', 'close', 'stuff', 'however', 'far', 'strong', 'let', 'theyre', 'put', 'shes',

 'high', 'told', 'year', 'wanted', 'music', 'comes', 'away', 'nothing', 'found', 'head', 'forum', 'start', 'totally',

 'making', 'either', 'test', 'without', 'relate', 'enjoy', 'using', 'rather', 'wouldnt', 'exactly', 'reading', 'big',

 'reason', 'fact', 'real', 'completely', 'problem', 'girl', 'remember', 'social', 'wish', 'change', 'welcome', 'looking',

 'interested', 'ni', 'past', 'along', 'whole', 'able', 'situation', 'emotions', 'old', 'figure', 'similar', 'done', 'show',

 'advice', 'havent', 'functions', 'important', 'felt', 'already', 'met', 'alone', 'saying', 'yet', 'seen', 'emotional', 'end',

 'family', 'place', 'book', 'ago', 'hear', 'open', 'example', 'live', 'sort', 'man', 'fi', 'oh', 'together', 'awesome', 'relationships',

 'ones', 'words', 'personally', 'arent', 'stop', 'mostly', 'matter', 'certain', 'job', 'crazy', 'wont', 'wasnt', 'answer', 'general',

 'took', 'although', 'honestly', 'gets', 'meet', 'months', 'means', 'group', 'working', 'started', 'recently', 'needs', 'course',

 'mom', 'days', 'conversation', 'less', 'based', 'questions', 'dad', 'easy', 'taking', 'interest', 'funny', 'saw', 'weird', 'depends',

 'posts', 'cool', 'mine', 'case', 'enneagram', 'extremely', 'goes', 'honest', 'hurt', 'youve', 'ne', 'night', 'next', 'consider', 'opinion',

 'whatever', 'absolutely', 'home', 'learn', 'explain', 'topic', 'prefer', 'play', 'appreciate', 'name', 'heard', 'must', 'seriously', 'self', 

 'likely', 'function', 'tried', 'couple', 'easily', 'wants', 'generally', 'week', 'known', 'ok', 'asking', 'trust', 'list', 'eyes', 'taken',

 'common', 'second', 'unless', 'posted', 'speak', 'write', 'instead', 'stay', 'meant', 'attention', 'ways', 'free', 'whether', 'college', 

 'female', 'seeing', 'says', 'parents', 'writing', 'fine', 'half', 'si', 'stand', 'god', 'watch', 'haha', 'today', 'personal', 'yeah', 'share',

 'become', 'thoughts', 'feels', 'went', 'favorite', 'side', 'kinda', 'curious', 'whats', 'deep', 'towards', 'please', 'video', 'asked',

 'possible', 'takes', 'act', 'problems', 'deal', 'etc', 'se', 'male', 'call', 'knew', 'wonder', 'women', 'dating', 'happens', 'face', 'intjs',

 'lets', 'works', 'story', 'perfect', 'anyway', 'realize', 'move', 'coming', 'introverted', 'word', 'mother', 'okay', 'movie', 'currently',

 'add', 'leave', 'happen', 'ideas', 'listen', 'noticed', 'came', 'game', 'called', 'picture', 'looks', 'stupid', 'understanding', 'issue',

 'level', 'issues', 'girls', 'liked', 'future', 'house', 'watching', 'money', 'spend', 'notice', 'three', 'class', 'short', 'view', 'cognitive',

 'te', 'order', 'sent', 'moment', 'difficult', 'left', 'sex', 'information', 'lots', 'cause', 'basically', 'fit', 'couldnt', 'ii',

 'happened', 'except', 'response', 'reasons', 'full', 'playing', 'brother', 'learning', 'late', 'child', 'books', 'decided', 'thei', 'imagine',

 'art', 'small', 'huge', 'wait', 'sleep', 'single', 'kids', 'description', 'top' 'song', 'heart', 'infjs', 'hell', 'sad', 'super',

 'human', 'giving', 'wondering', 'value', 'serious', 'youll', 'simply', 'clear', 'break', 'lost', 'learned', 'age', 'typing', 'sister',

 'hand', 'random', 'body', 'run', 'internet', 'living', 'character', 'due', 'hours', 'given', 'difference', 'online', 'telling', 'mentioned',

 'set', 'room', 'constantly', 'games', 'later', 'glad', 'control', 'perhaps','angry', 'ti', 'loved', 'gonna', 'men', 'fear', 'turn',

 'young', 'laugh', 'fall', 'food', 'perc', 'pick', 'confused', 'certainly', 'younger', 'normal', 'outside', 'across', 'physical', 'several',

 'simple', 'infps', 'energy', 'truly', 'intps', 'woman', 'older', 'toi', 'knows', 'aware', 'date', 'lack', 'suppose', 'choose', 'plan',

 'laughing', 'shit', 'kid', 'language', 'theory', 'eat', 'truth', 'gave', 'check', 'cannot', 'tests', 'amazing', 'peoples', 'boyfriend', 'focus',

 'comfortable', 'heres', 'hold','finding', 'respect', 'inside', 'movies', 'hair', 'bored', 'brain', 'situations', 'doubt', 'dominant', 'main',

 'worth', 'state', 'number', 'xd', 'bring', 'reply', 'characters', 'meaning', 'knowing', 'likes', 'nature', 'enfps', 'extroverted',

 'opposite', 'results', 'specific', 'worry', 'afraid', 'black', 'somewhere', 'behind', 'phone', 'process', 'romantic', 'shy', 'quiet','listening',

'power', 'accurate', 'negative', 'cold', 'entps','song','nf','nt','term','introvert','sj','dream', 'ex',

 'realized', 'finally', 'weeks', 'early','particular', 'annoying', 'boring', 'describe', 'vs', 'system', 'points', 'rest', 'english', 'avoid']





#Recreate document-term matrix

cv = CountVectorizer(max_features=1500, min_df=1, max_df=1.0, stop_words=add_stop_words)

data_cv = cv.fit_transform(data_clean.posts)

data_stop = pd.DataFrame(data_cv.toarray(), columns = cv.get_feature_names())

data_stop.index = data_clean.index

del data_stop.index.name

data_stop



# pickle it for later use

pickle.dump(cv, open("cv_stop.pkl", "wb"))

data_stop.to_pickle("data_post_stop.pkl")
data_stop = data_stop.transpose()
data_stop = data_stop.groupby(level=0, axis=1).sum()

data_stop
# Let's make some words clouds:

from wordcloud import WordCloud

import matplotlib

import matplotlib.pyplot as plt



# ADD STOP WORDS

extra_stop_words = ['im','like', 'think', 'people', 'dont', 'know', 'really', 'would', 'one', 'get',

 'feel', 'love', 'time', 'ive', 'much', 'say', 'something', 'good', 'start', 'girl',

 'things', 'want', 'see', 'way', 'someone', 'also', 'well', 'friends',

 'always', 'type', 'lot', 'could', 'make', 'go', 'thing', 'even', 'person', 'need',

 'find', 'right', 'never', 'youre', 'thats', 'going', 'life', 'friend',

 'pretty', 'though', 'sure', 'said', 'cant', 'first', 'actually', 'still', 'best', 'many', 'take', 'others', 'work', 'read', 'sometimes', 'got',

 'around', 'thought', 'try', 'back', 'makes', 'better', 'trying', 'didnt', 'agree', 'kind', 'mean', 'tell', 'post', 'two', 'probably', 'talk',

 'anything', 'since', 'maybe', 'understand', 'seems', 'ill', 'id', 'little', 'doesnt', 'thread', 'new', 'long', 'ever', 'years', 'hard', 'might',

 'types', 'us', 'everyone','different', 'look', 'usually', 'may', 'day', 'give', 'come', 'personality', 'guess', 'mind', 'relationship', 'bit', 'quite',

 'great', 'made', 'thinking', 'everything', 'school', 'seem', 'bad', 'every',

 'help', 'yes', 'definitely', 'believe', 'point', 'used', 'infp', 'guys', 'tend','hes', 'use', 'intj', 

 'often', 'getting', 'interesting', 'last', 'talking', 'infj', 'times', 'another', 'mbti', 'enfp', 'world','question','part', 'theres',

 'feeling', 'fun', 'intp', 'enough', 'isnt', 'else', 'hate', 'lol', 'keep',

 'anyone', 'nice', 'idea', 'sense','least','enfj', 'entj', 'entp', 'esfj', 'esfp', 'estj', 'estp',

 'isfj', 'isfp', 'istj', 'istp','sound','thank','enfjs', 'entjs', 'entps', 'esfjs', 'esfps', 'estjs', 'estps',

 'isfjs', 'isfps', 'istjs', 'istps','infjs','infps','intjs','enfps','intps','isnts', 'let', 'problem','im', 'fe',

 'feelings', 'happy', 'thanks', 'sorry', 'care', 'especially', 'guy', 'true', 'experience', 'almost',

 'hope', 'wrong', 'sounds', 'ask', 'close', 'stuff', 'however', 'far', 'strong', 'let', 'theyre', 'put', 'shes',

 'high', 'told', 'year', 'wanted', 'music', 'comes', 'away', 'nothing', 'found', 'head', 'forum', 'start', 'totally',

 'making', 'either', 'test', 'without', 'relate', 'enjoy', 'using', 'rather', 'wouldnt', 'exactly', 'reading', 'big',

 'reason', 'fact', 'real', 'completely', 'problem', 'girl', 'remember', 'social', 'wish', 'change', 'welcome', 'looking',

 'interested', 'ni', 'past', 'along', 'whole', 'able', 'situation', 'emotions', 'old', 'figure', 'similar', 'done', 'show',

 'advice', 'havent', 'functions', 'important', 'felt', 'already', 'met', 'alone', 'saying', 'yet', 'seen', 'emotional', 'end',

 'family', 'place', 'book', 'ago', 'hear', 'open', 'example', 'live', 'sort', 'man', 'fi', 'oh', 'together', 'awesome', 'relationships',

 'ones', 'words', 'personally', 'arent', 'stop', 'mostly', 'matter', 'certain', 'job', 'crazy', 'wont', 'wasnt', 'answer', 'general',

 'took', 'although', 'honestly', 'gets', 'meet', 'months', 'means', 'group', 'working', 'started', 'recently', 'needs', 'course',

 'mom', 'days', 'conversation', 'less', 'based', 'questions', 'dad', 'easy', 'taking', 'interest', 'funny', 'saw', 'weird', 'depends',

 'posts', 'cool', 'mine', 'case', 'enneagram', 'extremely', 'goes', 'honest', 'hurt', 'youve', 'ne', 'night', 'next', 'consider', 'opinion',

 'whatever', 'absolutely', 'home', 'learn', 'explain', 'topic', 'prefer', 'play', 'appreciate', 'name', 'heard', 'must', 'seriously', 'self', 

 'likely', 'function', 'tried', 'couple', 'easily', 'wants', 'generally', 'week', 'known', 'ok', 'asking', 'trust', 'list', 'eyes', 'taken',

 'common', 'second', 'unless', 'posted', 'speak', 'write', 'instead', 'stay', 'meant', 'attention', 'ways', 'free', 'whether', 'college', 

 'female', 'seeing', 'says', 'parents', 'writing', 'fine', 'half', 'si', 'stand', 'god', 'watch', 'haha', 'today', 'personal', 'yeah', 'share',

 'become', 'thoughts', 'feels', 'went', 'favorite', 'side', 'kinda', 'curious', 'whats', 'deep', 'towards', 'please', 'video', 'asked',

 'possible', 'takes', 'act', 'problems', 'deal', 'etc', 'se', 'male', 'call', 'knew', 'wonder', 'women', 'dating', 'happens', 'face', 'intjs',

 'lets', 'works', 'story', 'perfect', 'anyway', 'realize', 'move', 'coming', 'introverted', 'word', 'mother', 'okay', 'movie', 'currently',

 'add', 'leave', 'happen', 'ideas', 'listen', 'noticed', 'came', 'game', 'called', 'picture', 'looks', 'stupid', 'understanding', 'issue',

 'level', 'issues', 'girls', 'liked', 'future', 'house', 'watching', 'money', 'spend', 'notice', 'three', 'class', 'short', 'view', 'cognitive',

 'te', 'order', 'sent', 'moment', 'difficult', 'left', 'sex', 'information', 'lots', 'cause', 'basically', 'fit', 'couldnt', 'ii',

 'happened', 'except', 'response', 'reasons', 'full', 'playing', 'brother', 'learning', 'late', 'child', 'books', 'decided', 'thei', 'imagine',

 'art', 'small', 'huge', 'wait', 'sleep', 'single', 'kids', 'description', 'top' 'song', 'heart', 'infjs', 'hell', 'sad', 'super',

 'human', 'giving', 'wondering', 'value', 'serious', 'youll', 'simply', 'clear', 'break', 'lost', 'learned', 'age', 'typing', 'sister',

 'hand', 'random', 'body', 'run', 'internet', 'living', 'character', 'due', 'hours', 'given', 'difference', 'online', 'telling', 'mentioned',

 'set', 'room', 'constantly', 'games', 'later', 'glad', 'control', 'perhaps','angry', 'ti', 'loved', 'gonna', 'men', 'fear', 'turn',

 'young', 'laugh', 'fall', 'food', 'perc', 'pick', 'confused', 'certainly', 'younger', 'normal', 'outside', 'across', 'physical', 'several',

 'simple', 'infps', 'energy', 'truly', 'intps', 'woman', 'older', 'toi', 'knows', 'aware', 'date', 'lack', 'suppose', 'choose', 'plan',

 'laughing', 'shit', 'kid', 'language', 'theory', 'eat', 'truth', 'gave', 'check', 'cannot', 'tests', 'amazing', 'peoples', 'boyfriend', 'focus',

 'comfortable', 'heres', 'hold','finding', 'respect', 'inside', 'movies', 'hair', 'bored', 'brain', 'situations', 'doubt', 'dominant', 'main',

 'worth', 'state', 'number', 'xd', 'bring', 'reply', 'characters', 'meaning', 'knowing', 'likes', 'nature', 'enfps', 'extroverted',

 'opposite', 'results', 'specific', 'worry', 'afraid', 'black', 'somewhere', 'behind', 'phone', 'process', 'romantic', 'shy', 'quiet','listening',

'power', 'accurate', 'negative', 'cold', 'entps','song','nf','nt','term','introvert','sj','dream', 'ex',

 'realized', 'finally', 'weeks', 'early','particular', 'annoying', 'boring', 'describe', 'vs', 'system', 'points', 'rest', 'english', 'avoid']



wc = WordCloud(stopwords=extra_stop_words, background_color='white', colormap=matplotlib.cm.inferno,

              max_font_size=150, random_state=42)
data_stop['personality'] = data_stop.index

del data_stop['personality']

data_stop
data_stop.columns
Personality_type
' '.join(data_cleaning.posts[Personality_type])
import matplotlib.pyplot as plt

# Reset the output dimensions

plt.rcParams['figure.figsize'] = [20, 10]



personality_Names =['ENFJ','ENFP','ENTJ','ENTP','ESFJ','ESFP','ESTJ','ESTP','INFJ','INFP','INTJ','INTP','ISFJ','ISFP','ISTJ','ISTP']



# create subplots for each personality type

for index, Personality_type in enumerate(data_stop.columns):

    wc.generate(' '.join(data_cleaning.posts[Personality_type]))

    

    plt.subplot(4, 5, index+1)

    plt.imshow(wc, interpolation="bilinear")

    plt.axis("off")

    plt.title(personality_Names[index])
# Find the number of words each personality uses

# identify the non-zero items in the document-term matrix, meaning that the 

unique_list = []

for Personality_type in data_stop.columns:

    unique = data_stop[Personality_type].nonzero()[0].size

    unique_list.append(unique)

    

# Create a new DataFrame that contains this unique word count

words_data = pd.DataFrame(list(zip(personality_Names, unique_list)), columns=['Type','unique_words'])

sorted_words = words_data.sort_values(by='unique_words')

sorted_words
# Find the total number of words that a personality type uses

total_list = []

for Personality_type in data_stop.columns:

    totals = sum(data_stop[Personality_type])

    total_list.append(totals)

    

# Add columns to the DataFrame

words_data['total_words'] = total_list



# Sort the DataFrame by total number of words used to see who talks alot

sorted_words_number = words_data.sort_values(by='total_words')

sorted_words_number
# Let's plot our findings

y_position = np.arange(len(words_data))

plt.subplot(1, 2, 1)

plt.barh(y_position, sorted_words.unique_words, align = "center")

plt.yticks(y_position, sorted_words.Type)

plt.title('Number of Unique Words', fontsize = 20)



plt.subplot(1, 2, 2)

plt.barh(y_position, sorted_words_number.total_words, align = "center")

plt.yticks(y_position, sorted_words_number.Type)

plt.title('Total Number of Words To See Who Talks alot', fontsize = 20)



plt.tight_layout()
# Revisiting narcissism. Let's take a look at the most common words

Counter(words).most_common()
# We'll start by reading in the corpus, which preserves word order

import pandas as pd



datas = pd.read_pickle('corpus.pkl')

del datas.index.name

datas
datas = datas.transpose()

datas = datas.groupby(level=0, axis=1).sum()

datas = datas.transpose()

datas
# Let's add the personality names' as well

personality_Names = ['Extraverted iNtuitive Feeling Judging', 'Extraverted iNtuitive Feeling Perceiving', 'Extraverted iNtuitive Thinking Judging', 'Extraverted iNtuitive Thinking Perceiving', 'Extraverted Sensing Feeling Judging', 'Extraverted Sensing Feeling Perceiving',

              'Extraverted Sensing Thinking Judging', 'Extraverted Sensing Thinking Perceiving', 'Introverted iNtuitive Feeling Judging', 'Introverted iNtuitive Feeling Perceiving', 'Introverted iNtuitive Thinking Judging', 'Introverted iNtuitive Thinking Perceiving','Introverted Sensing Feeling Judging','Introverted Sensing Feeling Perceiving','Introverted Sensing Thinking Judging','Introverted Sensing Thinking Perceiving']



datas['personality_Names'] = personality_Names

datas
# Split each chat into 8 parts

import numpy as np

import math



def split_text(text, n=8):

    '''Takes in a string of text and splits into n equal parts, with a default of 8 equal parts.'''



    # Calculate length of text, the size of each chunk of text and the starting points of each chunk of text

    length = len(text)

    size = math.floor(length / n)

    start = np.arange(0, length, size)

    

    # Pull out equally sized pieces of text and put it into a list

    split_list = []

    for piece in range(n):

        split_list.append(text[start[piece]:start[piece]+size])

    return split_list
# Let's take a look at our data again

datas
# Let's read in our document-term matrix

data2 = pd.read_pickle('data_post_stop.pkl')

data2
# Import the necessary modules for LDA with gensim

from gensim import matutils, models

import scipy.sparse
# One of the required inputs is a term-document matrix

termdm = data2.transpose()

termdm = termdm.groupby(level=0, axis=1).sum()

termdm.head()
# We're going to put the term-document matrix into a new gensim format, from df --> sparse matrix --> gensim corpus

sparse_counts = scipy.sparse.csr_matrix(termdm)

corpus = matutils.Sparse2Corpus(sparse_counts)
# Gensim also requires dictionary of the all terms and their respective location in the term-document matrix

import pickle

cv = pickle.load(open("cv_stop.pkl", "rb"))

id2word = dict((v, k) for k, v in cv.vocabulary_.items())
# Now that we have the corpus (term-document matrix) and id2word (dictionary of location: term),

# we need to specify two other parameters as well - the number of topics and the number of passes

lda_topics_passes = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=2, passes=10)

lda_topics_passes.print_topics()
# LDA for num_topics = 3

lda_topics_passes = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=3, passes=10)

lda_topics_passes.print_topics()
# LDA for num_topics = 4

lda_topics_passes = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=4, passes=10)

lda_topics_passes.print_topics()
#In preparation for the algorithms to get models.

#X = pd.read_pickle('xdata.pkl')

X = data_x #if there is a problem run the previosu one



train = pd.read_csv('../input/edsa-mbti/train.csv', encoding='ISO-8859-1')

test = pd.read_csv('../input/edsa-mbti/test.csv', encoding='ISO-8859-1')
#output a csv with results

def output_csv(file_name, predictions):



    '''Converts the 16 class predictions to 4 class outputs and outputs the  predictions as csv. '''

    

    output=pd.DataFrame({'id':test_id.values, 'type':predictions  })

    

    output['mind'] = output['type'].apply(lambda x: 0 if x in mind_map else 1)

    output['energy'] = output['type'].apply(lambda x: 0 if x in energy_map else 1)

    output['nature'] = output['type'].apply(lambda x: 0 if x in nature_map else 1)

    output['tactics'] = output['type'].apply(lambda x: 0 if x in tactics_map else 1)

    

    output.drop(columns = ['type'], inplace = True)

    

    output.set_index('id', inplace = True)

    

    output.to_csv(file_name)
train['type'] = train['type'].map({'INFJ':0, 'ENTP':1, 'INTP':2, 'INTJ':3, 'ENTJ':4, 'ENFJ':5, 'INFP':6, 'ENFP':7,

       'ISFP':8, 'ISTP':9, 'ISFJ':10, 'ISTJ':11, 'ESTP':12, 'ESFP':13, 'ESTJ':14, 'ESFJ':15 })



#mapping the personality type to just 4 main classes

mind_map = [1,4,5,7,12,13,14,15]

energy_map = [0,1,2,3,4,5,6,7]

nature_map = [1,2,3,4,9,11,12,14]

tactics_map = [0,3,4,5,10,11,14,15]



#map the training the data to the 4 classes

train['mind'] = train['type'].apply(lambda x: 1 if x in mind_map else 0)

train['energy'] = train['type'].apply(lambda x: 1 if x in energy_map else 0)

train['nature'] = train['type'].apply(lambda x: 1 if x in nature_map else 0)

train['tactics'] = train['type'].apply(lambda x: 1 if x in tactics_map else 0)
#get the train type column as Y

Y = train['type']



#We preserve the id's in the test set for submission

test_id = test['id']



#we get the number of rows for each set to split the set during preprocessing

ntrain = train.shape[0]

ntest = test.shape[0]



train_final = X[:ntrain]

test_final = X[ntrain:]



X_train, X_test, y_train, y_test = train_test_split( train_final, Y, test_size = 0.2, random_state = 1111) 
#fit the model using train.csv data only just see if a better score isn't achieved compared to validated train_test_split dat

#logReg_All = LogisticRegression()

#logReg_All.fit(train_transformed, Y)
#fit the model using train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import log_loss

logReg = LogisticRegression(random_state = 1111, C=30)

logReg.fit(X_train, y_train)



y_pred_logReg_train = logReg.predict(X_train)

y_pred_logReg_test = logReg.predict(X_test)



y_hat_logReg_train = logReg.predict_proba(X_train)

y_hat_logReg_test = logReg.predict_proba(X_test)



y_pred_logReg_output = logReg.predict(test_final)



#print the train and test log loss:

print("The train log loss error for our model is: ",log_loss(y_train, y_hat_logReg_train))

print("The test log loss error for our model is: ",log_loss(y_test, y_hat_logReg_test))
output_csv('LogisticRegression.csv', y_pred_logReg_output)
#fit the model using train.csv and test.csv only just to see what happens

#multinomialNB_All = MultinomialNB()

#multinomialNB_All.fit(train_final, Y)
#fit the model using train.csv split for validation using train_test_split

multinomialNB = MultinomialNB()

multinomialNB.fit(X_train, y_train)



y_pred_MNB_test = multinomialNB.predict(X_test)

y_pred_MNB_train = multinomialNB.predict(X_train)



y_hat_MNB_train = multinomialNB.predict_proba(X_train)

y_hat_MNB_test = multinomialNB.predict_proba(X_test)



y_pred_MNB_output = multinomialNB.predict(test_final)
#print the train and test log loss error

print("The train log loss error for our model is: ",log_loss(y_train, y_hat_MNB_train))

print("The test log loss error for our model is: ",log_loss(y_test, y_hat_MNB_test))
output_csv('Multinomial Naive Bayes.csv', y_pred_MNB_output)
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)



y_pred_KNN_test = knn.predict(X_test)

y_pred_KNN_train = knn.predict(X_train)



y_hat_KNN_train = knn.predict_proba(X_train)

y_hat_KNN_test = knn.predict_proba(X_test)



y_pred_KNN_output = knn.predict(test_final)
print("The train log loss error for our model is: ",log_loss(y_train, y_hat_KNN_train))

print("The test log loss error for our model is: ",log_loss(y_test, y_hat_KNN_test))
y_pred_knn_output = knn.predict(test_final)

output_csv('KNN Classifier.csv', y_pred_KNN_output)
adaBoost = AdaBoostClassifier(random_state= 1111)

adaBoost.fit(X_train, y_train)
y_pred_adaBoost_train = adaBoost.predict(X_train)

y_pred_adaBoost_test = adaBoost.predict(X_test)



y_hat_adaBoost_train = adaBoost.predict_proba(X_train)

y_hat_adaBoost_test = adaBoost.predict_proba(X_test)
print("The train log loss error for our model is: ",log_loss(y_train, y_hat_adaBoost_train))

print("The test log loss error for our model is: ",log_loss(y_test, y_hat_adaBoost_test))
y_pred_adaBoost_output = adaBoost.predict(test_final)

output_csv('AdaBoost Classifier.csv', y_pred_adaBoost_output)
#svm_model = svm.SVC(gamma='scale')

#svm_model.fit(train_transformed, Y)
svm_model = svm.SVC(gamma='scale', random_state = 1111, probability=True)

svm_model.fit(X_train, y_train)
y_pred_SVM_test = svm_model.predict(X_test)

y_pred_SVM_train = svm_model.predict(X_train)



y_hat_SVM_train = svm_model.predict_proba(X_train)

y_hat_SVM_test = svm_model.predict_proba(X_test)
print("The train log loss error for our model is: ",log_loss(y_train, y_hat_SVM_train))

print("The test log loss error for our model is: ",log_loss(y_test, y_hat_SVM_test))
y_pred_SVM_output = svm_model.predict(test_final)

output_csv('SVM Classifier_2.csv', y_pred_SVM_output)
#model using train_test_split

forest = RandomForestClassifier( n_estimators=50, random_state = 1111 )

forest.fit(X_train, y_train)
y_pred_RF_train = forest.predict(X_train)

y_pred_RF_test = forest.predict(X_test)



y_hat_RF_train = forest.predict_proba(X_train)

y_hat_RF_test = forest.predict_proba(X_test)



print("The train log loss error for our model is: ",log_loss(y_train, y_hat_RF_train))

print("The test log loss error for our model is: ",log_loss(y_test, y_hat_RF_test))
y_pred_RF_output = forest.predict(test_transformed)

output_csv('Random Forest.csv', y_pred_RF_output)