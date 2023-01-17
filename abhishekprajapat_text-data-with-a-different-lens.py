import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

from sklearn.model_selection import StratifiedKFold

from tqdm.notebook import tqdm

from IPython.display import YouTubeVideo

tqdm.pandas()
df = pd.read_csv('../input/covid-19-nlp-text-classification/Corona_NLP_train.csv', encoding='latin8')

df.head()
def create_folds(X,y):

    

    df['kfold'] = -1

    

    splitter = StratifiedKFold(n_splits=5)

    

    for f, (t_, v_) in enumerate(splitter.split(X, y)):

        

        X.loc[v_, 'kfold'] = f

        

    return X
df = create_folds(df, df['Sentiment'])

df.head()
df = df[['OriginalTweet', 'Sentiment', 'kfold']]

df.head(2)
from tensorflow.keras.preprocessing.text import Tokenizer

from nltk.tokenize import TweetTokenizer, word_tokenize
sentences = df['OriginalTweet'][:5]
for i in sentences[2:3]:

    print("Original:\n")

    print(i)

    print('\nTensorflow Tokenizer\n:')

    a = Tokenizer()

    a.fit_on_texts([i])

    print(a.word_index)

    print("\nTweet Tokenizer:\n")

    print(TweetTokenizer().tokenize(i))

    print('\nNLTK word_tokenizer:\n')

    print(word_tokenize(i))
tweets = []



for i in tqdm(df['OriginalTweet']):

    

    tweet = TweetTokenizer().tokenize(i)

    tweet = ' '.join(tweet)

    tweets.append(tweet)
for i in tweets[:3]:

    print(i, '\n')
from gensim.models import KeyedVectors

from gensim import downloader



embedding_file = '../input/embeddings/GoogleNews-vectors-negative-300d.bin'



embedding_model =  KeyedVectors.load_word2vec_format(embedding_file, binary=True)
def build_vocab(sentences, verbose =  True):

    """

    :param sentences: list of list of words

    :return: dictionary of words and their count

    """

    vocab = {}

    for sentence in tqdm(sentences, disable = (not verbose)):

        for word in sentence:

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1

    return vocab
vocab = build_vocab([tweet.split() for tweet in tweets])

print({k: vocab[k] for k in list(vocab)[:5]})
import operator 



def check_coverage(vocab,embeddings_index):

    a = {}

    oov = {}

    k = 0

    i = 0

    for word in tqdm(vocab):

        try:

            a[word] = embeddings_index[word]

            k += vocab[word]

        except:



            oov[word] = vocab[word]

            i += vocab[word]

            pass



    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))

    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))

    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]



    return sorted_x
oov = check_coverage(vocab,embedding_model)
oov[:20]
def clean_text(x):



    x = str(x)

    for punct in "/-'":

        x = x.replace(punct, ' ')

    for punct in '&':

        x = x.replace(punct, f' {punct} ')

    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':

        x = x.replace(punct, '')

    return x
for index, tweet in enumerate(tweets):

    

    tweets[index] = clean_text(tweet)





vocab = build_vocab([tweet.split() for tweet in tweets])
oov = check_coverage(vocab,embedding_model)
oov[:20]
"crisis" in embedding_model
"distancing" in embedding_model
len(oov)
count = 0

index = 0



while((count != 30) and count < len(oov)):

    

    if len(oov[index][0]) > 3:

        print(oov[index])

        count += 1

        

    index += 1
to_replace = [('COVID', 'health crisis'),

            ('COVID19', 'health crisis'),

            ('Covid19', 'health crisis'),

            ('Covid', 'health crisis'),

            ('COVID2019', 'health crisis'),

            ('covid19', 'health crisis'),

            ('toiletpaper', 'toilet paper'),

            ('covid', 'health crisis'),

            ('CoronaCrisis', 'health crisis'),

            ('CoronaVirus', 'health crisis'),

            ('SocialDistancing', 'Social distancing'),

            ('2020', 'this year'),

            ('CoronavirusPandemic', 'health crisis'),

            ('CoronavirusOutbreak', 'health crisis'),

            ('StayHomeSaveLives', 'Stay Home Save Lives'),

            ('StayAtHome', 'Stay At Home'),

            ('StayHome', 'Stay Home'),

            ('panicbuying', 'Panic Buying'),

            ('socialdistancing', 'Social Distancing'),

            ('CoronaVirusUpdate', 'health crisis update'),

            ('StopHoarding', 'Stop Hoarding'),

            ('realDonaldTrump', 'real Donald Trump'),

            ('StopPanicBuying', 'Stop Panic Buying'),

            ('covid19UK', 'health crisis'),

            ('QuarantineLife', 'Quarantine life'),

            ('behaviour', 'behave')]
to_replace_dict = {}



for i in to_replace:

    

    to_replace_dict[i[0]] = i[1]
for index, tweet in tqdm(enumerate(tweets)):

    

    cleaned_tweet = []

    

    for word in tweet.split():

        

        if len(word) > 2:

            

            if word in to_replace_dict:              

                cleaned_tweet.append(to_replace_dict[word])

            else:

                cleaned_tweet.append(word)

                

    tweets[index] = ' '.join(cleaned_tweet)
vocab = build_vocab([tweet.split() for tweet in tweets])
oov = check_coverage(vocab,embedding_model)
count = 0

index = 0



while((count != 30) and count < len(oov)):

    

    if len(oov[index][0]) > 3:

        print(oov[index])

        count += 1

        

    index += 1
to_check = ['fuck', 'motherfucker', ':)', ":{", 'bastard', ':(']



for i in to_check:

    if i in embedding_model:

        print('yes')

    else:

        print('no')
TweetTokenizer().tokenize('This word has a :) face')
from nltk.stem import SnowballStemmer, WordNetLemmatizer



word = 'elegant'

stem_word = SnowballStemmer('english').stem(word)

lemma = WordNetLemmatizer().lemmatize(word)



print("Stem word: ", stem_word)

print("\nLemma: ", lemma)



print("\nIs stemmed word present in embedding :", stem_word in embedding_model)

print("\nIs lemma present in embedding :", lemma in embedding_model)
word1 = 'feet'

word2 = 'foot'



print(WordNetLemmatizer().lemmatize(word1))



print(word1 in embedding_model)

print(word2 in embedding_model)
!pip install nlpaug
import nlpaug.augmenter.word as naw
sent = 'All month there hasn been crowding the supermarkets restaurants however reducing all the hours and closing the malls means everyone now using the same entrance and dependent single supermarket manila lockdown covid2019 Philippines https tco HxWs9LAnF9'

print('original: ', sent)

print('\nAugmented: ', naw.SynonymAug(aug_src='wordnet').augment(sent))
YouTubeVideo('BBR3J2HI5xI')
YouTubeVideo('VpLAjOQHaLU')
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
'not' in stop_words
words = ['break the rules', 'free time', 'draw a conclusion', 'keep in mind', 'get ready']



for i in words:

    

    print(i in embedding_model)
from fuzzywuzzy import fuzz

from fuzzywuzzy import process
fuzz.ratio("this is a test", "this is a test!")
fuzz.partial_ratio("this is a test", "this is a test!")
fuzz.token_sort_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
fuzz.token_set_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
fuzz.token_sort_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
fuzz.token_set_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")