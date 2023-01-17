# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#!conda remove -y greenlet

#!pip uninstall allnlp
tf.__version__
import matplotlib.pyplot as plt

from statsmodels.sandbox.regression.predstd import wls_prediction_std

import sklearn

from sklearn import metrics

import sklearn.metrics

from sklearn.metrics import *



from sklearn import feature_selection

from sklearn.model_selection import train_test_split

from sklearn import preprocessing



from sklearn.preprocessing import KBinsDiscretizer



import scipy.stats as stats



import statsmodels.api as sm

import math



import seaborn as sns

import tensorflow as tf

from tensorflow.keras.utils import to_categorical

from statsmodels.stats.outliers_influence import variance_inflation_factor



from sklearn.preprocessing import LabelEncoder

import tensorflow_hub as hub

import re

import nltk

from sklearn.feature_extraction.text import CountVectorizer
# Filtering and Tokening Data text

#HappyEmoticons

emoticons_happy = list([

    (':-)', 'happy'), (':)', 'happy'), (';)', 'wink happy'), (';D', 'wink very happy'),(':o)', 'excited'), (':]', 'okay'), (':3', 'kiss'), (':c)', 'happy'), (':>', 'happy'), ('=]', 'happy'), ('8)', 'happy'), ('=)', 'happy'), (':}', 'happy'),

    (':^)', 'very happy'), (':-D', 'very happy'), (':D', 'very happy'), ('8-D', 'very happy'), ('8D', 'very happy'), ('x-D', 'very happy'), ('xD', 'very happy'), ('X-D', 'very happy'), ('XD', 'very happy'), ('=-D', 'very happy'), ('=D', 'very happy'),

    ('=-3', 'nice good'), ('=3', 'very happy'), (':-))', 'very happy'), (":'-)", 'very happy'), (":')",'very happy'), (':*', 'love you'),( ':^*', 'love you'), ('>:P', 'nice good'), (':-P', 'nice good'), (':P', 'nice good'), ('X-P', 'nice good'),

    ('x-p', 'happy'), ('xp', 'happy'), ('XP', 'happy'), (':-p', 'happy'), (':p', 'happy'), ('=p', 'happy'), (':-b', 'happy'), (':b', 'happy'), ('>:)', 'happy'), ('>;)', 'happy'), ('>:-)', 'happy'),

    ('<3', 'happy')

    ])



# Sad Emoticons

emoticons_sad = list([

    (':L', 'frustrated'), (':-/', 'upset'), ('>:/', 'frustrated'), (':S','angry'), ('>:[', 'frustrated'), (':@', 'angry'), (':-(', 'upset'), (':[', 'upset'), (':-||', 'astonished'), ('=L', 'angry'), (':<', 'upset'),

    (':-[', 'sad'), (':-<', 'upset'), ('=\\', 'confused'), ('=/', 'confused'), ('>:(', 'very sad'), (':(', 'sad'), ('>.<', 'frustrated'), (":'-(", 'very sad'), (":'(", 'very sad'), (':\\', 'confused'), (':-c', 'very upset'),

    (':c', 'sad'), (':{', 'very sad'), ('>:\\', 'frustrated'), (';(', 'frustrated')

    ])



emoticons_happy_set = set([i[0] for i in emoticons_happy])

emoticons_sad_set = set([i[0] for i in emoticons_sad])



allEmojis = set(emoticons_happy + emoticons_sad)



emojiDict = dict()

for i in allEmojis:

    emojiDict[i[0]] = i[1]

    



shortForms = dict()

shortForms[' s '] = ' is '

shortForms[' ve '] = ' have '

shortForms[' nt '] = ' not '

shortForms[' d '] = ' did '

shortForms[' ll '] = ' all '

shortForms[' m '] = ' am '





def multiple_replace(adict, text):

  # Create a regular expression from all of the dictionary keys

  regex = re.compile("|".join(map(re.escape, adict.keys(  ))))

  # For each match, look up the corresponding value in the dictionary

  return regex.sub(lambda match: adict[match.group(0)], text)



#Emoji patterns

emoji_pattern = re.compile("["

         u"\U0001F600-\U0001F64F"  # emoticons

         u"\U0001F300-\U0001F5FF"  # symbols & pictographs

         u"\U0001F680-\U0001F6FF"  # transport & map symbols

         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

         u"\U00002702-\U000027B0"

         u"\U000024C2-\U0001F251"

         "]+", flags=re.UNICODE)



#combine sad and happy emoticons

emoticons = emoticons_happy_set.union(emoticons_sad)



reFiltersDict = {"\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*|\w+\.com":"url", "#[a-z0-9]+":"hashtag", "@[a-z0-9]+":"person"}

reFiltersKeys = [re.compile(i) for i in reFiltersDict.keys()]



reFilters = re.compile("|".join(reFiltersDict.keys()))



def re_match(match):

    for i in reFiltersKeys:

        if re.search(i, match.string) is not None:

            return reFiltersDict[i.pattern]



def multiple_replace_re(text):

  # Create a regular expression from all of the dictionary keys

  # For each match, look up the corresponding value in the dictionary

    return reFilters.sub(re_match, text)



def tokenizer(data):

    badWords = ['cPs', 'D', 'P', 'p', 'o', 'O']

    #inputSet = [[x for x in i['reviewText'].replace(',',' ').split(' ')] for i in dataset[:1]]

    happy = dict()

    sad = dict()

    tmpCorpus = [set(i.split(' ')) for i in data]

    happyIntersecs = [emoticons_happy_set.intersection(i) for i in tmpCorpus]

    sadIntersecs = [emoticons_sad_set.intersection(i) for i in tmpCorpus]

    #emojis = [i.union(j) for i,j in zip(happyIntersecs, sadIntersecs)]

    

    #fixedEmoji = list()

    for i in range(0, len(data)):

        for j in happyIntersecs[i]:

            data[i] = data[i].replace(j, emojiDict[j])

        for j in sadIntersecs[i]:

            data[i] = data[i].replace(j, emojiDict[j])

        #fixedEmoji

    

    #inputSet = [re.sub("@[A-Za-z0-9]+", "person", i) for i in data]

    #inputSet = [re.sub("#[A-Za-z0-9]+", "hashtag", re.sub("@[A-Za-z0-9]+", "person", i)) for i in inputSet]

    #inputSet = [re.sub("\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*|\w+\.com", "url", re.sub("#[A-Za-z0-9]+", "hashtag", re.sub("@[A-Za-z0-9]+", "person", i))) for i in data]

    inputSet = [multiple_replace_re(i.lower()) for i in data]

    inputSet = [[(x) for x in re.split('\W+', i) if x != ''] for i in inputSet]

    for i in inputSet:

        for j in range(0,len(i)):

            #if re.match('^\d{4}$', i[j]):

                #print(i[j])

            #    i[j] = 'year'

            if re.search('[0-9]', i[j]):

                if re.search('[a-zA-Z]', i[j]):

                    i[j] = 'item'

                else:

                    i[j] = 'number'

            elif i[j] in badWords:

                i[j] = 'AXTREMOVE'

    inputSet = [re.sub('AXTREMOVE', '', '-'.join(i)).split('-') for i in inputSet]

    data = inputSet

    

    return data, happyIntersecs, sadIntersecs



def dataFilter(data):

    data = [multiple_replace(shortForms, ' '.join(i)) for i in tokenizer(data)]  

    #data = [multiple_replace(emojiDict, i) for i in data]

    return data





def dataStitch(data):

    data = [' '.join(i) for i in data]

    data = [multiple_replace(shortForms, i) for i in data]  

    #data = [multiple_replace(emojiDict, i) for i in data]

    return data
print("\nLoading embedding layer...")

embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")

#hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)



print("Fetched the hub module...")
df = pd.read_csv("../input/train.csv")

df = df.dropna()

df.head()

independent = [str(i) for i in df['text'].values]#df.drop([''], axis=1)

target = df['label'].values

target = [int(i) for i in target]

df.head()
#y = LabelEncoder().fit_transform(target)

import pickle

x = tokenizer(independent)#pickle.loads(open('proceesedText.pkl', 'rb').read())#dataFilter(independent)

y = tf.keras.utils.to_categorical(target)

#x = np.load('Text.npy')

#y = np.load('Labels.npy')

#sns.distplot(y)

plt.show()
dd = pd.DataFrame()

dd['text'] = dataStitch(x[0])

dd['happy'] = [len(i) for i in x[1]]

dd['sad'] = [len(i) for i in x[2]]
dd.to_csv('processedData.csv', header=True)
dd.iloc[0]['text']
dd['text'][0]
tokenizer(['i recommend this'])
X = dataStitch(x[0])
dd[dd['sad'] >= 1].shape
df.iloc[14]['text']
import pickle

f = open('../input/ut/proceesedText.pkl', 'wb')

f.write(pickle.dumps(x))