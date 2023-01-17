# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#import numpy as np # linear algebra

#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

   # for filename in filenames:

      #  print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd 

import os

from nltk.corpus import stopwords

import sys

import warnings



if not sys.warnoptions:

    warnings.simplefilter("ignore")

trainData = pd.read_table("train.tsv")



# trainData = trainData.head(50000)

print(trainData.shape)
trainData.info()
print(trainData['Sentiment'].value_counts())

trainData.head()
stop_words = set(stopwords.words('english'))

print(stop_words)

non_stop_words = {"not","isn't","don't"}

# # print(non_stop_words)

stop_words = stop_words - non_stop_words

# print(stop_words)

print(len(stop_words))

trainData.head()
def stopwords(text):

    '''a function for removing the stopword'''

    # removing the stop words and lowercasing the selected words

    text = [word.lower() for word in text.split() if word.lower() not in stop_words]

    # joining the list of words with space separator

    return " ".join(text)
trainData['Phrase'] = trainData['Phrase'].apply(stopwords)

trainData.head()
import string

def remove_punctuation(text):

    '''a function for removing punctuation'''    

    # replacing the punctuations with no space, 

    # which in effect deletes the punctuation marks 

    translator = str.maketrans('', '', string.punctuation)

    # return the text stripped of punctuation marks

    return text.translate(translator)



trainData['Phrase'] = trainData['Phrase'].apply(remove_punctuation)

trainData.head()
trainData.shape
trainData = trainData.drop_duplicates(subset = ['Phrase'])

print(trainData.shape)

trainData.head()
trainData.info()
print(trainData['Sentiment'].value_counts())
from sklearn.feature_extraction.text import CountVectorizer



countVectorizer = CountVectorizer()

countVectorizer.fit(trainData["Phrase"])

# collect the vocabulary items used in the vectorizer

dictionary = countVectorizer.vocabulary_.items()

count = []

vocab = []

# iterate through each vocab and count append the value to designated lists

for key, value in dictionary:

    vocab.append(key)

    count.append(value)

# store the count in panadas dataframe with vocab as index

vocab_bef_stem = pd.Series(count, index=vocab)

# sort the dataframe

vocab_bef_stem = vocab_bef_stem.sort_values(ascending=False)
from nltk.stem.snowball import SnowballStemmer

# create an object of stemming function

stemmer = SnowballStemmer("english")



def stemming(text):    

    '''a function which stems each word in the given text'''

    text = [stemmer.stem(word) for word in text.split()]

    return " ".join(text)



trainData["Phrase"] = trainData["Phrase"].apply(stemming)
def length(text):    

    '''a function which returns the length of text'''

    return len(text)



trainData['length'] = trainData['Phrase'].apply(length)

trainData.head(10)
#df = df.drop(df[df.score < 50].index)

trainData = trainData.drop(trainData[trainData.length == 0].index)

trainData.head(10)
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier



def applyKNN(new_trainData):

    new_trainData = dropNaNSentiments(new_trainData)

    Y = new_trainData['Y']

    X = new_trainData.drop(['Y'], axis = 1)

    print(Y.shape)

    print(X.shape)

    X["PhraseId"].fillna( method ='ffill', inplace = True)

    X["SeneteceId"].fillna( method ='ffill', inplace = True)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    neighbor = KNeighborsClassifier(n_neighbors = 5,algorithm = 'ball_tree')

    neighbor.fit(X_train,y_train)

    print(neighbor.score(X_test,y_test))
def appendPharseIDY(phraseDataframe):

    phraseDataframe['PhraseId'] = trainData['PhraseId']

    phraseDataframe['SeneteceId'] = trainData['SentenceId']

    phraseDataframe['Y'] = trainData['Sentiment']

    return phraseDataframe

# Drop rows with NaN entities in Sentiment.

def dropNaNSentiments(inputDataFrame):

    return inputDataFrame.dropna(subset = ['Y'])



vectorizer = CountVectorizer()

X = vectorizer.fit_transform(trainData['Phrase'])

X_D = X.todense()



new_trainData = pd.DataFrame(X_D)

new_trainData = appendPharseIDY(new_trainData)

print(new_trainData.shape)





print("Apply K-NN On Bag of Words:")

applyKNN(new_trainData)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(trainData['Phrase'])

X_D = X.todense()

new_trainData = pd.DataFrame(X_D)

new_trainData = appendPharseIDY(new_trainData)

print(new_trainData.shape)
print("Apply K-NN On TF-IDF:")

applyKNN(new_trainData)
from gensim.models import Word2Vec
model = Word2Vec(trainData['Phrase'], min_count = 1)
X = model[model.wv.vocab]

new_trainData = pd.DataFrame(X_D)

new_trainData = appendPharseIDY(new_trainData)

print(new_trainData.shape)
print("Apply K-NN On Word Embeddings:")

applyKNN(new_trainData)