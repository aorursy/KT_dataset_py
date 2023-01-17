import pandas as pd

import re

import nltk



from collections import Counter

import itertools 



import random



import sklearn
path = "../input/imdb.csv"

df = pd.read_csv(path)
df[(df['id'] == 'id_028642')].head()
def remove_tag(x):

    res = re.findall("<s ([a-zA-Z\s]*) \/s>", x)

    return res[0] if len(res) > 0 else None



def get_n_grams(x, n_gram):

    x = remove_tag(x)

    if x is None: return None

    x = x.split()

    if n_gram is 'unigrams':

        return x

    elif n_gram is 'bigrams':

        return [i for i in nltk.bigrams(x)]

    elif n_gram is 'trigrams':

        return [i for i in nltk.trigrams(x)]

    else: return None
df['unigrams'] = df['sentence_clean'].apply(lambda x: get_n_grams(x, 'unigrams'))

df['bigrams'] = df['sentence_clean'].apply(lambda x: get_n_grams(x, 'bigrams'))

df['trigrams'] = df['sentence_clean'].apply(lambda x: get_n_grams(x, 'trigrams'))
df[(df['id'] == 'id_028642')].head()
df = df.dropna()#remove incomplete rows from our dataset

print('Null Values per Feature')

print(df.isnull().sum())#show that there is no longer any missing data
df[(df['id'] == 'id_028642')].head()
positive = df.loc[df.pol == 1] #take all of the positive polarities and store in a new dataframe

negative = df.loc[df.pol == 0] #take all of the negative polarities and store in a new dataframe
positive.head()
negative.head()
#create a dataframe of positive flavored unigrams

posUni = pd.DataFrame([(k,v) for (k, v) in Counter(list(itertools.chain(*positive['unigrams']))).items()], columns = ['Word', "Count"])

posUni.nlargest(10, 'Count')
posBi = pd.DataFrame([(k,v) for (k, v) in Counter(list(itertools.chain(*positive['bigrams']))).items()], columns = ['Bigrams', "Count"])

posBi.nlargest(10, 'Count')
#create a dataframe of positive flavored unigrams

negUni = pd.DataFrame([(k,v) for (k, v) in Counter(list(itertools.chain(*negative['unigrams']))).items()], columns = ['Word', "Count"])

negUni.nlargest(10, 'Count')
negBi = pd.DataFrame([(k,v) for (k, v) in Counter(list(itertools.chain(*negative['bigrams']))).items()], columns = ['Bigrams', "Count"])

negBi.nlargest(10, 'Count')
text = "This movie is " #place holder text that starts our sentence

i = 0

while(i < 17):

    text = text + str(posUni['Word'][random.randint(0,len(posUni['Word']))]) + " " #grab 1 unigram and append it per round

    i = i + 1

    

text
text = "This movie is " #place holder text that starts our sentence

i = 0

while(i < 12):

    text = text + str(negUni['Word'][random.randint(0,len(negUni['Word']))]) + " " #grab 1 unigram and append it per round

    i = i + 1

    

text
text = "This movie is " #place holder text that starts our sentence

i = 0

while(i < 9):

    text = text + str(posBi['Bigrams'][random.randint(0,len(posBi['Bigrams']))]) + " " #grab 1 unigram and append it per round

    i = i + 1

    

text
text = "This movie is " #place holder text that starts our sentence

i = 0

while(i < 6):

    text = text + str(negBi['Bigrams'][random.randint(0,len(negBi['Bigrams']))]) + " " #grab 1 unigram and append it per round

    i = i + 1

    

text
import numpy as np

from sklearn.utils import shuffle



#shuffle our initial dataframe's indexes

df = shuffle(df)



#split test and train data 75 25 respectively

splitter = np.random.rand(len(df)) < 0.75 #cut data 75% to 25%

test = df[~splitter]

train = df[splitter]



print('Test Dimensions')

print(test.shape)

print('Test Preview')

print(test.head())



print('Train Dimensions')

print(train.shape)

print('Train Preview')

print(train.head())
from sklearn.feature_extraction.text import TfidfVectorizer #to use methods shown in class

from nltk.corpus import stopwords #to specify our stop words



#search for the 100 most common uni and bigrams while removing stop words

cols = TfidfVectorizer(max_features=100, min_df=5, max_df=0.5, ngram_range = (1,2), stop_words = stopwords.words('english'))  

features = cols.fit_transform(train['sentence_clean']).toarray()#use our cleaned test data for our features



#create a dataframe with our features (terms) and their respective probability

tfidf = pd.DataFrame(

    features,

    columns = cols.get_feature_names())



print(tfidf)
# Createand train Decision Tree classifer object

tree = DecisionTreeClassifier()

tree = clf.fit(tfidf, train['pol'])



#Predict the response for test dataset

validation = tree.predict(test)



display(validation)
display(confusion_matrix(test['pol'],validation))  

print(classification_report(test['pol'],validation))  

print(accuracy_score(test['pol'], validation))