# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier #loading random forest classifier library

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Load the data
trainjson = pd.read_json('../input/train.json', orient='columns')
#testjson = pd.read_json('../input/test.json', orient='columns')
#test_data = np.array(testjson)
train_data = np.array(trainjson)
#Train data
train_data_X = train_data[:3000,6] # request_text 
train_data_Y = train_data[:3000,22] #requester_received_pizza
        
print(len(train_data_X)) #3000
print(len(train_data_Y)) #3000


#Test data
test_data_X = train_data[3000:,6] # request_text 
test_data_Y = train_data[3000:,22] #requester_received_pizza

print(len(test_data_X)) #3000
print(len(test_data_Y)) #3000
#for i in test_data_Y:
#    test_data_Y = np.where(i is False,0,1)

i=train_data_Y
train_data_Y=np.where(i==True, 1, 0)

a=test_data_Y
test_data_Y=np.where(a==True, 1, 0)
import string
from nltk.corpus import stopwords
from nltk import PorterStemmer as Stemmer

#Preparation for futher work

#FOR TRAIN DATA
#make new array, for all processed trained data
array_for_train = []
new_train_data_X = ""

for s in train_data_X:
    #first, we need to remove punctuation from sentences
    exclude = set(string.punctuation)
    new_train_data_X = ''.join(ch for ch in s if ch not in exclude)
    #add to array in lower case, to make words same, for example, Pizza == pizza
    array_for_train.append(new_train_data_X.lower())


#FOR TEST DATA
array_for_test = []
new_test_data_X = ""

for ss in test_data_X:
    #first, we need to remove punctuation from sentences
    exclude = set(string.punctuation)
    new_test_data_X = ''.join(ch for ch in ss if ch not in exclude)
    #add to array in lower case, to make words same, for example, Pizza == pizza
    array_for_test.append(new_test_data_X.lower())
#Test
print(array_for_test[2])
#https://www.tutorialspoint.com/python/python_stemming_and_lemmatization.htm
#stemming, lemmatization 
from nltk.tokenize import sent_tokenize, word_tokenize
#from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

#porter=PorterStemmer()

# Snowball streamer - this algorithm was created by Martin Portrer.
# This algorithm consistently applies a series of rules of the English language -  cuts off the endings and suffixes
# It means, that plays, played and playing are the same words from word play
porter = SnowballStemmer("english", ignore_stopwords=False) #language that we are looking for is engish

def stemSentence(sentence):
    token_words=word_tokenize(sentence) # splitting sentecne for pieces
    stem_sentence=[]
    stopWords = set(stopwords.words('english')) #establish stopwords algorithm. Stop words -  words that are not necessary in sentence or don´t give any meaning
    #stop words can be articles (a, the), prepositions (in, on..) and etc
    for word in token_words:
        if word not in stopWords: #throwing away these stop words
            stem_sentence.append(porter.stem(word))
            stem_sentence.append(" ")
    return "".join(stem_sentence)


#For training data
new_array_for_train = []
for w in array_for_train:
    new_array_for_train.append(stemSentence(w))

#For testing data
new_array_for_test = []
for ww in array_for_test:
    new_array_for_test.append(stemSentence(ww))
    
#Test
print(array_for_train[1])
print("  After Stemming    ")    
print(new_array_for_train[1])
print(len(new_array_for_train))
print(len(new_array_for_test))
#print(type(new_array_for_train))
new_array_for_train = np.array(new_array_for_train)
new_array_for_test = np.array(new_array_for_test)
new_array_for_test[2]
from sklearn.feature_extraction.text import CountVectorizer
# Input
#new_array_for_train
#new_array_for_test
# Output - true/false
# train_data_Y
# test_data_Y

#print(np.array(new_array_for_train)[1])
#print(type(new_array_for_train))

#Initialize a CountVectorizer object
vect = CountVectorizer()
#Transform the training data
c_train = vect.fit_transform(new_array_for_train).toarray()

#Transform the test data
c_test = vect.transform(new_array_for_test).toarray()


# Printing first 20 features
# print(vect.get_feature_names()[:20])

print(type(c_train))
c_test.shape
c_train.shape
#creating  and printing Random Forest classifier
randforclass = RandomForestClassifier(n_estimators=10, n_jobs=2, random_state = 0)
randforclass.fit(c_train, train_data_Y)
#predict probability on test
randforclass.predict_proba(c_test)[0:20]
from sklearn import metrics # delete after uniting the code
randforpred = randforclass.predict(c_test) # random forest predict

# Calculate the accuracy score of random forest
score = metrics.accuracy_score(test_data_Y, randforpred)
print(score * 100 )

# Calculate the confusion of random forestmatrix
cm = metrics.confusion_matrix(test_data_Y, randforpred)
print(cm)