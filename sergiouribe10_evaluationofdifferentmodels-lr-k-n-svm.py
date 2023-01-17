# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import seaborn as sns
train_data = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

print("Data shape = ", train_data.shape)

train_data.head()

test_data = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

test_data
# Missing data

total = train_data.isnull().sum()

percentage = (train_data.isnull().sum()/train_data.isnull().count()*100)

missing_data = pd.concat([total, percentage], axis=1, keys = ['Total', 'Percentage'])

missing_data
train_data = train_data.drop(columns = ['id', 'location', 'keyword'])
import re

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer



def clean_data (data):

    corpus = []

    pstem = PorterStemmer()



    for i in range(data.shape[0]):

        #Remove unwanted words

        tweet = re.sub("[^a-zA-Z]", ' ', data['text'][i])



        #Lower case

        tweet = tweet.lower()

        tweet = tweet.split()

    

        #Remove stop words and steeming words(take the roots)

        tweet = [pstem.stem(word) for word in tweet if not word in set (stopwords.words('english'))]

        tweet = ' '.join(tweet)

    

        #Append clean tweet to corpus

        corpus.append(tweet)

    return corpus
# To reduce bag of words dimensionality, we should remove those words that are

# repeated very few times. So, we create a dictionary where key refer to word 

# and value refer to word frequents in all tweets







def bagOfWords(data, data_corpus):

    uniqueWordFrequents = {}



    for tweet in data_corpus: 

        for word in tweet.split():

            if(word in uniqueWordFrequents.keys()):

                uniqueWordFrequents[word] +=1

            else:

                uniqueWordFrequents[word] = 1



    # Convert dictionary to dataFrame

    uniqueWordFrequents = pd.DataFrame.from_dict(uniqueWordFrequents, orient = 'index', columns = ['Word Frequent'])

    #uniqueWordFrequents.sort_values(by=['Word Frequent'], inplace = True, ascending=False)

    # We take only words repeated more than 10 times

    uniqueWordFrequents = uniqueWordFrequents[uniqueWordFrequents['Word Frequent'] >= 20]

    

    # Create Bag of words --> they contain only unique words in corpus



    from sklearn.feature_extraction.text import CountVectorizer



    counVec = CountVectorizer(max_features = uniqueWordFrequents.shape[0])



    bagWords = counVec.fit_transform(data_corpus).toarray()

    return bagWords
train_data_corpus = clean_data(train_data)

train_data_bagWords = bagOfWords(train_data, train_data_corpus)

print(train_data_bagWords.shape)

train_data_bagWords

X = train_data_bagWords

y = train_data ['target']



print('X shape: ', X.shape)

print('y shape: ', y.shape)

X
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=55, shuffle=True)



X_train

print('X_train shape: ', X_train.shape)

print('y_train shape: ', y_train.shape)
from sklearn.tree import DecisionTreeClassifier



decisionTreeModel = DecisionTreeClassifier (criterion = 'entropy', max_depth=None, splitter ='best', random_state = 55)

decisionTreeModel.fit(X_train, y_train)
#Error: Number of features of the model must match the input. Model n_features is 1410 and input n_features is 700 



#You are supposed to pass numpy arrays and not lists as arguments to the DecisionTree, since your input was a list it gets trained as 70 features (1D list) and your test had list of 30 elements and the classifier sees it as 30 features.

#Nonetheless, you need to reshape your input numpy array and pass it as a matrix

#meaning: X_train.values.reshape(-1, 1) instead of X_train (it should be a numpy array not a list)

#c.fit(X_train.values.reshape(-1, 1), y_train)



#test_data_corpus = clean_data(test_data)



#test_data_bagWords = bagOfWords(test_data, test_data_corpus)

# PROBLEMS HERE.. THE MODEL HAS BEEN TRAINED WITH A DIFFERENT NUMBER OF FEATURES

# FROM THE TEST DATA

# I DON'T HAVE TIME TO SOLVE IT BUT THE ERROR I THINK IS SOMEWHERE IN THE BAG OF WORDS





#y_pred_test_data = decisionTreeModel.predict(test_data_bagWords)
#test_data = test_data.drop(columns=['keyword', 'location', 'text'])

#test_data
#test_data['target'] = y_pred_test_data
from sklearn.linear_model import LogisticRegression



LogisticRegression = LogisticRegression(penalty='l2',solver='saga', random_state = 55)

LogisticRegression.fit(X_train, y_train)
from sklearn.linear_model import SGDClassifier





SGDClassifier = SGDClassifier(loss = 'hinge', 

                              penalty = 'l1',

                              learning_rate = 'optimal',

                              random_state = 55, 

                              max_iter=100)



SGDClassifier.fit(X_train,y_train)
from sklearn.svm import SVC



SVClassifier = SVC(kernel= 'linear',

                   degree=3,

                   max_iter=10000,

                   C=2, 

                   random_state = 55)



SVClassifier.fit(X_train,y_train)
from sklearn.naive_bayes import GaussianNB



gaussianNBModel = GaussianNB()

gaussianNBModel.fit(X_train,y_train)
from sklearn.naive_bayes import MultinomialNB



multinomialNBModel = MultinomialNB(alpha=0.1)

multinomialNBModel.fit(X_train,y_train)
from sklearn.naive_bayes import BernoulliNB



bernoulliNBModel = BernoulliNB(alpha=0.1)

bernoulliNBModel.fit(X_train,y_train)
from sklearn.neighbors import KNeighborsClassifier



KNeighborsModel = KNeighborsClassifier(n_neighbors = 7,

                                       weights = 'distance',

                                      algorithm = 'brute')



KNeighborsModel.fit(X_train,y_train)
from sklearn.ensemble import GradientBoostingClassifier



gradientBoostingModel = GradientBoostingClassifier(loss = 'deviance',

                                                   learning_rate = 0.01,

                                                   n_estimators = 100,

                                                   max_depth = 30,

                                                   random_state=55)



gradientBoostingModel.fit(X_train,y_train)
from sklearn.ensemble import VotingClassifier



modelsNames = [('LogisticRegression',LogisticRegression),

               ('SGDClassifier',SGDClassifier),

               ('SVClassifier',SVClassifier),

               ('bernoulliNBModel',bernoulliNBModel),

               ('multinomialNBModel',multinomialNBModel)]



votingClassifier = VotingClassifier(voting = 'hard',estimators= modelsNames)

votingClassifier.fit(X_train,y_train)
from sklearn.metrics import f1_score

models = [decisionTreeModel, gradientBoostingModel, KNeighborsModel, LogisticRegression, 

          SGDClassifier, SVClassifier, bernoulliNBModel, gaussianNBModel, multinomialNBModel, votingClassifier]



for model in models:

    print(type(model).__name__,' Train Score is   : ' ,model.score(X_train, y_train))

    print(type(model).__name__,' Test Score is    : ' ,model.score(X_test, y_test))

    

    y_pred = model.predict(X_test)

    print(type(model).__name__,' F1 Score is      : ' ,f1_score(y_test,y_pred))

    print('--------------------------------------------------------------------------')