# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import confusion_matrix

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        os.path.join(dirname, filename)

       # print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Reading the datasets from excel

data_senti2 = pd.read_excel('../input/twitter-senti-data/senti2.xlsx')

data_senti1 = pd.read_excel('../input/twitter-senti-data/sentim.xlsx')

data_senti2 = data_senti2.dropna()

data_senti1 = data_senti1.dropna()

data_senti1['Sentiment'].value_counts()
to_drop = ['Neutral', 'neutral','Sentiment','neuta','  neutral']

data_senti1 = data_senti1[~data_senti1['Sentiment'].isin(to_drop)]

data_senti1['Sentiment'].value_counts()
data_senti1.head()
data_senti2.head()
tweets_1 = data_senti1.iloc[:,[0]].values

sentiments_1 =data_senti1.iloc[:,[1]].values

tweets_2 = data_senti2.iloc[:,[0]].values

sentiments_2 =data_senti2.iloc[:,[1]].values



print(len(tweets_1) , len( tweets_2) , len(sentiments_1) , len(sentiments_2) )
tweets = np.concatenate((tweets_1 , tweets_2))

score = np.concatenate((sentiments_1 , sentiments_2))

print(len(tweets) , len(score))

score = score.astype('str')

score = np.char.lower(score)
# Replacing neg and pos with 0 & 1

error = []

def cov_pos_nev(data):

    #print(data[2970])

    for i in range(0,len(data)):

        if data[i] == 'positive':

            data[i]=1

        if data[i] == 'negative':

            data[i]=0

        if data[i] == 'positve':

            data[i]=1

        else:

            #error.append('error at: ' + i )

            pass



cov_pos_nev(score)



print(score)
#converting to dataframe 

df_tweets = pd.DataFrame({'reviews':tweets[:, 0]})

df_score = pd.DataFrame({'score':score[:, 0]})

twitter_data = pd.concat([df_tweets, df_score], axis=1, sort=False)

twitter_data.head()
# Reading test files from tar.gz and arranging them

import os             

test_pos = os.listdir("../input/imdb-data/aclimdb_v1/aclImdb/test/pos/")  

test_neg = os.listdir("../input/imdb-data/aclimdb_v1/aclImdb/test/neg/")

train_pos = os.listdir("../input/imdb-data/aclimdb_v1/aclImdb/train/pos/")

train_neg = os.listdir("../input/imdb-data/aclimdb_v1/aclImdb/train/neg/")

print(len(test_pos),len(test_neg),len(train_pos),len(train_neg))

#print(len(test_pos))



test_pos_list , test_neg_list , train_pos_list , train_neg_list = ([] for i in range(4))

def creating_arrays(data , new_list , category , pos_neg):

    for i in range(0,len(data)):

        #print(i)

        f = open('../input/imdb-data/aclimdb_v1/aclImdb/'+category+'/'+pos_neg+'/'+data[i],'r',encoding='Latin-1')

        content = f.readline()

        new_list.append(content)
#Calling the function created read the data from text files

creating_arrays(test_pos , test_pos_list , 'test' , 'pos')

creating_arrays(test_neg , test_neg_list , 'test' , 'neg')

creating_arrays(train_pos , train_pos_list , 'train' , 'pos')

creating_arrays(train_neg , train_neg_list , 'train','neg')
#converting textfiles to dataframes 

test_pos_list = pd.DataFrame(test_pos_list, columns = ['reviews'])

test_neg_list = pd.DataFrame(test_neg_list, columns = ['reviews'])

train_pos_list = pd.DataFrame(train_pos_list, columns = ['reviews'])

train_neg_list = pd.DataFrame(train_neg_list, columns = ['reviews'])



print(len(test_pos_list),len(test_neg_list),len(train_pos_list),len(train_neg_list))



train_pos_score = [] 

test_pos_score = [] 

train_neg_score = [] 

test_neg_score = []

for i in range(0,len(train_pos_list)):

    train_pos_score.append(1)

    test_pos_score.append(1)

    train_neg_score.append(0)

    test_neg_score.append(0)

train_pos_list['score'] = train_pos_score

test_pos_list['score'] = test_pos_score

train_neg_list['score'] = train_neg_score

test_neg_list['score'] = test_neg_score



frames = [train_pos_list, test_pos_list, train_neg_list,test_neg_list,twitter_data]

complete_data = pd.concat(frames)

complete_data.head()
print(len(test_pos_list),len(test_neg_list),len(train_neg_list),len(train_pos_list))
complete_data.head()
review_list = complete_data.iloc[:,[0]].values

review_list[0]
#data cleaning

import re

import nltk

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer





remove_words = ['com','http','https','www']

corpus =[]

for i in range(0,len(review_list)):

    #print(i)

    tweet = re.sub('[^a-zA-Z]' , ' ' , str(review_list[i]))

    tweet = tweet.lower()

    #tweet = tweet.split()

    #lemmatizer = WordNetLemmatizer()

    #tweet = [lemmatizer.lemmatize(words) for words in tweet if not words in set(stopwords.words('english'))] 

    #tweet = [word for word in tweet if word not in remove_words]

    #tweet = ' '.join(tweet)

    corpus.append(tweet)

corpus[0]
complete_data['score'].value_counts()
score_list = complete_data.iloc[:,[1]].values

print(len(score_list) , len(corpus))


# # Bag of words(spars matrix)

# from sklearn.feature_extraction.text import CountVectorizer

# cv = CountVectorizer(max_features = 1500)

# X = cv.fit_transform(corpus).toarray()

# Y = df_score.iloc[:,[0]].values



#Bag of words(sparc matrix)

from sklearn.feature_extraction.text import TfidfVectorizer

#cv = TfidfVectorizer(ngram_range=(1,2))

cv = TfidfVectorizer(max_features=5000)

X = cv.fit_transform(corpus).toarray()

Y = complete_data.iloc[:,1].values

#Y = list(map(int, Y))

Y = Y.astype(int)

#Y=Y.astype(int)

X.shape

# train test split

from sklearn.model_selection import train_test_split

X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.20 , random_state = 123)
# Naive bayes

from sklearn.metrics import confusion_matrix

from sklearn.naive_bayes import MultinomialNB



classifier = MultinomialNB(alpha = 10 , fit_prior=True, class_prior=None)

classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, y_pred)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test , y_pred)

accuracy
#training - Logistic regression

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(penalty='l1',C=1)

classifier.fit(X_train , Y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, y_pred)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test , y_pred)

accuracy
# Fitting Random Forest Classification to the Training set

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'gini', random_state = 0 , max_depth=100)

classifier.fit(X_train , Y_train)

# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, y_pred)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test, y_pred)

accuracy
# Fitting SGDClassifier Classification to the Training set

from sklearn.linear_model import SGDClassifier

classifier = SGDClassifier(alpha = 1e-17 , loss='hinge')

classifier.fit(X_train, Y_train)

# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, y_pred)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test, y_pred)

accuracy