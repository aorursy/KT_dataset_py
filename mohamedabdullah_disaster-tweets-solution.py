import numpy as np

import pandas as pd 

import seaborn as sns

import re

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import BernoulliNB

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import VotingClassifier

from sklearn.metrics import f1_score



import warnings

warnings.filterwarnings('ignore')



print("Important libraries loaded successfully")
data_train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

print("Data shape = ",data_train.shape)

data_train.head()
#get total count of data including missing data

total = data_train.isnull().sum().sort_values(ascending=False)



#get percent of missing data relevant to all data

percent = (data_train.isnull().sum()/data_train.isnull().count()).sort_values(ascending=False)



missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(data_train.shape[1])
data_train = data_train.drop(['location','keyword'], axis=1)

print("location and keyword columns droped successfully")
data_train = data_train.drop('id', axis=1)

print("id column droped successfully")
data_train.columns
data_train["text"].head(10)
corpus  = []

pstem = PorterStemmer()

for i in range(data_train['text'].shape[0]):

    #Remove unwanted words

    tweet = re.sub("[^a-zA-Z]", ' ', data_train['text'][i])

    #Transform words to lowercase

    tweet = tweet.lower()

    tweet = tweet.split()

    #Remove stopwords then Stemming it

    tweet = [pstem.stem(word) for word in tweet if not word in set(stopwords.words('english'))]

    tweet = ' '.join(tweet)

    #Append cleaned tweet to corpus

    corpus.append(tweet)

    

print("Corpus created successfully")    
print(pd.DataFrame(corpus)[0].head(10))
rawTexData = data_train["text"].head(10)

cleanTexData = pd.DataFrame(corpus, columns=['text after cleaning']).head(10)



frames = [rawTexData, cleanTexData]

result = pd.concat(frames, axis=1, sort=False)

result
#Create our dictionary 

uniqueWordFrequents = {}

for tweet in corpus:

    for word in tweet.split():

        if(word in uniqueWordFrequents.keys()):

            uniqueWordFrequents[word] += 1

        else:

            uniqueWordFrequents[word] = 1

            

#Convert dictionary to dataFrame

uniqueWordFrequents = pd.DataFrame.from_dict(uniqueWordFrequents,orient='index',columns=['Word Frequent'])

uniqueWordFrequents.sort_values(by=['Word Frequent'], inplace=True, ascending=False)

uniqueWordFrequents.head(10)
uniqueWordFrequents['Word Frequent'].unique()
uniqueWordFrequents = uniqueWordFrequents[uniqueWordFrequents['Word Frequent'] >= 20]

print(uniqueWordFrequents.shape)

uniqueWordFrequents
counVec = CountVectorizer(max_features = uniqueWordFrequents.shape[0])

bagOfWords = counVec.fit_transform(corpus).toarray()
X = bagOfWords

y = data_train['target']

print("X shape = ",X.shape)

print("y shape = ",y.shape)



X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.20, random_state=55, shuffle =True)

print('data splitting successfully')
decisionTreeModel = DecisionTreeClassifier(criterion= 'entropy',

                                           max_depth = None, 

                                           splitter='best', 

                                           random_state=55)



decisionTreeModel.fit(X_train,y_train)



print("decision Tree Classifier model run successfully")
gradientBoostingModel = GradientBoostingClassifier(loss = 'deviance',

                                                   learning_rate = 0.01,

                                                   n_estimators = 100,

                                                   max_depth = 30,

                                                   random_state=55)



gradientBoostingModel.fit(X_train,y_train)



print("gradient Boosting Classifier model run successfully")
KNeighborsModel = KNeighborsClassifier(n_neighbors = 7,

                                       weights = 'distance',

                                      algorithm = 'brute')



KNeighborsModel.fit(X_train,y_train)



print("KNeighbors Classifier model run successfully")
LogisticRegression = LogisticRegression(penalty='l2', 

                                        solver='saga', 

                                        random_state = 55)  



LogisticRegression.fit(X_train,y_train)



print("LogisticRegression Classifier model run successfully")
SGDClassifier = SGDClassifier(loss = 'hinge', 

                              penalty = 'l1',

                              learning_rate = 'optimal',

                              random_state = 55, 

                              max_iter=100)



SGDClassifier.fit(X_train,y_train)



print("SGDClassifier Classifier model run successfully")
SVClassifier = SVC(kernel= 'linear',

                   degree=3,

                   max_iter=10000,

                   C=2, 

                   random_state = 55)



SVClassifier.fit(X_train,y_train)



print("SVClassifier model run successfully")
bernoulliNBModel = BernoulliNB(alpha=0.1)

bernoulliNBModel.fit(X_train,y_train)



print("bernoulliNB model run successfully")
gaussianNBModel = GaussianNB()

gaussianNBModel.fit(X_train,y_train)



print("gaussianNB model run successfully")
multinomialNBModel = MultinomialNB(alpha=0.1)

multinomialNBModel.fit(X_train,y_train)



print("multinomialNB model run successfully")
modelsNames = [('LogisticRegression',LogisticRegression),

               ('SGDClassifier',SGDClassifier),

               ('SVClassifier',SVClassifier),

               ('bernoulliNBModel',bernoulliNBModel),

               ('multinomialNBModel',multinomialNBModel)]



votingClassifier = VotingClassifier(voting = 'hard',estimators= modelsNames)

votingClassifier.fit(X_train,y_train)

print("votingClassifier model run successfully")
#evaluation Details

models = [decisionTreeModel, gradientBoostingModel, KNeighborsModel, LogisticRegression, 

          SGDClassifier, SVClassifier, bernoulliNBModel, gaussianNBModel, multinomialNBModel, votingClassifier]



for model in models:

    print(type(model).__name__,' Train Score is   : ' ,model.score(X_train, y_train))

    print(type(model).__name__,' Test Score is    : ' ,model.score(X_test, y_test))

    

    y_pred = model.predict(X_test)

    print(type(model).__name__,' F1 Score is      : ' ,f1_score(y_test,y_pred))

    print('--------------------------------------------------------------------------')