import pandas as pd

import numpy as np

train=pd.read_csv('../input/train_2kmZucJ.csv')

test=pd.read_csv('../input/test_oJQbWVk.csv')
print(train.info())

print(train.head())
print(test.info())

print(test.head())
tweet_train=train.iloc[:,2]

tweet_test=test.iloc[:,1]
label=train.iloc[:,1]
# Removing urls,numbers,symbols

# Removing the urls

tweet_train=tweet_train.str.replace(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*',' ')

tweet_test=tweet_test.str.replace(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*',' ')

print(tweet_train[:5])

print(tweet_test[:5])
# Removing the Symbols

tweet_train=tweet_train.str.replace(r'[^\w]',' ')

tweet_test=tweet_test.str.replace(r'[^\w]',' ')
print(tweet_train[:5])

print(tweet_test[:5])
# Removing the Numbers

tweet_train=tweet_train.str.replace(r'[0-9]+',' ')

tweet_test=tweet_test.str.replace(r'[0-9]+',' ')
print(tweet_train[:5])

print(tweet_test[:5])
# Turning them into lower case

tweet_train=tweet_train.str.lower()

tweet_test=tweet_test.str.lower()
print(tweet_train[:5])

print(tweet_test[:5])
# Removing Whitespace

# Replace whitespace between terms with a single space

tweet_train = tweet_train.str.replace(r'\s+', ' ')

tweet_test = tweet_test.str.replace(r'\s+', ' ')



# Remove leading and trailing whitespace

tweet_train = tweet_train.str.replace(r'^\s+|\s+?$', '')

tweet_test = tweet_test.str.replace(r'^\s+|\s+?$', '')
print(tweet_train[:5])

print(tweet_test[:5])
# Removing the Stop Words

from nltk.corpus import stopwords



stop_words=set(stopwords.words('english'))



tweet_train = tweet_train.apply(lambda x: ' '.join(

    term for term in x.split() if term not in stop_words))
from nltk.corpus import stopwords



stop_words=set(stopwords.words('english'))



tweet_test = tweet_test.apply(lambda x: ' '.join(

    term for term in x.split() if term not in stop_words))
# Stemming the Words using Porter Stemmer

import nltk

from nltk.stem.porter import PorterStemmer

ps=nltk.PorterStemmer()



tweet_train=tweet_train.apply(lambda x:' '.join(ps.stem(term) for term in x.split()))

tweet_test=tweet_test.apply(lambda x:' '.join(ps.stem(term) for term in x.split()))

print(tweet_train.shape)

print(tweet_test.shape)
from nltk.tokenize import word_tokenize



# Creating a Bag of Words Model

all_words=[]



for text in tweet_train:

    words=word_tokenize(text)

    for w in words:

        all_words.append(w)

        

for text in tweet_test:

    words=word_tokenize(text)

    for w in words:

        all_words.append(w)



all_words=nltk.FreqDist(all_words)
print(len(all_words))
word_features=list(all_words.keys())[:5000]
def find_features(text):

    words=word_tokenize(text)

    features={}

    for word in word_features:

        features[word]=(word in words)

        

    return features



features=find_features(tweet_train[0])



for key,values in features.items():

    if values==True:

        print(key)
featuresset_train=[(find_features(tweet)) for (tweet) in tweet_train]

featuresset_test=[(find_features(tweet)) for (tweet) in tweet_test]
featureset=list(zip(featuresset_train,label))
from sklearn.model_selection import train_test_split

training,validation=train_test_split(featureset,test_size=0.1,random_state=365)
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.svm import SVC



model = SklearnClassifier(SVC(kernel = 'linear'))



# train the model on the training data

model.train(training)



# and test on the testing dataset!

accuracy = nltk.classify.accuracy(model, validation)*100

print("SVC Accuracy: {}".format(accuracy))
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import SVC

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix



names=['KNeighbors Classifier','DecisionTree Classifier','RandomForest Classifier','Logistic Regression','SGD Classifier','Multinomial NB','SVC']

classifier=[KNeighborsClassifier(),DecisionTreeClassifier(),RandomForestClassifier(),LogisticRegression(),SGDClassifier(max_iter=100),MultinomialNB(),SVC(kernel='linear')]



models = zip(names, classifier)



for name,classifier in models:

    nltk_model=SklearnClassifier(classifier)

    nltk_model.train(training)

    accuracy=nltk.classify.accuracy(nltk_model,validation)*100

    print('{} Accuracy : {}'.format(name,accuracy))
# Ensemble methods - Voting classifier

from sklearn.ensemble import VotingClassifier



names = [ "Random Forest", "Logistic Regression", "SGD Classifier",

         "Naive Bayes", "SVM Linear"]



classifier = [

    RandomForestClassifier(),

    LogisticRegression(),

    SGDClassifier(max_iter = 100),

    MultinomialNB(),

    SVC(kernel = 'linear')]



models = list(zip(names, classifier))



nltk_ensemble = SklearnClassifier(VotingClassifier(estimators = models, voting = 'hard', n_jobs = -1))

nltk_ensemble.train(training)

accuracy = nltk.classify.accuracy(nltk_model, validation)*100

print("Voting Classifier: Accuracy: {}".format(accuracy))
prediction=nltk_ensemble.classify_many(featuresset_test)
prediction
test_ID=test.iloc[:,0]

print(test_ID)
submission_1=pd.DataFrame(test_ID,columns=['id'])
print(submission_1.head())
submission_1['label']=prediction
submission_1.head()
submission_1.to_csv('Submission.csv')