import sys
import nltk
import sklearn
import pandas as pd 
import numpy as np 
df = pd.read_table('../input/SMSSpamCollection',header = None, encoding = 'utf-8')
#print useful information about the data set
print(df.info())
print(df.head())
#chack class distribution
classes = df[0]
print(classes.value_counts())
#convert  class label to binary values, 0 = hum , 1 =spam
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(classes)
print(y[:10])
#store the SMS message data
text_messages = df[1]
print(text_messages[:10])
# use regular expressions to replace e mail addresses , urls,phone number,other numbers,symbols 
# replace e mail addresses with 'emailaddr'
processed= text_messages.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$','emailaddr')
#replace urls with 'webaddress'
processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\s*)?$','webaddress')

# replace money symbols with 'moneysymb'
processed = processed.str.replace(r'£/\$','moneysymbol')

#replace 10 digit phone number with 'phonenumber'
processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$','phonenumber')

#replace number 
processed = processed.str.replace(r'\d+(\.\d+)?','number')
#remove punctuation
processed = processed.str.replace(r'[^\w\d\s]',' ')

#replace witespace between term with a single space
processed = processed.str.replace(r'\s+',' ')

#remove leading and trailing whitespace
processed = processed.str.replace(r'^\s+|\s+?$', '')
#changing words to lower case
processed = processed.str.lower()
print(processed)
#remove stop words from text object

from nltk.corpus import stopwords

stop_words= set(stopwords.words('english'))

processed = processed.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))
#remove word stems using a porter stemer 

ps = nltk.PorterStemmer()
processed = processed.apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))
from nltk.tokenize import word_tokenize
all_words = []
for message in processed:
    words = word_tokenize(message)
    for w in words:
        all_words.append(w)
        
all_words = nltk.FreqDist(all_words)
#print the total no of words and 15 most common 
print('Number of words: {}'.format(len(all_words)))
print('Number of common words: {}'.format(all_words.most_common(15)))
#use the 1500 most common words as feature 
word_features = list(all_words.keys())[:1500]
#define a find feature function 
def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in word_features:
        features[word] =(word in words)
        
    return features
features =find_features(processed[0])
for key,value in features.items():
    if value == True:
        print(key)
processed[0]
features
#find features fro all messages
messages = zip(processed,y)
#define a seed for reproducibility
seed = 1
np.random.seed = seed
#np.random.shuffle(messages)

#call find features function for each sms 
featuresets = [(find_features(text), label) for (text,label) in messages]
#split traing and testing dataset
from sklearn import model_selection
training,testing = model_selection.train_test_split(featuresets, test_size = 0.25, random_state = seed)
print(len(training))
print(len(testing))
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
names = ['K Nearest Neighbors', ' Decision Tree', 'Random Forest', ' Logistic Regression', ' SGD Classifier',' Naive Bayes', ' SVM Linear']
classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel='linear')
]
models = zip(names, classifiers)
print(models)
from nltk.classify.scikitlearn import SklearnClassifier
for name,model in models:
    nltk_model = SklearnClassifier(model)
    nltk_model.train(training)
    accuracy = nltk.classify.accuracy(nltk_model,testing) * 100
    print('{}: Accuracy: {}'.format(name,accuracy))
# Ensemble methods - Voting classifier

from sklearn.ensemble import VotingClassifier

 

names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear"]
classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]
models = zip(names, classifiers)
models=list(models)
nltk_ensemble = SklearnClassifier(VotingClassifier(estimators = models, voting = 'hard', n_jobs = -1))
nltk_ensemble.train(training)
accuracy = nltk.classify.accuracy(nltk_model, testing)*100
print("Voting Classifier: Accuracy: {}".format(accuracy))
txt_features,labels = zip(*testing)
prediction = nltk_ensemble.classify_many(txt_features)
print(classification_report(labels,prediction))

pd.DataFrame(
    confusion_matrix(labels,prediction),
    index = [['actual','actual'], ['ham', 'spam']],
    columns = [['predicted', 'predicted'],['ham','spam']]
)
