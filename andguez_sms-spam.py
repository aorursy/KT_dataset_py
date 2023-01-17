import sys
import nltk
#nltk.download('stopwords')
import sklearn
import pandas as pd
import numpy as np



print("Python:{}".format(sys.version))
print("NLTK:{}".format(nltk.__version__))
print("sklearn:{}".format(sklearn.__version__))
print("pandas:{}".format(pd.__version__))
print("numpy:{}".format(np.__version__))


import pandas as pd
import numpy as np
# Load the dataset of sms messages
df = pd.read_table('../input/SMSSpamCollection',header= None, encoding='utf-8')
print(df.info())
print(df.head())
# check class distribution
classes = df[0]
print(classes.value_counts())
# conver class labels to binary values, 0 = ham, 1 = spam

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
Y = encoder.fit_transform(classes)

print(classes[:10])
print(Y[:10])
# store the SMS message data
text_messages = df[1]
print(text_messages[:10])
# use regular expressions to remplace email addresses, urls, phone numbers, other numbers, symbols

# replace email addresses with 'emailaddr'

processed = text_messages.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$','emailaddr')

# replace urls with 'webaddress'
processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$','webaddress')

#replace  mony symbols with 'moneysymb'
processed = processed.str.replace(r'£|\$','moneysymb')

#replace 10 digit phone number with 'phonenumber'
processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$', 'phonenumber')

# replace normal numbers with 'numbr'
processed = processed.str.replace(r'\d+(\.\d+)?','numbr')
# remove punctation

processed = processed.str.replace(r'[^\w\d\s]',' ')

# replace whitespave between terms with a single space
processed = processed.str.replace(r'\s+',' ')

#remove leading and trailing whitespace
processed = processed.str.replace(r'^\s+|\s+?$','')
# change word to lower case - HolLe, HELLO, hello are all the same
processed = processed.str.lower()
# remove stop words from text messages

from nltk.corpus import stopwords


stop_words = set(stopwords.words('english'))

processed = processed.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))
#remove word stems using a Porter stemmer

ps = nltk.PorterStemmer()

processed = processed.apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))
print(processed)
from nltk.tokenize import word_tokenize
#nltk.download('punkt')
#Creating a bag of words
all_words = []

for message in processed:
    words = word_tokenize(message)
    for w in words:
        all_words.append(w)

all_words = nltk.FreqDist(all_words)
# print the total number of words and the  most common words
print('Number of words: {}'.format(len(all_words)))
print('Most common words: {}'.format(all_words))
#use the 1500 most common words as features
words_features = [x[0] for x in list(all_words.most_common(3000))] 
#define a find_features function
def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in words_features:
        features[word] = (word in words)
    
    return features

#lets see an example
features = find_features(processed[0])
print(processed[0])
for key,value in features.items():
    if value == True:
        print(key)
# find features for all messages
messages = list(zip(processed,Y))

#define a seed for repriducibility
seed = 1
np.random.seed = seed
np.random.shuffle(messages)

#call find_features function for SMS messages
featuresets = [(find_features(text),label) for (text,label) in messages]
#split training and testing data sets using sklearn
from sklearn import model_selection

training,testing = model_selection.train_test_split(featuresets,test_size = 0.25,random_state = seed)
print('Training: {}'.format(len(training)))
print('Testing: {}'.format(len(testing)))
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
# Define models to train
names = ['K Nearest Neighbors', 'Decision Tree','Random Forest','Logistic Regression','SGD Classifier','Navie Bayes','SVM Line']

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(), 
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = list(zip(names,classifiers))
# wrap models in NLTK
from nltk.classify.scikitlearn import SklearnClassifier

for name,model in models:
    nltk_model = SklearnClassifier(model)
    nltk_model.train(training)
    accuracy = nltk.classify.accuracy(nltk_model,testing) * 100
    print('{}: Accurency: {}'.format(name,accuracy))
# ensemble method - voting classifier
from sklearn.ensemble import VotingClassifier

# Define models to train
names = ['K Nearest Neighbors', 'Decision Tree','Random Forest','Logistic Regression','SGD Classifier','Navie Bayes','SVM Line']

classifiers = [
    #KNeighborsClassifier(),
    #DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(), 
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = list(zip(names,classifiers))

nltk_esemble = SklearnClassifier(VotingClassifier(estimators = models,voting = 'hard',n_jobs = -1))
nltk_esemble.train(training)
accuracy = nltk.classify.accuracy(nltk_esemble,testing) * 100
print('Emsable Accurency: {}'.format(accuracy))
# make class label prediction for testing set
text_features, labels = list(zip(*testing))

prediction = nltk_esemble.classify_many(text_features)
# print a confusion matrix and a classification report
print(classification_report(labels,prediction))

pd.DataFrame(
    confusion_matrix(labels,prediction),
    index= [['actual','actual'],['ham','spam']],
    columns = [['predicted','predicted'],['ham','spam']]
)
