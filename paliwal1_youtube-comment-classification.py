import numpy as np # linear algebra

import pandas as pd

import matplotlib.pyplot as plt
Psy = pd.read_csv('../input/youtube-spam-classifiedcomments/Youtube01-Psy.csv')

Katy = pd.read_csv('../input/youtube-spam-classifiedcomments/Youtube02-KatyPerry.csv')

Eminem = pd.read_csv('../input/youtube-spam-classifiedcomments/Youtube04-Eminem.csv')

Shakira = pd.read_csv('../input/youtube-spam-classifiedcomments/Youtube05-Shakira.csv')

LMFAO = pd.read_csv('../input/youtube-spam-classifiedcomments/Youtube03-LMFAO.csv')
df = pd.concat([Shakira, Eminem, Katy, Psy, LMFAO])

df.drop('DATE', axis=1, inplace=True)



df.shape
df.head()
df['CLASS'].value_counts().plot(kind='bar')
classes = df['CLASS']

print(classes.value_counts())
text_messages = df["CONTENT"]
# use regular expressions to replace email addresses, URLs, phone numbers, other numbers



# Replace email addresses with 'email'

processed = text_messages.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$',

                                 'emailaddress')



# Replace URLs with 'webaddress'

processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$',

                                  'webaddress')



# Replace money symbols with 'moneysymb' (£ can by typed with ALT key + 156)

processed = processed.str.replace(r'£|\$', 'moneysymb')

    

# Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'

processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',

                                  'phonenumbr')

    

# Replace numbers with 'numbr'

processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')
print(text_messages[:10])
processed = processed.str.replace(r'[^\w\d\s]', ' ')



# Replace whitespace between terms with a single space

processed = processed.str.replace(r'\s+', ' ')



# Remove leading and trailing whitespace

processed = processed.str.replace(r'^\s+|\s+?$', '')
# change words to lower case - Hello, HELLO, hello are all the same word

processed = processed.str.lower()

print(processed)
import nltk
from nltk.corpus import stopwords



# remove stop words from text messages



stop_words = set(stopwords.words('english'))



processed = processed.apply(lambda x: ' '.join(

    term for term in x.split() if term not in stop_words))
# Remove word stems using a Porter stemmer

ps = nltk.PorterStemmer()



processed = processed.apply(lambda x: ' '.join(

    ps.stem(term) for term in x.split()))
from nltk.tokenize import word_tokenize



# create bag-of-words

all_words = []



for message in processed:

    words = word_tokenize(message)

    for w in words:

        all_words.append(w)

        

all_words = nltk.FreqDist(all_words)
# print the total number of words and the 15 most common words

print('Number of words: {}'.format(len(all_words)))

print('Most common words: {}'.format(all_words.most_common(15)))
# use the 1500 most common words as features

word_features = list(all_words.keys())[:1500]
# The find_features function will determine which of the 1500 word features are contained in the review

def find_features(message):

    words = word_tokenize(message)

    features = {}

    for word in word_features:

        features[word] = (word in words)



    return features



# Lets see an example!

features = find_features(str(processed[0]))

for key, value in features.items():

    if value == True:

        print (key)
# Now lets do it for all the messages

messages = list(zip(processed, classes))



# define a seed for reproducibility

seed = 1

np.random.seed = seed

np.random.shuffle(messages)



# call find_features function for each SMS message

featuresets = [(find_features(text), label) for (text, label) in messages]
# we can split the featuresets into training and testing datasets using sklearn

from sklearn import model_selection



# split the data into training and testing datasets

training, testing = model_selection.train_test_split(featuresets, test_size = 0.25, random_state=seed)
print(len(training))

print(len(testing))
# import warnings filter

from warnings import simplefilter

# ignore all future warnings

simplefilter(action='ignore', category=FutureWarning)
# We can use sklearn algorithms in NLTK

from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.svm import SVC



model = SklearnClassifier(SVC(kernel = 'linear'))



# train the model on the training data

model.train(training)



# and test on the testing dataset!

accuracy = nltk.classify.accuracy(model, testing)*100

print("SVC Accuracy: {}".format(accuracy))
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import SVC

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier



# Define models to train

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

names1 = []

results = []



for name, model in models:

    nltk_model = SklearnClassifier(model)

    nltk_model.train(training)

    accuracy = nltk.classify.accuracy(nltk_model, testing)*100

    print("{} Accuracy: {}".format(name, accuracy))

    names1.append(name)

    results.append(accuracy)
txt_features, labels = zip(*testing)
prediction = nltk_model.classify_many(txt_features)
# print a confusion matrix and a classification report

print(classification_report(labels, prediction))



pd.DataFrame(

    confusion_matrix(labels, prediction),

    index = [['actual', 'actual'], ['ham', 'spam']],

    columns = [['predicted', 'predicted'], ['ham', 'spam']])