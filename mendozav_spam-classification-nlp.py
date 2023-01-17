import numpy as np

import pandas as pd

import sklearn

import nltk
csv = pd.read_csv('../input/sms-spam-collection-dataset/spam.csv', encoding="ISO-8859-1")

df = pd.DataFrame(csv)
# check class distribution

classes = df[df.columns[0]]

print(classes.value_counts())
# convert class labels to binary values, 0 = ham  1 = spam



from sklearn.preprocessing import LabelEncoder



encoder = LabelEncoder()

Y = encoder.fit_transform(classes)



# quick check

print(classes[:10])

print(Y[:10])
# store SMS message data

text_messages = df[df.columns[1]]

print(text_messages[:10])
# expressions can be found at https://regexlib.com/

# use regular expressions to replace email addresses, urls, phone numbers, etc.



# replace email addresses with 'emailaddr

processed = text_messages.str.replace(r'^\w+@[a-zA-Z_]+?\.[a-zA-Z]{2,3}$', 'emailaddr')



# replace urls with 'webaddress'

processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$','webaddress')

# replace money symbols with 'moneysymb'

processed = text_messages.str.replace(r'£|\$', 'moneysymb')



# replace 10 digit phone numbers with 'phonenum'

processed = text_messages.str.replace(r'^[2-9]\d{2}-\d{3}-\d{4}$', 'phonenum')



# replace normal numbers with 'num'

processed = text_messages.str.replace(r'\d+(\.\d+)?', 'num')
# remove punctuation

processed = processed.str.replace(r'[^\w\d\s]', ' ')



# replace whitespace between terms with a single space

processed = processed.str.replace(r'\s+', ' ')



# remove leading and trailing whitespace

processed = processed.str.replace(r'^\s+|\s+?$', '')
# change all words to lower case

processed = processed.str.lower()

processed.head()
from nltk.corpus import stopwords



# remove stop words from text messages

# stop words are basically a set of commonly used words in any language such as i, me, to, it, etc.



stop_words = set(stopwords.words('english'))



processed = processed.apply(lambda x: ' '.join(

    term for term in x.split() if term not in stop_words))
# remove word stems using a Porter stemmer

# stemming is the process of reducing a word to its word stem such as removing -ing

ps = nltk.PorterStemmer()



processed = processed.apply(lambda x: ' '.join(

    ps.stem(term) for term in x.split()))



processed.head()
from nltk.tokenize import word_tokenize



# create bag-of-words

all_words = []



for message in processed:

    words = word_tokenize(message)

    for w in words:

        all_words.append(w)



# FreqDist class is used to encode “frequency distributions”, which count the number of times that each outcome of an experiment occurs



all_words = nltk.FreqDist(all_words)
# print the total number of words and the 15 most common words

print('Number of words: {}'.format(len(all_words)))

print('Most common words: {}'.format(all_words.most_common(10)))
# use the 1500 most common words as features

word_features = list(all_words.keys())[:1500]
# find_features function will determine which of the 1500 word features are contained in the email/message

def find_features(message):

    words = word_tokenize(message)

    features = {}

    for word in word_features:

        features[word] = (word in words)



    return features



# example

features = find_features(processed[0])

for key, value in features.items():

    if value == True:

        print(key)
# do it for all the messages

messages = list(zip(processed, Y))



# define a seed for reproducibility

seed = 1

np.random.seed = seed

np.random.shuffle(messages)



# call find_features function for each SMS message

featuresets = [(find_features(text), label) for (text, label) in messages]
from sklearn import model_selection



# split the data into training and testing datasets

training, testing = model_selection.train_test_split(featuresets, test_size = 0.25, random_state=seed)
print('Training:',len(training))

print('Testing:',len(testing))
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.svm import SVC



model = SklearnClassifier(SVC(kernel = 'linear'))



# train the model on the training data

model.train(training)



# and test on the testing dataset!

accuracy = nltk.classify.accuracy(model, testing)*100

print("SVC Accuracy: {}".format(accuracy))
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix



# define models to train

names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression"]



classifiers = [

    KNeighborsClassifier(),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    LogisticRegression()

]



models = zip(names, classifiers)



for name, model in models:

    nltk_model = SklearnClassifier(model)

    nltk_model.train(training)

    accuracy = nltk.classify.accuracy(nltk_model, testing)*100

    print("{} Accuracy: {}".format(name, accuracy))
# ensemble methods - Voting classifier

from sklearn.ensemble import VotingClassifier



names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression"]



classifiers = [

    KNeighborsClassifier(),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    LogisticRegression()

]



models = list(zip(names, classifiers))



nltk_ensemble = SklearnClassifier(VotingClassifier(estimators = models, voting = 'hard', n_jobs = -1))

nltk_ensemble.train(training)

accuracy = nltk.classify.accuracy(nltk_model, testing)*100

print("Voting Classifier: Accuracy: {}".format(accuracy))
# class label prediction for testing set

txt_features, labels = zip(*testing)



prediction = nltk_ensemble.classify_many(txt_features)
# print a confusion matrix and a classification report

print(classification_report(labels, prediction))



pd.DataFrame(

    confusion_matrix(labels, prediction),

    index = [['actual', 'actual'], ['ham', 'spam']],

    columns = [['predicted', 'predicted'], ['ham', 'spam']])