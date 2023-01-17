# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import nltk
#dataset
data = pd.read_csv('../input/sms-spam-collection-dataset/spam.csv',delimiter=',',encoding='latin-1')
data.head()
#some of the columns are of no use, therefor it will be ideal to drop them
data=data.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"], axis=1)
data.head()
#There is no missing values 
import missingno
missingno.matrix(data, figsize=(5,5))
# value count of span/ham in the data set
data["v1"].value_counts()
# Label Enciding the data ham=0, spam=1
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(data["v1"])
print(data.v1[:10])
print(y)
text = data.iloc[:,1]
text[:10]
# replace all the email address with "emailaddress" 
processed = text.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$','emailaddress')
# replace all the url with "webaddress"
processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$','webaddress')
# Replace money symbols with 'moneysymb' (£ can by typed with ALT key + 156)
processed = processed.str.replace(r'£|\$', 'moneysymb')
# Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'
processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$','phonenumbr')
# Replace numbers with 'numbr'
processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')
# Remove punctuation
processed = processed.str.replace(r'[^\w\d\s]', ' ')
# Replace whitespace between terms with a single space
processed = processed.str.replace(r'\s+', ' ')
# Remove leading and trailing whitespace
processed = processed.str.replace(r'^\s+|\s+?$', '')
# change aal the words into lower case
processed = processed.str.lower()
print(processed)
# remove all the stop words
from nltk.corpus import stopwords
stop_words= set(stopwords.words("english"))
processed = processed.apply(lambda x: " ".join(term for term in x.split() if term not in stop_words))
# breaking the word to its bas word with the help of Porter Stemer
from nltk.stem import PorterStemmer
ps= PorterStemmer()
processed = processed.apply(lambda x: " ".join(ps.stem(term) for term in x.split()))
from nltk.tokenize import word_tokenize
all_words = []

for message in processed:
    words = word_tokenize(message)
    for w in words:
        all_words.append(w)
        
all_words = nltk.FreqDist(all_words)       
print(f"Total number of words : {len(all_words)}")
print(f"Most common words:{all_words.most_common(15)}")
word_freature = list(all_words.keys())[:1500]
# The find_features function will determine which of the 1500 word features are contained in the review
def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in word_freature:
        features[word] = (word in words)

    return features

# Lets see an example!
features = find_features(processed[0])
for key, value in features.items():
    if value == True:
        print (key)
# Now lets do it for all the messages
messages = zip(processed, y)
featuresets = [(find_features(text), label) for (text, label) in messages]
# we can split the featuresets into training and testing datasets using sklearn
from sklearn import model_selection

# split the data into training and testing datasets
training, testing = model_selection.train_test_split(featuresets, test_size = 0.25, random_state=1)
print(len(training))
print(len(testing))
# We can use sklearn algorithms in NLTK
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC
model = SklearnClassifier(SVC(kernel = 'linear'))
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

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
    SVC(kernel = 'linear')]

models = zip(names, classifiers)

for name, model in models:
    nltk_model = SklearnClassifier(model)
    nltk_model.train(training)
    accuracy = nltk.classify.accuracy(nltk_model, testing)*100
    print("{} Accuracy: {}".format(name, accuracy))



