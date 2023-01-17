#import libraries
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
df = pd.read_csv('../input/spam.csv', encoding="latin-1")
#Reading head of dataset
df.head()
#Removing unwanted column from the datset
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df.head()
df['v1'] = df['v1'].map({'ham': 0, 'spam': 1})
df.head()
df['v2'].tail()
#Cleaning test msgs
corpus = []
for i in range(0,5572):
    msg = re.sub('[^a-zA-Z]', ' ', df['v2'][i]) #Keeping only words, removing all numbers, punctuation
    msg = msg.lower() #converting all words in lower case
    msg = msg.split() #spliting each words to its own
    ps = PorterStemmer() #To get root meaning of each words
    msg = [ps.stem(word) for word in msg if not word in set(stopwords.words('english'))] 
    #Removing stoping words form the msgs
    msg = ' '.join(msg) #Keeping cleaned words together
    corpus.append(msg)

# Checking random msgs
corpus[101]
#Spliting dataset to train and test
cv = CountVectorizer()
X = cv.fit_transform(corpus) # Fit the Data
y = df['v1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
#Naive Bayes Classifier
clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
#Making Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm = pd.DataFrame(cm)
cm
#checking accuracy score
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


