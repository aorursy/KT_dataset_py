# importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import sklearn
from sklearn.metrics import accuracy_score
import nltk
import re
# loading our dataset
df = pd.read_csv("../input/spam.csv", encoding = 'latin-1')
# checking the first 5 rows of our data
df.head()
# dropping the unnecessary columns Unnamed: 2, Unnamed: 3, Unnamed: 4
df = df.drop(labels = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)

# renaming the existing column v1 as type, v2 as message
df = df.rename(columns = {'v1': 'type', 'v2': 'message'})

df.head()
# bar chart
count_types = pd.value_counts(df['type'])
count_types.plot(kind = 'bar', color = ['blue', 'orange'])
plt.title('Bar Chart')
plt.show()
# pie chart
count_types.plot(kind = 'pie', autopct='%1.0f%%')
plt.ylabel('')
plt.title('Pie Chart')
plt.show()
# labeling ham as 0 and spam as 1
df['type'] = df.type.map({'ham':0, 'spam':1})
# splitting the columns
X = df['message']
y = df['type']
# converting X and y to numpy array
X = np.array(X)
y = np.array(y)
# coverting the message into lower case, as hello, Hello, HELLO means the same
for i in range(len(X)):
    X[i] = X[i].lower()
    
# first 5 messages after converting text into lower case
print(X[:5])
# removing the extra spaces, digits and non word characters like punctuations, ascii etc.
for i in range(len(X)):
    X[i] = re.sub(r'\W',' ',X[i])
    X[i] = re.sub(r'\d',' ',X[i])
    X[i] = re.sub(r'\s+',' ',X[i])

# first 5 messages after removing the extras
print(X[:5])
# removing the stop words
from nltk.corpus import stopwords
for i in range(len(X)):
    words = nltk.word_tokenize(X[i])
    new_words = [word for word in words if word not in stopwords.words('english')]
    X[i] = ' '.join(new_words)
    
# first 5 messages after removing the stopwords
print(X[:5])
# stemming - get the root of each word
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

for i in range(len(X)):
    words = nltk.word_tokenize(X[i])
    new_words = [stemmer.stem(word) for word in words]
    X[i] = ' '.join(new_words)

# first 5 messages after stemming
print(X[:5])
# creating the tf-idf model
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer('english')
X = vectorizer.fit_transform(X)
# splitting the dataset into test and train data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
# running multinomial naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha = 0.2)
clf = clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print('Accuracy for Multinomial Naive Bayes Classifier: ', accuracy)
# running linear support vector machine classifier
from sklearn.svm import SVC
clf = SVC(kernel = "linear")
clf = clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print('Accuracy for Linear SVM Classifier: ', accuracy)
# running decision tree classifier
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print('Accuracy for Decision Tree Classifier: ', accuracy)