import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))
data = pd.read_csv("../input/ExtractedTweets.csv")
data.head()
data['numClass'] = data['Party'].map({'Democrat':0, 'Republican':1})
data.tail()
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

stopset = set(stopwords.words("english"))
print(stopset)
vectorizer = TfidfVectorizer(stop_words = stopset, binary = True)

X = vectorizer.fit_transform(data.Tweet)
y = data.numClass
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size = 0.3, train_size=0.7, random_state = 42)
#Import models from sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score

classifiers = ['Multi-NB', 'DecisionTrees', '1-NN', '3-NN', 'AdaBoost', 'RandomForest']
#Initialize the models
A = MultinomialNB(alpha = 0.1, fit_prior = True)
B = DecisionTreeClassifier(random_state = 42)
C = KNeighborsClassifier(n_neighbors = 1)
D = KNeighborsClassifier(n_neighbors = 3)
E = AdaBoostClassifier(n_estimators = 100)
F = RandomForestClassifier(n_estimators = 10, max_depth = None, min_samples_split = 2, random_state = 37)

clfs = [A,B,C,D,E,F]
#Classifier function
def classifierFunction(clf, X, y):
    clf.fit(X,y)
    
#Predict function
def predictFunction(clf, X):
    return clf.predict(X)
predictScores = []

for i in range(len(clfs)):
    classifierFunction(clfs[i], X_train, Y_train)
    Y_pred = predictFunction(clfs[i], X_test)
    score = f1_score(Y_test, Y_pred)
    predictScores.append(score)
    print(score)    
# ploating data for F1 Score
import matplotlib.pyplot as plt

Y_pos = np.arange(len(classifiers))
Y_val = [ x for x in predictScores]
plt.bar(Y_pos,Y_val, align='center', alpha=0.7)
plt.xticks(Y_pos, classifiers)
plt.ylabel('Accuracy Score')
plt.title('Accuracy of Models')
plt.show()