import pandas as pd 
import random

data = pd.read_csv("/kaggle/input/boardgamegeek-reviews/bgg-13m-reviews.csv") 
# subset_percent = 0.01 # for running algo on this percent of data (used to reduce execution time)
# data = pd.read_csv("/kaggle/input/boardgamegeek-reviews/bgg-13m-reviews.csv", skiprows=lambda i: i>0 and random.random() > subset_percent, header=0) 
data.head()

data.isnull().sum()
data.dropna(how='any', subset=['comment'], inplace=True)
data.drop(['user', 'ID', 'name'], axis=1, inplace=True)
data.head()
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)


vectorizer = CountVectorizer()
# Didn't used stopwords or stemming as the reviews are from other languages as well.
# Accuracy after removing stopwords and using stemmer: ~0.30
# Accuracy with not removing stopwords nor using stemmer: ~0.32


train_features = vectorizer.fit_transform(data_train['comment'])
test_features = vectorizer.transform(data_test['comment'])

test_label = [round(r) for r in data_test['rating']]
train_label = [round(r) for r in data_train['rating']]


# Below code was used to manipulate number of classes.Got good accuracy when the 
# number of classes were only 2 (positive and negative). But since it was not the goal, it was removed

# for i in range(len(test_label)):
#     if test_label[i] > 5:
#         test_label[i] = 1
#     else:
#         test_label[i] = 0

# for i in range(len(train_label)):
#     if train_label[i] > 5:
#         train_label[i] = 1
#     else:
#         train_label[i] = 0
algos = ["MulitnomialNB", "KNN", "MLP", "SVM"]
algo_scores = []
nb = MultinomialNB(alpha=1)
nb.fit(train_features, train_label)

score = nb.score(test_features, test_label)

algo_scores.append(score)
print(score)
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(train_features, train_label)
knnscore = neigh.score(test_features, test_label)
algo_scores.append(knnscore)
print(knnscore)

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='adam', alpha=1e-4, random_state=1, max_iter=500) # Increased max_iter to 500 as 200 was not enough
clf.fit(train_features, train_label)
score = clf.score(test_features, test_label)

algo_scores.append(score)
print(score)

from sklearn import svm
clf = svm.SVC()
clf.fit(train_features, train_label)
score = clf.score(test_features, test_label)

algo_scores.append(score)
print(score)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(algos,algo_scores)
plt.show()

# Below accuracy was calculated by relaxing the output class by 1.
# For ex. if the predicted value was 4 but the actual value was 3, then too it was considered correct.
# Accuracy almost doubles when calculated in this way

predictions = nb.predict(test_features)

def custom_accuracy(preds, actual):
    count = 0
    n = len(preds)
    for i in range(n):
        if abs(preds[i] - actual[i] < 1):
            count += 1
    return count / n

# Compute the error
print("accuracy s " + str(custom_accuracy(predictions, test_label)))

from sklearn.model_selection import cross_val_score
import numpy as np

alpha_range = list(np.arange(1,50,5))
len(alpha_range)

alpha_scores=[]

for a in alpha_range:
    clf = MultinomialNB(alpha=a)
    scores = cross_val_score(clf, train_features, train_label, cv=5, scoring='accuracy')
    alpha_scores.append(scores.mean())
    print(a,scores.mean())

import matplotlib.pyplot as plt

MSE = [1 - x for x in alpha_scores]


optimal_alpha_bnb = alpha_range[MSE.index(min(MSE))]

# plot misclassification error vs alpha
plt.plot(alpha_range, MSE)

plt.xlabel('hyperparameter alpha')
plt.ylabel('Misclassification Error')
plt.show()

import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import metrics

cm_test = confusion_matrix(test_label, predictions)

sns.heatmap(cm_test,annot=True,fmt='d')

nb.class_count_
test_review = "with in this is it"
print(nb.predict(vectorizer.transform([test_review])))
print(nb.predict_proba(vectorizer.transform([test_review])))
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

vectorizer = CountVectorizer()

train_features = vectorizer.fit_transform(data['comment'])

train_label = [round(r) for r in data['rating']]

nb = MultinomialNB(alpha=1)
nb.fit(train_features, train_label)

test_review = "This is a sample review"
print(nb.predict(vectorizer.transform([test_review])))
print(nb.predict_proba(vectorizer.transform([test_review])))

import pickle
with open('/kaggle/working/model.pk', 'wb') as file:
    pickle.dump(nb, file)
with open('/kaggle/working/vect.pk', 'wb') as file:
    pickle.dump(vectorizer, file)

