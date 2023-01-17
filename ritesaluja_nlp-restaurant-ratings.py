# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

from itertools import product
import matplotlib.pyplot as plt
# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
df.head()
#word cloud - Visualizing Reviews
wordcloud = (WordCloud(width=1440, height=1080, relative_scaling=0.5, stopwords=stopwords).generate_from_frequencies(df['Review'].value_counts()))

fig = plt.figure(1,figsize=(15, 15))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
#cleaning text and preparing a corpus
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords)]
    review = ' '.join(review)
    corpus.append(review)
# Creating the Bag of Words model
cv = CountVectorizer(max_features = 1500)
X =cv.fit_transform(corpus).toarray()
y = df.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set -  80-20 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
#using logistic classifier
Lclassifier = LogisticRegression(random_state = 0)
Lclassifier.fit(X_train, y_train)
#using Cross validation to get the mean Accuracy
accuracies = cross_val_score(estimator = Lclassifier, X = X_train, y = y_train, cv = 10)
print('Logisitc Regression- Mean Accuracy: {0:.2f}, Std of Accuracy: {1:.2f}'.format(accuracies.mean(),accuracies.std()))
# Fitting Naive Bayes to the Training set
Gclassifier = GaussianNB()
Gclassifier.fit(X_train, y_train)
#using Cross validation to get the mean Accuracy
accuracies = cross_val_score(estimator = Gclassifier, X = X_train, y = y_train, cv = 10)
print('Gaussian Naive Bayes- Mean Accuracy: {0:.2f}, Std of Accuracy: {1:.2f}'.format(accuracies.mean(),accuracies.std()))
# Predicting the Test set results
y_pred = Lclassifier.predict(X_test)
#function from SKLEARN Examples to print Confusion Matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Plot non-normalized confusion matrix
class_names = ['Positive', 'Negative']
plt.figure()
plot_confusion_matrix(cm, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.show()
