#Importing required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
# Reading the dataset
dataset = pd.read_csv('../input/reviews/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
dataset.head()
dataset.shape
# Preprocessing
nltk.download('stopwords')
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
# Creating the Bag of Words model
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
#Clasiification

# Naive Bayes
NB_classifier = GaussianNB()
NB_classifier.fit(X_train, y_train)
y_pred_NB = NB_classifier.predict(X_test)

# Random Forest
rf_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)

#Support Vector Machine
SVC_classifier = SVC(kernel = 'rbf')
SVC_classifier.fit(X_train, y_train)
y_pred_SVC = SVC_classifier.predict(X_test)
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
accuracy_score(y_test, y_pred_NB)
print(classification_report(y_test, y_pred_NB))
accuracy_score(y_test, y_pred_rf)
print(classification_report(y_test, y_pred_rf))
accuracy_score(y_test, y_pred_SVC)
print(classification_report(y_test, y_pred_SVC))
#CAP Analysis
total = len(y_test) 
one_count = np.sum(y_test) 
zero_count = total - one_count 
lm_NB = [y for _, y in sorted(zip(y_pred_NB, y_test), reverse = True)] 
lm_SVC = [y for _, y in sorted(zip(y_pred_SVC, y_test), reverse = True)] 
lm_RandFor = [y for _, y in sorted(zip(y_pred_rf, y_test), reverse = True)] 
x = np.arange(0, total + 1) 
y_NB = np.append([0], np.cumsum(lm_NB)) 
y_SVC = np.append([0], np.cumsum(lm_SVC)) 
y_RandFor = np.append([0], np.cumsum(lm_RandFor)) 

plt.figure(figsize = (10, 10))
plt.title('CAP Curve Analysis')
plt.plot([0, total], [0, one_count], c = 'k', linestyle = '--', label = 'Random Model')
plt.plot([0, one_count, total], [0, one_count, one_count], c = 'grey', linewidth = 2, label = 'Perfect Model') 
plt.plot(x, y_SVC, c = 'y', label = 'SVC', linewidth = 2)
plt.plot(x, y_NB, c = 'b', label = 'Naive Bayes', linewidth = 2)
plt.plot(x, y_RandFor, c = 'r', label = 'Rand Forest', linewidth = 2)
plt.legend()