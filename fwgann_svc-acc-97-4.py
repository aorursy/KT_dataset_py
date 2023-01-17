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
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
filename = '../input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv'
dataset = pd.read_csv(filename)
dataset.info()
dataset.describe()
dataset[dataset['fraudulent'] == 1]
dataset = dataset[dataset['description'].notna()]
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer 
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
# preprocess the text
lancaster=LancasterStemmer() 
stop_words = stopwords.words('english')

def preprocess(text):
    # tokenize 
    text = re.sub('[^a-zA-Z\s]', '', text)
    # lowercase 
    text = text.lower()
    # removing stop word and stemming using Lancaster
    split = text.split()
    for word in split :
      if word in stop_words :
        word = ''
      else :
        lancaster.stem(word)
    return ' '.join([word for word in split])

dataset['description'] = dataset['description'].apply(preprocess)
dataset['description'].sample(10)
# prepare train & data sets 
train_x, test_x, train_y, test_y = model_selection.train_test_split(dataset['description'], dataset['fraudulent'],test_size=0.2)
# encode categories so machine can understand
encoder = LabelEncoder()
train_y = encoder.fit_transform(train_y)
test_y = encoder.fit_transform(test_y)
# transform data to count matrix
MAX = 3000 # after many trials, we landed on this number. also, because we love thầy Minh 3000 <3
vectorizer = TfidfVectorizer(max_features = MAX)
vectorizer.fit(train_x)

# vectorize train and test sets
train_x_vec = vectorizer.transform(train_x)
test_x_vec = vectorizer.transform(test_x)
from sklearn.linear_model import LogisticRegression

# fit data using logistic regression
log_regress = LogisticRegression()
log_regress.fit(train_x_vec, train_y)

# predict output of the test data set
predicted = log_regress.predict(test_x_vec)

# get accuracy
print("accuracy score of Logistic Regression:", accuracy_score(predicted, test_y), "\n")

# creating a confusion matrix 
cm = confusion_matrix(test_y, predicted)
print("confusion matrix of Logistic Regression:\n", cm, "\n")

# create a classifcation report
print("classification report:\n", classification_report(test_y, predicted), "\n")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# cross validation to find best n_neighbors

val_error_rate = []
neighbors_range = range(1,100,2)

for i in neighbors_range:
    knn = KNeighborsClassifier(n_neighbors=i, weights='distance')
    val_error = 1 - cross_val_score(knn, train_x_vec, train_y, cv=5).mean()
    val_error_rate.append(val_error)
    
plt.figure(figsize=(15,7))
plt.plot(neighbors_range, val_error_rate, color='orange', linestyle='dashed', marker='o',
         markerfacecolor='black', markersize=5, label='Validation Error')

plt.xticks(np.arange(neighbors_range.start, neighbors_range.stop, neighbors_range.step), rotation=45)
plt.grid()
plt.legend()
plt.title('Validation Error vs. K Value')
plt.xlabel('K')
plt.ylabel('Validation Error')
plt.show()

N_NEIGHBORS = 37

# fit data using KNN
knn = KNeighborsClassifier(n_neighbors = N_NEIGHBORS, weights='distance')
knn.fit(train_x_vec, train_y)

# predict output of the test data set
predicted = knn.predict(test_x_vec)

# get accuracy
print("accuracy score of KNN, k = {k}: {asc}".format(k = N_NEIGHBORS, asc = accuracy_score(predicted, test_y)), "\n")

# creating a confusion matrix 
cm = confusion_matrix(test_y, predicted)
print("confusion matrix of KNN:\n", cm, "\n")

# create a classifcation report
print("classification report:\n", classification_report(test_y, predicted), "\n")
from sklearn.svm import SVC

# fit data using SVC
svc = SVC(kernel='rbf')
svc.fit(train_x_vec, train_y)

# predict output of the test data set
predicted = svc.predict(test_x_vec)

# get accuracy
print("accuracy score of Support Vector Machine:", accuracy_score(predicted, test_y), "\n")

# creating a confusion matrix 
cm = confusion_matrix(test_y, predicted)
print("confusion matrix of SVM:\n", cm, "\n")

# create a classifcation report
print("classification report:\n", classification_report(test_y, predicted), "\n")
models = [
  LogisticRegression(),
  KNeighborsClassifier(n_neighbors = N_NEIGHBORS, weights='distance'),
  SVC(kernel='rbf')
]

for model in models:
  accuracy = cross_val_score(model, train_x_vec, train_y, scoring='accuracy', cv=5).mean()
  print(model.__class__.__name__, ':', accuracy)