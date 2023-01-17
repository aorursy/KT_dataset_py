#Loading the libraries required for loading dataset and visualization 
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
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
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session#loading the dataset using pandas library
df = pd.read_csv('../input/boardgamegeek-reviews/bgg-13m-reviews.csv')
#Shuffle the Dataset
df = df.sample(frac=1)
len(df)
df.dropna(subset=['comment'], inplace=True)
len(df)
plt.hist(df['rating'])
plt.show()
import emoji
import re
def clean_data(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    emoji_pattern = re.compile("["
    u"\U0001F600-\U0001F64F"  
    u"\U0001F300-\U0001F5FF"  
    u"\U0001F680-\U0001F6FF"  
    u"\U0001F1E0-\U0001F1FF"  
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE)
    text = re.sub(r'\n',' ', text) 
    text = re.sub('\s+', ' ', text).strip() 
    return emoji_pattern.sub(r'', text)
    return url.sub(r'',text)
df['comment'] = df['comment'].apply(clean_data)
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
count_vectorizer = feature_extraction.text.TfidfVectorizer(stop_words='english')

%%time
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(df['comment'], 15)

langs=[]
students=[]
for word, freq in common_words:
    langs.append(word)
    students.append(freq)
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(langs,students)
plt.show()
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(df['comment'], 15)
langs=[]
students=[]
for word, freq in common_words:
    langs.append(word)
    students.append(freq)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(langs,students)
plt.show()
#Store the review in X and the rating i.e. the class label in y
X = df['comment'].values
y = df['rating'].values
len(df['rating'].unique())
y = np.round(y)
len(df['rating'].unique())
split_size = 520000
X_train = X[:-split_size]
y_train = y[:-split_size]

X_test = X[-split_size:]
y_test = y[-split_size:]
train_vectors = count_vectorizer.fit_transform(X_train)
test_vectors = count_vectorizer.transform(X_test)
#Load the required libraries
import sklearn
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve
def range_accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i] or actual[i] == (predicted[i]+1) or actual[i] == (predicted[i]-1) :
            correct += 1
    return correct / float(len(actual)) * 100.0
%%time
clf = RidgeClassifier()
clf.fit(train_vectors, y_train)
%%time
y_pred = np.round(clf.predict(test_vectors))
acc = accuracy_score(np.round(y_pred), np.round(y_test))
print('Accuracy on Ridge Classifier : {} %'.format(acc))
range_ridge_score=range_accuracy_metric(y_test, np.round(y_pred))
print('Range Accuracy on Ridge Classifier : {} %'.format(range_ridge_score))
mse1 = mean_squared_error(y_test, y_pred, squared=False)
mse1
print(metrics.classification_report(np.round(y_test), np.round(y_pred)))
%%time
clf2 = MultinomialNB()
clf2.fit(train_vectors, y_train)
%%time
y_pred_nb = np.round(clf2.predict(test_vectors))
acc2 = accuracy_score(y_pred_nb, np.round(y_test))
print('Range Accuracy on Multinomial Naive Bayes : {} %'.format(acc2))
range_MNB_score=range_accuracy_metric(y_test, np.round(y_pred_nb))
print('Range Accuracy on Multinomial Naive Bayes : {} %'.format(range_MNB_score))
mse2 = mean_squared_error(y_test, np.round(y_pred_nb), squared=False)
mse2
print(metrics.classification_report(np.round(y_test), np.round(y_pred_nb)))
%%time
clf3 = LinearSVC()
clf3.fit(train_vectors, y_train)
%%time
y_pred_svc = np.round(clf3.predict(test_vectors))
acc3 = accuracy_score(y_pred_svc, np.round(y_test))
print('Range Accuracy on Linear SVC : {} %'.format(acc3))
range_svc_score=range_accuracy_metric(y_test, np.round(y_pred_svc))
print('Range Accuracy on Linear SVC : {} %'.format(range_svc_score))
mse3 = mean_squared_error(y_test, np.round(y_pred_svc), squared=False)
mse3
print(metrics.classification_report(np.round(y_test), np.round(y_pred_svc)))
%%time
from sklearn.ensemble import VotingClassifier
estimators = []
estimators.append(('ridge', clf))
estimators.append(('MNB', clf2))
# create the ensemble model
ensemble = VotingClassifier(estimators)
ensemble.fit(train_vectors, y_train)
y_pred_ensemble = np.round(ensemble.predict(test_vectors))
acc4 = accuracy_score(y_pred_ensemble, np.round(y_test))
print('Range Accuracy on Ensemble model : {} %'.format(acc4))
range_ensemble_score=range_accuracy_metric(y_test, np.round(y_pred_ensemble))
print('Range Accuracy on Ensemble model : {} %'.format(range_ensemble_score))
mse4 = mean_squared_error(y_test, y_pred_ensemble, squared=False)
mse4
print(metrics.classification_report(np.round(y_test), np.round(y_pred_ensemble)))
labels = ['Ridge', 'MultinomialNB', 'LinearSVC', 'Ensemble']
accuracy = np.round([acc*100, acc2*100, acc3*100, acc4*100])
range_accuracy = np.round([range_ridge_score, range_MNB_score, range_svc_score, range_ensemble_score])

x = np.arange(len(labels))  # the label locations
width = 0.3  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, accuracy, width, label='Accuracy')
rects2 = ax.bar(x + width/2, range_accuracy, width, label='Range Accuracy')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy vs Range Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
autolabel(rects1)
autolabel(rects2)
fig.tight_layout()
plt.show()
from sklearn.externals import joblib
joblib.dump(clf, 'ridge_model.sav')
joblib.dump(count_vectorizer, 'vector.sav')
joblib.dump(clf2, 'multi_naive_bayes.sav')
joblib.dump(clf3, 'linear_SVC.sav')
