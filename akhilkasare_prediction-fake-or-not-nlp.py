# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import string

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
baseLoc = '/kaggle/input/nlp-getting-started/'

train_data = pd.read_csv(baseLoc + 'train.csv')

train_data.shape
train.head()
train.info()
train.describe()
train.isnull().sum()
train.shape
train.tail()
test.head()
target_value_counts = train['target'].value_counts()

target_value_counts
non_disasterous = train[train['target'] == 0]

non_disasterous
disasterous = train[train['target'] == 1]

disasterous
sns.barplot(x = target_value_counts.index, y = target_value_counts)
keywords_value_counts = train['keyword'].value_counts()

keywords_value_counts[:30]
figure = plt.figure(figsize=(7,6))

sns.barplot(x = keywords_value_counts[:20], y = keywords_value_counts.index[:20])
location_value_counts = train['location'].value_counts()

location_value_counts[:20]
figure = plt.figure(figsize=(7,6))

sns.barplot(x = location_value_counts[:20], y = location_value_counts.index[:20])
import re
# Removing punctuation, html tags, symbols, numbers, etc



def remove_noise(text):

    # Dealing with Punctuation

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text
train['text'] = train['text'].apply(lambda x:remove_noise(x))

test['text'] = test['text'].apply(lambda x:remove_noise(x))
# Converting the upper case to lower case

train['text'] = train['text'].apply(lambda x : x.lower())

test['text'] = test['text'].apply(lambda x : x.lower())
train['text'][2000]
test['text'][1000]
train.head()
import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize



stop = stopwords.words('english')
def remove_stopwords(text):

    text = [item for item in text.split() if item not in stop]

    return ' '.join(text)



train['cleaned_text'] = train['text'].apply(remove_stopwords)

test['cleaned_text'] = test['text'].apply(remove_stopwords)
train.head()
test.head()
from nltk.stem.porter import PorterStemmer



stemmer = PorterStemmer()



def stemming(text):

    text = [stemmer.stem(word) for word in text.split()]

    return ' '.join(text)

    

train['stemed_text'] = train['cleaned_text'].apply(stemming)

test['stemed_text'] = test['cleaned_text'].apply(stemming)



train.head()
from wordcloud import WordCloud

import matplotlib.pyplot as plt

fig, (ax1) = plt.subplots(1, figsize=[7, 7])

wordcloud = WordCloud( background_color='white',

                        width=600,

                        height=600).generate(" ".join(train['stemed_text']))

ax1.imshow(wordcloud)

ax1.axis('off')

ax1.set_title('Frequent Words',fontsize=16);
from sklearn.feature_extraction.text import CountVectorizer



count_vectorizer = CountVectorizer(analyzer='word', binary=True)

count_vectorizer.fit(train['stemed_text'])



train_vectors = count_vectorizer.fit_transform(train['stemed_text'])

test_vectors = count_vectorizer.transform(test['stemed_text'])



print(train_vectors[0].todense())
y = train['target']
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    See full source and example: 

    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, f1_score

import numpy as np 

import itertools
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(train_vectors, y , test_size=0.30, random_state=42)
from sklearn.naive_bayes import MultinomialNB



multi_nb = MultinomialNB(alpha=1.6)



multi_nb.fit(x_train, y_train)
pred = multi_nb.predict(x_test)
multi_f1_score = f1_score(y_test, pred)

print("The F1 score for MultinomialNB is : {}".format(multi_f1_score))



acc_score = accuracy_score(y_test, pred)

("The accuracy for MultinomialNB is : ",acc_score)
conf_matrix = confusion_matrix(y_test, pred)

plot_confusion_matrix(conf_matrix, classes=['FAKE', 'REAL'])
from sklearn.model_selection import cross_val_score

score = cross_val_score(multi_nb, train_vectors, y, cv=3, scoring='f1', )

score
multinb_classifier = MultinomialNB(alpha=0.1)
previous_score = 0



# We are taking values from 0 to 1 with an increament of 0.1 



for alpha in np.arange(0,2,0.1):

    sub_classifier = MultinomialNB(alpha=alpha)

    sub_classifier.fit(x_train, y_train)

    y_pred = sub_classifier.predict(x_test)

    score = accuracy_score(y_test, y_pred)

    

    if score> previous_score:

        classifier = sub_classifier

        print("Alpha is : {} & Accuracy is : {}".format(alpha, score))
from sklearn.linear_model import LogisticRegression

log_regression = LogisticRegression()



log_regression.fit(x_train, y_train)



log_pred = log_regression.predict(x_test)



log_acc_score = accuracy_score(y_test, log_pred)

print("The accuracy score for logistic regression is : ",log_acc_score)



log_reg_f1_score = f1_score(y_test, pred)

print("The F1 score for Logistic Regression is : {}".format(log_reg_f1_score))
conf_matrix = confusion_matrix(y_test, log_pred)

plot_confusion_matrix(conf_matrix, classes=['FAKE', 'REAL'])
from sklearn.tree import DecisionTreeClassifier



dtree = DecisionTreeClassifier()



dtree.fit(x_train, y_train)

dtree_predict = dtree.predict(x_test)



dtree_score = accuracy_score(y_test, dtree_predict)

print("The accuracy score for Decision Tree is : {}".format(dtree_score))



conf_matrix = confusion_matrix(y_test, dtree_predict)

plot_confusion_matrix(conf_matrix, classes=['FAKE', 'REAL'])



decision_tree_f1_score = f1_score(y_test, pred)

print("The F1 score for Decision Tree is : {}".format(decision_tree_f1_score))
from sklearn.ensemble import RandomForestClassifier



random_forest = RandomForestClassifier()

random_forest.fit(x_train, y_train)



random_forest_predict = random_forest.predict(x_test)



random_forest_score = accuracy_score(y_test, random_forest_predict)

print("The accuracy score for Decision Tree is : {}".format(random_forest_score))



conf_matrix = confusion_matrix(y_test, random_forest_predict)

plot_confusion_matrix(conf_matrix, classes=['FAKE', 'REAL'])



random_forest_f1_score = f1_score(y_test, pred)

print("The F1 score for Random Forest Classifier is : {}".format(random_forest_f1_score))
import xgboost as xgb



xgboost = xgb.XGBClassifier()



xgboost.fit(x_train, y_train)

xgboost_pred = xgboost.predict(x_test)



xgboost_score = accuracy_score(y_test, xgboost_pred)

print("The accuracy score for XGboost is : {}".format(xgboost_score))



conf_matrix = confusion_matrix(y_test, xgboost_pred)

plot_confusion_matrix(conf_matrix, classes=['FAKE', 'REAL'])



xgboost_f1_score = f1_score(y_test, pred)

print("The F1 score for XGBoost is : {}".format(xgboost_f1_score))
from sklearn.neighbors import KNeighborsClassifier



knn_clf = KNeighborsClassifier(n_neighbors = 3)

knn_clf.fit(x_train, y_train)

y_pred_knn = knn_clf.predict(x_test)

acc_knn = knn_clf.score(x_train, y_train)



print("The accuracy score for knn is : {}".format(acc_knn))



conf_matrix = confusion_matrix(y_test, y_pred_knn)

plot_confusion_matrix(conf_matrix, classes=['FAKE', 'REAL'])



knn_f1_score = f1_score(y_test, pred)

print("The F1 score for K-NN Classifier is : {}".format(knn_f1_score))
from sklearn.svm import SVC, LinearSVC



svm_clf = SVC()

svm_clf.fit(x_train, y_train)

y_pred_svc = svm_clf.predict(x_test)

acc_svc = svm_clf.score(x_train, y_train)



print("The accuracy score for SVM is : {}".format(acc_svc))



conf_matrix = confusion_matrix(y_test, y_pred_svc)

plot_confusion_matrix(conf_matrix, classes=['FAKE', 'REAL'])



svm_f1_score = f1_score(y_test, pred)

print("The F1 score for SVM is : {}".format(svm_f1_score))
linear_svc_clf = LinearSVC()

linear_svc_clf.fit(x_train, y_train)

y_pred_linear_svc = linear_svc_clf.predict(x_test)

acc_linear_svc =linear_svc_clf.score(x_train, y_train)



print("The accuracy score for SVM is : {}".format(acc_linear_svc))



conf_matrix = confusion_matrix(y_test, y_pred_linear_svc)

plot_confusion_matrix(conf_matrix, classes=['FAKE', 'REAL'])



linear_svm_f1_score = f1_score(y_test, pred)

print("The F1 score for Linear SVM is : {}".format(linear_svm_f1_score))
from sklearn.linear_model import SGDClassifier



sgdc_clf = SGDClassifier(max_iter=5, tol=None)

sgdc_clf.fit(x_train, y_train)

y_pred_sgd = sgdc_clf.predict(x_test)

acc_sgd = sgdc_clf.score(x_train, y_train)



print("The accuracy score for SVM is : {}".format(acc_sgd))



conf_matrix = confusion_matrix(y_test, y_pred_sgd)

plot_confusion_matrix(conf_matrix, classes=['FAKE', 'REAL'])



sgdc_f1_score = f1_score(y_test, pred)

print("The F1 score for SGD is : {}".format(sgdc_f1_score))
models = pd.DataFrame({

    'Model': ['MultinomialNB', 'Decision Tree', 'Random Forest', 'XgBoost', 'K-NN Classifier', 'SVM', 'linear SVM', 'SGD'],

    

    'Score': [acc_score, dtree_score, random_forest_score, xgboost_score, acc_knn, acc_svc, acc_svc, acc_sgd],

    

    'F1 Score' : [multi_f1_score, decision_tree_f1_score, random_forest_f1_score, xgboost_f1_score, knn_f1_score, svm_f1_score,

                 linear_svm_f1_score, sgdc_f1_score]

    })



models.sort_values(by='Score', ascending=False)
sample_submission = pd.read_csv(baseLoc + "sample_submission.csv")

# Predicting model with the test data that was vectorized (test_vectors)

sample_submission['target'] = multi_nb.predict(test_vectors)
sample_submission.to_csv("submission3.csv", index=False)

sample_submission.head(20)