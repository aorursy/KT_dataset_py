# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Reviews.csv',nrows=100000)

#df = df_full[:10000]

print("Dimension",df.shape)

df.head()
df.describe()
# Calculating length of reviews and adding it as a column

df['TextLength'] = df['Text'].apply(lambda x: len(x)-x.count(' '))

df.head()
df.Score.value_counts().sort_index().plot.bar(alpha=0.7, grid=True, color = 'seagreen', width = 0.9)

plt.xlabel('Score')

plt.ylabel('Number Of Reviews')

plt.title('Distribution of reviews over each score')

plt.show()
# How many empty length texts are there ?

df[df['TextLength']==0].Text.count()
plt.scatter(df['Score'], df['TextLength'])

plt.xlabel('Score')

plt.ylabel('Text Lengths')

plt.title('Relation B/W Text Length and Scores')

plt.show()
sns.pairplot(df)

plt.show()
import nltk

import string

from nltk.stem import PorterStemmer

import re

from nltk.corpus import stopwords

stopwords_en = set(stopwords.words('english'))

ps = PorterStemmer()
def clean_text(text):

    text = text.lower() #converting text to lowercase

    text = ' '.join([i for i in nltk.word_tokenize(text) if i not in stopwords_en and i not in string.punctuation]) #stopword and punct removal

    text = re.sub('[^a-z]+', ' ', text) #removal of anything other than English letters

    text = ' '.join([ps.stem(i) for i in nltk.word_tokenize(text)]) #stemming

    return text
# Apply the cleanup and create a new column

df['CleanText'] = df['Text'].apply(lambda x: clean_text(x))

df.head(3)
def partition(val):

    if(val>2):

        return 1

    return 0

df['Positivity']=df['Score'].apply(lambda x: partition(x))

df.head(3)
required_columns = ['CleanText', 'Positivity']

df = df[required_columns]

df.head()
df.Positivity.value_counts().plot.bar(alpha=0.5, grid=True)

plt.title('Distribution Of Positive & Negative Reviews')

plt.ylabel('Counts')

plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['CleanText'], df['Positivity'], test_size=0.25, random_state=42, shuffle=True, stratify=df['Positivity'])

print("Dataset Splitted ... \nTrain Set Size = {}\nTest Set Size  = {}".format(X_train.shape[0], X_test.shape[0]))
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectoriser = TfidfVectorizer()



# Training % Feature Extraction On Entire Dataset, Used For Cross Validation & Model Comparison

features = tfidf_vectoriser.fit_transform(df['CleanText'])

labelss = df['Positivity'].astype(int)
# Training On Only Train Set Now

tfidf_vectoriser.fit(X_train)

X_train_tf = tfidf_vectoriser.transform(X_train)

X_test_tf = tfidf_vectoriser.transform(X_test)

X_train_tf.shape, X_test_tf.shape
import random

print("Ten Random Words from Training Set ...\n",*random.sample(tfidf_vectoriser.get_feature_names(),10))
from sklearn.feature_selection import chi2

chi2score = chi2(X_train_tf, y_train)[0]

plt.figure(figsize=(16,8))

scores = list(zip(tfidf_vectoriser.get_feature_names(), chi2score))

chi2 = sorted(scores, key=lambda x:x[1])

topchi2 = list(zip(*chi2[-20:]))

x = range(len(topchi2[1]))

labels = topchi2[0]

plt.barh(x,topchi2[1], align='center', alpha=0.5)

plt.plot(topchi2[1], x, '-o', markersize=5, color='r')

plt.yticks(x, labels)

plt.xlabel('$\chi^2$')

plt.show();
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score
models = [

    LinearSVC(),

    MultinomialNB(),

    LogisticRegression(random_state=0),

    RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)

]

CV = 5

cv_df = pd.DataFrame(index=range(CV * len(models)))

entries = []

for model in models:

    model_name = model.__class__.__name__

    accuracies = cross_val_score(model, features ,labelss, scoring='accuracy', cv=CV)

    for fold_idx, accuracy in enumerate(accuracies):

        entries.append((model_name, fold_idx, accuracy))

cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
plt.figure(figsize=(10,6))

sns.boxplot(x='model_name', y='accuracy', data=cv_df)

sns.stripplot(x='model_name', y='accuracy', data=cv_df, 

              size=8, jitter=True, edgecolor="gray", linewidth=2)

plt.show()
from sklearn.model_selection import GridSearchCV

param_grid = {

    'C':[0.5,0.8,1.0,1.5]

}

svm = LinearSVC(max_iter=1500)

svm_cv = GridSearchCV(svm, param_grid, cv=5)

svm_cv.fit(features, labelss)

print("Best Parameters :", svm_cv.best_params_)

print("Best Score :",svm_cv.best_score_)
svm = LinearSVC(C=0.5, max_iter=2000)

svm.fit(X_train_tf, y_train)
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve

y_pred = svm.predict(X_test_tf)

print(classification_report(y_test, y_pred, target_names=['Negative','Positive']))

print("Accuracy :",accuracy_score(y_test, y_pred), end='\n\n')

conf_mat = confusion_matrix(y_test, y_pred)

print(conf_mat)
fig = plt.figure(figsize=(6,6))

ax = fig.add_subplot(111)

cax = ax.matshow(conf_mat, alpha=0.8)

plt.title('Confusion Matrix of the Linear SVC Classifier')

fig.colorbar(cax)

plt.xlabel('Predicted')

plt.ylabel('Actual')

plt.show()
cm_normalized = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

fig = plt.figure(figsize=(7,7))

ax = fig.add_subplot(111)

cax = ax.matshow(cm_normalized, alpha=0.8)

plt.title('Normalized Confusion Matrix of the Linear SVC Classifier')

fig.colorbar(cax)

plt.xlabel('Predicted')

plt.ylabel('Actual')

plt.show()
from sklearn.calibration import CalibratedClassifierCV

cclf = CalibratedClassifierCV(base_estimator=svm, cv="prefit")

cclf.fit(X_train_tf, y_train)
y_pred_prob = cclf.predict_proba(X_test_tf)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr, label='Linear SVC')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Linear SVC ROC Curve')

plt.show();

print("ROC AUC Score :", roc_auc_score(y_test, y_pred_prob))