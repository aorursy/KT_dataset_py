# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier





from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,

                             roc_curve, recall_score, classification_report, f1_score,

                             precision_recall_fscore_support, roc_auc_score)







sns.set_style('dark')

sns.set_context('talk')# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



pd.set_option('display.max_colwidth', -1)

        # Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/spam.csv', encoding = 'latin1')

df.info()
df.head()
# last 3 cols have most values Nans. Dropping them

df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace = True)
# Renaming columns

df.columns = ['label', 'text']
pd.set_option('display.width', 1000)
df.head()
df['length'] = df['text'].apply(lambda row: len(row))
sns.distplot(df['length'])

plt.show()
sns.countplot(df['label'])
# Encoding label to 0,1

df['label'] = df['label'].map({'spam' : 1, 'ham' :0})
df.head()
df[df.label == 1].text
df.label.value_counts()
words = ['free', 'winner', 'prize', 'won', 'win']

def count_words(row):

    count = 0

    for word in words:

        if word in row.lower():

            count +=1

    return count
df['bad_words'] = df['text'].apply(count_words )
df.head()
df.groupby(['bad_words', 'label']).count()
sns.heatmap(df.corr(), annot = True)

plt.show()
X = df.drop(['label'], axis=1)

y = df['label']
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=12)
print(X_train.shape, len(y_train))

print(X_test.shape, len(y_test))
vectorizer = TfidfVectorizer(lowercase=True, min_df=20, use_idf=True, )
train_tfidf = vectorizer.fit_transform(X_train.text)

test_tfidf = vectorizer.transform(X_test.text)
print(train_tfidf.shape)

print(test_tfidf.shape)
clf = RandomForestClassifier(n_estimators=10, n_jobs=-1, class_weight='balanced')
clf.fit(train_tfidf, y_train)
y_pred = clf.predict_proba(test_tfidf)[:,1]

y_pred_binary = clf.predict(test_tfidf)
fpr, tpr, thres = roc_curve(y_test, y_pred)

plt.plot(fpr, tpr, label = 'AUC') 

plt.plot([0,1], [0,1], ':', label = 'Random') 

plt.legend() 

plt.grid() 

plt.ylabel("TPR") 

plt.xlabel("FPR") 

plt.title('ROC') 

plt.show()
LABELS = ['Ham', 'Spam']

conf_matrix = confusion_matrix(y_test, y_pred_binary)

cm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

sns.heatmap(cm, xticklabels=LABELS, yticklabels=LABELS, annot=True, cmap='Greens');

plt.title("Confusion matrix")

plt.ylabel('True class')

plt.xlabel('Predicted class')

plt.show()
print("F1-Score : {:.2f}".format(f1_score(y_test, y_pred_binary)))

print("AUC-ROC  : {:.2f}".format(roc_auc_score(y_test, y_pred_binary)))