import matplotlib.pyplot as plt

import seaborn as sns
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
%%time

JIGSAW_PATH = "../input/"

train = pd.read_csv(os.path.join(JIGSAW_PATH,'train.csv'), index_col='id')

test = pd.read_csv(os.path.join(JIGSAW_PATH,'test.csv'), index_col='id')
train.head()
train.describe()
print("shape of test - {} and train - {}".format(test.shape, train.shape ))
plt.figure(figsize=(12,6))

plt.title("distribution of target in train set")

sns.distplot(train['target'], kde=True,  hist=True, label='Target')

plt.legend; plt.show()
def plot_features_distribution(features, title):

    plt.figure(figsize=(12,6))

    plt.title(title)

    for feature in features:

        sns.distplot(train[feature],  kde=True,  hist=False, label='Target')

    plt.xlabel('')

    plt.legend()

    plt.show()
features = ['severe_toxicity', 'obscene','identity_attack','insult','threat']

plot_features_distribution(features, "Distribution of additional toxicity features in the train se")
train.groupby(by='rating').count()
#testing with tfidfvectorizer

from sklearn.feature_extraction.text import TfidfVectorizer



Vectorize = TfidfVectorizer(stop_words='english', token_pattern=r'\w{1,}', max_features=35000)
X = Vectorize.fit_transform(train["comment_text"])

y = np.where(train['target'] >= 0.5, 1, 0)



test_X = Vectorize.transform(test["comment_text"])
X.shape
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, cross_val_score



from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LogisticRegression(C=32, dual=False, n_jobs=-2, solver='sag')



lr.fit(X_train, y_train)
y_predict=lr.predict(X_test)

print("Model accuracy ", accuracy_score(y_test, y_predict)*100)
print(classification_report(y_test, y_predict))
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt



fpr, tpr, thr = roc_curve(y_test, lr.predict_proba(X_test)[:,1])

plt.figure(figsize=(10, 8))

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic Plot')

auc = auc(fpr, tpr) * 100

plt.legend(["AUC {0:.3f}".format(auc)]);
predictions = lr.predict_proba(test_X)
sub = pd.read_csv('../input/sample_submission.csv')
sub['prediction'] = predictions

sub.to_csv('submission.csv', index=False)

sub.head()