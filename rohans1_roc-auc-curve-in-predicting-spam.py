# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dir = '../input/sms-spam-collection-dataset/spam.csv'

import pandas as pd

df = pd.read_csv(dir, encoding='ISO-8859-1')

df.head()
import numpy as np

y = np.array([(1 if i=='spam' else 0) for i in df.v1.tolist()])

X = np.array(df.v2.tolist())
from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(

    n_splits=1, test_size=0.3, random_state=0)

for train_index, test_index in splitter.split(X, y):

    X_train_pre_vectorize, X_test_pre_vectorize = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

X_train = vectorizer.fit_transform(X_train_pre_vectorize)

X_test = vectorizer.transform(X_test_pre_vectorize)
from sklearn.linear_model import LogisticRegression



classifier = LogisticRegression()

classifier.fit(X_train, y_train)
y_score = classifier.predict_proba(X_test)

y_score = np.array(y_score)

print(y_score)
from sklearn.preprocessing import label_binarize

y_test_bin = label_binarize(y_test, neg_label=0, pos_label=1, classes=[0,1])

y_test_bin = np.hstack((1 - y_test_bin, y_test_bin))

print(y_test_bin)
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

fpr = dict()

tpr = dict()

roc_auc = dict()

for i in [0,1]:

    # collect labels and scores for the current index

    labels = y_test_bin[:, i]

    scores = y_score[:, i]

    

    # calculates FPR and TPR for a number of thresholds

    fpr[i], tpr[i], thresholds = roc_curve(labels, scores)

    

    # given points on a curve, this calculates the area under it

    roc_auc[i] = auc(fpr[i], tpr[i])
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())

roc_auc['micro'] = auc(fpr["micro"], tpr["micro"])
plt.figure()

lw = 2

plt.plot(fpr[1], tpr[1], color='darkorange',

         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
