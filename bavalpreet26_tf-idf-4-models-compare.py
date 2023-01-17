# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

from collections import Counter
from imblearn.over_sampling import SMOTE
import matplotlib                  # 2D Plotting Library
import matplotlib.pyplot as plt
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
df = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
import seaborn as sns
fig, plots = plt.subplots(2,3,figsize=(18,12))
plot1, plot2, plot3, plot4, plot5, plot6 = plots.flatten()
sns.countplot(df['obscene'], palette= 'deep', ax = plot1)
sns.countplot(df['threat'], palette= 'muted', ax = plot2)
sns.countplot(df['insult'], palette = 'pastel', ax = plot3)
sns.countplot(df['identity_hate'], palette = 'dark', ax = plot4)
sns.countplot(df['toxic'], palette= 'colorblind', ax = plot5)
sns.countplot(df['severe_toxic'], palette= 'bright', ax = plot6)

rslt_df = df[(df['toxic'] == 0) & (df['severe_toxic'] == 0) & (df['obscene'] == 0) & (df['threat'] == 0) & (df['insult'] == 0) & (df['identity_hate'] == 0)]
rslt_df2 = df[(df['toxic'] == 1) & (df['severe_toxic'] == 0) & (df['obscene'] == 0) & (df['threat'] == 0) & (df['insult'] == 0) & (df['identity_hate'] == 0)]
new1 = rslt_df[['id', 'comment_text', 'toxic']].iloc[:23891].copy() 
new2 = rslt_df2[['id', 'comment_text', 'toxic']].iloc[:946].copy()
new = pd.concat([new1, new2], ignore_index=True)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.95, min_df=5)
Xv = vectorizer.fit(new['comment_text'])
import pickle
#test train split
import numpy as np
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new["comment_text"], new['toxic'], test_size=0.33)
from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.95, min_df=5)
X1 = vectorizer.transform(X_train)
X_test1= vectorizer.transform(X_test)
print('Original dataset shape %s' % Counter(y_train))
sm = SMOTE(random_state=12)
x_train_res, y_train_res = sm.fit_sample(X1, y_train)
print('Resampled dataset shape %s' % Counter(y_train_res))
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
clf2 = LogisticRegression(C=0.1, solver='sag')
scores = cross_val_score(clf2, x_train_res,y_train_res, cv=5,scoring='f1_weighted')
scores
y_p1 = clf2.fit(x_train_res, y_train_res).predict(X_test1)
from sklearn.metrics import accuracy_score

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, y_p1)
print('Accuracy: %f' % accuracy)
import numpy as np

z=1.96
interval = z * np.sqrt( (0.908137 * (1 - 0.908137)) / y_test.shape[0])
interval
from sklearn.svm import SVC
from sklearn import svm
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf,x_train_res,y_train_res, cv=5)
scores
from sklearn.svm import SVC

y_p2 = clf.fit(x_train_res, y_train_res).predict(X_test1)
from sklearn.metrics import accuracy_score

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, y_p2)
print('Accuracy: %f' % accuracy)
import numpy as np

z=1.96
interval = z * np.sqrt( (0.963279 * (1 - 0.963279)) / y_test.shape[0])
interval
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

clf3 = RandomForestClassifier() #Initialize with whatever parameters you want to

# 10-Fold Cross validation
scores = cross_val_score(clf3,x_train_res,y_train_res, cv=5)
scores
y_p3 = clf3.fit(x_train_res, y_train_res).predict(X_test1)
accuracy = accuracy_score(y_test, y_p3)
print('Accuracy: %f' % accuracy)
import numpy as np

z=1.96
interval = z * np.sqrt( (0.9629 * (1 - 0.9629)) / y_test.shape[0])
interval
from sklearn.naive_bayes import MultinomialNB
clf4 = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
scores = cross_val_score(clf4,x_train_res,y_train_res, cv=5)
y_pred4 = clf4.fit(x_train_res, y_train_res).predict(X_test1)
scores
accuracy = accuracy_score(y_test, y_pred4)
print('Accuracy: %f' % accuracy)
import numpy as np

z=1.96
interval = z * np.sqrt( (0.893376 * (1 - 0.893376)) / y_test.shape[0])
interval