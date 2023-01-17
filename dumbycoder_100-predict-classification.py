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
data = pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")

df = data.copy()
df.head()
df.info()
# no null data so we're gonna transform object datas to int



from sklearn.preprocessing import LabelEncoder



lbe = LabelEncoder()



for gez in df.columns:

    df[gez] = lbe.fit_transform(df[gez])
df.head(2)
df.info()
# Now, i am gonna groupin' classes 



df.groupby('class').size()



# 1 is poison
# Dependent and Independent Variables



y = df['class']

X = df.drop(['class'], axis = 1)
# correlation



df.corr()
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split, GridSearchCV



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 30)
from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier
lr = LogisticRegression().fit(X_train, y_train)

y_pred = lr.predict(X_test)

accuracy_score(y_test, y_pred)
# tuning the Logistic Regression 



lr = LogisticRegression()

lr_params = {'C':np.arange(1,10,1),

             'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],

            'verbose': [0,1,2]}

lr_cv = GridSearchCV(lr, lr_params, cv = 10, n_jobs=-1).fit(X_train, y_train)
lr_cv.best_params_
lr_tuned = LogisticRegression(C=9, solver='newton-cg', verbose=0).fit(X_train, y_train)

y_pred = lr_tuned.predict(X_test)

accuracy_score(y_test, y_pred)
nb = GaussianNB().fit(X_train, y_train)

y_pred = nb.predict(X_test)

accuracy_score(y_test, y_pred)
knn = KNeighborsClassifier().fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy_score(y_test, y_pred)
knn
# model tuning



knn_params = {'n_neighbors':np.arange(1,10,1),

              'weights': ['uniform', 'distance'],

              'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute']}

knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, knn_params, cv = 10, n_jobs=-1).fit(X_train, y_train)
knn_cv.best_params_
knn_tuned = KNeighborsClassifier(algorithm='brute',n_neighbors=1,weights='uniform').fit(X_train, y_train)

y_pred = knn_tuned.predict(X_test)

accuracy_score(y_test, y_pred)
rf = RandomForestClassifier().fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy_score(y_test, y_pred)
classifiers = [

    KNeighborsClassifier(algorithm='brute',n_neighbors=1,weights='uniform'),

    LogisticRegression(C=9, solver='newton-cg', verbose=0),

    RandomForestClassifier(),

    GaussianNB()]
log_cols=["Classifier", "Accuracy"]

log = pd.DataFrame(columns=log_cols)
for clf in classifiers:

    clf.fit(X_train, y_train)

    name = clf.__class__.__name__

    

    print("="*30)

    print(name)

    

    print('****Results****')

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("Accuracy: {:.4%}".format(acc))

    

    log_entry = pd.DataFrame([[name, acc*100]], columns=log_cols)

    log = log.append(log_entry)

    

print("="*30)
import seaborn as sns

import matplotlib.pyplot as plt



sns.set_color_codes("muted")

sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")



plt.xlabel('Accuracy %')

plt.title('Classifier Accuracy')

plt.show()