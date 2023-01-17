import pandas as pd

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import LabelEncoder

import scipy.sparse as sp

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn import metrics

from sklearn.metrics import accuracy_score,recall_score,confusion_matrix

from sklearn.model_selection import KFold
df = pd.read_csv("../input/data.csv", header=None, dtype=str)
df['label'] = df[df.shape[1]-1]

df.drop([df.shape[1]-2], axis=1, inplace=True)



X = np.array(df.drop(['label'], axis = 1), dtype='<U13')

y = np.array(df['label'])



X = X[1:,2:]

y = y[1:]

kf = kf = KFold(n_splits=2)

kf.get_n_splits(X)

y = list(map(int, y))





X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state= 100)



        

X_train = list(X_train)

new_X_train = []

for i in X_train:

    for j in i:

        new_X_train.append(j)     

        

X_test = list(X_test)

new_X_test = []

for i in X_test:

    for j in i:

        new_X_test.append(j)

        

        

        

count_vect = CountVectorizer(lowercase=False)

X_train_counts = count_vect.fit_transform(new_X_train)

X_test_counts = count_vect.transform(new_X_test)



rf = RandomForestClassifier()

rf.fit(X_train_counts,y_train)

y_pred = rf.predict(X_test_counts)

acc = accuracy_score(y_test,y_pred)

print ("Accuracy", float("{0:.2f}".format(acc*100)))