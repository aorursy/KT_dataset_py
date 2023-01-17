# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
test_df = pd.read_csv('../input/test.csv')

train_df = pd.read_csv('../input/train.csv')
train_df.head()

#test_df.head()
# put dataset into array

X = train_df.drop(['id', 'species'], axis=1).values

y = train_df['species'].values

test = test_df.drop(['id'], axis=1).values
#preprocessing data and using KNeighbors Classifier

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import LabelEncoder



le = LabelEncoder().fit(train_df.species) 

labels = le.transform(train_df.species)           # encode species strings

classes = list(le.classes_)  



X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

                 

clf = KNeighborsClassifier(n_neighbors=4)

clf.fit(X_train, y_train)



# record the accuracy

training_accuracy = clf.score(X_train, y_train)

test_accuracy = clf.score(X_test, y_test)

    

print("Training Accuracy: ", training_accuracy)

print("Test Accuracy: ", test_accuracy)
# predict the test data

y_pred = clf.predict_proba(test)



y_pred
# Format DataFrame

submission = pd.DataFrame(y_pred, columns=classes)

submission.insert(0, 'id', test_df.id)

submission.reset_index()



# Export Submission

submission.to_csv('submission.csv', index = False)

submission.tail()