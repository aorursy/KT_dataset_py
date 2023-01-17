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



# Displaying all columns

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
#import sklearn classes

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import roc_auc_score
#fix seed for ranomization

SEED = 5471242
# Read imput files

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

submission = pd.read_csv("../input/sample_submission.csv")
#show the first rows of train DataFrame

train.head()
# show submission sample with bash command

!head "../input/sample_submission.csv"
# Malware is a column with classification target

Y_train_all = train["Malware"]

# Let drop Malware class labels column and other text columsn

X_train_all = train.drop(["Malware", "Category", "Package"], axis=1)



# Split train dataset into X_train, Y_train, X_val, Y_val

X_train, X_val, Y_train, Y_val = train_test_split(X_train_all,

                                                  Y_train_all,

                                                  test_size=0.20,

                                                  random_state=SEED)
# initiate classifier object

clf = KNeighborsClassifier(n_neighbors=5, weights='uniform', p=2)



# train model on train set

clf.fit(X_train, Y_train)
# predict probablility on validation set 

Y_predicted_val = clf.predict_proba(X_val)



# the Y_predicted_val contains probablility of all labels, 

# the column with index 1 is describing pobablility 

# of package beeing malware

Y_predicted_val_pobablility_of_malware = Y_predicted_val[:,1]
# calulate AUC score on validation set

auc = roc_auc_score(Y_val, Y_predicted_val_pobablility_of_malware)

print(auc)
# init final test classifier

clf = KNeighborsClassifier(n_neighbors=5, weights='uniform', p=2)



#let learn on full data set

clf.fit(X_train_all, Y_train_all)



#prepare test data

knn_submission = pd.DataFrame(test['Package'])

X_test = test.drop(['Category', 'Package'], axis=1)



#predict on test dataset

Y_test_predicted = clf.predict_proba(X_test)



#take probabliliy of malware 

Y_test_predicted_pobablility_of_malware = Y_test_predicted[:,1]



# save submission file

knn_submission['Probability_of_Malware'] = Y_test_predicted_pobablility_of_malware

knn_submission.to_csv('submission_knn_n5.csv', index=False)
#show submission file

!head submission_knn_n5.csv
# please remnember to commit the changes to submit the file