# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from sklearn import metrics

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.







import pandas as pd

import numpy as np

from IPython.display import display

import matplotlib.pyplot as plt

from sklearn.utils import shuffle









#load shuffled train dataset

train_data = shuffle(pd.read_csv(r"../input/train.csv"))



#load shuffled testing dataset

test_data = shuffle(pd.read_csv(r"../input/test.csv"))



#sample dataset

display(train_data.head())



#shape of train and test datasets

print(train_data.shape)

print(test_data.shape)



# Label Data

training_labels = train_data['Activity']

testing_labels = test_data['Activity']





# Data for the subject [Who carried the observations]

subject_train_data = train_data['subject']

subject_test_data = test_data['subject']





#Drop Subject and Acivity from data

train_data = train_data.drop(['subject', 'Activity'], axis=1)

test_data = test_data.drop(['subject', 'Activity'], axis=1)



from sklearn import preprocessing

from keras.utils.np_utils import to_categorical







#Label Encoded format

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le = le.fit(["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"])

training_labels_LE = le.transform(training_labels)

testing_labels_LE = le.transform(testing_labels)



# One Hot Encoded format

training_labels_C = to_categorical(training_labels_LE)

test_labels__C = to_categorical(testing_labels_LE)





#########################Tuning Hyper Parameter with Accuracy Metrics[Test Data]#############################################

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

from sklearn import svm



model = svm.SVC() 





tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],

                     'C': [1, 10, 100, 1000]},

                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

score = 'accuracy'

print("Tuning Hyper Parameter with Accuracy Metrics")

clf = GridSearchCV(model, tuned_parameters, cv=5, scoring=score)

clf.fit(train_data,training_labels_LE)

print("Report:")

y_true, y_pred = testing_labels_LE, clf.predict(test_data)

print(classification_report(y_true, y_pred))

#########################Tuning Hyper Parameter with Accuracy Metrics[Test Data]#############################################

prediction = clf.predict(test_data)

    

print(metrics.accuracy_score(y_true,prediction))