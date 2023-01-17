"""

@author: Alok Kumar Sahu

@Linkedin : www.linkedin.com/in/alokkrsahu

"""

import os

import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt



path = ("../input/st4035-2019-assignment-1/")

train = pd.read_csv(os.path.join(path,"vehicle_train.csv"))

test = pd.read_csv(os.path.join(path,"vehicle_test.csv"))

label = pd.read_csv(os.path.join(path,"vehicle_training_labels.csv"))
train = train.drop(['ID'],axis =1)

test = test.drop(['ID'],axis =1)
logreg = LogisticRegression()

logreg.fit(train, label)
svm = pd.DataFrame(logreg.predict(test))

svm.index.name = 'ID' 

svm.index += 1

svm.to_csv('./Submission.csv', index = True,header=['Class'])