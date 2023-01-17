

"""

@author: Alok Kumar Sahu

@email : alok.kr.sahu@outlook.com

@Linkedin : www.linkedin.com/in/alokkrsahu

"""

import os

import pandas as pd

from sklearn.ensemble import RandomForestClassifier



path = ("../input/st4035-2019-assignment-1/")

train = pd.read_csv(os.path.join(path,"vehicle_train.csv"))

test = pd.read_csv(os.path.join(path,"vehicle_test.csv"))

label = pd.read_csv(os.path.join(path,"vehicle_training_labels.csv"))

train = train.drop(['ID'],axis = 1)

test = test.drop(['ID'],axis = 1)
clf = RandomForestClassifier()

clf.fit(train,label)
rand_forest = pd.DataFrame(clf.predict(test))

rand_forest.index.name = 'ID' 

rand_forest.index += 1

rand_forest.to_csv('./Submission.csv', index = True,header=['Class'])