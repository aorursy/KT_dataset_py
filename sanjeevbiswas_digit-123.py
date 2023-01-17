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
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
sample_data = pd.read_csv("../input/sample_submission.csv")





train_data.head()



print(train_data.shape)

print(test_data.shape)


from sklearn.model_selection import train_test_split
train_new,validate = train_test_split(train_data,test_size=0.3,random_state=100)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
train_x = train_new.drop('label',axis=1)
train_y = train_new['label']

test_x = validate.drop('label',axis=1)
test_y = validate['label']

model = DecisionTreeClassifier(random_state=100,max_depth=5)
model.fit(train_x,train_y)

pred_test = model.predict(test_x)

print(accuracy_score(test_y,pred_test))
print(classification_report(pred_test, test_y))

from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(random_state = 100,
                                 n_estimators=300)
model_rf.fit(train_x,train_y)
pred = model_rf.predict(test_x)

print(accuracy_score(test_y,pred))
print(classification_report(test_y,pred))
from sklearn.ensemble import AdaBoostClassifier
model_ab = AdaBoostClassifier(random_state = 100,
                                 n_estimators=300)
model_ab.fit(train_x,train_y)
pred_ab = model_ab.predict(test_x)

print(accuracy_score(test_y,pred_ab))
print(classification_report(test_y,pred_ab))
model_rf = RandomForestClassifier(random_state = 100,
                                 n_estimators=300)
model_rf.fit(train_x,train_y)
pred = model_rf.predict(test_data)

pred = pd.DataFrame({'ImageId':range(1,28001),'Label':pred})
pred.to_csv('Predictions.csv',index=False)
