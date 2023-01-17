# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
digit_recognisation = pd.read_csv("../input/train.csv")
digit_recognisation_test = pd.read_csv("../input/test.csv")
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# Any results you write to the current directory are saved as output.
digit_locker = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
train1,validate = train_test_split(digit_locker,test_size = 0.3,random_state = 100)
train_y = train1['label']
train_x = train1.drop('label',axis = 1)
validate_y = validate['label']
validate_x = validate.drop('label',axis = 1)
model = KNeighborsClassifier()
model.fit(train_x,train_y)
validate_predict = model.predict(validate_x)
print(accuracy_score(validate_y,validate_predict))
test_pred = model.predict(test_data)
df_test_predict = pd.DataFrame(test_pred,columns = ['Label'])
df_test_predict['ImageId'] = test_data.index+1
df_test_predict[['ImageId','Label']].to_csv('submission_1.csv',index = False)

digit_locker = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
train1,validate = train_test_split(digit_locker,test_size = 0.3,random_state = 100)
train_y = train1['label']
train_x = train1.drop('label',axis = 1)
validate_y = validate['label']
validate_x = validate.drop('label',axis = 1)
model_random = RandomForestClassifier(random_state = 100)
model_random.fit(train_x,train_y)
validate_predict = model_random.predict(validate_x)
print(accuracy_score(validate_y,validate_predict))
test_pred = model_random.predict(test_data)
df_test_predict = pd.DataFrame(test_pred,columns = ['Label'])
df_test_predict['ImageId'] = test_data.index+1
df_test_predict[['ImageId','Label']].to_csv('submission_1.csv',index = False)
digit_locker = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
train1,validate = train_test_split(digit_locker,test_size = 0.3,random_state = 100)
train_y = train1['label']
train_x = train1.drop('label',axis = 1)
validate_y = validate['label']
validate_x = validate.drop('label',axis = 1)
model_ada= AdaBoostClassifier(random_state = 100)
model_ada.fit(train_x,train_y)
validate_predict = model_ada.predict(validate_x)
print(accuracy_score(validate_y,validate_predict))
test_pred = model_ada.predict(test_data)
df_test_predict = pd.DataFrame(test_pred,columns = ['Label'])
df_test_predict['ImageId'] = test_data.index+1
df_test_predict[['ImageId','Label']].to_csv('submission_1.csv',index = False)
digit_locker = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
train1,validate = train_test_split(digit_locker,test_size = 0.3,random_state = 100)
train_y = train1['label']
train_x = train1.drop('label',axis = 1)
validate_y = validate['label']
validate_x = validate.drop('label',axis = 1)
model_dt= DecisionTreeClassifier(random_state = 100)
model_dt.fit(train_x,train_y)
validate_predict = model_dt.predict(validate_x)
print(accuracy_score(validate_y,validate_predict))
test_pred = model_dt.predict(test_data)
df_test_predict = pd.DataFrame(test_pred,columns = ['Label'])
df_test_predict['ImageId'] = test_data.index+1
df_test_predict[['ImageId','Label']].to_csv('submission_1.csv',index = False)

