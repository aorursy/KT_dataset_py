# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")

# Any results you write to the current directory are saved as output.
# splitting train data into train and train1
from sklearn.model_selection import train_test_split
train1, validate = train_test_split(train,test_size=0.3 ,random_state = 100)

# Input and Output variables
train_x = train1.drop('label', axis=1)
train_y = train1['label']
validate_x = validate.drop('label', axis=1)
validate_y = validate['label']

# Decision tree
from sklearn.tree import DecisionTreeClassifier
model1 = DecisionTreeClassifier(max_depth=5)
model1.fit(train_x, train_y)
test_pred1 = model1.predict(validate_x)
df1 = pd.DataFrame({'actual':validate_y, 'pred':test_pred1})
df1['status'] = df1['actual'] == df1['pred']
df1['status'].value_counts()/df1['status'].shape[0]
# Random forest classifier
from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier()
model2.fit(train_x, train_y)
test_pred2 = model2.predict(validate_x)
df2 = pd.DataFrame({'actual':validate_y, 'pred':test_pred2})
df2['status'] = df2['actual'] == df2['pred']
df2['status'].value_counts()/df2['status'].shape[0]
# AdaBoost classifier
from sklearn.ensemble import AdaBoostClassifier
model3 = AdaBoostClassifier()
model3.fit(train_x, train_y)
test_pred3 = model3.predict(validate_x)
df3 = pd.DataFrame({'actual':validate_y, 'pred':test_pred3})
df3['status'] = df3['actual'] == df3['pred']
df3['status'].value_counts()/df3['status'].shape[0]
# K Nearest Neibours
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(train_x, train_y)
test_pred = model.predict(validate_x)
df = pd.DataFrame({'actual':validate_y, 'pred':test_pred})
df['status'] = df['actual'] == df['pred']
df['status'].value_counts()/df['status'].shape[0]
test = pd.read_csv("../input/test.csv")
# Applying Random Forest on test data
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(train_x, train_y)
pred = model.predict(test)
index = test.index + 1
df = pd.DataFrame({'ImageId':index, 'Label':pred})
import pandas as pd
df.to_csv("prediction.csv", index = False)
