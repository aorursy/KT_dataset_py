# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
digit = pd.read_csv("../input/train.csv")
####Import all required packages from sckit learn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
###Split the data into train and test
train1,validate = train_test_split(digit,test_size=0.3,random_state=100) 
##Drop label from input variables
train_x = train1.drop('label', axis=1)
train_y = train1['label']

test_x = validate.drop('label', axis=1)
test_y = validate['label']


# Any results you write to the current directory are saved as output.


## Decision Tree
##Fit the decision tree model
model = DecisionTreeClassifier(random_state=100,max_depth=15)
model.fit(train_x, train_y)

##Predict the decision tree model

test_pred = model.predict(test_x)
##Accuracy of decision tree
dt_acc =accuracy_score(test_y,test_pred)
dt_acc

## Random Forest
##Fit the random forest model
model_randomforest = RandomForestClassifier(random_state=100,n_estimators=200)
model_randomforest.fit(train_x, train_y)
##Predict the random forest model
test_pred_rf = model_randomforest.predict(test_x)
##Accuracy of random forest 
rf_acc = accuracy_score(test_y,test_pred_rf)
rf_acc

## AdaBoost

model_ab  = AdaBoostClassifier(random_state=100)
model_ab.fit(train_x,train_y)
test_pred = model_ab.predict(test_x)
df_pred2 = pd.DataFrame({'actual' : test_y,
                       'predicted' : test_pred})
df_pred2['pred_status'] = df_pred2['actual'] == df_pred2['predicted']
acc_ab = df_pred2['pred_status'].sum() / df_pred2.shape[0] * 100
acc_ab
test = pd.read_csv("../input/test.csv")
test_digit_recognizer = model_randomforest.predict(test)

df_predict_digit = pd.DataFrame(test_digit_recognizer,columns=['Label'])
df_predict_digit['ImageId'] = test.index + 1
df_predict_digit.head(10)
### writing it to csv file

df_predict_digit[['ImageId','Label']].to_csv('ML-Assignment1.csv',index=False)