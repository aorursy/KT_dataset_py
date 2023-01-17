# importing packages

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import os

# reading test and train datasets
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
test.head()

train_dummies=pd.get_dummies(train)
train1,validate=train_test_split(train_dummies,test_size=0.3,random_state=100)
train_x=train1.drop('label',axis=1)
train_y = train1['label']
validate_x=validate.drop('label',axis=1)
validate_y=validate["label"]
model_rf=RandomForestClassifier(random_state=100,n_estimators=300)
model_rf.fit(train_x,train_y)
validate_pred=model_rf.predict(validate_x)

confusion_matrix(validate_y,validate_pred,labels=[1,2])



Acc=accuracy_score(validate_y,validate_pred)
pred_results_rf=pd.DataFrame({'actual':validate_y,
                           'predicted':validate_pred})
Acc
test_pred = model_rf.predict(test)
df_test_pred = pd.DataFrame(test_pred, columns=['Label'])
df_test_pred['ImageId'] = test.index + 1
df_test_pred[['ImageId','Label']].to_csv("Submission.csv",index=False)