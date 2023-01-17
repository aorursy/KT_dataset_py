import csv
import pandas as pd
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
train=pd.read_csv("../input/7e966ce0-e-dataset/dataset/train.csv")
test=pd.read_csv("../input/7e966ce0-e-dataset/dataset/test.csv")
print(train.columns)
train['VIOLATION CODE'].fillna("00A", inplace=True)
test['VIOLATION CODE'].fillna("00A", inplace=True)
mapping_dictionary = {'CRITICAL FLAG':{ "Critical": 1, "Not Critical": 0,"Not Applicable": 2}}
train = train.replace(mapping_dictionary)
features_temp=train.values[:,[11]]
length=len(train.values)
for i in range(0,length):
    s=str(features_temp[i])
    features_temp[i] = (ord(s[2])-ord('0')) * 100 + (ord(s[3])-ord('0')) * 10 + (ord(s[4])-ord('A'))
features=features_temp
labels=train.values[:,[19]]
labels=labels.flatten()
labels=labels.astype('int')
from sklearn.linear_model import LogisticRegression
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.33)
model = LogisticRegression(multi_class='auto',solver='lbfgs')
print("TRAINING THE MODEL")
model.fit(features_train,labels_train)
print("MODEL SUCCESSFULLY TRAINED")
pred = model.predict(features_test)
print("Accuracy Score : ", accuracy_score(labels_test,pred))
features_test_final=test.values[:,[11]]
length=len(test.values)
for i in range(0,length):
    s=str(features_test_final[i])
    features_test_final[i] = (ord(s[2])-ord('0')) * 100 + (ord(s[3])-ord('0')) * 10 + (ord(s[4])-ord('A'))
features_test_final
pred=model.predict(features_test_final)
print("finished")
mapping_dictionary2 = {'CRITICAL FLAG':{ 1: "Critical", 0: "Not Critical", 2: "Not Applicable"}}
final=pd.DataFrame({'UIDX':test['UIDX'],'CRITICAL FLAG':pred})
final = final.replace(mapping_dictionary2)
final.to_csv("submission.csv",index=False)
ls