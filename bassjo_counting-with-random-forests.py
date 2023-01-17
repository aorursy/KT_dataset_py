#Import necessary packages

import pandas as pd

import numpy as np

import sklearn

import matplotlib.pyplot as plt

import math

import random as rand



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split,GridSearchCV
filepath="../input/"

train_path = filepath+"train.csv"

training = pd.read_csv(train_path)

training.describe()
#Based on this, the picture size will be 28x28

size=int(math.sqrt(training.shape[1]-1))

size
#Pull a random integer between 0 and number of rows in dataaset to test

rand_num =rand.randint(0,training.shape[0])



#Create array based on pulling row corresponding to random int

number=np.array(training.loc[rand_num][1:],dtype='uint8')



#Create 2-D array based on image size

number=number.reshape((size,size))



print("Labeled value is",str(training.loc[rand_num][0]))

plt.imshow(number,cmap='Greys')

plt.show()
training_labels=training['label']

training_without_labels=training.drop(labels='label',axis=1)

train_data, test_data, train_labels, test_labels = train_test_split(training_without_labels, training_labels, test_size=0.7)
#rf = RandomForestClassifier()

parameters = {'n_estimators':[100,200],'max_features':['auto','sqrt',None]}



#gs_cv_rf = GridSearchCV(rf,parameters,cv=3)

#gs_cv_rf.fit(train_data,train_labels)

#gs_cv_rf.best_params_

#best params: max_features = 'sqrt', n_estimators = 200



rf = RandomForestClassifier(n_estimators=200,max_features='sqrt')

rf.fit(train_data,train_labels)
predict=gs_cv_rf.predict(test_data)

check=pd.DataFrame(predict,columns=["Predict"])

check['true']=test_labels.reset_index()['label']



check['Accuracy']=0

check['Accuracy']=check['Accuracy'].where(check['Predict']!=check['true'],1)

print("Accuracy: ",(check.sum()[2]/check.shape[0]).round(3))

check.head()
check['true'][check['Accuracy']==0].value_counts()
#The random forest seems to think that 4s and 3s are often 9s

plt.hist(check['Predict'][(check['Accuracy']==0)&(check['true']==9)])
test_path = filepath+"test.csv"

testing = pd.read_csv(test_path)

solutions=rf.predict(testing)

submission=pd.DataFrame(solutions,columns=['Label'])

submission.index.name = "ImageID"

submission.index += 1



submission.to_csv('submission.csv')