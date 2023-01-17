import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
#Read train data

train_data=pd.read_csv('../input/digit-recognizer/train.csv')

train_data.head()
#No.of Train data's row and column

train_data.shape
#Read test data

test_data=pd.read_csv('../input/digit-recognizer/test.csv')

test_data.head()
#NO. of Test data's row and column

test_data.shape
#viewing the 4th row of train_data

a=train_data.iloc[3,1:].values

a=a.reshape(28,28)

plt.imshow(a)
#Label of 4th row

train_data.iloc[3,0]
x=train_data.iloc[:,1:]

y=train_data.iloc[:,0]
#Creating training and testing samples from train_data with a ratio of 7:3(train:test)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=42)
model=RandomForestClassifier(n_estimators=200,max_samples=0.5)

model.fit(x_train,y_train)
#Predicting the testing sample of train_data

pred=model.predict(x_test)
#accuracy of training sample of train_data

model.score(x_train,y_train)
#accuracy of testing sample of train_data

model.score(x_test,y_test)
confusion_matrix(pred,y_test)
print(classification_report(pred,y_test))
#first 5 values of testing sample of train_data

y_test[0:5]
#first 5 values of predicting samples

pred[0:5]
prediction=model.predict(test_data)

prediction
prediction.shape
#Visualizing the 3rd row of test_data

b=test_data.iloc[2,0:].values

b=b.reshape(28,28)

plt.imshow(b)
#The 3rd value of prediction data

prediction[3]
#Visualizing the 1st row of test_data

b1=test_data.iloc[0,0:].values

b1=b1.reshape(28,28)

plt.imshow(b1)
#1st value of prediction data

prediction[0]
print("Predicted "+ str(y_test.iloc[np.where(y_test!=pred)[0][3]]) + " as "+str(pred[np.where(y_test!=pred)[0][3]]) )

plt.imshow(np.array(x_test.iloc[np.where(y_test!=pred)[0][3]]).reshape(28,28))
np.where(train_data['label']==5)
b=train_data.iloc[51,1:].values

b=b.reshape(28,28)

plt.imshow(b)