import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
#read train data

train_data=pd.read_csv('../input/digit-recognizer/train.csv')

train_data.head()
train_data.shape
train_data.info()
#read test data

test_data=pd.read_csv('../input/digit-recognizer/test.csv')

test_data.head()
test_data.shape
test_data.info()
label=train_data.iloc[2,1:].values

label=label.reshape(28,28)

plt.imshow(label)
#Label of 4th row

train_data.iloc[2,0]
X=train_data.iloc[:,1:]

Y=train_data.iloc[:,0]
#Creating training and testing samples from train_data with a ratio of 7:3(train:test)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,Y,train_size=0.7,random_state=42)
model=RandomForestClassifier(n_estimators=200,max_samples=0.5)

model.fit(x_train,y_train)
pred=model.predict(x_test)

pred
pred.shape
#accuracy of training sample of train_data

model.score(x_train,y_train)
#accuracy of testing sample of train_data

model.score(x_test,y_test)
confusion_matrix(pred,y_test)
print(classification_report(pred,y_test))
#random values of testing sample of train_data

y_test[5:10]
pred[5:10]
test_data_prediction=model.predict(test_data)

test_data_prediction
test_data_prediction.shape
#Visualizing the 39th row of test_data

b=test_data.iloc[39,0:].values

b=b.reshape(28,28)

plt.imshow(b)
#The 39th value of prediction data

test_data_prediction[39]
print("Predicted "+ str(y_test.iloc[np.where(y_test!=pred)[0][8]]) + " as "+str(pred[np.where(y_test!=pred)[0][8]]) )

plt.imshow(np.array(x_test.iloc[np.where(y_test!=pred)[0][8]]).reshape(28,28))
np.where(train_data['label']==9)
b=train_data.iloc[28,1:].values

b=b.reshape(28,28)

plt.imshow(b)