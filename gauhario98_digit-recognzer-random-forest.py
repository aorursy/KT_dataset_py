import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#read train and test files

train_file = pd.read_csv('../input/train.csv')
test_file = pd.read_csv('../input/test.csv')
#view first 5 records of train_file
train_file.head()
#view first 5 records of test_file
test_file.head()
#list of all digits that are going to be predicted
np.sort(train_file.label.unique())
#define the number of samples for training set and for validation set
num_train,num_validation = int(len(train_file)*0.8),int(len(train_file)*0.2)
num_train,num_validation
#generate training data from train_file
x_train,y_train=train_file.iloc[:num_train,1:].values,train_file.iloc[:num_train,0].values
x_validation,y_validation=train_file.iloc[num_train:,1:].values,train_file.iloc[num_train:,0].values
print(x_train.shape)
print(y_train.shape)
print(x_validation.shape)
print(y_validation.shape)
index=3
print("Label: " + str(y_train[index]))
plt.imshow(x_train[index].reshape((28,28)),cmap='gray')
plt.show()
#fit a Random Forest classifier
clf=RandomForestClassifier()
clf.fit(x_train,y_train)
#predict value of label using classifier
prediction_validation = clf.predict(x_validation)

print("Validation Accuracy: " + str(accuracy_score(y_validation,prediction_validation)))
x_test=test_file
#predict test data
prediction_test = clf.predict(x_test)
index=5
print("Predicted " + str(prediction_test[index]))
plt.imshow(x_test.iloc[index].values.reshape((28,28)),cmap='gray')

