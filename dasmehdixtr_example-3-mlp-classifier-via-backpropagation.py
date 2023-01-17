from sklearn.neural_network import MLPClassifier

import pandas as pd

import numpy as np



data_train = pd.read_table("/kaggle/input/data-to-work-on-basic-examples/class3_tr.txt",header=None)

data_test = pd.read_table("/kaggle/input/data-to-work-on-basic-examples/class3_test.txt",header=None)
x_train = data_train.loc[:,:1]

y_train = data_train.loc[:,2:4]

x_test = data_test.loc[:,:1]

y_test = data_test.loc[:,2:4]
model = MLPClassifier(hidden_layer_sizes=(300,),early_stopping=True, verbose=1,activation='tanh')

model.fit(x_train,y_train)
print("Train Accuracy:  ",model.score(x_train,y_train))

print("Test Accuracy: ",model.score(x_test,y_test))

print('Classes: ', model.classes_)
from sklearn.metrics import classification_report

predict_test = model.predict(x_test)

print(classification_report(y_test,predict_test))