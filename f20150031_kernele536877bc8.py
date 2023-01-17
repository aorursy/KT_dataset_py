# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
benign=pd.read_csv('../input/opcode_frequency_benign.csv')

malware=pd.read_csv('../input/opcode_frequency_malware.csv')

test=pd.read_csv('../input/Test_data.csv')
benign['y']=0

malware['y']=1


FileName=test['FileName']

test=test.drop(columns=['FileName','Unnamed: 1809'])

test.head()
data=pd.concat([benign,malware])
data.head()
y=data['y']

X=data.drop(columns=['FileName','y'])
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=10,stratify=y)
from sklearn.linear_model import LogisticRegression

model=LogisticRegression()

model.fit(X_train,y_train)

y_reg=model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_reg)
from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeClassifier(max_depth=35,max_features=300)

model.fit(X_train,y_train)

y_reg=model.predict(X_test)

accuracy_score(y_test,y_reg)

from sklearn.ensemble  import RandomForestClassifier

model=RandomForestClassifier()

model.fit(X_train,y_train)

y_reg=model.predict(X_test)

accuracy_score(y_test,y_reg)
import matplotlib.pyplot as plt
# class Perceptron:

  

#     def __init__ (self):

#         self.w = None

#         self.b = None

    

#     def model(self, x):

#         return 1 if (np.dot(self.w, x) >= self.b) else 0

    

#     def predict(self, X):

#         Y = []

#         for x in X:

#             result = self.model(x)

#             Y.append(result)

#         return np.array(Y)

    

#     def fit(self, X, Y, epochs = 1, lr = 1):

    

#         self.w = np.ones(X.shape[1])

#         self.b = 0



#         accuracy = {}

#         max_accuracy = 0



#         wt_matrix = []



#         for i in range(epochs):

#             for x, y in zip(X, Y):

#                 y_pred = self.model(x)

#                 if y == 1 and y_pred == 0:

#                     self.w = self.w + lr * x

#                     self.b = self.b - lr * 1

#                 elif y == 0 and y_pred == 1:

#                     self.w = self.w - lr * x

#                     self.b = self.b + lr * 1



#             wt_matrix.append(self.w)    



#             accuracy[i] = accuracy_score(self.predict(X), Y)

#             if (accuracy[i] > max_accuracy):

#                 max_accuracy = accuracy[i]

#                 chkptw = self.w

#                 chkptb = self.b



#         self.w = chkptw

#         self.b = chkptb



#         print(max_accuracy)



#         plt.plot(accuracy.values())

#         plt.ylim([0, 1])

#         plt.show()



#         return np.array(wt_matrix)
# X_train=X_train.values

# y_train=y_train.values
# perceptron=Perceptron()

# wt_matrix = perceptron.fit(X_train, y_train, 1000, 0.001)
# y_pred=perceptron.predict(X_test.values)

# accuracy_score(y_pred,y_test)
model.fit(X,y)

sub=model.predict(test)

out=sub.astype(int)

output = pd.DataFrame( { 'FileName': FileName , 'Class': out } )

output.to_csv( '../input/dataset.csv' , index = False )