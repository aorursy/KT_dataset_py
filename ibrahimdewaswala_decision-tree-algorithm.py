import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/falldeteciton.csv")
X_train=train_data.iloc[:,[1,2,3,4,5,6]].values

y_train=train_data.iloc[:,0].values
from sklearn.model_selection import train_test_split

X_train_main, X_train_eval, y_train_main, y_train_eval=train_test_split(X_train,y_train,test_size=0.25)
from sklearn.tree import DecisionTreeClassifier

classifier=DecisionTreeClassifier(criterion='entropy')

classifier.fit(X_train_main,y_train_main)
y_predicted=classifier.predict(X_train_eval)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_train_eval,y_predicted)
total=0

for i in range(0,len(cm)):

    for j in range(0,len(cm)):

        if(i==j):

           total=total+cm[i][j] 
total