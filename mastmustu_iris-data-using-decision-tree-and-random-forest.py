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
import numpy as np # importing numpy package 

import pandas as pd # importing pandas package



data = pd.read_csv('../input/Iris.csv')

data.head() #printing top 5 rows of the DataFrame

data = data.drop(['Id'], axis=1) #dropping 'Id' column from DataFrame

data.head() #printing top 5 rows of the DataFrame

data.shape
data.info()
data.describe()
data['Species'].value_counts()
import seaborn as sns

sns.pairplot(data,hue ='Species')
X = data.drop(['Species'], axis=1) #Input data

y = data['Species'] #Output data

print(X.shape) #dimensions of input data

print(y.shape) #dimensions of output data
from sklearn.model_selection import train_test_split  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
from sklearn.tree import DecisionTreeClassifier  

classifier = DecisionTreeClassifier()  

classifier.fit(X_train, y_train) 

y_pred = classifier.predict(X_test)  

print(y_pred)
from sklearn.metrics import classification_report, confusion_matrix  

print(confusion_matrix(y_test, y_pred))  

print(classification_report(y_test, y_pred)) 
from sklearn.metrics import accuracy_score #importing accuracy_score function from sklearn.metrics package

acc = accuracy_score(y_test,y_pred)

print("Accuracy for this model {} %".format(acc*100))
# important features 



print(classifier.feature_importances_)



#SepalLenCm   SepalWidCm    PetalLenCm     PetalWidCm
from sklearn import tree

from sklearn.tree import export_graphviz



from sklearn.datasets import load_iris



iris = load_iris()



tree.export_graphviz(classifier,out_file='tree.dot',feature_names = iris.feature_names,

class_names = iris.target_names,rounded = True, proportion = False, precision = 2, filled = True)  





!dot -Tpng tree.dot -o tree.png
from IPython.display import Image

Image(filename = 'tree.png')
from sklearn.ensemble import RandomForestClassifier

rclf = RandomForestClassifier(n_estimators=100, max_depth=10,random_state=0)



rclf.fit(X_train, y_train)
ry_pred = rclf.predict(X_test)  

print(ry_pred)
from sklearn.metrics import classification_report, confusion_matrix  

print(confusion_matrix(y_test, ry_pred))  

print(classification_report(y_test, ry_pred)) 
from sklearn.metrics import accuracy_score #importing accuracy_score function from sklearn.metrics package

acc = accuracy_score(y_test,ry_pred)

print("Accuracy for this model {} %".format(acc*100))
from sklearn.svm import SVC



clf = SVC()

clf.fit(X, y) 

svm_pred = clf.predict(X_test)  

print(svm_pred)
from sklearn.metrics import classification_report, confusion_matrix  

print(confusion_matrix(y_test, svm_pred))  

print(classification_report(y_test, svm_pred)) 
from sklearn.metrics import accuracy_score #importing accuracy_score function from sklearn.metrics package

acc = accuracy_score(y_test,svm_pred)

print("Accuracy for this model {} %".format(acc*100))
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

lr.fit(X_train,y_train)



lr_pred = lr.predict(X_test)



from sklearn.metrics import classification_report, confusion_matrix  

print(confusion_matrix(y_test, lr_pred))  

print(classification_report(y_test, lr_pred))



from sklearn.metrics import accuracy_score #importing accuracy_score function from sklearn.metrics package

acc = accuracy_score(y_test,lr_pred)

print("Accuracy for this model {} %".format(acc*100))

from sklearn.svm import SVC



svc_2 = SVC()

svc_2.fit(X_train[['PetalLenCm','PetalWidCm']],y_train)



svc_2_pred = svc_2.predict(X_test[['PetalLenCm','PetalWidCm']])



from sklearn.metrics import classification_report, confusion_matrix  

print(confusion_matrix(y_test, svc_2_pred))  

print(classification_report(y_test, svc_2_pred))



from sklearn.metrics import accuracy_score #importing accuracy_score function from sklearn.metrics package

acc = accuracy_score(y_test,svc_2_pred)

print("Accuracy for this model {} %".format(acc*100))