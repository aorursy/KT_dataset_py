# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn import decomposition
import matplotlib.pyplot as plt
#dataset = pd.read_csv("roo_data.csv")
dataset =pd.read_csv('/kaggle/input/careerproject/career_data.csv',encoding= 'latin-1')
data = dataset.iloc[:,:-1].values
label = dataset.iloc[:,-1].values
len(data[0])
dataset.iloc[:,3:13]

dataset.iloc[:,:3]

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
for i in range(3,13):
    data[:,i] = labelencoder.fit_transform(data[:,i])
data[:5]
data[:5,3:]
from sklearn.preprocessing import Normalizer
data1=data[:,:3]
normalized_data = Normalizer().fit_transform(data1)
print(normalized_data.shape)
normalized_data
data2=data[:,3:]
data2.shape

df1 = np.append(normalized_data,data2,axis=1)

df1.shape

X1 = pd.DataFrame(df1,columns=['HSLC', 'HS',
       'UG','Self-learning capability?', 'Extra-courses', 'Certifications',
       'Workshops','Interested subjects', 'Interested career', 'Job/Higher Studies?',
       'Type of  profession want to settle in?','Profession type','Worked in teams ?'])
X1.head()
label = labelencoder.fit_transform(label)
print(len(label))
y=pd.DataFrame(label,columns=["Suggested Job Role"])
y.head()
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.metrics import accuracy_score
X_train,X_test,y_train,y_test=train_test_split(X1,y,test_size=0.2,random_state=10)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix,accuracy_score
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
print("confusion matrics=",cm)
print("  ")
print("accuracy=",accuracy*100)
X_test
single_test_input = X_test.iloc[[30]]
single_test_input
y_pred_new = clf.predict(single_test_input)
y_pred_new
labelencoder.inverse_transform(y_pred_new)
#Decision tree with entropy
clf_entropy = tree.DecisionTreeClassifier(criterion = "entropy", random_state = 10)
clf_entropy.fit(X_train, y_train)
entropy_y_pred=clf_entropy.predict(X_test)
cm_entopy = confusion_matrix(y_test,entropy_y_pred)
entropy_accuracy = accuracy_score(y_test,entropy_y_pred)
print("confusion matrics=",cm_entopy)
print("  ")
print("accuracy=",entropy_accuracy*100)

#SVM (Support vector machine) classifier
from sklearn import svm
clf = svm.SVC()
clf.fit(X_train, y_train)
svm_y_pred = clf.predict(X_test)
svm_cm = confusion_matrix(y_test,svm_y_pred)
svm_accuracy = accuracy_score(y_test,svm_y_pred)
print("confusion matrics=",svm_cm)
print("  ")
print("accuracy=",svm_accuracy*100)
#xgboost
#changing data type to int64
X_train,X_test,y_train,y_test=train_test_split(X1,y,test_size=0.3,random_state=10)
X_train.shape
X_train=pd.to_numeric(X_train.values.flatten())
X_train=X_train.reshape((14000,13))
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)
xgb_y_pred = clf.predict(X_test)
xgb_cm = confusion_matrix(y_test,xgb_y_pred)
xgb_accuracy = accuracy_score(y_test,xgb_y_pred)
print("confusion matrics=",xgb_cm)
print("  ")
print("accuracy=",xgb_accuracy*100)
