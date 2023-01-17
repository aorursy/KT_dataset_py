# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")
data.head()
data.info()
data.describe()
data.loc[:,'class'].value_counts()
# train test split

x,y = data.loc[:,data.columns!='class',],data.loc[:,'class']

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)

print("With KNN (k=3) accuracy is: ", knn.score(x_test,y_test))
# model complexity

neig = np.arange(1,25)

train_accuracy = []

test_accuracy = []

# Loop over different values of k

for i, k in enumerate(neig):

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(x_train,y_train)

    train_accuracy.append(knn.score(x_train,y_train))

    test_accuracy.append(knn.score(x_test,y_test))

# Plot

plt.figure(figsize=(13,8))

plt.plot(neig, test_accuracy, label = "Testing Accuracy")

plt.plot(neig, train_accuracy, label = "Training Accuracy")

plt.legend()

plt.title('-value VS Accuracy')

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')

plt.xticks(neig)

plt.savefig('graph.png')

plt.show()

print("Best accuracy is {} with K={}". format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))
data = pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")
data["class"] = [1 if each =="Abnormal" else 0 for each in data["class"]]

y = data["class"].values

x_data = data.drop(["class"],axis=1)
# normalization

x = ((x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)))
# train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=1)
# SVM

from sklearn.svm import SVC

svm = SVC(random_state=1)

svm.fit(x_train,y_train)
print("Accuracy of svm algo: ", svm.score(x_test,y_test))
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

print("test score with naive bayes: ", nb.score(x_test,y_test))
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)

print("test score with decision tree: ", dt.score(x_test,y_test))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=1)

rf.fit(x_train,y_train)

print("test score with random forest classification: ", rf.score(x_test,y_test))
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train,y_train)

print("test score with logistic regression: ", lr.score(x_test,y_test))
X = x.values   # ann needs array as input

print(type(X))


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=1)
import tensorflow as tf

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=3, activation='relu')) 

ann.add(tf.keras.layers.Dense(units=3, activation='relu'))  

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

ann.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])

ann.fit(X_train, y_train, batch_size = 32, epochs = 100)
y_pred = ann.predict(X_test)

y_pred = (y_pred > 0.5)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
#We get the highest accuracy with Random Forest Classification which is %87.

#Lets evaluate Random Forest prediction with confusion matrix

from sklearn.ensemble import RandomForestClassifier 

rf = RandomForestClassifier(n_estimators=100, random_state=1) 

rf.fit(x_train,y_train) 

print("test score with random forest classification: ", rf.score(x_test,y_test))



y_pred = rf.predict(x_test)

y_true = y_test



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)

print(cm)
# confusion matrix visualiation

f,ax = plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot=True,linewidths=0.5, linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show