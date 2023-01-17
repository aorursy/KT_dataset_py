import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
df=pd.read_csv("../input/heart.csv")
df.head()
df.describe()
sns.countplot(x="target", data=df)
y = df.target.values
x_data = df.drop(["target"],axis=1)

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
x_data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.20,random_state=56)
lr=LogisticRegression()
#Training the model

lr.fit(x_train,y_train)

print("test accuracy {}".format(lr.score(x_test,y_test)))

lr_score=lr.score(x_test,y_test)
#Testing the model

y_prediction = lr.predict(x_test)

y_actual=y_test

cm = confusion_matrix(y_actual,y_prediction)
#Visualizing the results

sns.heatmap(cm, annot=True)

plt.xlabel("Predictions Y")

plt.ylabel("Actual Y")

plt.show()
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()

#Training the model

knn.fit(x_train,y_train)

print("test accuracy {}".format(knn.score(x_test,y_test)))



knn_prediction_score=knn.score(x_test,y_test)
#Testing the model

y_prediction = knn.predict(x_test)

y_actual=y_test

cm = confusion_matrix(y_actual,y_prediction)
#Visualizing the result

sns.heatmap(cm, annot=True)

plt.xlabel("Predictions Y")

plt.ylabel("Actual Y")

plt.show()
from sklearn.svm import SVC
svm = SVC()

#Training the model

svm.fit(x_train,y_train)

# prediction and accuracy 

print("print accuracy of svm algo: ",svm.score(x_test,y_test))



svm_score = svm.score(x_test,y_test)
#Testing the model

y_prediction = knn.predict(x_test)

y_actual=y_test

cm = confusion_matrix(y_actual,y_prediction)
#Visualizing the results

sns.heatmap(cm, annot=True)

plt.xlabel("Predictions Y")

plt.ylabel("Actual Y")

plt.show()
from sklearn.tree import DecisionTreeClassifier



dt = DecisionTreeClassifier()

#Training the Model

dt.fit(x_train,y_train)



print("score: ", dt.score(x_test,y_test))



dt_score=dt.score(x_test,y_test)
#Testing the model

y_prediction = knn.predict(x_test)

y_actual=y_test

cm = confusion_matrix(y_actual,y_prediction)
#Visulaizing the results

sns.heatmap(cm, annot=True)

plt.xlabel("Predictions Y")

plt.ylabel("Actual Y")

plt.show()
class_name = ("Logistic Regression","KNN","SVM","Decision Tree")

class_score = (lr_score,knn_prediction_score,svm_score,dt_score)

y_pos= np.arange(len(class_score))

colors = ("red","gray","purple","green")

plt.figure(figsize=(20,12))

plt.bar(y_pos,class_score,color=colors)

plt.xticks(y_pos,class_name,fontsize=20)

plt.yticks(np.arange(0.00, 1.05, step=0.05))

plt.ylabel('Accuracy')

plt.title(" Confusion Matrix Comparision of the Classes",fontsize=15)

plt.savefig('graph.png')

plt.show()