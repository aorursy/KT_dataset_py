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
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



wine = pd.read_csv("../input/winequality-red.csv")

wine.shape
wine['quality'] = np.where(wine['quality']<6.5, 'Bad', 'Good')

wine[:10]
#One hot encoding

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

#Since many ML alogrithms operate with numerical data well than categorical data,

#encoding the target variable to have 0 & 1 in place of Good and Bad

Encoder = LabelEncoder()

wine['quality'] = Encoder.fit_transform(wine['quality'])

wine[:10]
import matplotlib.pyplot as plt
X = wine.drop('quality',axis = 1)



y  = wine.quality

wine[:10]
import seaborn as sns



wine.quality.value_counts()

sns.countplot(wine.quality)

plt.show()

#The distribution of the target population classes is skewed
#Train test split

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

x_train.shape

#Exploring Classification algorithms

#Random forest

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt



from sklearn.metrics import confusion_matrix

rf = RandomForestClassifier(random_state = 10)

rf.fit(x_train,y_train)

pred_rfc = rf.predict(x_test)



from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test, pred_rfc))

#Random forest gives 93% accuracy

print(confusion_matrix(y_test, pred_rfc))

plt.figure(figsize=(5,5))



cm_rf = confusion_matrix(y_test,pred_rfc)



plt.suptitle("Confusion Matrixes",fontsize=20)



#plt.subplot(2,3,1)

plt.title("Random Forest Confusion Matrix")

sns.heatmap(cm_rf,cbar=False,annot=True,cmap="Greens",fmt="d")

#Read more about precision and recall in this article

#https://towardsdatascience.com/model-evaluation-techniques-for-classification-models-eac30092c38b#:~:text=In%20machine%20learning%2C%20we%20often,predicted%20result%20of%20population%20data.&text=The%20training%20dataset%20trains%20the,Decision%20tree%2C%20Naive%20Bayes%20etc.
predctn_RF =pd.DataFrame({'Actual':y_test, 'Predicted':pred_rfc})

predctn_RF
print( np.unique( pred_rfc) )

#Model is predicting for both 0 & 1

#Decision Tree

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state = 10)

dt.fit(x_train,y_train)

pred_dt = dt.predict(x_test)

print(classification_report(y_test,pred_dt))

print(confusion_matrix(y_test, pred_dt))

plt.figure(figsize=(5,5))

cm_dt = confusion_matrix(y_test,pred_dt)

plt.suptitle("Confusion Matrixes",fontsize=20)



#plt.subplot(2,3,1)

plt.title("Random Forest Confusion Matrix")

sns.heatmap(cm_dt,cbar=False,annot=True,cmap="Blues",fmt="d")

predctn_DT =pd.DataFrame({'Actual':y_test, 'Predicted':pred_dt})

predctn_DT
print( np.unique( pred_dt) )

#Model is predicting for both 0 & 1
from sklearn.linear_model import LogisticRegression

logit = LogisticRegression()

logit.fit(x_train,y_train)
logit_pred = logit.predict(x_test)

print(classification_report(y_test,logit_pred))

print(confusion_matrix(y_test, logit_pred))

print( np.unique( logit_pred) )

#Model is predicting for both 0 & 1
plt.figure(figsize=(5,5))



cm_lg = confusion_matrix(y_test,logit_pred)



plt.suptitle("Confusion Matrixes",fontsize=20)



#plt.subplot(2,3,1)

plt.title("Random Forest Confusion Matrix")

sns.heatmap(cm_lg,cbar=False,annot=True,cmap="Oranges",fmt="d")
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV



svc1= SVC(random_state = 42, C = 10, gamma = 1, kernel = 'rbf')

svc1.fit(x_train, y_train)



ac = accuracy_score(y_test,svc1.predict(x_test))

#accuracies['SVM'] = ac





print('Accuracy is: ',ac, '\n')

cm = confusion_matrix(y_test,svc1.predict(x_test))

sns.heatmap(cm,annot=True,fmt="d")



print('SVM report\n',classification_report(y_test, svc1.predict(x_test)))
plt.scatter(x_train.iloc[:, 0], x_train.iloc[:, 1], c=y_train, s=100, cmap='autumn')



plt.scatter(model.support_vectors_[:,0],model.support_vectors_[:,1])