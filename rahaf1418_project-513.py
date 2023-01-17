# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# plotly library



from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/xAPI-Edu-Data/xAPI-Edu-Data.csv")


from sklearn.model_selection import train_test_split

# create new data containing only numerical values

data_new=data.loc[:,["gender","raisedhands","VisITedResources","AnnouncementsView","Discussion"]]

# write 1 for male and 0 for female

data_new.gender=[1 if i=="M" else 0 for i in data_new.gender]

#y: binary output 

y=data_new.gender.values

#x_data: rest of the data (i.e. features of data except gender)

x_data=data_new.drop("gender",axis=1)

# normalize the values in x_data

x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

#* create x_train, y_train, x_test and  y_test arrays with train_test_split method

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=52)



#KNN Classification

from sklearn.neighbors import KNeighborsClassifier

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)



knn=KNeighborsClassifier(n_neighbors=3)



#fit

knn.fit(x_train,y_train)



#prediction

prediction=knn.predict(x_test)

#prediction score (accuracy)

print('KNN (K=3) accuracy is: ',knn.score(x_test,y_test)) 





#Support Vector Machine (SVM) Classification

from sklearn.svm import SVC



svm=SVC(random_state=1)

svm.fit(x_train,y_train)

#accuracy

print("accuracy of svm algorithm: ",svm.score(x_test,y_test))

from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()

nb.fit(x_train,y_train)



# test accuracy

print("Accuracy of naive bayees algorithm: ",nb.score(x_test,y_test))

from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()

dt.fit(x_train,y_train)



print("Accuracy score for Decision Tree Classification: " ,dt.score(x_test,y_test))
#RandomForest

from sklearn.ensemble import RandomForestClassifier



rf=RandomForestClassifier(n_estimators=100,random_state=1)

rf.fit(x_train,y_train)



print("random forest algorithm accuracy: ",rf.score(x_test,y_test))

#Confusion matrix of Random Forest Classf

#y_pred:  results that we predict

#y_test: our real values

y_pred=rf.predict(x_test)

y_true=y_test



#cm

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_true,y_pred)



#cm visualization

f,ax=plt.subplots(figsize=(8,8))

sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("predicted value")

plt.ylabel("real value")

plt.show()
#Confusion matrix of KNN Classf.

y_pred1=knn.predict(x_test)

y_true=y_test

#cm

cm1=confusion_matrix(y_true,y_pred1)



#cm visualization

f,ax=plt.subplots(figsize=(8,8))

sns.heatmap(cm1,annot=True,linewidths=0.5,linecolor="blue",fmt=".0f",ax=ax)

plt.xlabel("predicted value")

plt.ylabel("real value")

plt.show()
#Confusion matrix of Decision Tree Classf.

y_pred2=dt.predict(x_test)

y_true=y_test

#cm

cm2=confusion_matrix(y_true,y_pred2)



#cm visualization

f,ax=plt.subplots(figsize=(8,8))

sns.heatmap(cm2,annot=True,linewidths=0.5,linecolor="green",fmt=".0f",ax=ax)

plt.xlabel("predicted value")

plt.ylabel("real value")

plt.show()
# naive bayes

y_pred3=nb.predict(x_test)

y_true=y_test

#cm

cm3=confusion_matrix(y_true,y_pred3)



#cm visualization

f,ax=plt.subplots(figsize=(8,8))

sns.heatmap(cm3,annot=True,linewidths=0.5,linecolor="yellow",fmt=".0f",ax=ax)

plt.xlabel("predicted value")

plt.ylabel("real value")

plt.show()
dictionary={"model":["KNN","SVM","NB","DT","RF"],"score":[knn.score(x_test,y_test),svm.score(x_test,y_test),nb.score(x_test,y_test),dt.score(x_test,y_test),rf.score(x_test,y_test)]}

df1=pd.DataFrame(dictionary)

print(df1)