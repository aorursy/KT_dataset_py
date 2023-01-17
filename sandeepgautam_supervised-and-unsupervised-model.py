# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

data1=pd.read_csv("../input/Admission_Predict_Ver1.1.csv")

data2=pd.read_csv("../input/Admission_Predict.csv")



# Any results you write to the current directory are saved as output.
data1.head()

data1.shape

data2.shape
data1.head()

data1.corr()
corrmat = data1.corr() 

import seaborn as sns

import matplotlib.pyplot as plt

  

f, ax = plt.subplots(figsize =(9, 8)) 

sns.heatmap(corrmat, ax = ax, linewidths = 0.1) 
import sys

print("There are",len(data1.columns),"columnsand",len(data1.index),"rows in the train datasets")

print("Name of the columns are " )

data1.columns



    

data1.info()
data1.head()
data1.describe()

import sys

sns.boxplot(data=data1[['GRE Score',"TOEFL Score"]])



def func(x):

    if 0 < x <= 0.7:

        return 'Less'

    else :

        return 'High'

# I had changed the name to Chance_acceptance first so I am again changing it ,you can change directly from"Chance of Admit" 

data1=data1.rename(columns = {'Chance of Admit ':'Chance_of_Admit'})

data1.columns

    



data1['Prob'] = data1['Chance_of_Admit'].apply(func)

data1.columns

data1.Prob.value_counts()
# converting our new variable "Prob" into categorical variable

data1['Prob']=data1['Prob'].astype('category')

data1.info()



sns.boxplot(x='Prob',y='GRE Score',data=data1)

plt.xlabel("Probability of selection")
sns.boxplot(x='Prob',y='University Rating',data=data1)

plt.xlabel("Probability of Selection")
sns.countplot('University Rating',data=data1[data1.Prob=='High'])

plt.xlabel("University Rating")

plt.ylabel("Candidates")
sns.countplot('University Rating',data=data1[data1.Prob=='Less'])

plt.xlabel("University Rating")

plt.ylabel("Candidates")
sns.boxplot(x='Prob',y='CGPA',data=data1)

plt.xlabel("Probability of selection")
data1.columns
# Now we will see whether or not research played a significant role for Chance of Admit

sns.countplot(x='Research',hue='Prob',data=data1)

plt.title("Student with/without research")

plt.xlabel("Research ")

plt.ylabel("Frequency")
pd.crosstab(data1.Research, data1.Prob)

y=data1['Chance_of_Admit']

x=data1.drop(['Serial No.',"Prob","Chance_of_Admit"],axis=1)



# separating train (80%) and test (%20) sets

from sklearn.model_selection import train_test_split



x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.20,random_state = 42)

# normalization

from sklearn.preprocessing import MinMaxScaler

scalerX = MinMaxScaler(feature_range=(0, 1))

x_train[x_train.columns] = scalerX.fit_transform(x_train[x_train.columns])

x_test[x_test.columns]=scalerX.fit_transform(x_test[x_test.columns])

x_train.head()
from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(x_train,y_train)

pred=lr.predict(x_test)

lr.score(x_train,y_train) # 82 .10 percent r squared 
from sklearn.metrics import r2_score

print("r_square score in Linear Regreesion for test data: ", r2_score(pred,y_test)) # r_square score in Linear Regreesion for test data:  0.8108289642028199



print("r_square score in Linear Regression for training data: ", r2_score(lr.predict(x_train),y_train)) #r_square score in Linear Regression for training data:  0.7820727988987459



pred_train=lr.predict(x_train)

from sklearn.ensemble import RandomForestRegressor

Rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

Rf.fit(x_train,y_train)

pred_Rf = Rf.predict(x_test)

from sklearn.metrics import r2_score

print("r_square score in random forest for test data: ", r2_score(pred_Rf,y_test))

#r_square score in random forest for test data:  0.7430833922546942

print("r_square score in random forest for train data: ", r2_score(Rf.predict(x_train),y_train))

# r_square score in random forest for train data:  0.9642185975347183



from sklearn.tree import DecisionTreeRegressor

Dt = DecisionTreeRegressor(random_state = 42)

Dt.fit(x_train,y_train)

pred_Dt = Dt.predict(x_test) 



from sklearn.metrics import r2_score

print("r_square score in decision tree for test data: ", r2_score(pred_Dt,y_test))

# r_square score in decision tree for test data:  0.6440141183163999

print("r_square score in decision tree for train data: ", r2_score(Dt.predict(x_train),y_train))

# r_square score in decision tree for train data:  1.0. It is overfitting i.e worked extremely well for training data but can't 

#      better in test data





y = np.array([r2_score(pred_Dt,y_test),r2_score(pred_Rf,y_test),r2_score(pred,y_test)])

x = ["DecisionTreeReg.","Random Forest Regression","Linear Regression"]

plt.bar(x,y)

plt.title("Comparison of Regression Algorithms")

plt.xlabel("Regressor")

plt.ylabel("r2_score")

plt.show()

y1=data1['Prob']

x1=data1.drop(['Serial No.',"Prob","Chance_of_Admit"],axis=1)



# Spliting data for classification problem

from sklearn.model_selection import train_test_split



x_train1, x_test1,y_train1, y_test1 = train_test_split(x1,y1,test_size = 0.20,random_state = 42)



# Normalizing Data

x_train1[x_train1.columns] = scalerX.fit_transform(x_train1[x_train1.columns])

x_test1[x_test1.columns]=scalerX.fit_transform(x_test1[x_test1.columns])





x_train[x_train.columns] = scalerX.fit_transform(x_train[x_train.columns])

x_test[x_test.columns]=scalerX.fit_transform(x_test[x_test.columns])

from sklearn.linear_model import LogisticRegression

lrc = LogisticRegression()

lrc.fit(x_train1,y_train1)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test1,lrc.predict(x_test1))

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test1, lrc.predict(x_test1)))

# 85 percent accuracy
from sklearn.svm import SVC

svm = SVC(random_state = 1)

svm.fit(x_train1,y_train1)

svm_pred =svm.predict(x_test1)

svm_pred

from sklearn.metrics import accuracy_score

print( accuracy_score(y_test1, svm_pred))

# accuracy of 86 percent

confusion_matrix(y_test1,svm_pred)

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train1,y_train1)

nb_pred=nb.predict(x_test1)



from sklearn.metrics import accuracy_score

print( accuracy_score(y_test1, nb_pred)) # 89 percent accuracy

confusion_matrix(y_test1,nb_pred)
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc.fit(x_train1,y_train1)

dt_pred=dtc.predict(x_test1)



from sklearn.metrics import accuracy_score

print( accuracy_score(y_test1, dt_pred))

# 83 percent accuracy

confusion_matrix(y_test1,dt_pred)
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 100,random_state = 1)

rfc.fit(x_train1,y_train1)

rfc_pred=rfc.predict(x_test1)



from sklearn.metrics import accuracy_score

print( accuracy_score(y_test1, rfc_pred))

# 87 percent accuracy

confusion_matrix(y_test1,rfc_pred)
from sklearn.neighbors import KNeighborsClassifier

# finding optimum k value

scores = []

for each in range(1,50):

    knn_n = KNeighborsClassifier(n_neighbors = each)

    knn_n.fit(x_train1,y_train1)

    scores.append(knn_n.score(x_test1,y_test1))

    

    

   

plt.plot(range(1,50),scores)

plt.xlabel("k")

plt.ylabel("accuracy")

plt.show()



# We can see that at around k=8 , we have the highest accuracy.so we will build a model with k=9
knn_model=KNeighborsClassifier(n_neighbors=8)

knn_model.fit(x_train1,y_train1)

knn_predi=knn_model.predict(x_test1)

from sklearn.metrics import accuracy_score

print( accuracy_score(y_test1, knn_predi))

#87 percent accuracy

confusion_matrix(y_test1,knn_predi)

y = np.array([accuracy_score(y_test1, lrc.predict(x_test1)),accuracy_score(y_test1, svm_pred),accuracy_score(y_test1,nb_pred),accuracy_score(y_test1, dt_pred),accuracy_score(y_test1, rfc_pred),accuracy_score(y_test1, knn_predi)])

#x = ["LogisticRegression","SVM","GaussianNB","DecisionTreeClassifier","RandomForestClassifier","KNeighborsClassifier"]

x = ["LogisticReg.","SVM","GNB","Dec.Tree","Ran.Forest","KNN"]



plt.bar(x,y)

plt.title("Comparison of Classification Algorithms")

plt.xlabel("Classfication")

plt.ylabel("Score")

plt.show()






