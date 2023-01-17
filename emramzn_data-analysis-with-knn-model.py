

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import pandas_profiling as pp

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV, train_test_split

df=pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")

df.head(10)



df.describe()

sns.countplot(x="class",data=df)

df.loc[:,"class"].value_counts()


pp.ProfileReport(df)
x,y=df.loc[:,df.columns != "class"], df.loc[:,"class"]
y=[1 if each =="Abnormal" else 0 for each in y]  #Converting feature of "Class" to binary ( Abnormal = 1 and Normal = 0 )
x=(x- np.min(x))/(np.max(x)- np.min(x))
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
KnnModel=KNeighborsClassifier(n_neighbors=9)

KnnModel.fit(x_train,y_train)

print("Model Acc. : ",KnnModel.score(x_test,y_test))
grid={"n_neighbors":np.arange(1,50)}
KnnCV=KNeighborsClassifier()

KnnCVGrid=GridSearchCV(KnnCV,grid,cv=10)

KnnCVGrid.fit(x_train,y_train)

print("Best Parameter : ",KnnCVGrid.best_params_,"\nBest Score : ",KnnCVGrid.best_score_)
neigh=np.arange(1,25)

train_score=[]

test_score=[]



for neighbors in neigh:

    KnnM=KNeighborsClassifier(n_neighbors=neighbors)

    KnnM.fit(x_train,y_train)

    

    train_score.append(KnnM.score(x_train,y_train))

    test_score.append(KnnM.score(x_test,y_test))

    



plt.figure(figsize=[13,8])

plt.plot(neigh,test_score, label="Test Score" ,color="red")

plt.plot(neigh, train_score, label="Train Score", color="black")

plt.legend()

plt.xlabel("Number of Neighbors")

plt.ylabel("Accuarcy")

plt.show()

print("Best accuarcy {} with K = {}  ".format(np.max(test_score), test_score.index(np.max(test_score))))


