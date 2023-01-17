import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.ensemble import AdaBoostClassifier

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib as mlt

from sklearn.model_selection import cross_val_score
Data=pd.read_csv("../input/diabetes.csv")

Data=Data.dropna(thresh=9)
M_Data=Data

Outcome=M_Data['Outcome']

M_Data.drop('Outcome',axis=1,inplace=True)
Data=pd.read_csv("../input/diabetes.csv")

Positives=Data[Data['Outcome']==1]

Negatives=Data[Data['Outcome']==0]
#Import data from csv file

Data.sample(frac=0.1).head(n=5)
Data.describe()
#For curve fitting

from scipy import stats
fig, ax =plt.subplots(1,3)

sns.distplot(Positives['Pregnancies'],rug=True,kde=False,color='r',fit=stats.gamma,ax=ax[0])

sns.distplot(Positives['BloodPressure'],rug=True,kde=False,color='r',fit=stats.gamma,ax=ax[1])

sns.distplot(Positives['Age'],rug=True,kde=False,color='r',fit=stats.gamma,ax=ax[2])

fig.show()
fig, ax =plt.subplots(1,3)

sns.distplot(Negatives['Pregnancies'],rug=True,kde=False,color='g',fit=stats.gamma,ax=ax[0])

sns.distplot(Negatives['BloodPressure'],rug=True,kde=False,color='g',fit=stats.gamma,ax=ax[1])

sns.distplot(Negatives['Age'],rug=True,kde=False,color='g',fit=stats.gamma,ax=ax[2])

fig.show()
Corr=Data[Data.columns].corr()

sns.heatmap(Corr,annot=True)
Data.Outcome.value_counts()
(500/float(len(Data)))*100
#Import the required libraries for machie learning algorithms



from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn import tree
#Naive Bayes Algorithm 



gNB=GaussianNB()

scores=cross_val_score(gNB, M_Data,Outcome, cv=5)

print("Accuracy: ",scores.mean())


scores=[]

for i in range(1,31):

    neigh=KNeighborsClassifier(n_neighbors=i)

    scores.append(cross_val_score(neigh,M_Data,Outcome,cv=5).mean())

    

max_a=0

k_max=0



for i in range(0,30):

    

    if(scores[i]>=max_a):

        

        max_a=scores[i]

        

        if(i>k_max):

                

            k_max=i

        

print("K is maximum in Knn for ",k_max," with a accuracy of ",max_a)       

 
clf=svm.SVC(kernel='linear')

print("Accuracy: ",cross_val_score(clf, M_Data,Outcome, cv=5).mean())
clf_r=svm.SVC(kernel='rbf')

print("Accuracy: ",cross_val_score(clf_r,M_Data,Outcome, cv=5).mean())
from sklearn import tree

cl=tree.DecisionTreeClassifier()

print("Accuracy: ",cross_val_score(cl,M_Data,Outcome, cv=5).mean())
Rf=RandomForestClassifier()

print("Accuracy: ",cross_val_score(Rf,M_Data,Outcome, cv=5).mean())
AB=AdaBoostClassifier()

print("Accuracy: ",cross_val_score(AB,M_Data,Outcome, cv=5).mean())
from keras.models import Sequential 

from keras.layers import Dense,Activation

from sklearn.model_selection import StratifiedKFold
Numpy_Matrix=M_Data.as_matrix()

Numpy_Outcome=Outcome.as_matrix()
from keras.layers.normalization import BatchNormalization

from keras.layers import Dropout



K_fold=StratifiedKFold(n_splits=4,shuffle=True)

cv_scores=[]



for train,test in K_fold.split(M_Data,Outcome):

    

        model=Sequential()

        model.add(Dense(10,activation='relu',input_dim=8))

        model.add(BatchNormalization())

        

        model.add(Dense(12,activation='relu',input_dim=8))

        model.add(BatchNormalization())

        

        model.add(Dense(12,activation='relu'))

        model.add(BatchNormalization())

        

        model.add(Dense(12,activation='relu'))

        model.add(BatchNormalization())

        

        

        model.add(Dense(1,activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(Numpy_Matrix[train],Numpy_Outcome[train], epochs=280, batch_size=32)

        scores=model.evaluate(Numpy_Matrix[test],Numpy_Outcome[test])

        cv_scores.append(scores[1]*100)
def Mean(Scores):

    Sum=0

    for i in Scores:

        

        Sum+=i

        

    return(Sum/len(Scores))
print(Mean(cv_scores))
Corr.mean()
Data_P=Data

Data_P.drop('Pregnancies',axis=1,inplace=True)

Data_P.drop('Outcome',axis=1,inplace=True)

clf=svm.SVC(kernel='linear')

print("Accuracy: ",cross_val_score(clf,Data_P,Outcome,cv=5).mean())
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(Data_P,Outcome,test_size=0.5,random_state=0)

clf.fit(x_train,y_train)

Conf=confusion_matrix(y_test,clf.predict(x_test))

sns.heatmap(Conf,annot=True,)
print(classification_report(y_test, clf.predict(x_test)))