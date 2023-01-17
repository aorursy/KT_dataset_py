import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
Train_Data=pd.read_csv("../input/train.csv")

Test_Data=pd.read_csv("../input/test.csv")
len(Train_Data)
len(Test_Data)
Train_Data.sample(frac=0.1).head(n=5)
Train_Data.describe()
list(Train_Data.columns.values)
#Split the Data for machine learning algorithms



def Featurize_Outcome(Data):

    

        '''The activity variable is in the form of strings , which cannot be directly given as input to the algorithm , 

        for the purpose we will convert them to numerical features'''

        

        Labels=[]

        Key={}



        Prev=None

        Index=0



        for i in Data:

        

                    if(i in Key):



                         Labels.append(Key[i])

        

                    else:

        

                         Key[i]=Index

                         Labels.append(Key[i])

                         Index+=1

                    

        return(Labels)

    

Train_Data_X=Train_Data

Train_Data_Y=Train_Data['Activity']

Train_Data_Y=Featurize_Outcome(Train_Data_Y)

Train_Data_X.drop('Activity',axis=1,inplace=True)



Test_Data_X=Test_Data

Test_Data_Y=Test_Data['Activity']

Test_Data_Y=Featurize_Outcome(Test_Data_Y)

Test_Data_X.drop('Activity',axis=1,inplace=True)

#Importing required libraries for developing machine learning models



from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import cross_val_score

from sklearn import tree

from sklearn.svm import LinearSVC

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

Rf=RandomForestClassifier()

print("Accuracy: ",cross_val_score(Rf,Train_Data_X,Train_Data_Y, cv=5).mean())
gNB=GaussianNB()

scores=cross_val_score(gNB,Train_Data_X,Train_Data_Y, cv=5)

print("Accuracy: ",scores.mean())
cl=tree.DecisionTreeClassifier()

print("Accuracy: ",cross_val_score(cl,Train_Data_X,Train_Data_Y,cv=5).mean())
clf=svm.SVC(kernel='rbf')

print("Accuracy: ",cross_val_score(clf,Train_Data_X,Train_Data_Y, cv=5).mean())
clf=svm.SVC(kernel='linear')

print("Accuracy: ",cross_val_score(clf,Train_Data_X,Train_Data_Y, cv=5).mean())
scores=[]

for i in range(1,31):

    neigh=KNeighborsClassifier(n_neighbors=i)

    scores.append(cross_val_score(neigh,Train_Data_X,Train_Data_Y,cv=5).mean())

    

max_a=0

k_max=0



for i in range(0,30):

    

    if(scores[i]>=max_a):

        

        max_a=scores[i]

        

        if(i>k_max):

                

            k_max=i

        

print("K is maximum in Knn for ",k_max," with a accuracy of ",max_a) 
AB=AdaBoostClassifier()

print("Accuracy: ",cross_val_score(AB,Train_Data_X,Train_Data_Y, cv=5).mean())
#Import the requird library required for feature elimination based on variance. 

from sklearn.feature_selection import VarianceThreshold
sel=VarianceThreshold(.9 * (1 - .9))

FS_Train_Data_X=sel.fit_transform(Train_Data_X)
FS_Train_Data_X.shape
clf=svm.SVC(kernel='linear')

print("Accuracy: ",cross_val_score(clf,FS_Train_Data_X,Train_Data_Y, cv=5).mean())
from sklearn.feature_selection import RFECV

from sklearn.model_selection import StratifiedKFold
clf=svm.SVC(kernel='linear')

rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(2),

              scoring='accuracy')

rfecv.fit(Train_Data_X,Train_Data_Y)
print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores

plt.figure()

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score (nb of correct classifications)")

plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

plt.show()
print(max(rfecv.grid_scores_))
print("Accuracy: ",cross_val_score(rfecv,Test_Data_X,Test_Data_Y,cv=5).mean())