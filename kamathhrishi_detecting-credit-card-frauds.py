import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential

from keras.layers import Dense

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import svm

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split

from sklearn.metrics import recall_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
W_Data=pd.read_csv("../input/creditcard.csv")

W_Data.dropna(thresh=284315)

Data=W_Data
Data.sample(frac=0.1).head(n=5)
Data.describe()
Positives=W_Data[W_Data['Class']==1]

Negatives=W_Data[W_Data['Class']==0]
print((len(Positives)/len(W_Data))*100,"%")
sns.kdeplot(Positives['Amount'],shade=True,color="red")
sns.kdeplot(Negatives['Amount'],shade=True,color="green")
sns.kdeplot(Negatives['Time'],shade=True,color="red")
sns.kdeplot(Positives['Time'],shade=True,color="green")
Negatives=Data[Data['Class']==0]

Positives=Data[Data['Class']==1]
Train_Data=Data[1:50000]

Target=Train_Data['Class']

Train_Data.drop('Class',axis=1,inplace=True)
x_train,x_test,y_train,y_test=train_test_split(Train_Data,Target,test_size=0.5,random_state=0)
clf_l=svm.SVC(kernel='linear')

clf_l.fit(x_train,y_train)

print(classification_report(y_test,clf_l.predict(x_test)))
clf = RandomForestClassifier(max_depth=2, random_state=0)

clf.fit(x_train, y_train)

print(classification_report(y_test,clf.predict(x_test)))

E_Data=pd.read_csv("../input/creditcard.csv")

E_Data.dropna(thresh=284315)

E_Train_Data=E_Data

E_Target=E_Train_Data['Class']

E_Train_Data.drop('Class',axis=1,inplace=True)

x_train_E,x_test_E,y_train_E,y_test_E=train_test_split(E_Train_Data,E_Target,test_size=0.5,random_state=0)
clf_E = RandomForestClassifier(max_depth=2, random_state=0)

clf_E.fit(x_train, y_train)

print(classification_report(y_test_E,clf_E.predict(x_test_E)))
from sklearn.covariance import EllipticEnvelope

from sklearn.ensemble import IsolationForest
W_Data=pd.read_csv("../input/creditcard.csv")

W_Data.dropna(thresh=284315)

Data=W_Data[1:50000]
Negatives=Data[Data['Class']==0]

Positives=Data[Data['Class']==1]
#RBF Kernel

clf_AD = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)

clf_AD.fit(Negatives)
#Linear Kernel

clf_AD_L = svm.OneClassSVM(nu=0.1, kernel="linear", gamma=0.1)

clf_AD_L.fit(Negatives)
IFA=IsolationForest()

IFA.fit(Negatives)
train_AD_L=clf_AD_L.predict(Negatives)

test_AD_L=clf_AD_L.predict(Positives)
train_IFA=IFA.predict(Negatives)

test_IFA=IFA.predict(Positives)
train_AD=clf_AD.predict(Negatives)

test_AD=clf_AD.predict(Positives)
def Train_Accuracy(Mat):

   

   Sum=0

   for i in Mat:

    

        if(i==1):

        

           Sum+=1.0

            

   return(Sum/len(Mat)*100)



def Test_Accuracy(Mat):

   

   Sum=0

   for i in Mat:

    

        if(i==-1):

        

           Sum+=1.0

            

   return(Sum/len(Mat)*100)

print("Training: One Class SVM (RBF) : ",(Train_Accuracy(train_AD)),"%")

print("Test: One Class SVM (RBF) : ",(Test_Accuracy(test_AD)),"%")
print("Training: Isolation Forest: ",(Train_Accuracy(train_IFA)),"%")

print("Test: Isolation Forest: ",(Test_Accuracy(test_IFA)),"%")
print("Training: One Class SVM (Linear) : ",(Train_Accuracy(train_AD_L)),"%")

print("Test: One Class SVM (Linear) : ",(Test_Accuracy(test_AD_L)),"%")
W_Data=pd.read_csv("../input/creditcard.csv")

W_Data.dropna(thresh=284315)

Data=W_Data

Positives_E=W_Data[W_Data['Class']==1]

Negatives_E=W_Data[W_Data['Class']==0]
IFA=IsolationForest()

IFA.fit(Negatives_E)

train_IFA=IFA.predict(Negatives)

test_IFA=IFA.predict(Positives)
print("Training: Isolation Forest: ",(Train_Accuracy(train_IFA)),"%")

print("Test: Isolation Forest: ",(Test_Accuracy(test_IFA)),"%")
plt.figure(figsize=(20,18))

Corr=Data[Data.columns].corr()

sns.heatmap(Corr,annot=True)
from imblearn.over_sampling import SMOTE 



W_Data=pd.read_csv("../input/creditcard.csv")

W_Data.dropna(thresh=284315)



sm = SMOTE(random_state=42)

X_res, y_res = sm.fit_sample(W_Data, W_Data['Class'])
S_Positives=[]

S_Negatives=[]



for i in range(0,len(X_res)):

    if(y_res[i]==0):

        S_Negatives.append(X_res[i])

    else:

        S_Positives.append(X_res[i])
IFA=IsolationForest()

IFA.fit(S_Negatives)

S_train_IFA=IFA.predict(S_Negatives)

S_test_IFA=IFA.predict(S_Positives)
print("Training: Isolation Forest: ",(Train_Accuracy(S_train_IFA)),"%")

print("Test: Isolation Forest: ",(Test_Accuracy(S_test_IFA)),"%")
E_Data=pd.read_csv("../input/creditcard.csv")

E_Data.dropna(thresh=284315)

Outcome=E_Data['Class']

E_Data.drop('Class',axis=1,inplace=True)

X_res, y_res = sm.fit_sample(E_Data,Outcome)

x_train_E,x_test_E,y_train_E,y_test_E=train_test_split(X_res,y_res,test_size=0.5,random_state=0)

x_train_O,x_test_O,y_train_O,y_test_O=train_test_split(E_Data,Outcome,test_size=0.5,random_state=0)
clf = RandomForestClassifier(max_depth=2, random_state=0)

clf.fit(x_train_E, y_train_E)

print(classification_report(y_test_O,clf.predict(x_test_O)))