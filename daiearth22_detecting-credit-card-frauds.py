import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential

from keras.layers import Dense

from sklearn import svm

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import cross_val_score
W_Data=pd.read_csv("../input/creditcard.csv")

W_Data.dropna(thresh=284315)

Data=W_Data[1:15000]

Data.sample(frac=0.1).head(n=5)
Data.describe()
Train_Data=Data

Target=Data['Class']

Train_Data.drop('Class',axis=1,inplace=True)
clf_r=svm.SVC(kernel='rbf')

print("Accuracy: ",cross_val_score(clf_r,Train_Data,Target, cv=4).mean())
clf_l=svm.SVC(kernel='linear')

print("Accuracy: ",cross_val_score(clf_r,Train_Data,Target, cv=4).mean())
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(Train_Data,Target,test_size=0.5,random_state=0)
from sklearn.metrics import classification_report

clf_r.fit(x_train,y_train)

clf_l.fit(x_train,y_train)

print(classification_report(y_test,clf_l.predict(x_test)))
from sklearn.covariance import EllipticEnvelope

from sklearn.ensemble import IsolationForest
W_Data=pd.read_csv("../input/creditcard.csv")

W_Data.dropna(thresh=284315)

Data=W_Data[1:15000]
Negatives=Data[Data['Class']==0]

Positives=Data[Data['Class']==1]
len(Positives)
train_Pos=Positives[1:30]

test_Pos=Positives[30:61]
#RBF Kernel



clf_AD = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)

clf_AD.fit(Negatives)
#Linear Kernel



clf_AD_L = svm.OneClassSVM(nu=0.1, kernel="linear", gamma=0.1)

clf_AD_L.fit(Negatives)
IFA=IsolationForest()

IFA.fit(Negatives)
train_AD=clf_AD.predict(Negatives)

train_IFA=IFA.predict(Negatives)

train_AD_L=clf_AD_L.predict(Negatives)
test_AD=clf_AD.predict(Positives)

test_IFA=IFA.predict(Positives)

test_AD_L=clf_AD_L.predict(Positives)
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



print()

print("Training Accuracy: ")

print()



print("One Class SVM (RBF) : ",(Train_Accuracy(train_AD)),"%")

print("One Class SVM (Linear) : ",(Train_Accuracy(train_AD_L)),"%")

print("Isolation Forest: ",(Train_Accuracy(train_IFA)),"%")



print()

print("Test Accuracy: ")

print()



print("One Class SVM (RBF) : ",(Test_Accuracy(test_AD)),"%")

print("One Class SVM (Linear) : ",(Test_Accuracy(test_AD_L)),"%")

print("Isolation Forest: ",(Test_Accuracy(test_IFA)),"%")
from keras.models import Sequential 

from keras.layers import Dense,Activation
model=Sequential()

model.add(Dense(10,activation='relu',input_dim=30))

model.add(Dense(12,activation='relu',input_dim=30))

model.add(Dense(8,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train.as_matrix(),y_train.as_matrix(), epochs=5, batch_size=10)
print(model.evaluate(x_test.as_matrix(),y_test.as_matrix()))

scores=model.predict(x_test.as_matrix())
X_Test_Result=[]

for i in scores:

   

    if(i[0]>0.01):

        

            X_Test_Result.append(1)

            

    else:

        

            X_Test_Result.append(0)

        

           
plt.figure(figsize=(20,18))

Corr=Data[Data.columns].corr()

sns.heatmap(Corr,annot=True)