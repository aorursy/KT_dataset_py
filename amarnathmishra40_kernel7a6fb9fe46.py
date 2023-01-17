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



# Any results you write to the current directory are saved as output.
import pandas as pd

train = pd.read_csv("../input/mishra-ji/train.csv")

train.describe() # describe function is use to show brief description about datase
import matplotlib.pyplot as plt

plt.scatter(train['Loan_Status'],train['ApplicantIncome'])

plt.xlabel('Loan_Status')

plt.ylabel('ApplicantIncome')

plt.title("ApplicantIncome Vs Loan_Status")

plt.show()
train[train.Loan_Amount_Term >= 245].plot(kind='scatter', x='ApplicantIncome', y='CoapplicantIncome',color="red")

plt.xlabel("ApplicantIncome")

plt.ylabel("CoapplicantIncome")

plt.title("Loan_Amount_Term >= 245")

plt.grid(True)

plt.show()
train.hist(bins=50,figsize=(20,20))

plt.show()
co=train.corr()

co
import seaborn as sns 

sns.heatmap(co)
#spliting dataset into X and Y



X=train.drop(['Loan_Status','Loan_ID'],axis=1) #data set

Y=train['Loan_Status'] #labels
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder() 

Y=label_encoder.fit_transform(Y)



#this process can also be done by maping

#Y=train['Loan_Status'].map({'N':0,'Y':1}) this is the alternative way of label encoding
X #this is new input dataset
X.info() #use to check the datatype of all columns
X.isnull().sum() # this function is use to check null values in dataset
#now we are filling null values in dataset
X['Gender'].value_counts() # count all unique values in column
X['Gender'].fillna('Male',inplace=True) #filling null values with most occuring values
X['Married'].value_counts()
X['Married'].fillna('Yes',inplace=True)
X['Dependents'].value_counts()
X['Dependents'].fillna(0,inplace=True)
X['Self_Employed'].value_counts()
X['Self_Employed'].fillna('No',inplace=True)
mean_loan=X['LoanAmount'].mean()

X['LoanAmount'].fillna(mean_loan,inplace=True)
X['Loan_Amount_Term'].fillna(X['Loan_Amount_Term'].mean(),inplace=True)

X['Credit_History'].fillna(X['Credit_History'].mean(),inplace=True)
X.isnull().sum() #now we have repalced all the null values with respective mode in textual feilds and mean in numeric feilds
X=pd.get_dummies(X) #changes the alphabetical data into numerical data in adding more feild(this is done for machine to learn more properly and accuratly
X.head()
#now we are going to split dataset into traing data and testing data
from sklearn.model_selection import train_test_split #sklearn is a package in  python and model_selection is a class of this package from which train_test_split function is called

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.25)
X_train.shape

X_test.shape
Y_train.shape
Y_test.shape
#now applying logistic regression model for training the machine

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(X_train,Y_train)
#now applying SVC model for training the machine



from sklearn.svm import SVC

svc = SVC()



svc.fit(X_train, Y_train)
#now applying GaussianNB model for training the machine



from sklearn.naive_bayes import GaussianNB

n_b = GaussianNB()

n_b.fit(X_train, Y_train)
#now applying DecisionTreeClassifier model for training the machine



from sklearn.tree import DecisionTreeClassifier

dtf = DecisionTreeClassifier()

dtf.fit(X_train, Y_train)

#now applying KNeighborsClassifier model for training the machine



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()  

knn.fit(X_train, Y_train)
#now showing score of every model
print(lr.score(X_test, Y_test))

print(dtf.score(X_test, Y_test))

print(n_b.score(X_test, Y_test))

print(knn.score(X_test, Y_test))

print(svc.score(X_test, Y_test))
#printing confusion_matrix( we need confusion matrix when data set is imbalance)

from sklearn.metrics import confusion_matrix

y_predict=n_b.predict(X_test)

results=confusion_matrix(Y_test,y_predict)

print("Linear rergression confusion matrix:")

print(results)

y_predict=dtf.predict(X_test)

results=confusion_matrix(Y_test,y_predict)

print("decision Tree forest confusion matrix:")

print(results)
y_predict=lr.predict(X_test)

results=confusion_matrix(Y_test,y_predict)

print("decision Tree forest confusion matrix:")

print(results)
y_predict=dtf.predict(X_test)

results=confusion_matrix(Y_test,y_predict)

print("decision Tree forest confusion matrix:")

print(results)

y_predict=svc.predict(X_test)

results=confusion_matrix(Y_test,y_predict)

print("decision Tree forest confusion matrix:")

print(results)
proba=lr.predict_proba(X_test)[0:5] #this shows the probability of any result being No or Yes

print(proba)
from sklearn.metrics import roc_curve,auc,roc_auc_score
p=lr.predict(X_test)

p
Y_test
fpr,tpr,threshold=roc_curve(Y_test,lr.predict_proba(X_test)[:,1])
print('fpr:',fpr[0:5]),print('tpr:',tpr[0:5]),print('threshold:',threshold[0:5])
roc_auc=roc_auc_score(Y_test,lr.predict(X_test))

roc_auc
import matplotlib.pyplot as plt # matplotlib is used for plotting the curve

plt.figure()

lw=2

plt.plot(fpr,tpr,color='blue',lw=lw,label='ROC curve (area=%.5f100)'% roc_auc)

plt.plot([0,1],[0,1],color="navy",lw=lw,linestyle='--')

plt.xlim([0.0,1.1])

plt.ylim([0.0,1.1])

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.legend(loc="lower_right")

plt.show()
fpr1,tpr1,threshold1=roc_curve(Y_test,dtf.predict_proba(X_test)[:,1])
print('fpr:',fpr1[0:5]),print('tpr:',tpr1[0:5]),print('threshold:',threshold1[0:5])
roc_auc=roc_auc_score(Y_test,dtf.predict(X_test))

roc_auc
import matplotlib.pyplot as plt # matplotlib is used for plotting the curve

plt.figure()

lw=2

plt.plot(fpr1,tpr1,color='purple',lw=lw,label='ROC curve (area=%.5f100)'% roc_auc)

plt.plot([0,1],[0,1],color="navy",lw=lw,linestyle='--')

plt.xlim([0.0,1.1])

plt.ylim([0.0,1.1])

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.legend(loc="lower_right")

plt.show()
fpr2,tpr2,threshold2=roc_curve(Y_test,n_b.predict_proba(X_test)[:,1])
print('fpr:',fpr2[0:5]),print('tpr:',tpr2[0:5]),print('threshold:',threshold2[0:5])
roc_auc=roc_auc_score(Y_test,n_b.predict(X_test))

roc_auc
import matplotlib.pyplot as plt # matplotlib is used for plotting the curve

plt.figure()

lw=2

plt.plot(fpr2,tpr2,color='green',lw=lw,label='ROC curve (area=%.5f100)'% roc_auc)

plt.plot([0,1],[0,1],color="navy",lw=lw,linestyle='--')

plt.xlim([0.0,1.1])

plt.ylim([0.0,1.1])

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.legend(loc="lower_right")

plt.show()
fpr2,tpr2,threshold2=roc_curve(Y_test,n_b.predict_proba(X_test)[:,1])
print('fpr:',fpr2[0:5]),print('tpr:',tpr2[0:5]),print('threshold:',threshold2[0:5])
roc_auc=roc_auc_score(Y_test,n_b.predict(X_test))

roc_auc
import matplotlib.pyplot as plt # matplotlib is used for plotting the curve

plt.figure()

lw=2

plt.plot(fpr2,tpr2,color='red',lw=lw,label='ROC curve (area=%.5f100)'% roc_auc)

plt.plot([0,1],[0,1],color="navy",lw=lw,linestyle='--')

plt.xlim([0.0,1.1])

plt.ylim([0.0,1.1])

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.legend(loc="lower_right")

plt.show()
fpr3,tpr3,threshold3=roc_curve(Y_test,knn.predict_proba(X_test)[:,1])
print('fpr:',fpr3[0:5]),print('tpr:',tpr3[0:5]),print('threshold:',threshold3[0:5])
roc_auc=roc_auc_score(Y_test,knn.predict(X_test))

roc_auc
import matplotlib.pyplot as plt # matplotlib is used for plotting the curve

plt.figure()

lw=2

plt.plot(fpr3,tpr3,color='orange',lw=lw,label='ROC curve (area=%.5f100)'% roc_auc)

plt.plot([0,1],[0,1],color="navy",lw=lw,linestyle='--')

plt.xlim([0.0,1.1])

plt.ylim([0.0,1.1])

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.legend(loc="lower_right")

plt.show()
y_predict=knn.predict_proba(X_test)

y_predict[0:5]
from sklearn.preprocessing import binarize #important for converting the probability values in binary form
y_pred=binarize(y_predict,0.60)

y_pred[0:5]
y_pred1=y_pred[:,1]

y_pred1 # the output is in float we need to convert this output into integer type
y_pred2=y_pred1.astype(int)

y_pred2 # now the output is in integer type
tuned_result=confusion_matrix(Y_test,y_pred2)

print("confusion martics after tuning:")

print(tuned_result)
from sklearn.metrics import accuracy_score

accuracy_score(Y_test, y_pred2)