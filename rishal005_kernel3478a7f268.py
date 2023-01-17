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

from sklearn.metrics import roc_curve,auc,roc_auc_score

train = pd.read_csv("../input/train.csv")

train.head() #head is used to view top 5 row
train['Loan_Status'].value_counts() # the data set is imbalanced
import matplotlib.pyplot as plt

plt.scatter(train['Loan_Status'],train['ApplicantIncome'])

plt.xlabel('Loan_Status')

plt.ylabel('ApplicantIncome')

plt.title("ApplicantIncome Vs Loan_Status")

plt.show()
train.hist(bins=50,figsize=(20,20))

plt.show
train.describe() # Describe function is used to take a brief description about the data (like mean ,stander deaviation ,etc)
co=train.corr()

co
import seaborn as sns 

sns.heatmap(co)
#now spliting the data into data set and labels as X and y respectively

X=train.drop(['Loan_Status','Loan_ID'], axis=1) #data set

y=train['Loan_Status'].map({'N':0,'Y':1})# we can do this task by label encoding(label_encoder=preprocessing.LablelEncoder()// y=label_encoder.fit_transform(y))

#y=train['Loan_Status'] #labels 
X 
X.isnull().sum() # used to check the null values in each field
X.info() #its used to check the data type of the fields
#now felling the null values with suitable values
List=X.columns.tolist()

List
List.remove("ApplicantIncome")

List.remove("CoapplicantIncome")

List.remove("LoanAmount")

List
def count(x):

    t=X[x].value_counts()

    print(t)

def fill(m):

    X[m].fillna(X[m].mode()[0],inplace=True)
for i in List:

    count(i)

    fill(i)
mean_loan=X["ApplicantIncome"].mean()

X["ApplicantIncome"].fillna(mean_loan,inplace=True)
mean_loan=X["CoapplicantIncome"].mean()

X["CoapplicantIncome"].fillna(mean_loan,inplace=True)
mean_loan=X['LoanAmount'].mean()

X['LoanAmount'].fillna(mean_loan,inplace=True)
X.isnull().sum() # now we have replaced all the null values with respective mode in textual fields and mean in numaric fields
X=pd.get_dummies(X) # changes the alphabatical data into numarical data and adding more fields ( this is done for machine to learn more properly and accuratlyX)
from sklearn.preprocessing import MinMaxScaler

scale=MinMaxScaler()

X[X.columns]=scale.fit_transform(X[X.columns])
X.head()
X.shape
# now we are going to split the data into training data and testing data 
from sklearn.model_selection import train_test_split # sk learn is the package in python and model selection is the class of this package from wich train _test_split function is called



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.30)
X_train.shape,X_test.shape
y_train.shape,y_test.shape
# now applying logistic regression model for train the machine

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,y_train)

# now applying svm model for train the machine

from sklearn.svm import SVC

svc = SVC()



svc.fit(X_train, y_train)
# now applying DecisionTreeClassifier model for train the machine

from sklearn.tree import DecisionTreeClassifier

dtf = DecisionTreeClassifier()

dtf.fit(X_train, y_train)
# now applying GaussianNB model for train the machine

from sklearn.naive_bayes import GaussianNB

n_b = GaussianNB()

n_b.fit(X_train, y_train)
# now applying KNeighborsClassifier model for train the machine

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()  

knn.fit(X_train, y_train)
# now checking the accuracy for each model below and choosing the best out of it
print(lr.score(X_test, y_test))

print(dtf.score(X_test, y_test))

print(n_b.score(X_test, y_test))

print(knn.score(X_test, y_test))

print(svc.score(X_test, y_test))
#printing confusion_matrix( we need confusion matrix when data set is imbalance)

from sklearn.metrics import confusion_matrix

y_predict=lr.predict(X_test)

results=confusion_matrix(y_test,y_predict)

print("Linear rergression confusion matrix:")

print(results)
y_predict=dtf.predict(X_test)

results=confusion_matrix(y_test,y_predict)

print("decision Tree forest confusion matrix:")

print(results)
y_predict=n_b.predict(X_test)

results=confusion_matrix(y_test,y_predict)

print("n_b confusion matrix:")

print(results)
y_predict=knn.predict(X_test)

results=confusion_matrix(y_test,y_predict)

print("knn confusion matrix:")

print(results)
y_predict=svc.predict(X_test)

results=confusion_matrix(y_test,y_predict)

print("svc confusion matrix:")

print(results)
#CONFUSION MATRIX OF LOGISTIC REGRESSION MODEL IS GOOD 
proba=lr.predict_proba(X_test)[0:5] #this shows the probability of any result being No or Yes

print(proba)
p=lr.predict_proba(X_test)[:,1]

p[0:5]
fpr,tpr,threshold=roc_curve(y_test,p)
print("fpr",fpr[0:5])

print("tpr",tpr[0:5])

print("threshold",threshold[0:5])
roc_auc=roc_auc_score(y_test,lr.predict(X_test))
roc_auc
import matplotlib.pyplot as plt # matplotlib is used for plotting the curve

plt.figure()

lw=2

plt.plot(fpr,tpr,color='orange',lw=lw,label='ROC curve (area=%.5f100)'% roc_auc)

plt.plot([0,1],[0,1],color="navy",lw=lw,linestyle='--')

plt.xlim([0.0,1.1])

plt.ylim([0.0,1.1])

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.legend(loc="lower_right")

plt.show()
y_predict=lr.predict_proba(X_test)

y_predict[0:5]
from sklearn.preprocessing import binarize #important for converting the probability values in binary form
y_pred=binarize(y_predict,0.495)

y_pred[0:5]
y_pred1=y_pred[:,1]

y_pred1 # the output is in float we need to convert this output into integer type
y_pred2=y_pred1.astype(int)

y_pred2 # now the output is in integer type
tuned_result=confusion_matrix(y_test,y_pred2)

print("confusion martics after tuning:")

print(tuned_result)
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred2)
proba=dtf.predict_proba(X_test)[0:5] #this shows the probability of any result being No or Yes

print(proba)
p=dtf.predict_proba(X_test)[:,1]

p[0:5]
fpr,tpr,threshold=roc_curve(y_test,p)
print("fpr",fpr[0:5])

print("tpr",tpr[0:5])

print("threshold",threshold[0:5])
roc_auc=roc_auc_score(y_test,lr.predict(X_test))
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
y_predict=dtf.predict_proba(X_test)

y_predict[0:5]
from sklearn.preprocessing import binarize #important for converting the probability values in binary form
y_pred=binarize(y_predict,0.495)

y_pred[0:5]
y_pred1=y_pred[:,1]

y_pred1 # the output is in float we need to convert this output into integer type
y_pred2=y_pred1.astype(int)

y_pred2 # now the output is in integer type
tuned_result=confusion_matrix(y_test,y_pred2)

print("confusion martics after tuning:")

print(tuned_result)
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred2)
proba=n_b.predict_proba(X_test[0:5]) #this shows the probability of any result being No or Yes

print(proba)
from sklearn.metrics import roc_curve,auc,roc_auc_score
fpr,tpr,threshold=roc_curve(y_test,n_b.predict_proba(X_test)[:,1])
n_b.predict_proba(X_test)[:,1][0:5]
print("fpr:",fpr[0:5])

print("tpr:",tpr[0:5])

print("threshold:",threshold[0:5])
roc_auc=roc_auc_score(y_test,n_b.predict(X_test))
roc_auc
import matplotlib.pyplot as plt # matplotlib is used for plotting the curve

plt.figure()

lw=2

plt.plot(fpr,tpr,color='red',lw=lw,label='ROC curve (area=%.5f100)'% roc_auc)

plt.plot([0,1],[0,1],color="navy",lw=lw,linestyle='--')

plt.xlim([0.0,1.1])

plt.ylim([0.0,1.1])

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.legend(loc="lower_right")

plt.show()
y_predict=n_b.predict_proba(X_test)

y_predict[0:5]
from sklearn.preprocessing import binarize #important for converting the probability values in binary form
y_pred=binarize(y_predict,0.53)

y_pred[0:5]
y_pred1=y_pred[:,1]

y_pred1 # the output is in float we need to convert this output into integer type
y_pred1=y_pred[:,1]

y_pred1 # the output is in float we need to convert this output into integer type
y_pred2=y_pred1.astype(int)

y_pred2 # now the output is in integer type
tuned_result=confusion_matrix(y_test,y_pred2)

print("confusion martics after tuning:")

print(tuned_result)
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred2)

proba=knn.predict_proba(X_test[0:5]) #this shows the probability of any result being No or Yes

print(proba)
from sklearn.metrics import roc_curve,auc,roc_auc_score
fpr,tpr,threshold=roc_curve(y_test,knn.predict_proba(X_test)[:,1])
knn.predict_proba(X_test)[:,1][0:5]
print("fpr:",fpr[0:5])

print("tpr:",tpr[0:5])

print("threshold:",threshold[0:5])
roc_auc=roc_auc_score(y_test,knn.predict(X_test))
roc_auc
import matplotlib.pyplot as plt # matplotlib is used for plotting the curve

plt.figure()

lw=2

plt.plot(fpr,tpr,color='green',lw=lw,label='ROC curve (area=%.5f100)'% roc_auc)

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

y_pred=binarize(y_predict,0.53)

y_pred[0:5]
y_pred1=y_pred[:,1]

y_pred1 # the output is in float we need to convert this output into integer type
y_pred2=y_pred1.astype(int)

y_pred2 # now the output is in integer type
tuned_result=confusion_matrix(y_test,y_pred2)

print("confusion martics after tuning:")

print(tuned_result)
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred2)
