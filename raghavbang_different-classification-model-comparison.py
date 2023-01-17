import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

data_train=pd.read_csv("../input/loan-prediction-data/train.csv")

data_test=pd.read_csv("../input/loan-prediction-data/test.csv")
data_test.info()
data_train.info()
data_train.isnull().sum()
data_test.isnull().sum()
data_train["Gender"].value_counts()
data_train["Gender"].fillna("Male",inplace=True)

data_test["Gender"].fillna("Male",inplace=True)
data_train["Married"].value_counts()
data_train["Married"].fillna("Yes",inplace=True)
data_train["Dependents"].value_counts()
data_train["Dependents"].fillna("0",inplace=True)

data_test["Dependents"].fillna("0",inplace=True)
data_train["Self_Employed"].value_counts()
data_train["Self_Employed"].fillna("No",inplace=True)

data_test["Self_Employed"].fillna("No",inplace=True)
data_train["LoanAmount"].describe()
data_train["LoanAmount"].fillna(146,inplace=True)

data_test["LoanAmount"].describe()
data_test["LoanAmount"].fillna(136,inplace=True)

data_test["Loan_Amount_Term"].describe()
data_train["Loan_Amount_Term"].describe()
data_train["Loan_Amount_Term"].fillna(342,inplace=True)

data_test["Loan_Amount_Term"].fillna(342,inplace=True)
data_train["Credit_History"].describe()
data_test["Credit_History"].describe()
data_train["Credit_History"].fillna(1,inplace=True)

data_test["Credit_History"].fillna(1,inplace=True)
data_test.isnull().sum()
data_train.isnull().sum()
data_train
plt.bar(data_train["Gender"].unique(),data_train["Gender"].value_counts(),color="Red")

plt.xlabel("Gender")

plt.ylabel("Count")

plt.title("Gender Count")

plt.show()



plt.pie(data_train["Married"].value_counts(),shadow=True,autopct="%1.1f%%",radius=1.2,startangle=120,labels=["Yes","No"])

plt.title("Married %")

plt.show()
import seaborn as sns

sns.catplot('Gender', col='Married',data=data_train, kind = 'count')

plt.show()
data_train["CoapplicantIncome"].value_counts()
data_train=data_train.drop(["Loan_ID","CoapplicantIncome"],axis=1)

data_test=data_test.drop(["Loan_ID"],axis=1)
import seaborn as sns

sns.catplot('Gender', col='Loan_Status',data=data_train, kind = 'count',height=3)

sns.catplot('Married', col='Loan_Status',data=data_train, kind = 'count',height=3)

sns.catplot('Dependents', col='Loan_Status',data=data_train, kind = 'count',height=3)

sns.catplot('Education', col='Loan_Status',data=data_train, kind = 'count',height=3)

sns.catplot('Self_Employed', col='Loan_Status',data=data_train, kind = 'count',height=3)

sns.catplot('Property_Area', col='Loan_Status',data=data_train, kind = 'count',height=3)
data_train=pd.get_dummies(data_train,drop_first=True)
data_train
print(pd.crosstab(index=data_train["Gender_Male"], columns= "Loan_Status", normalize=True))

print("#####################################################################################")

print(pd.crosstab(index=data_train["Married_Yes"], columns= "Loan_Status", normalize=True))
from sklearn.model_selection  import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC,LinearSVC

from sklearn import preprocessing

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
x=data_train.drop(["Loan_Status_Y"],axis=1)

y=data_train["Loan_Status_Y"].copy()
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.4)

X_train=preprocessing.scale(X_train)

X_test=preprocessing.scale(X_test)
random_forest = RandomForestClassifier(n_estimators=40)

random_forest.fit(X_train, Y_train)

Y_predict = random_forest.predict(X_test)

Y1_predict=random_forest.predict(X_train)

acc_rand=random_forest.score(X_test, Y_test)

print("###########Train##############")

print("Confusion Matrix")

print(confusion_matrix(Y_train, Y1_predict))

print ('Accuracy Score :',accuracy_score(Y_train, Y1_predict) )

print ('Report : ')

print (classification_report(Y_train, Y1_predict) )

print("###########Test##############")

print("Confusion Matrix")

print(confusion_matrix(Y_test, Y_predict))

print ('Accuracy Score :',accuracy_score(Y_test, Y_predict) )

print ('Report : ')

print (classification_report(Y_test, Y_predict) )
logreg = LogisticRegression(solver='lbfgs')

logreg.fit(X_train, Y_train)

Y_predict = logreg.predict(X_test)

Y1_predict=logreg.predict(X_train)

acc_log=logreg.score(X_test, Y_test)

print("###########Train##############")

print("Confusion Matrix")

print(confusion_matrix(Y_train, Y1_predict))

print ('Accuracy Score :',accuracy_score(Y_train, Y1_predict) )

print ('Report : ')

print (classification_report(Y_train, Y1_predict) )

print("###########Test##############")

print("Confusion Matrix")

print(confusion_matrix(Y_test, Y_predict))

print ('Accuracy Score :',accuracy_score(Y_test, Y_predict) )

print ('Report : ')

print (classification_report(Y_test, Y_predict) )
decision_tree = DecisionTreeClassifier() 

decision_tree.fit(X_train, Y_train)  

Y_predict = decision_tree.predict(X_test)

Y1_predict=decision_tree.predict(X_train)

acc_dec=decision_tree.score(X_test, Y_test)

print("###########Train##############")

print("Confusion Matrix")

print(confusion_matrix(Y_train, Y1_predict))

print ('Accuracy Score :',accuracy_score(Y_train, Y1_predict) )

print ('Report : ')

print (classification_report(Y_train, Y1_predict) )

print("###########Test##############")

print("Confusion Matrix")

print(confusion_matrix(Y_test, Y_predict))

print ('Accuracy Score :',accuracy_score(Y_test, Y_predict) )

print ('Report : ')

print (classification_report(Y_test, Y_predict) )
classifier = KNeighborsClassifier(n_neighbors = 30)

classifier.fit(X_train, Y_train)

Y_predict = classifier.predict(X_test)

Y1_predict=classifier.predict(X_train)

acc_knn=classifier.score(X_test, Y_test)

print("###########Train##############")

print("Confusion Matrix")

print(confusion_matrix(Y_train, Y1_predict))

print ('Accuracy Score :',accuracy_score(Y_train, Y1_predict) )

print ('Report : ')

print (classification_report(Y_train, Y1_predict) )

print("###########Test##############")

print("Confusion Matrix")

print(confusion_matrix(Y_test, Y_predict))

print ('Accuracy Score :',accuracy_score(Y_test, Y_predict) )

print ('Report : ')

print (classification_report(Y_test, Y_predict) )
linear_svc=LinearSVC(max_iter=10000)

linear_svc.fit(X_train, Y_train)

Y_predict = linear_svc.predict(X_test)

Y1_predict=linear_svc.predict(X_train)

acc_svm=linear_svc.score(X_test, Y_test)

print("###########Train##############")

print("Confusion Matrix")

print(confusion_matrix(Y_train, Y1_predict))

print ('Accuracy Score :',accuracy_score(Y_train, Y1_predict) )

print ('Report : ')

print (classification_report(Y_train, Y1_predict) )

print("###########Test##############")

print("Confusion Matrix")

print(confusion_matrix(Y_test, Y_predict))

print ('Accuracy Score :',accuracy_score(Y_test, Y_predict) )

print ('Report : ')

print (classification_report(Y_test, Y_predict) )
model=["RandomForestClass","Logistic REG","KNN","Decision Tree","SVM"]

Acc=[acc_rand*100,acc_log*100,acc_knn*100,acc_dec*100,acc_svm*100]

plt.barh(model,Acc)

plt.title("Test Accuracy")

plt.show()