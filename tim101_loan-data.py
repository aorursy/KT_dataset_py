import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

from pandas import DataFrame, read_csv
import os

print(os.listdir("../input"))
train_data = read_csv('../input/train.csv')

train_data.head()
train_data.describe()
train_data.isnull().sum()
train_data['Credit_History'] = train_data['Credit_History'].fillna(train_data['Credit_History'].mode()[0])
loan_mean = np.nanmean(train_data['LoanAmount'])

term_mean = np.nanmean(train_data['Loan_Amount_Term'])
train_data['Gender'] = train_data['Gender'].fillna(train_data['Gender'].mode()[0])

train_data['Married'] = train_data['Married'].fillna(train_data['Married'].mode()[0])

train_data['Dependents'] = train_data['Dependents'].fillna(train_data['Dependents'].mode()[0])

train_data['Self_Employed'] = train_data['Self_Employed'].fillna(train_data['Self_Employed'].mode()[0])

train_data['Credit_History'] = train_data['Credit_History'].fillna(train_data['Credit_History'].mode()[0])

train_data['LoanAmount'] = train_data['LoanAmount'].fillna(loan_mean)

train_data['Loan_Amount_Term'] = train_data['Loan_Amount_Term'].fillna(term_mean)
train_data.isnull().sum()
from sklearn.preprocessing import LabelEncoder

features = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']

le = LabelEncoder()

for i in features:

    train_data[i] = le.fit_transform(train_data[i])
train_data = train_data.drop(['Loan_ID'], axis = 1)
train_data.head()
# Algorithms

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB



# linear algebra

import numpy as np 



# data processing

import pandas as pd 



# data visualization

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style



import re
X = train_data.iloc[:, :11].values  

y = train_data.iloc[:, 11].values
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()  

X_train = sc.fit_transform(X_train)  

X_test = sc.transform(X_test)  
#Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)



Y_pred = logreg.predict(X_test)



acc_log = round(logreg.score(X_train, y_train) * 100, 2)
#Stochastic Gradient Descent (SGD):



sgd = linear_model.SGDClassifier(max_iter=5, tol=None)

sgd.fit(X_train, y_train)

Y_pred = sgd.predict(X_test)



sgd.score(X_train, y_train)



acc_sgd = round(sgd.score(X_train, y_train) * 100, 2)
#Gaussian Naive Bayes:



gaussian = GaussianNB() 

gaussian.fit(X_train, y_train)  

Y_pred = gaussian.predict(X_test)  

acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)
#Perceptron:



perceptron = Perceptron(max_iter=5)

perceptron.fit(X_train, y_train)



Y_pred = perceptron.predict(X_test)



acc_perceptron = round(perceptron.score(X_train, y_train) * 100, 2)


#Linear Support Vector Machine:



linear_svc = LinearSVC()

linear_svc.fit(X_train, y_train)



Y_pred = linear_svc.predict(X_test)



acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 2)
#Decision Tree:



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, y_train)  

Ydt_pred = decision_tree.predict(X_test)  

acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)

# KNN 

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)  

Y_pred = knn.predict(X_test)  

acc_knn = round(knn.score(X_train, y_train) * 100, 2)
#Random Forest

random_forest = RandomForestClassifier(n_estimators=100, random_state = 1)

random_forest.fit(X_train, y_train)



Y_prediction = random_forest.predict(X_test)



random_forest.score(X_train, y_train)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
#Which is the best Model ?

results = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 

              'Decision Tree'],

    'Score': [acc_linear_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_decision_tree]})

result_df = results.sort_values(by='Score', ascending=False)

result_df = result_df.set_index('Score')

result_df.head(9)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



print(confusion_matrix(y_test,Y_prediction))  

print(classification_report(y_test,Y_prediction))  

print(accuracy_score(y_test, Y_prediction))
print(confusion_matrix(y_test,Ydt_pred))  

print(classification_report(y_test,Ydt_pred))  

print(accuracy_score(y_test, Ydt_pred))
forest = random_forest

importances = forest.feature_importances_

features = pd.DataFrame({

    'Model': ['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome',

              'CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area'],

    'Score': importances})



imp_df = features.sort_values(by='Score', ascending=False)

imp_df
Total_income = pd.DataFrame(data=train_data['ApplicantIncome']+train_data['CoapplicantIncome'])
train_data['Total_income'] = pd.DataFrame(data=Total_income)
#data = data.drop(['Gender', 'Self_Employed', 'Education'], axis = 1)
train_data = train_data.drop(['ApplicantIncome', 'CoapplicantIncome'], axis = 1)
train_data['Debt_Income_Ratio'] = train_data['Total_income'] / train_data['LoanAmount']
len(train_data[train_data.Total_income > 30000])
train_data = train_data[train_data.Total_income < 30000]
train_data.head()
y = train_data.pop('Loan_Status')
y = y.iloc[:].values
train_data.head()
X = train_data.iloc[:].values

X
train_data.isnull().sum()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(n_estimators=100, random_state = 1)

clf.fit(X_train, y_train)  

y_pred = clf.predict(X_test)
result = pd.DataFrame(data=y_pred, columns = ['Prediction_RF'])

result["Actual"] = pd.DataFrame(data=y_test)
result.head()
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



print(confusion_matrix(y_test,y_pred))  

print(classification_report(y_test,y_pred))  

print(accuracy_score(y_test, y_pred))
arr = []

for i in train_data:

    arr.append(i)



forest = clf

importances = forest.feature_importances_

features = pd.DataFrame({

    'Features': arr,

    'Score': importances})



imp_df = features.sort_values(by='Score', ascending=False)

imp_df
data2 = train_data.drop(['Gender', 'Married', 'Education'], axis = 1)
data2.head()
X = data2.iloc[:].values
from sklearn.model_selection import train_test_split



X2_train, X2_test, y2_train, y2_test = train_test_split(X, y, test_size=0.3, random_state=0)
#from sklearn import preprocessing



#std_scale = preprocessing.StandardScaler().fit(X2_train)

#X2_train = std_scale.transform(X2_train)

#X2_test = std_scale.transform(X2_test)



# However, for random forest we don't really need scaling
from sklearn.ensemble import RandomForestClassifier



clf2 = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=1)

clf2.fit(X2_train, y2_train)
y2_pred = clf2.predict(X2_test)
result = pd.DataFrame(data=y_pred, columns = ['Prediction_RF'])

result["Actual"] = pd.DataFrame(data=y2_test)

result.head()
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



print(confusion_matrix(y2_test,y2_pred))  

print(classification_report(y2_test,y2_pred))  

print(accuracy_score(y2_test, y2_pred))
arr = []

for i in data2:

    arr.append(i)



forest = clf2

importances = forest.feature_importances_

features = pd.DataFrame({

    'Model': arr,

    'Score': importances})



imp_df = features.sort_values(by='Score', ascending=False)

imp_df