import pandas as pd
data  = pd.read_csv('../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')

vdata  = pd.read_csv('../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')
vdata.dropna(inplace=True)
data.head(10)
data.info()
data.dropna(inplace=True)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
datatype = data.dtypes==object

columns = data.columns[datatype].tolist()
data[columns]=data[columns].apply(lambda val : le.fit_transform(val))
data.head()
data.drop('Loan_ID',axis=1,inplace=True)
import seaborn as sns

import matplotlib.pyplot as plt
plt.figure(figsize=(12,7))

sns.heatmap(data.corr(),annot=True,fmt='.2f')
plt.figure(figsize=(12,7))

sns.scatterplot(data=data,y='ApplicantIncome',x='LoanAmount',hue='Loan_Status')
plt.figure(figsize=(12,7))

sns.scatterplot(data=data,y='ApplicantIncome',x='CoapplicantIncome',hue='Loan_Status')
plt.figure(figsize=(12,7))

sns.scatterplot(data=data,y='ApplicantIncome',x='LoanAmount',hue='Credit_History')
plt.figure(figsize=(12,7))

sns.barplot(data=vdata,x='Loan_Status',y='ApplicantIncome',hue='Married')
plt.figure(figsize=(12,7))

sns.boxplot(data=data,y='ApplicantIncome')
plt.figure(figsize=(12,7))

sns.boxplot(data=data,y='CoapplicantIncome')
plt.figure(figsize=(12,7))

sns.boxplot(data=data,y='LoanAmount')
desc = data['ApplicantIncome'].describe()

q1 = desc[4]

q3 = desc[6]

iqr = q3-q1

low = q1 -0.5*iqr

up = q3 +0.5*iqr

print('Lower limit is ',low,'Upper lim is ',up)

print('New DataFrame')

data = data[(data['ApplicantIncome']>low)&(data['ApplicantIncome']<up)]

data.head()
# Checking outlier

plt.figure(figsize=(12,7))

sns.boxplot(data=data,y='ApplicantIncome')
desc = data['CoapplicantIncome'].describe()

q1 = desc[4]

q3 = desc[6]

iqr = q3-q1

low = q1 -0.5*iqr

up = q3 +0.5*iqr

print('Lower limit is ',low,'Upper lim is ',up)

print('New DataFrame')

data = data[(data['CoapplicantIncome']>low)&(data['CoapplicantIncome']<up)]

data.head()
# Checking outlier

plt.figure(figsize=(12,7))

sns.boxplot(data=data,y='CoapplicantIncome')
desc = data['LoanAmount'].describe()

q1 = desc[4]

q3 = desc[6]

iqr = q3-q1

low = q1 -0.5*iqr

up = q3 +0.5*iqr

print('Lower limit is ',low,'Upper lim is ',up)

print('New DataFrame')

data = data[(data['LoanAmount']>low)&(data['LoanAmount']<up)]

data.head()
# Checking outlier

plt.figure(figsize=(12,7))

sns.boxplot(data=data,y='LoanAmount')
data
data.info()
plt.figure(figsize=(12,7))

sns.countplot(data=vdata,x='Property_Area',hue='Loan_Status')
vdata['Loan_Amount_Term'].nunique()
plt.figure(figsize=(12,7))

sns.countplot(data=vdata,x='Loan_Amount_Term',hue='Loan_Status')
data.info()
plt.figure(figsize=(12,7))

sns.countplot(data=vdata,x='Gender',hue='Loan_Status')
plt.figure(figsize=(12,7))

sns.countplot(data=vdata,x='Self_Employed',hue='Loan_Status')
plt.figure(figsize=(12,7))

sns.countplot(data=vdata,x='Dependents',hue='Loan_Status')
data.head()
plt.figure(figsize=(12,7))

sns.heatmap(data=data.corr(),annot=True,fmt='.2f',cmap='winter')
x = data.loc[:,['Gender','Married','CoapplicantIncome','LoanAmount','Credit_History']]

y = data['Loan_Status']
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.1,random_state=101)
from sklearn.svm import SVC
svr = SVC()
svr.fit(train_x,train_y)
predict = svr.predict(test_x)

from sklearn.metrics import accuracy_score,confusion_matrix
import numpy as np
accuracy_score(test_y,predict)
confusion_matrix(test_y,predict)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_x,train_y)
predict = lr.predict(test_x)
accuracy_score(test_y,predict)*100
confusion_matrix(test_y,predict)
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(train_x,train_y)
predict = dtc.predict(test_x)
accuracy_score(test_y,predict)
confusion_matrix(test_y,predict)
from sklearn.neighbors import KNeighborsClassifier
knn =KNeighborsClassifier(n_neighbors=5)

knn.fit(train_x,train_y)
predict = knn.predict(test_x)
accuracy_score(test_y,predict)
confusion_matrix(test_y,predict)
error = []

for k in range(1,41):

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(train_x,train_y)

    predict = knn.predict(test_x)

    error.append(np.mean(test_y!=predict))
plt.figure(figsize=(10,6))

plt.plot(range(1,41),error,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
knn =KNeighborsClassifier(n_neighbors=37)

knn.fit(train_x,train_y)

predict = knn.predict(test_x)

print('Accuracy ',accuracy_score(test_y,predict))

print('Confusion Matrix \n',confusion_matrix(test_y,predict))