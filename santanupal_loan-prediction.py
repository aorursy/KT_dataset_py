import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split # it is used for spliting the data
from sklearn.tree import DecisionTreeClassifier # it is tool we are using mathametical coditions line 'Entropy'
from sklearn.metrics import accuracy_score # accuracy b/w training and testing data
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("../input/test.csv")
data1 = pd.read_csv("../input/train.csv")
# printing the datalength
print('DataLength test:-',data.shape) # shape will show lines and columns
print('DataLength train:-',data1.shape)
# data head for train
data.head()
# data head for test
data1.head()
data.describe()
# Box Plot for understanding the distributions and to observe the outliers.
%matplotlib inline
# Histogram of variable ApplicantIncome
data['ApplicantIncome'].hist()
# checking the null values in data set
data.isnull().sum()
# Impute missing values for Gender
data['Gender'].fillna(data['Gender'].mode()[0],inplace=True)
# Impute missing values for Married
data['Married'].fillna(data['Married'].mode()[0],inplace=True)
# Impute missing values for Dependents
data['Dependents'].fillna(data['Dependents'].mode()[0],inplace=True)
# Impute missing values for Credit_History
data['Credit_History'].fillna(data['Credit_History'].mode()[0],inplace=True)
# Convert all non-numeric values to number

cat=['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area']
for var in cat:
    le = LabelEncoder()
    data[var]=le.fit_transform(data[var].astype('str'))
data.dtypes
#Imputing Missing values with mean for continuous variable
data['LoanAmount'].fillna(data['LoanAmount'].mean(), inplace=True)
data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mean(), inplace=True)
data['ApplicantIncome'].fillna(data['ApplicantIncome'].mean(), inplace=True)
data['CoapplicantIncome'].fillna(data['CoapplicantIncome'].mean(), inplace=True)
# Box Plot for variable ApplicantIncome by variable Education of training data set
data.boxplot(column='ApplicantIncome', by = 'Education')
# Histogram of variable LoanAmount
data['LoanAmount'].hist(bins=50)
# Box Plot for variable LoanAmount of training data set
data.boxplot(column='LoanAmount')
# Add both ApplicantIncome and CoapplicantIncome to TotalIncome
data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']

# Looking at the distribtion of TotalIncome
data['LoanAmount'].hist(bins=20)
# separating the target variable from 'test'
array = data.values
X = array[:,6:11]
Y = array[:,11]
x_train, x_test, y_train, y_test =train_test_split(X, Y, test_size=0.2, random_state=7)
 #K-NEAREST NEIGHBOR(kNN) ALGORITHM
kNN=KNeighborsClassifier()
kNN. fit(x_train,y_train)
#Predict values for cv data
pred_cv=kNN.predict(x_test)
pre=accuracy_score(y_test,pred_cv )*100 
print("accuracy score is:-",pre)
#naive bayes
nb=GaussianNB()
nb.fit(x_train,y_train)
#Predict values 
pred_cv1=nb.predict(x_test)
#Evaluate accuracy of model
pred=accuracy_score(y_test,pred_cv1)*100 
print("the accuracy score is ",pred)
# Logistic Regression
logmodel=LogisticRegression()
logmodel.fit(x_train,y_train)
Pred=logmodel.predict(x_test)
Lf=accuracy_score(y_test,Pred)*100
print("Accurate Prediction for Logistic Regrassion:-",Lf)
# Decision Tree Prediction
# formula of entropy is k∑i=1  p(value).log2(p(value i))
model = DecisionTreeClassifier()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
Df=accuracy_score(y_test, predictions)*100
print('Accurate Prediction for Decision Tree:-',Df)
# Box plotting for the Accuracy of all algorithms 
plt.bar('KNN',[pre],label='K Nearest Neighbors')
plt.bar('NB',[pred],label='Naive Bayes')
plt.bar('LR',[Lf],label='Logistic Regression')
plt.bar('DST',[Df],label='Decision Tree')
plt.title("Accuracy")
plt.legend()
plt.show