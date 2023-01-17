# importing libraries

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt
# Loading customer churn data set into cust_churn dataframe:

cust_churn = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
# Top 5 records:

cust_churn.head()
# Using column indexing: c_5 = cust_churn.loc['Dependents']

# Below is the integer indexing example we are using (syntax-> df.iloc[rows,columns])

c_5 = cust_churn.iloc[:,4]
c_5.head()
# Similary we are extracting 15th column:

c_15 = cust_churn.iloc[:,14]

c_15.head()
senior_male_electronic = cust_churn[(cust_churn['gender'] == 'Male') & (cust_churn['SeniorCitizen'] == 1) & (cust_churn['PaymentMethod'] == 'Electronic check')]
senior_male_electronic.head()
customer_total_tenure = cust_churn[(cust_churn['tenure']>70) | (cust_churn['MonthlyCharges']> 100)]
customer_total_tenure.head()
two_mail_yes = cust_churn[(cust_churn['Contract']=='Two year')&(cust_churn['PaymentMethod']=='Mailed check')&(cust_churn['Churn']=='Yes')]
two_mail_yes.head()
custumer_333 = cust_churn.sample(n=333)

custumer_333.head()
cust_churn['Churn'].value_counts()
# similarly we can calculate for Contract column:

cust_churn['Contract'].value_counts()
# plt.bar(arg1, arg2, color = 'red')

# arg1 is distinct values of InternetService columns: cust_churn['InternetService'].value_counts().keys().to_list()

# arg2 is the count of Internetservice columns



plt.bar(cust_churn['InternetService'].value_counts().keys().tolist(),cust_churn['InternetService'].value_counts().tolist(), color = 'red')



# Now we need label and title:

plt.xlabel('Categories of Internet Service')

plt.ylabel('Count')

plt.title('Distribution of Internet Service')
# plt.hist(arg1, bins, color)

plt.hist(cust_churn['tenure'], bins = 30 ,color = 'green')



plt.title('Distribution of tenure')
# plt.scatter(x, y)



plt.scatter(cust_churn['tenure'], cust_churn['MonthlyCharges'])



plt.xlabel('Tenure')

plt.ylabel('Monthly Charges')

plt.title('MonthlyCharges vs tenure')
# DF.boxplot(column='y-axis', by='x-axis')



cust_churn.boxplot(column=['tenure'], by=['Contract'])



plt.xlabel('Contract')

plt.ylabel('Tenure')

plt.title('Contract vs Tenure')
# importing ML libraries

from sklearn import linear_model

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
y = cust_churn[['MonthlyCharges']]

x = cust_churn[['tenure']]
#y.head(),x.head()
# Dividing the dataset into training dataset and testing dataset

# train_test_split gives us four results so we store them in differnet datasets

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.30, random_state=0)
x_train.shape,y_train.shape,x_test.shape,y_test.shape
# Creating instatnce/object of LinearRegression Class:

regressor = LinearRegression()



# Fitting the train data sets

regressor.fit(x_train, y_train)
# Now predict the values based on test data set

y_predict = regressor.predict(x_test)
# for RMS we will use numpy function (np.sqrt) and we will use our test data set

from sklearn.metrics import mean_squared_error



np.sqrt(mean_squared_error(y_test,y_predict))
print(y_predict[:5]) # Predicted Values

print(y_test[:5]) # Actual Values
# importing libraries

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
# seperating dependent and independent variables

x = cust_churn[['MonthlyCharges']]

y = cust_churn[['Churn']]
# Dividing dataset into training and test data set

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.35,random_state=0)



# Let's check size of our train and test data sets

x_train.shape,y_train.shape,x_test.shape,y_test.shape
# creating instantce/object of Logistic regression class

regressor = LogisticRegression()



# Fitting the model on our training data sets:

regressor.fit(x_train,y_train)
# Predicting values

y_predict = regressor.predict(x_test)



# Checking first 5 values

y_predict[:5]
# importing libraries

from sklearn.metrics import confusion_matrix, accuracy_score
# Confusion matrix:

confusion_matrix(y_test, y_predict)
# Accuracy Score:

accuracy_score(y_test, y_predict)



# (1815+0)/(1815+0+651+0)
# Importing libraries

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
# Seperating dependent and Independent Variables

x = cust_churn[['MonthlyCharges','tenure']]

y = cust_churn[['Churn']]
# Dividing data set into training and test data sets

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)



# Checking shape of train and test data sets

x_train.shape,y_train.shape,x_test.shape,y_test.shape
# Creating instance/object of Logistic Regression class

regressor = LogisticRegression()



# Fitting data sets to our model

regressor.fit(x_train,y_train)
# Predicting values

y_predict = regressor.predict(x_test)



# Checking first 5 values

y_predict[:5]
from sklearn.metrics import confusion_matrix, accuracy_score
# Confusion Matrix

confusion_matrix(y_test,y_predict)
# Accuracy score

accuracy_score(y_test, y_predict)



# (935+157)/(935+157+211+106)
# Importing Libraries

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
# Seperating dependent and independent variables

x = cust_churn[['tenure']]

y = cust_churn[['Churn']]
# Dividing data set into training and test data sets

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)



# Checking shape of train and test data sets

x_train.shape,y_train.shape,x_test.shape,y_test.shape
# Creating instance/object of Logistic Regression class

DTree = DecisionTreeClassifier()



# Fitting data sets to our model

DTree.fit(x_train,y_train)
# Predicting Values

y_predict = DTree.predict(x_test)



# Checking first 5 Values

y_predict[:5]
# Now we will check confusion matrix and accuracy score

from sklearn.metrics import confusion_matrix, accuracy_score
# Confusion matrix:

confusion_matrix(y_test,y_predict)
# Accuracy Score

accuracy_score(y_test,y_predict)
# Importing random forest classifier

from sklearn.ensemble import RandomForestClassifier
# Creating instance/object

rf = RandomForestClassifier()



# Fitting model

rf.fit(x_train,y_train)
# Predicting values

y_predict = rf.predict(x_test)



# Checking first 5 values

y_predict[:5]
# confusion matrix

confusion_matrix(y_test,y_predict)
# Accuracy Score

accuracy_score(y_test,y_predict)