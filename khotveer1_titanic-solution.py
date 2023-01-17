#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
train_data.head()
data = train_data
tdata = test_data
#dropping unimportant feautres
data = data.drop(labels=['Name' ,'Ticket','Cabin'],axis=1)
tdata = tdata.drop(labels=['Name' ,'Ticket','Cabin'],axis=1)
data.shape
#checking null Values
data.isnull().sum()
#filling null values
data = data.fillna(method='bfill')
tdata = tdata.fillna(method='bfill')
tdata['Age'] = tdata.Age.fillna(method='ffill')
tdata.isnull().sum()
#handelling catagorical data
data['Sex'] = data.Sex.replace({'male':1,'female':0})
tdata['Sex'] = tdata.Sex.replace({'male':1,'female':0})
#creating dummy variables
Embarked_dummy=pd.get_dummies(data.Embarked,prefix='Embarked')
Embarked_dummy_test=pd.get_dummies(tdata.Embarked,prefix='Embarked')
data = pd.concat([data , Embarked_dummy] , axis=1, sort = False)
tdata = pd.concat([tdata , Embarked_dummy_test] , axis=1, sort = False)
data = data.drop('Embarked' , axis=1)
tdata = tdata.drop('Embarked' , axis=1)
tdata.head()
#normalizing train data
from sklearn.preprocessing import MinMaxScaler
min_max_scalar = MinMaxScaler()
scaled_data = min_max_scalar.fit_transform(data)
#normalizing test data
from sklearn.preprocessing import MinMaxScaler
min_max_scalar_1 = MinMaxScaler()
scaled_data_test = min_max_scalar.fit_transform(tdata) 
normalized_data = pd.DataFrame(scaled_data)
normalized_data_test = pd.DataFrame(scaled_data_test)
normalized_data.columns = data.columns
normalized_data_test.columns = tdata.columns

normalized_data.head()
#selecting dependent and independent variables
X = normalized_data.drop(['Survived'] , axis=1)
y = normalized_data['Survived']
#splitting data train , test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
#classification using logistic regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train , y_train)
y_pred_log_reg = log_reg.predict(X_test)
#checking accuracy using test data
from sklearn.metrics import accuracy_score , confusion_matrix
acc_logistic_regression  =accuracy_score(y_test , y_pred_log_reg)
print(acc_logistic_regression)
print("Confusion matrix\n",confusion_matrix(y_test , y_pred_log_reg))
#classification using random forest
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train , y_train)
random_forest_predicted = random_forest.predict(X_test)
#accuracy
acc_random_forest = accuracy_score(y_true=y_test , y_pred=random_forest_predicted)
print(acc_random_forest)
print("Confusion matrix\n",confusion_matrix(y_test , random_forest_predicted))
#feature importance 
importance = random_forest.feature_importances_
plt.barh(range(len(importance)), importance)
plt.yticks(range(len(X.columns)), X.columns,fontsize=15,color='blue')
plt.xlabel("relative importance" , fontsize=15)
plt.ylabel("feature importnace" , fontsize=15)
plt.show()
#support vector machine classifier
from sklearn.svm import SVC
svc = SVC(C=1.0,kernel='rbf')
svc.fit(X_train , y_train)
svc_predicted = svc.predict(X_test)
#accuracy
acc_svm = accuracy_score(y_test , svc_predicted)
print(acc_svm)
print("Confusion matrix\n",confusion_matrix(y_test , svc_predicted))
#model seleection
models = pd.DataFrame({'model':['logistic regression','random forest','SVM'],
                      'accuracy':[acc_logistic_regression,acc_random_forest,acc_svm]})
#we choose model with highest accuracy
models
