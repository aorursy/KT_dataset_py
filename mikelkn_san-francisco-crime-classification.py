import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,SGDClassifier

from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score as score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

 


#loading and looing at the data 
train_data =pd.read_csv("../input/train.csv", parse_dates =['Dates'])
test_data = pd.read_csv("../input/test.csv", parse_dates =['Dates'])

print("The size of the train data is:", train_data.shape)
print("The size of the test data is:", test_data.shape)
#Lets take a look at the train set
train_data.head()
#Take a look at the test data set
test_data.head()
train_data.dtypes.value_counts()
test_data.dtypes.value_counts()
#First we check for any nans or missing values
train_data.isnull().sum()
test_data.isnull().sum()
train_data.columns
test_data.columns
#The category is what we need to predict
train_data.Category.value_counts()
#encoding the Category features
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

train_data['Category'] = le.fit_transform(train_data.Category)
train_data.Category.head()

train_data.PdDistrict.value_counts()
#i will use the panda get dummies to one hot encode the remaining categorical features
feature_cols =['DayOfWeek', 'PdDistrict']
train_data = pd.get_dummies(train_data, columns=feature_cols)
test_data = pd.get_dummies(test_data, columns=feature_cols)

train_data
test_data
for x in [train_data, test_data]:
    x['years'] = x['Dates'].dt.year
    x['months'] = x['Dates'].dt.month
    x['days'] = x['Dates'].dt.day
    x['hours'] = x['Dates'].dt.hour
    x['minutes'] = x['Dates'].dt.minute
    x['seconds'] = x['Dates'].dt.second
train_data.head()

test_data.head()
#no need for Dtes column anymore so we drop it
#i will also dropthe adresses on both data
train_data = train_data.drop(['Dates', 'Address','Resolution'], axis = 1)
train_data = train_data.drop(['Descript'], axis = 1)
train_data.head()
test_data = test_data.drop(['Dates', 'Address'], axis = 1)
test_data.head()

#First up spitting the data into train and validation sets

feature_cols = [x for x in train_data if x!='Category']
X = train_data[feature_cols]
y = train_data['Category']
X_train, x_test,y_train, y_test = train_test_split(X, y)

#Logisticregressioncv
LR_L2 = LogisticRegression()
LR_L2 = LR_L2.fit(X_train, y_train)
y_pred_LR = LR_L2.predict(x_test)
y_pred_test_LR = LR_L2.predict(X_train)


print("score is {:.3f}".format (score(y_test, y_pred_LR, average = 'micro')*100))
print("Accuracy for the test data is: {:.3f} ".format (accuracy_score(y_test, y_pred_LR)*100))
print("Accuracy for the train data is: {:.3f} ".format (accuracy_score(y_train, y_pred_test_LR)*100))

#Naive bayes - best for very large datasets
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(x_test)
y_pred_test_gnb = gnb.predict(X_train)

print("score is {:.3f}".format (score(y_test, y_pred_gnb, average = 'micro')*100))
print("Accuracy for the test data is {:.3f}".format (accuracy_score(y_test, y_pred_gnb)*100))
print("Accuracy for the train data is {:.3f} ".format (accuracy_score(y_train, y_pred_test_gnb)*100))

#SVC since there are so manyfaetures (>100k) i will use it with a nystroem kernel
from sklearn.kernel_approximation import Nystroem
nystroemSVC = Nystroem(kernel = 'rbf')
sgd = SGDClassifier()

X_train_svc = nystroemSVC.fit_transform(X_train)
X_test_svc = nystroemSVC.transform(x_test)

linSVC = sgd.fit(X_train_svc, y_train)
y_pred_svc = linSVC.predict(X_test_svc)
y_pred_test_svc = linSVC.predict(X_train_svc)

print("score is {:.3f}".format (score(y_test, y_pred_svc, average = 'micro')*100))
print("Accuracy for the test data is {:.3f}".format (accuracy_score(y_test, y_pred_svc)*100))
print("Accuracy for the train data is {:.3f}".format (accuracy_score(y_train, y_pred_test_svc)*100))


#Decision Tree
DTC = DecisionTreeClassifier(criterion = 'gini', max_features = 10, max_depth = 5)
DTC = DTC.fit(X_train, y_train)
y_pred_DTC = DTC.predict(x_test)
y_pred_test_DTC = DTC.predict(X_train)

print("score is {:.3f}".format (score(y_test, y_pred_DTC, average = 'micro')*100))
print("Accuracy for the test data is {:.3f} ".format (accuracy_score(y_test, y_pred_DTC)*100))
print("Accuracy for the train data is {:.3f} ".format (accuracy_score(y_train, y_pred_test_DTC)*100))

# #Random Forest
# RC =RandomForestClassifier(n_estimators = 100, max_features = 10)
# RC =RC.fit(X_train, y_train)
# y_pred_RC = RC.predict(x_test)
# y_pred_proba_RC = RC.predict_proba(x_test)

# print("score is {}".format (score(y_test, y_pred_RC, average = 'micro')))
# print("Accuracy is {}".format (accuracy_score(y_test, y_pred_RC)))

#so we know we are going to use the decision tree
#lets creat a submission form in a form needed

#for now the best model with an accuracy of ~24 is the deciaion tree

X_test =test_data.drop(['Id'], axis = 1)

my_prediction = DTC.predict(X_test)
SFCC_submission_final = pd.DataFrame({'Id': test_data.Id, 'Category': my_prediction})
print(SFCC_submission_final.shape)
SFCC_submission_final.to_csv('SFCC_prediction.csv', index = False)




