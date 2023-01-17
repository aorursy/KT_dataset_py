# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
loan=pd.read_csv("../input/LoanStats3a.csv", skiprows=[0],low_memory=False )
loan.head()
half_count = len(loan) / 2
loan = loan.dropna(thresh=half_count,axis=1) # Drop columns with more than 50% missing values
loan = loan.drop(['desc'],axis=1)  #dropping columns that are not useful
loan.shape
loan.dtypes
#removing the columns that are not useful for prediction and/or are dependent on any other column
drop_list = ['funded_amnt','funded_amnt_inv',
             'int_rate','sub_grade','emp_title','issue_d','zip_code','out_prncp','out_prncp_inv',
             'total_pymnt','total_pymnt_inv','total_rec_prncp','total_rec_int', 'total_rec_late_fee',
             'recoveries', 'collection_recovery_fee', 'last_pymnt_d',
             'last_pymnt_amnt']
loan = loan.drop(drop_list,axis=1)
#loan status is the target column since loan_status will provide data whether a person will pay the loan or not
loan["loan_status"].value_counts()
#only fully paid and charged off status will help in predicting the outcome so remove all the loans that doesn't have fully paid and charged off
loan = loan[(loan["loan_status"] == "Fully Paid") |
                            (loan["loan_status"] == "Charged Off")]

mapping = {"loan_status":{"Fully Paid": 1, "Charged Off": 0}}
loan = loan.replace(mapping)

#Visualizing the Target Column Outcomes
fig, axs = plt.subplots(1,2,figsize=(14,7))
sns.countplot(x='loan_status',data=loan,ax=axs[0])
axs[0].set_title("Frequency of each Loan Status")
loan.loan_status.value_counts().plot(x=None,y=None, kind='pie', ax=axs[1],autopct='%1.2f%%')
axs[1].set_title("Percentage of each Loan status")
plt.show()
#remove columns with one value
loan = loan.loc[:,loan.apply(pd.Series.nunique) != 1]
#some columns with more than one unique values but one of the values has insignificant frequency in the dataset
for col in loan.columns:
    if (len(loan[col].unique()) < 4):
        print(loan[col].value_counts())
        print()
#finding categorical columns
print(loan.dtypes.value_counts())
object_columns_df = loan.select_dtypes(include=['object'])
print(object_columns_df.iloc[0])
loan['revol_util'] = loan['revol_util'].str.rstrip('%').astype('float')

cols = ['home_ownership', 'grade','verification_status', 'emp_length', 'term', 'addr_state']
for name in cols:
    print(name,':')
    print(object_columns_df[name].value_counts(),'\n')
# addr_state conatins too many categorical values, so drop it
loan = loan.drop('addr_state',axis=1)
for name in ['purpose','title']:
    print("Unique Values in column: {}\n".format(name))
    print(loan[name].value_counts(),'\n')
#It appears that purpose and title columns do contain overlapping information but the purpose column contains fewer discrete values and is cleaner,so we'll keep it and drop title.
loan = loan.drop('title',axis=1)
#dropping debt_settlement_flag becasue Y values are very less
print(loan['debt_settlement_flag'].value_counts(),'\n')
loan = loan.drop('debt_settlement_flag',axis=1)
#Convert Categorical Columns to Numeric Features

mapping_dict = {
    "emp_length": {
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0,
        "n/a": 0

    },
    "grade":{
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 4,
        "E": 5, "F": 6,
        "G": 7
    }
}

loan = loan.replace(mapping_dict)
loan[['emp_length','grade']].head()
# Encoding nominal values
nominal_columns = ["home_ownership", "verification_status", "purpose", "term"]
dummy_df = pd.get_dummies(loan[nominal_columns])
loan = pd.concat([loan, dummy_df], axis=1)
loan = loan.drop(nominal_columns, axis=1)
loan.head()
loan.info()
#Finding percentage null in each column
non_null_data=[]
missing_value=loan.isnull().sum()*100/len(loan.iloc[:,1])
print(missing_value)
#Removing rows with null values
loan =loan.dropna(how='any',axis=0)

#Finding percentage null in each column
missing_value=loan.isnull().sum()*100/len(loan.iloc[:,1])
print(missing_value)
loan.earliest_cr_line = pd.to_datetime(loan['earliest_cr_line'],format="%b-%Y").dt.year
loan.last_credit_pull_d = pd.to_datetime(loan['last_credit_pull_d'],format="%b-%Y").dt.year
loan_final=loan.columns.values.tolist()
y=["loan_status"]

X=[i for i in loan_final if i not in y]
y=loan[y]
X=loan[X]
X.info()
#dividing data into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=30)
#Create a svm Classifier
clf = LinearSVC(random_state=0) # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy: how often is the classifier correct
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

confusion_matrix = metrics.confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)
print('\nConfusion Matrix\n',confusion_matrix)
# Model Precision: what percentage of positive tuples are labeled as such?
#print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
#print("Recall:",metrics.recall_score(y_test, y_pred))
print('\nclassification_report\n',classification_report(y_test, y_pred))
#logistic regression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Accuracy : {:.2f}'.format(logreg.score(X_test, y_test)))

kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
#print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
print('\nConfusion Matrix\n',confusion_matrix)

print('\nclassification_report\n',classification_report(y_test, y_pred))
logreg.get_params()
#logreg.predict_proba(x_test)

#logreg.densify()

#logreg.decision_function(x_test)

#logreg.AIC
logreg.coef_
#Random forest classifier

random_clf=RandomForestClassifier(n_estimators=100)
random_clf.fit(X_train,y_train)

y_pred=random_clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
print('\nConfusion Matrix\n',confusion_matrix)

print('\nclassification_report\n',classification_report(y_test, y_pred))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred,pos_label=1)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc
#feature_imp = pd.Series(clf.feature_importances_).sort_values(ascending=False)
#feature_imp
#XGboost classifier

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
#decision tree
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
# Function to perform training with giniIndex.
def train_using_gini(X_train, X_test, y_train):
 
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion = "gini",
            random_state = 100,max_depth=3, min_samples_leaf=5)
 
    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini

# Function to perform training with entropy.
def tarin_using_entropy(X_train, X_test, y_train):
 
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
            criterion = "entropy", random_state = 100,
            max_depth = 3, min_samples_leaf = 5)
 
    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy
 
# Function to make predictions
def prediction(X_test, clf_object):
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred
# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ",
        confusion_matrix(y_test, y_pred))
     
    print ("Accuracy : ",
    accuracy_score(y_test,y_pred)*100)
     
    print("Report : ",
    classification_report(y_test, y_pred))
    
def main():
     
    # Building Phase
    clf_gini = train_using_gini(X_train, X_test, y_train)
    clf_entropy = tarin_using_entropy(X_train, X_test, y_train)
     
    # Operational Phase
    print("Results Using Gini Index:")
     
    # Prediction using gini
    y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(y_test, y_pred_gini)
     
    print("Results Using Entropy:")
    # Prediction using entropy
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)
     
main()
#accuracy:
#svm classifier: 86.056%
#logistic regression: 86%
#random forest: 86.04%
#XGboost classifier: 86.77%
#decision tree classifier: 85.89%
#from the point of view of accuracy, XGboost classifier gives the best model.

