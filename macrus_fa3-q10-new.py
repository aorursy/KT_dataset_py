# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import matplotlib.pyplot as plt
loan=pd.read_csv("../input/LoanStats3a.csv", skiprows=[0],low_memory=False )
loan.head()
unknown = loan[loan['loan_status'].isnull()]
len(unknown)

loan = loan.ix[~(loan['loan_status'].isnull())]
loan.head(3)

half_count = len(loan) / 2
loan = loan.dropna(thresh=half_count,axis=1) # Drop any column with more than 50% missing values
loan = loan.drop(['desc'],axis=1)      # These columns are not useful for our purposes
#to identify number of columns and rows
loan.shape
loan.dtypes
#removing the unwanted columns that are not useful in prediction and our dependent on any other column
#int_rate and sub_grade are dependent on grade column
#funded_amnt , funded_amnt_inv , issue_d, 'total_rec_prncp','total_rec_int', 'total_rec_late_fee',
#'recoveries', 'collection_recovery_fee', 'last_pymnt_d' will leak data in future means it will make
#model think the data in a different direction
drop_list = ['funded_amnt','funded_amnt_inv',
             'int_rate','sub_grade','emp_title','issue_d','zip_code','out_prncp','out_prncp_inv',
             'total_pymnt','total_pymnt_inv','total_rec_prncp','total_rec_int', 'total_rec_late_fee',
             'recoveries', 'collection_recovery_fee', 'last_pymnt_d',
             'last_pymnt_amnt']
loan = loan.drop(drop_list,axis=1)
#finding the target value: since loan_status is the only term that will provide data whether a person will
#pay the loan or not
loan["loan_status"].value_counts()
#only fully paid and charged off status will help in predicting the outcome
#Remove all the loans that doesn't have fully paid and charged off
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
loan = loan.loc[:,loan.apply(pd.Series.nunique) != 1]
for col in loan.columns:
    if (len(loan[col].unique()) < 4):
        print(loan[col].value_counts())
        print()
print(loan.dtypes.value_counts())
# We have 12 object columns
object_columns_df = loan.select_dtypes(include=['object'])
print(object_columns_df.iloc[0])
# revol_util contains numerical value
loan['revol_util'] = loan['revol_util'].str.rstrip('%').astype('float')
# invetigating  'home_ownership', 'grade','verification_status', 'emp_length', 'term', 'addr_state'
cols = ['home_ownership', 'grade','verification_status', 'emp_length', 'term', 'addr_state']
for name in cols:
    print(name,':')
    print(object_columns_df[name].value_counts(),'\n')
# addr_state conatins too many categorical values, so dropping it.
loan = loan.drop('addr_state',axis=1)

for name in ['purpose','title']:
    print("Unique Values in column: {}\n".format(name))
    print(loan[name].value_counts(),'\n')

#It appears the purpose and title columns do contain overlapping information,
#but the purpose column contains fewer discrete values and is cleaner,so we'll keep it and drop title.

loan = loan.drop('title',axis=1)
#dropping debt_settlement_flag becasue Y values are very less
print(loan['debt_settlement_flag'].value_counts(),'\n')
loan = loan.drop('debt_settlement_flag',axis=1)

#Ordinal Values grade,emp_length
#Nominal Values home_ownership, verification_status,purpose,term
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
        "E": 5,
        "F": 6,
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
#Removing rows with null values
loan =loan.dropna(how='any',axis=0) 
#Finding percentage null in each column
missing_value=loan.isnull().sum()*100/len(loan.iloc[:,1])
print(missing_value)


loan.earliest_cr_line = pd.to_datetime(loan['earliest_cr_line'],format="%b-%Y").dt.year

loan.last_credit_pull_d = pd.to_datetime(loan['last_credit_pull_d'],format="%b-%Y").dt.year
# LOGISTIC REGRESSION MODEL
loan_final=loan.columns.values.tolist()
y=["loan_status"]

X=[i for i in loan_final if i not in y]

y=loan[y]
X=loan[X]
X.info()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

# Evaluating the model
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))

# RANDOM FOREST MODEL

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X_train, y_train)
y_Pred=regressor.predict(X_train)
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_train, y_Pred)
r_squared = r2_score(y_train, y_Pred)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)
regressor.score(X_train, y_train)
y_Pred=regressor.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_Pred)
r_squared = r2_score(y_test, y_Pred)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)
# SVM MODEL
from sklearn import svm
clf1 = svm.SVC(decision_function_shape='ovo')
clf1.fit(X_train,y_train)
y_pred = clf1.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Evaluating the model
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))


#XGBoost Model
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
