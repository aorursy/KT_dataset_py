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

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import f1_score,accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score,accuracy_score
### Reading the Train Data

train_X1 = pd.read_csv("../input/Train-1564659747836.csv", na_values=['',' ','-','NaN'])

train_X2 = pd.read_csv("../input/Train_Complaints-1564659763354.csv", na_values=['',' ','-','NaN'])
## Merging the train Data

train_x = pd.merge(train_X1,train_X2, left_on='InsurerID', right_on='InsurerID',how = 'left')

train_x.head()
#### Making the DRC Classes Into numeric

def cat(train_x):

    if train_x['DRC'] == 'poor':

        return 1

    elif train_x['DRC'] == 'average':

        return 2

    elif train_x['DRC'] == 'outstanding':

        return 3

        

train_x['DRC'] = train_x.apply(lambda x : cat(x), axis=1)
categorical_Columns = train_x.select_dtypes(include= "object").copy()

### Box_Plot of Categorical Coloumn in Reasons

sns.set(font_scale=0.9)

plt.figure(figsize=(16,5)) 

sns.countplot(data = categorical_Columns, x = 'Reason')
### Bar_Plot of Categorical Coloumn in Sub-Reasons

sns.set(font_scale=1.5)

plt.figure(figsize=(20,30))

sns.countplot(data = categorical_Columns, y = 'SubReason')
## Bar_Plot of Categorical Coloumn in Enforcement-Action 



sns.set(font_scale=1)

plt.figure(figsize=(12,12))

sns.countplot(data = categorical_Columns, y = 'EnforcementAction')

#### Bar_Plot for Categorical Coloumn in Conclusion

sns.set(font_scale=1.0)

plt.figure(figsize=(12,5)) 

sns.countplot(data = categorical_Columns, x = 'Conclusion')

#### Dropping the Columns 

train_x = train_x.drop(['Company', 'FileNo', 'ComplaintID', 'State'], axis=1)

train_x.head()



#### Finding null values count

train_x.isnull().sum(axis = 0)

#### Converting  Date into Day Format

train_x["DateOfRegistration"] = pd.to_datetime(train_x["DateOfRegistration"],format="%Y-%m-%d", utc=True)

train_x["DateOfResolution"] = pd.to_datetime(train_x["DateOfResolution"],format="%Y-%m-%d", utc=True)



train_x["Duration_Days"] = train_x['DateOfResolution'] - train_x['DateOfRegistration']
#####    Droping   DateOfRegistration ,DateOfResolution   columns

train_x1 = train_x.drop(['DateOfRegistration', 'DateOfResolution'], axis =1)

train_x1.head()
#### Filling NA Values¶

train_x1['Coverage'] = train_x1['Coverage'].fillna('No')

train_x1['SubCoverage'] = train_x1['SubCoverage'].fillna('None')

train_x1['Duration_Days'] = train_x1['Duration_Days'].fillna(2500)

train_x1.isnull().sum()

#### Converting the DurationDays column into integer¶

train_x1["Duration_Days"]=pd.to_timedelta(train_x1["Duration_Days"]).dt.days.astype('int64')

train_x1.dtypes

### Converting DRC Column into Category¶

train_x1['DRC'] = train_x1['DRC'].astype('category')

### Copying and Dropping the InsurerID Column and cheeking the Shape

train_x2 = train_x1.InsurerID.copy()

train_x2 = train_x1.drop(['InsurerID'], axis=1)

train_x2.shape





### Changing the object type Columns into category with 'COLUMN'¶

column =train_x2.dtypes[train_x2.dtypes == 'object'].index

train_x2[column]=train_x2[column].astype('category')

train_x2.dtypes
cat_col = ['Coverage','SubCoverage', 'Reason', 'SubReason', 'EnforcementAction', 'Conclusion','ResolutionStatus']

#####   Dummification

dummies_train = pd.get_dummies(train_x2,columns = cat_col ,drop_first=True)

dummies_train.shape
### Copy and Drop DRC

X = dummies_train.copy().drop("DRC",axis=1)

y = dummies_train["DRC"]
### Train_Test_Spilt

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
print(y_test.shape)

y_train = pd.DataFrame(y_train)

y_test = pd.DataFrame(y_test)
#### Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier



DTclassifier =  DecisionTreeClassifier(criterion= 'entropy', max_depth= 5 )

DTclassifier.fit(X_train,y_train)



y_pred = DTclassifier.predict(X_test)



from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))



print('Train Accuracy =',DTclassifier.score(X_train, y_train))

print('Test Accuracy =',DTclassifier.score(X_test, y_test))

from sklearn.metrics import f1_score,accuracy_score

dt_f1_test = f1_score(y_pred, y_test, labels=None, average='weighted')



print('DecisionTreeClassifier F1_score on test: %.4f' %dt_f1_test)
#### Random Forest Classifier¶

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score,accuracy_score



RFC = RandomForestClassifier(n_estimators=120,max_depth=10,min_samples_split=5,random_state=52)

RFC.fit(X_train, y_train)

pred_ytest_RF = RFC.predict(X_test)

#pred_test_rfc = RFC.predict(test_merge)



RF_f1_test = f1_score(pred_ytest_RF, y_test,labels=None, average='weighted')  

RF_accuracy_test = accuracy_score(pred_ytest_RF, y_test)



print('Random Forest Classifier F1_score on test: %.4f' %RF_f1_test)

print('Random Forest Classifier Accuracy on test: %.4f' %RF_accuracy_test)
from sklearn.metrics import f1_score,accuracy_score

RF_f1_test = f1_score(y_pred, y_test, labels=None, average='weighted')



print('Random Forest Classifier F1_score on test: %.4f' %dt_f1_test)
### Logistic regression 

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



lr_model = LogisticRegression()

lr_model.fit(X_train,y_train)

pred_train = lr_model.predict(X_train)

pred_test = lr_model.predict(X_test)

print("Accuracy on train is:",accuracy_score(y_train,pred_train))

print("Accuracy on test is:",accuracy_score(y_test,pred_test))


test_X1 = pd.read_csv("../input/Test-1565162240834.csv", na_values=['',' ','-','NaN'])

test_X2 = pd.read_csv("../input/Test_Complaints-1565162197608.csv", na_values=['',' ','-','NaN'])
test_x = pd.merge(test_X1,test_X2, left_on='InsurerID', right_on='InsurerID',how = 'right')

test_x.head()
test_x1 = test_x.drop(['Company','FileNo','ComplaintID','State'], axis=1)

test_x1.head()
test_x1.isnull().sum(axis = 0)
test_x1["DateOfRegistration"] = pd.to_datetime(test_x1["DateOfRegistration"],format="%Y-%m-%d", utc=True)

test_x1["DateOfResolution"] =   pd.to_datetime(test_x1["DateOfResolution"],format="%Y-%m-%d", utc=True)



test_x1["Duration_Days"] = test_x1['DateOfResolution'] - test_x1['DateOfRegistration']
test_x2 = test_x1.drop(['DateOfRegistration', 'DateOfResolution'], axis =1)

test_x2.head()
test_x2['Coverage'] = test_x2['Coverage'].fillna('No')

test_x2['SubCoverage'] = test_x2['SubCoverage'].fillna('None')

test_x2['Duration_Days'] = test_x2['Duration_Days'].fillna(2500)

test_x2.isnull().sum()
test_x2["Duration_Days"]=pd.to_timedelta(test_x2["Duration_Days"]).dt.days.astype('int64')
test_Insurers = test_x2.InsurerID.copy()

test_x3 = test_x2.drop(['InsurerID'], axis=1)

test_x3.head()
column = test_x3.dtypes[test_x3.dtypes == 'object'].index

test_x3[column]=test_x3[column].astype('category')

test_x3.dtypes
cat_col = ['Coverage','SubCoverage', 'Reason', 'SubReason', 'EnforcementAction', 'Conclusion','ResolutionStatus']
#Dummification

dummies_test = pd.get_dummies(test_x3,columns = cat_col ,drop_first=True)

dummies_test.shape
print(dummies_test.shape)

X_train, dummies_test = X_train.align(dummies_test, join = 'left', axis = 1)

dummies_test.fillna(0, inplace=True)

print(dummies_test.shape)

print(X_train.shape)
predict_test = DTclassifier.predict(dummies_test)

predict_test
submission_dt = pd.DataFrame()

submission_dt['InsurerID'] = test_Insurers

submission_dt.shape
final_predictions_dt = predict_test

submission_dt['DRC'] = final_predictions_dt

submission_dt.to_csv('submission_dt.csv')