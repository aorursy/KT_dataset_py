## Import neccessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder ## For label encoding(converting categorical values to label)
from xgboost import XGBClassifier ## XG boost model
from sklearn.model_selection import GridSearchCV ##For Grid search(cross validation)
from sklearn.tree import DecisionTreeClassifier,export_graphviz ## DecisionTree model
from sklearn.naive_bayes import GaussianNB ## Naive Nayes Model
from sklearn.neighbors import KNeighborsClassifier ## KNN Model
from sklearn.ensemble import RandomForestClassifier ## Random Forest  Model
from sklearn.ensemble import BaggingClassifier ## Bagging Model
from sklearn.ensemble import AdaBoostClassifier ## AdaBoost Model
from sklearn.ensemble import GradientBoostingClassifier ## GradientBoost Model
from sklearn.svm import SVC ## SVC Model
from keras.models import Sequential, Model ## Sequential Model(Neural Network)
from keras.layers import Input, Dense ## Innput  and Fully connected(o/P) layer 
from sklearn.metrics import confusion_matrix,classification_report ## For classifier metrics(accuracy,TPR,TNR)
from sklearn.metrics import accuracy_score ## For getting accuracy value
from sklearn.model_selection import train_test_split ## For splitting data into train and validation
from datetime import datetime ## For converting time into date format
import warnings ## Ignorinng warning
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt ## For Visualization
# Magic Command
%matplotlib inline 
import os ## connnectinng to operating system
## set max how many rows and columns u want to display in jupyter notebook
pd.options.display.max_columns = 200 
pd.get_option('display.max_rows') 
pd.set_option('display.max_rows',None) 
## Get the file path and file name from kaggel 
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
## Read data sets
data = pd.read_csv("/kaggle/input/train.csv",header='infer',sep=',')
test_data = pd.read_csv('/kaggle/input/test.csv',header='infer',sep=',')
## Check dimensions of train and test data
print('Train Shapes',data.shape)
print('Test Shapes',test_data.shape)
## Check first 5 records of train data
data.head()
## Check first 5 records of test data
test_data.head()
## Check last 5 records of train data
data.tail()
## Check last 5 records of test data
test_data.tail()
## Check summary statistics of train data
data.describe(include='all')
## Check summary statistics of test data
test_data.describe(include='all')
## Check data types of train data
data.dtypes
## Check data types of test data
test_data.dtypes
## Check column names of train data
data.columns
## Check column names of test data
test_data.columns
## Check index range for train data
data.index
## Check index rangle for test data
test_data.index
## Check NA values in train data
data.isna().sum()
## Check NA values in test data
test_data.isna().sum()
## this method will return number of levels,null values,unique values,data types

def Observations(df):
    return(pd.DataFrame({'dtypes' : df.dtypes,
                         'levels' : [df[x].unique() for x in df.columns],
                         'null_values' : df.isnull().sum(),
                         'Unique Values': df.nunique()
                        }))
## Check number of levels,null values,unique values,data types for train data
Observations(data)
## Check number of levels,null values,unique values,data types for test data
Observations(test_data)
## Converting starTime and endTime into datetime format for train data
data['start_datetime'] = pd.to_datetime(data['startTime'])
data['end_datetime'] = pd.to_datetime(data['endTime'])
## Converting starTime and endTime into datetime format for test data
test_data['start_datetime'] = pd.to_datetime(test_data['startTime'])
test_data['end_datetime'] = pd.to_datetime(test_data['endTime'])
## Extract date and time from start_datetime,end_datetime for train data
data['start_date'] = [d.date() for d in data['start_datetime']]
data['start_time'] = [d.time() for d in data['start_datetime']]
data['end_date'] = [d.date() for d in data['end_datetime']]
data['end_time'] = [d.time() for d in data['end_datetime']]
## Extract date and time from start_datetime,end_datetime for test data
test_data['start_date'] = [d.date() for d in test_data['start_datetime']]
test_data['start_time'] = [d.time() for d in test_data['start_datetime']]
test_data['end_date'] = [d.date() for d in test_data['end_datetime']]
test_data['end_time'] = [d.time() for d in test_data['end_datetime']]
## Drop startTime, endTime, start_datetime, end_datetime columns from train data beacuse we have extracted features from them
## so those columns are not required.
data.drop(['startTime', 'endTime', 'start_datetime', 'end_datetime'], axis=1, inplace=True)
## Drop startTime, endTime, start_datetime, end_datetime columns from test data beacuse we have extracted features from them
## so those columns are not required.
test_data.drop(['startTime', 'endTime', 'start_datetime', 'end_datetime'], axis=1, inplace=True)
## Extract day,month,year features from start_date,end_date columns of train data and also drop 
## start_date, end_date columns after feature extraction
data['start_Y'] = data['start_date'].apply(lambda x: x.year)
data['start_M'] = data['start_date'].apply(lambda x: x.month)
data['start_D'] = data['start_date'].apply(lambda x: x.day)
data['end_Y'] = data['end_date'].apply(lambda x: x.year)
data['end_M'] = data['end_date'].apply(lambda x: x.month)
data['end_D'] = data['end_date'].apply(lambda x: x.day)
data.drop(['start_date', 'end_date'], axis=1, inplace=True)
## Extract day,month,year features from start_date,end_date columns of test data and also drop 
## start_date, end_date columns after feature extraction
test_data['start_Y'] = test_data['start_date'].apply(lambda x: x.year)
test_data['start_M'] = test_data['start_date'].apply(lambda x: x.month)
test_data['start_D'] = test_data['start_date'].apply(lambda x: x.day)
test_data['end_Y'] = test_data['end_date'].apply(lambda x: x.year)
test_data['end_M'] = test_data['end_date'].apply(lambda x: x.month)
test_data['end_D'] = test_data['end_date'].apply(lambda x: x.day)
test_data.drop(['start_date', 'end_date'], axis=1, inplace=True)
## Extract second,minute,hour features from start_time,end_time columns of train data and also drop 
## start_time,end_time columns after feature extraction
data['start_hour'] = data['start_time'].apply(lambda x: x.hour)
data['start_min'] = data['start_time'].apply(lambda x: x.minute)
data['start_sec'] = data['start_time'].apply(lambda x: x.second)
data['end_hour'] = data['end_time'].apply(lambda x: x.hour)
data['end_min'] = data['end_time'].apply(lambda x: x.minute)
data['end_sec'] = data['end_time'].apply(lambda x: x.second)
data.drop(['start_time', 'end_time'], axis=1, inplace=True)
## Extract second,minute,hour features from start_time,end_time columns of test data and also drop 
## start_time,end_time columns after feature extraction
test_data['start_hour'] = test_data['start_time'].apply(lambda x: x.hour)
test_data['start_min'] = test_data['start_time'].apply(lambda x: x.minute)
test_data['start_sec'] = test_data['start_time'].apply(lambda x: x.second)
test_data['end_hour'] = test_data['end_time'].apply(lambda x: x.hour)
test_data['end_min'] = test_data['end_time'].apply(lambda x: x.minute)
test_data['end_sec'] = test_data['end_time'].apply(lambda x: x.second)
test_data.drop(['start_time', 'end_time'], axis=1, inplace=True)
## Check head of train data after feature extraction
data.head()
## Check head of test data after feature extraction
test_data.head()
## Check null values for train data
data.isnull().sum()
## Check null values for test data
test_data.isnull().sum()
## Get first item from product list column
data['ProductList'][0]
temp = data['ProductList'].str.split(';')
temp[0]
len(temp)
temp.apply(len)[:5]
data.index.repeat(temp.apply(len))
data.reindex(data.index.repeat(temp.apply(len))).head()
np.hstack(temp)
## The products list are separated by semi-colon. We can split each product into a new row(for Train data)

temp = data['ProductList'].str.split(';') ## It splits each record with semocolon separator and forms a 1D array 

## temp.apply(len) ---> Invoke function on values of Series.
## Can be ufunc (a NumPy function that applies to the entire Series)
## or a Python function that only works on single values.

## data.index.repeat(temp.apply(len)) ---> Returns a new Index where each element of the current Index
## is repeated consecutively a given number of times.
data = data.reindex(data.index.repeat(temp.apply(len))) ## Conform DataFrame to new index with optional filling logic, placing
                                                        ## NA/NaN in locations having no value in the previous index. A new object
                                                        ## is produced unless the new index is equivalent to the current one 
data['product_data'] = np.hstack(temp) ## tack arrays in sequence horizontally (column wise).
## The products list are separated by semi-colon. We can split each product into a new row(for Test data)
temp = test_data['ProductList'].str.split(';')
test_data = test_data.reindex(test_data.index.repeat(temp.apply(len)))
test_data['product_data'] = np.hstack(temp)
## Check first 5 records after adding product_data column for train data
data.head()
## Check first 5 records after adding product_data column for test data
test_data.head()
## Check last 5 records after adding product_data column for train data
data.tail()
## Check last 5 records after adding product_data column for test data
test_data.tail()
## The product data is separated by forward slash as follows: category id, sub category id, sub sub category id and product id.
## We can split this data into 4 columns and drop ProductList, product_data after extarcting features(for train data)

data['category'] = data['product_data'].str.split('/').str[0]
data['sub_category'] = data['product_data'].str.split('/').str[1]
data['sub_sub_category'] = data['product_data'].str.split('/').str[2]
data['product'] = data['product_data'].str.split('/').str[3]

data.drop(['ProductList', 'product_data'], axis=1, inplace=True)
## The product data is separated by forward slash as follows: category id, sub category id, sub sub category id and product id.
## We can split this data into 4 columns and drop ProductList, product_data after extarcting features(for test data)
test_data['category'] = test_data['product_data'].str.split('/').str[0]
test_data['sub_category'] = test_data['product_data'].str.split('/').str[1]
test_data['sub_sub_category'] = test_data['product_data'].str.split('/').str[2]
test_data['product'] = test_data['product_data'].str.split('/').str[3]

test_data.drop(['ProductList', 'product_data'], axis=1, inplace=True)
## Check first 5 records after extarcting features for train data
data.head()
## Check first 5 records after extarcting features for test data
test_data.head()
## Check last 5 records after extracting features for train data
data.tail()
## Check last 5 records after extracting features for test data
test_data.tail()
## Check categories count for train data
data['category'].value_counts() ## There are only 11 categories across the entire dataset
## Check categories count for test data
test_data['category'].value_counts() ## There are only 11 categories across the entire dataset
## Check number of levels,null values,unique values,data types for train data
Observations(data)
## Check number of levels,null values,unique values,data types for test data
Observations(test_data)
## Check Gender column values count for train data
data.gender.value_counts()
## Store features into train_data (for train data)
train_data = data.drop('gender', axis=1)
## Store traget into y(for train data)
y = data['gender']
## Split the train data into  train and validation
X_train,X_test,y_train,y_test = train_test_split(train_data,y,test_size=0.2,random_state =1234)
## To perform label encoding, we need to append train and test data and fit label encoder on it
## (beacuse train and  test will not have same lebel so that is reason we combined both  data and build label encoder and
## will transform train and  test individuallly)
combined_data = train_data.append(test_data)
## Instantiate label encoder
le_cat = LabelEncoder()
le_subcat = LabelEncoder()
le_subsubcat = LabelEncoder()
le_product = LabelEncoder()
le_gender = LabelEncoder()
le_session = LabelEncoder()
le_start_Y = LabelEncoder()
le_start_M = LabelEncoder()
le_start_D = LabelEncoder()
le_end_Y = LabelEncoder()
le_end_M = LabelEncoder()
le_end_D = LabelEncoder()
le_start_hour = LabelEncoder()
le_start_min = LabelEncoder()
le_start_sec = LabelEncoder()
le_end_hour = LabelEncoder()
le_end_min = LabelEncoder()
le_end_sec = LabelEncoder()
## Fit the label encoder model
combined_data['category'] = le_cat.fit_transform(combined_data['category'])
combined_data['sub_category'] = le_subcat.fit_transform(combined_data['sub_category'])
combined_data['sub_sub_category'] = le_subsubcat.fit_transform(combined_data['sub_sub_category'])
combined_data['product'] = le_product.fit_transform(combined_data['product'])
combined_data['session_id'] = le_session.fit_transform(combined_data['session_id'])
combined_data['start_Y'] = le_start_Y.fit_transform(combined_data['start_Y'])
combined_data['start_M'] = le_start_M.fit_transform(combined_data['start_M'])
combined_data['start_D'] = le_start_D.fit_transform(combined_data['start_D'])
combined_data['end_Y'] = le_end_Y.fit_transform(combined_data['end_Y'])
combined_data['end_M'] = le_end_M.fit_transform(combined_data['end_M'])
combined_data['end_D'] = le_end_D.fit_transform(combined_data['end_D'])
combined_data['start_hour'] = le_start_hour.fit_transform(combined_data['start_hour'])
combined_data['start_min'] = le_start_min.fit_transform(combined_data['start_min'])
combined_data['start_sec'] = le_start_sec.fit_transform(combined_data['start_sec'])
combined_data['end_hour'] = le_end_hour.fit_transform(combined_data['end_hour'])
combined_data['end_min'] = le_end_min.fit_transform(combined_data['end_min'])
combined_data['end_sec'] = le_end_sec.fit_transform(combined_data['end_sec'])
y = le_gender.fit_transform(y)
## Check first 5 records of combined data after doing label encoding
combined_data.head()
## Do labe encoding for trian data 
X_train['category'] = le_cat.transform(X_train['category'])
X_train['sub_category'] = le_subcat.transform(X_train['sub_category'])
X_train['sub_sub_category'] = le_subsubcat.transform(X_train['sub_sub_category'])
X_train['product'] = le_product.transform(X_train['product'])
X_train['session_id'] = le_session.transform(X_train['session_id'])
X_train['start_Y'] = le_start_Y.fit_transform(X_train['start_Y'])
X_train['start_M'] = le_start_M.fit_transform(X_train['start_M'])
X_train['start_D'] = le_start_D.fit_transform(X_train['start_D'])
X_train['end_Y'] = le_end_Y.fit_transform(X_train['end_Y'])
X_train['end_M'] = le_end_M.fit_transform(X_train['end_M'])
X_train['end_D'] = le_end_D.fit_transform(X_train['end_D'])
X_train['start_hour'] = le_start_hour.fit_transform(X_train['start_hour'])
X_train['start_min'] = le_start_min.fit_transform(X_train['start_min'])
X_train['start_sec'] = le_start_sec.fit_transform(X_train['start_sec'])
X_train['end_hour'] = le_end_hour.fit_transform(X_train['end_hour'])
X_train['end_min'] = le_end_min.fit_transform(X_train['end_min'])
X_train['end_sec'] = le_end_sec.fit_transform(X_train['end_sec'])
y_train = le_gender.transform(y_train)
## Do labe encoding for validation data 
X_test['category'] = le_cat.transform(X_test['category'])
X_test['sub_category'] = le_subcat.transform(X_test['sub_category'])
X_test['sub_sub_category'] = le_subsubcat.transform(X_test['sub_sub_category'])
X_test['product'] = le_product.transform(X_test['product'])
X_test['session_id'] = le_session.transform(X_test['session_id'])
X_test['start_Y'] = le_start_Y.fit_transform(X_test['start_Y'])
X_test['start_M'] = le_start_M.fit_transform(X_test['start_M'])
X_test['start_D'] = le_start_D.fit_transform(X_test['start_D'])
X_test['end_Y'] = le_end_Y.fit_transform(X_test['end_Y'])
X_test['end_M'] = le_end_M.fit_transform(X_test['end_M'])
X_test['end_D'] = le_end_D.fit_transform(X_test['end_D'])
X_test['start_hour'] = le_start_hour.fit_transform(X_test['start_hour'])
X_test['start_min'] = le_start_min.fit_transform(X_test['start_min'])
X_test['start_sec'] = le_start_sec.fit_transform(X_test['start_sec'])
X_test['end_hour'] = le_end_hour.fit_transform(X_test['end_hour'])
X_test['end_min'] = le_end_min.fit_transform(X_test['end_min'])
X_test['end_sec'] = le_end_sec.fit_transform(X_test['end_sec'])
y_test = le_gender.transform(y_test)
## Do labe encoding for test data 
test_data['category'] = le_cat.transform(test_data['category'])
test_data['sub_category'] = le_subcat.transform(test_data['sub_category'])
test_data['sub_sub_category'] = le_subsubcat.transform(test_data['sub_sub_category'])
test_data['product'] = le_product.transform(test_data['product'])
test_data['session_id'] = le_session.transform(test_data['session_id'])
test_data['start_Y'] = le_start_Y.fit_transform(test_data['start_Y'])
test_data['start_M'] = le_start_M.fit_transform(test_data['start_M'])
test_data['start_D'] = le_start_D.fit_transform(test_data['start_D'])
test_data['end_Y'] = le_end_Y.fit_transform(test_data['end_Y'])
test_data['end_M'] = le_end_M.fit_transform(test_data['end_M'])
test_data['end_D'] = le_end_D.fit_transform(test_data['end_D'])
test_data['start_hour'] = le_start_hour.fit_transform(test_data['start_hour'])
test_data['start_min'] = le_start_min.fit_transform(test_data['start_min'])
test_data['start_sec'] = le_start_sec.fit_transform(test_data['start_sec'])
test_data['end_hour'] = le_end_hour.fit_transform(test_data['end_hour'])
test_data['end_min'] = le_end_min.fit_transform(test_data['end_min'])
test_data['end_sec'] = le_end_sec.fit_transform(test_data['end_sec'])
## Check first 5 records of train data after doing label encoding
X_train.head()
## Check first 5 records of validation data after doing label encoding
X_test.head()
## Check first 5 records of test data after doing label encoding
test_data.head()
test = test_data.copy()
test.head()
xgb = XGBClassifier() ## Instantiate XGBClassifier model

optimization_dict = {'max_depth': [2,3,4,5,6,7], ## trying with different max_depth,n_estimators to find best model
                      'n_estimators': [50,60,70,80,90,100,150,200]} 

## Build best model with Grid Search params
model = GridSearchCV(xgb, ## XGB model
                     optimization_dict, ## dictory with different max_depth,n_estimators
                     scoring='accuracy', ## on which parameter we are interested
                     verbose=1, ## for messaging purpose
                     n_jobs=-1) ## Number of jobs to run in parallel. ''-1' means use all processors

%time model.fit(X_train, y_train) ## Fit a model
print(model.best_score_) ## Display best score calues
print(model.best_params_) ## Display best parameters
## The best params were found after grid search CV (above code)
model = XGBClassifier(max_depth=7, n_estimators=200)
%time model.fit(X_train, y_train)
## Get the important features from model and sort them  based on value
results=pd.DataFrame()
results['columns']=X_train.columns
results['importances'] = model.feature_importances_
results.sort_values(by='importances',ascending=False,inplace=True)
## Check first 5 records of results
results.head()
## Get the predictions on train data
train_pred = model.predict(X_train)
## Display accuracy value for train data
print("Train Accuracy :",accuracy_score(y_train,train_pred))
## Get the predictions on validation data
validation_pred = model.predict(X_test)
## Display  accuracy value for validation data
print("Validation Accuracy :",accuracy_score(y_test,validation_pred))
## Get the confusion matrix for train data
confusion_matrix_train = confusion_matrix(y_train, train_pred)
print(confusion_matrix_train)
## Get the confusion matrix for validation data
confusion_matrix_test = confusion_matrix(y_test, validation_pred)
print(confusion_matrix_test)
Accuracy_Train=(confusion_matrix_train[0,0]+confusion_matrix_train[1,1])/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1]+confusion_matrix_train[1,0]+confusion_matrix_train[1,1])
TNR_Train= confusion_matrix_train[0,0]/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1])
TPR_Train= confusion_matrix_train[1,1]/(confusion_matrix_train[1,0]+confusion_matrix_train[1,1])

print("Train TNR: ",TNR_Train)
print("\n")
print("Train TPR: ",TPR_Train)
print("\n")
print("Train Accuracy: ",Accuracy_Train)
Accuracy_Test=(confusion_matrix_test[0,0]+confusion_matrix_test[1,1])/(confusion_matrix_test[0,0]+confusion_matrix_test[0,1]+confusion_matrix_test[1,0]+confusion_matrix_test[1,1])
TNR_Test= confusion_matrix_test[0,0]/(confusion_matrix_test[0,0] +confusion_matrix_test[0,1])
TPR_Test= confusion_matrix_test[1,1]/(confusion_matrix_test[1,0] +confusion_matrix_test[1,1])

print("Test TNR: ",TNR_Test)
print("\n")
print("Test TPR: ",TPR_Test)
print("\n")
print("Test Accuracy: ",Accuracy_Test)
## Get the predictions on test data
y_pred = model.predict(test_data)
y_pred
## Do inverse transform on y_ped so that we will get it's original values(male,female)
test_data['gender'] = le_gender.inverse_transform(y_pred)
## Do inverse transform on session_id so that we will get it's original values
test_data['session_id'] = le_session.inverse_transform(test_data['session_id'])
## Check first 5 records after doing inverse transformation
test_data.head()
test_data.drop_duplicates(subset=['session_id']).head()
test_data[test_data['session_id']=='u24492']
## Check the dimensions of the test_data before dropping dupliccate data
test_data.shape
## Drop duplicate session_id values
test_data = test_data.drop_duplicates(subset=['session_id'])
test_data[test_data['session_id']=='u24492']
## Check the dimensions of the test_data after dropping dupliccate data
test_data.shape
## Get session_id,gender from test_data and store into to_submit
to_submit = test_data[['session_id', 'gender']]
## Check first 5 records of to_submit
to_submit.head()
## Get gender count value 
to_submit.gender.value_counts()
## store to_submit into csv file with name xgb_model 
to_submit.to_csv('xgb_model.csv',index = False)
## Instantiate Decision Tree and fit the model
clf = DecisionTreeClassifier(max_depth=8)
clf = clf.fit(X_train, y_train)
## Get important features from model
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
names=np.asarray(X_train.columns[indices])
Important=pd.DataFrame(np.sort(importances)[::-1],index=names,columns=['Imp'])
Important
### Plot importances features
Important.plot(kind='bar')
## Get the predictions on train and test
train_pred = clf.predict(X_train)
validation_pred = clf.predict(X_test)
## Get confusion matrix for train and validation data
confusion_matrix_train = confusion_matrix(y_train, train_pred)
confusion_matrix_validation = confusion_matrix(y_test, validation_pred)

print(confusion_matrix_train)
print(confusion_matrix_validation)
Accuracy_Train=(confusion_matrix_train[0,0]+confusion_matrix_train[1,1])/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1]+confusion_matrix_train[1,0]+confusion_matrix_train[1,1])
TNR_Train= confusion_matrix_train[0,0]/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1])
TPR_Train= confusion_matrix_train[1,1]/(confusion_matrix_train[1,0]+confusion_matrix_train[1,1])

print("Train TNR: ",TNR_Train)
print("\n")
print("Train TPR: ",TPR_Train)
print("\n")
print("Train Accuracy: ",Accuracy_Train)
Accuracy_Test=(confusion_matrix_validation[0,0]+confusion_matrix_validation[1,1])/(confusion_matrix_validation[0,0]+confusion_matrix_validation[0,1]+confusion_matrix_validation[1,0]+confusion_matrix_validation[1,1])
TNR_Test= confusion_matrix_validation[0,0]/(confusion_matrix_validation[0,0] +confusion_matrix_validation[0,1])
TPR_Test= confusion_matrix_validation[1,1]/(confusion_matrix_validation[1,0] +confusion_matrix_validation[1,1])

print("Validation TNR: ",TNR_Test)
print("\n")
print("Validation TPR: ",TPR_Test)
print("\n")
print("Validation Accuracy: ",Accuracy_Test)
## Display classification metrics for train data
print(classification_report(y_true=y_train,y_pred=train_pred))
## Display classification metrics for validation data
print(classification_report(y_true=y_test,y_pred=validation_pred))
## Try with different parameters to find best mode
param_grid = {"criterion": ["gini", "entropy"],
              "min_samples_split": [2, 10, 20],
              "max_depth": [2, 5, 10],
              "min_samples_leaf": [1, 5, 10],
              "max_leaf_nodes": [5, 10, 20],
              }
## Instantiate model and send to Grid search to find best model and fit a model
dt = DecisionTreeClassifier()
clf = GridSearchCV(dt, param_grid, cv=10)
%time clf.fit(X_train, y_train)
## Get best parameters from model
clf.best_params_
## Instantiate Decision Tree with best parameters and fit a model
dt = DecisionTreeClassifier(criterion='entropy',max_depth=10,max_leaf_nodes=20)
%time dt.fit(X_train, y_train)
## Get prediction on train and validation 
train_pred = dt.predict(X_train)
validation_pred= dt.predict(X_test)
## Get confusion matrix on train and validation 
confusion_matrix_train = confusion_matrix(y_train, train_pred)
confusion_matrix_validation = confusion_matrix(y_test, validation_pred)

print(confusion_matrix_train)
print(confusion_matrix_validation)
Accuracy_Train=(confusion_matrix_train[0,0]+confusion_matrix_train[1,1])/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1]+confusion_matrix_train[1,0]+confusion_matrix_train[1,1])
TNR_Train= confusion_matrix_train[0,0]/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1])
TPR_Train= confusion_matrix_train[1,1]/(confusion_matrix_train[1,0]+confusion_matrix_train[1,1])

print("Train TNR: ",TNR_Train)
print("\n")
print("Train TPR: ",TPR_Train)
print("\n")
print("Train Accuracy: ",Accuracy_Train)
Accuracy_Test=(confusion_matrix_validation[0,0]+confusion_matrix_validation[1,1])/(confusion_matrix_validation[0,0]+confusion_matrix_validation[0,1]+confusion_matrix_validation[1,0]+confusion_matrix_validation[1,1])
TNR_Test= confusion_matrix_validation[0,0]/(confusion_matrix_validation[0,0] +confusion_matrix_validation[0,1])
TPR_Test= confusion_matrix_validation[1,1]/(confusion_matrix_validation[1,0] +confusion_matrix_validation[1,1])

print("Validation TNR: ",TNR_Test)
print("\n")
print("Validation TPR: ",TPR_Test)
print("\n")
print("Validation Accuracy: ",Accuracy_Test)
## Display classification metrics for train data
print(classification_report(y_true=y_train,y_pred=train_pred))
## Display classification metrics for validation data
print(classification_report(y_true=y_test,y_pred=validation_pred))
## Get copy of test data
test_data = test.copy()
## Get the predictions on test data
y_pred = dt.predict(test_data)
y_pred
## Do inverse transform on y_ped so that we will get it's original values(male,female)
test_data['gender'] = le_gender.inverse_transform(y_pred)
## Do inverse transform on session_id so that we will get it's original values
test_data['session_id'] = le_session.inverse_transform(test_data['session_id'])
## Check first 5 records after doing inverse transformation
test_data.head()
## Check the dimensions of the test_data before dropping dupliccate data
test_data.shape
## Drop duplicate session_id values
test_data = test_data.drop_duplicates(subset=['session_id'])
## Check the dimensions of the test_data after dropping dupliccate data
test_data.shape
## Get session_id,gender from test_data and store into to_submit
to_submit = test_data[['session_id', 'gender']]
## Check first 5 records of to_submit
to_submit.head()
## Get gender count value 
to_submit.gender.value_counts()
## store to_submit into csv file with name decisiontree_model 
to_submit.to_csv('decisiontree_model.csv',index = False)
## Instantiate Navie Bayes Model
model = GaussianNB()
## Fit a model
model.fit(X_train,y_train)
## Get prediction on train and validation data
predict_train = model.predict(X_train)
predict_validation = model.predict(X_test)
## Get confusion matrix for train data
confusion_matrix_train = confusion_matrix(y_train, predict_train)
print(confusion_matrix_train)
## Get confusion matrix for validation data
confusion_matrix_validation = confusion_matrix(y_test, predict_validation)
print(confusion_matrix_validation)
Accuracy_Train=(confusion_matrix_train[0,0]+confusion_matrix_train[1,1])/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1]+confusion_matrix_train[1,0]+confusion_matrix_train[1,1])
TNR_Train= confusion_matrix_train[0,0]/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1])
TPR_Train= confusion_matrix_train[1,1]/(confusion_matrix_train[1,0]+confusion_matrix_train[1,1])

print("Train TNR: ",TNR_Train)
print("\n")
print("Train TPR: ",TPR_Train)
print("\n")
print("Train Accuracy: ",Accuracy_Train)
Accuracy_Test=(confusion_matrix_validation[0,0]+confusion_matrix_validation[1,1])/(confusion_matrix_validation[0,0]+confusion_matrix_validation[0,1]+confusion_matrix_validation[1,0]+confusion_matrix_validation[1,1])
TNR_Test= confusion_matrix_validation[0,0]/(confusion_matrix_validation[0,0] +confusion_matrix_validation[0,1])
TPR_Test= confusion_matrix_validation[1,1]/(confusion_matrix_validation[1,0] +confusion_matrix_validation[1,1])

print("Validation TNR: ",TNR_Test)
print("\n")
print("Validation TPR: ",TPR_Test)
print("\n")
print("Validation Accuracy: ",Accuracy_Test)
## Display classification metrics for train data
print(classification_report(y_true=y_train,y_pred=train_pred))
## Display classification metrics for validation data
print(classification_report(y_true=y_test,y_pred=validation_pred))
## Get copy of test data
test_data = test.copy()
## Get the predictions on test data
y_pred = model.predict(test_data)
y_pred
## Do inverse transform on y_ped so that we will get it's original values(male,female)
test_data['gender'] = le_gender.inverse_transform(y_pred)
## Do inverse transform on session_id so that we will get it's original values
test_data['session_id'] = le_session.inverse_transform(test_data['session_id'])
## Check first 5 records after doing inverse transformation
test_data.head()
## Check the dimensions of the test_data before dropping dupliccate data
test_data.shape
## Drop duplicate session_id values
test_data = test_data.drop_duplicates(subset=['session_id'])
## Check the dimensions of the test_data after dropping dupliccate data
test_data.shape
## Get session_id,gender from test_data and store into to_submit
to_submit = test_data[['session_id', 'gender']]
## Check first 5 records of to_submit
to_submit.head()
## Get gender count value 
to_submit.gender.value_counts()
## store to_submit into csv file with name naivebayes_model 
to_submit.to_csv('naivebayes_model.csv',index = False)
## Instantiate KNN model
knn = KNeighborsClassifier(algorithm = 'brute', n_neighbors = 3,
                           metric = "euclidean")
## Fit a model
knn.fit(X_train, y_train)
## Get prediction on trian and validation data
train_pred = knn.predict(X_train)
validation_pred = knn.predict(X_test)
## Get confusion matrix for train data
confusion_matrix_train = confusion_matrix(y_train, train_pred)
print(confusion_matrix_train)
## Get confusion matrix for validation data
confusion_matrix_validation = confusion_matrix(y_test, validation_pred)
print(confusion_matrix_validation)
Accuracy_Train=(confusion_matrix_train[0,0]+confusion_matrix_train[1,1])/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1]+confusion_matrix_train[1,0]+confusion_matrix_train[1,1])
TNR_Train= confusion_matrix_train[0,0]/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1])
TPR_Train= confusion_matrix_train[1,1]/(confusion_matrix_train[1,0]+confusion_matrix_train[1,1])

print("Train TNR: ",TNR_Train)
print("\n")
print("Train TPR: ",TPR_Train)
print("\n")
print("Train Accuracy: ",Accuracy_Train)
Accuracy_Test=(confusion_matrix_validation[0,0]+confusion_matrix_validation[1,1])/(confusion_matrix_validation[0,0]+confusion_matrix_validation[0,1]+confusion_matrix_validation[1,0]+confusion_matrix_validation[1,1])
TNR_Test= confusion_matrix_validation[0,0]/(confusion_matrix_validation[0,0] +confusion_matrix_validation[0,1])
TPR_Test= confusion_matrix_validation[1,1]/(confusion_matrix_validation[1,0] +confusion_matrix_validation[1,1])

print("Validation TNR: ",TNR_Test)
print("\n")
print("Validation TPR: ",TPR_Test)
print("\n")
print("Validation Accuracy: ",Accuracy_Test)
## Display classification metrics for train data
print(classification_report(y_true=y_train,y_pred=train_pred))
## Display classification metrics for validation data
print(classification_report(y_true=y_test,y_pred=validation_pred))
## Get copy of test data
test_data = test.copy()
## Get the predictions on test data
y_pred = knn.predict(test_data)
y_pred
## Do inverse transform on y_ped so that we will get it's original values(male,female)
test_data['gender'] = le_gender.inverse_transform(y_pred)
## Do inverse transform on session_id so that we will get it's original values
test_data['session_id'] = le_session.inverse_transform(test_data['session_id'])
## Check first 5 records after doing inverse transformation
test_data.head()
## Check the dimensions of the test_data before dropping dupliccate data
test_data.shape
## Drop duplicate session_id values
test_data = test_data.drop_duplicates(subset=['session_id'])
## Check the dimensions of the test_data after dropping dupliccate data
test_data.shape
## Get session_id,gender from test_data and store into to_submit
to_submit = test_data[['session_id', 'gender']]
## Check first 5 records of to_submit
to_submit.head()
## Get gender count value 
to_submit.gender.value_counts()
## store to_submit into csv file with name knn_model 
to_submit.to_csv('knn_model.csv',index = False)
## Instantiate RandoForest model
rc = RandomForestClassifier(n_estimators=30,max_depth=10,n_jobs=-1)
## Fit a model
rc.fit(X_train,y_train)
## Get predictions on train and validation data
train_pred = rc.predict(X_train)
validation_pred = rc.predict(X_test)
## Get confusion matrix for train data
confusion_matrix_train = confusion_matrix(y_train, train_pred)
print(confusion_matrix_train)
## Get confusion matrix for validation data
confusion_matrix_validation = confusion_matrix(y_test, validation_pred)
print(confusion_matrix_validation)
Accuracy_Train=(confusion_matrix_train[0,0]+confusion_matrix_train[1,1])/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1]+confusion_matrix_train[1,0]+confusion_matrix_train[1,1])
TNR_Train= confusion_matrix_train[0,0]/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1])
TPR_Train= confusion_matrix_train[1,1]/(confusion_matrix_train[1,0]+confusion_matrix_train[1,1])

print("Train TNR: ",TNR_Train)
print("\n")
print("Train TPR: ",TPR_Train)
print("\n")
print("Train Accuracy: ",Accuracy_Train)
Accuracy_Test=(confusion_matrix_validation[0,0]+confusion_matrix_validation[1,1])/(confusion_matrix_validation[0,0]+confusion_matrix_validation[0,1]+confusion_matrix_validation[1,0]+confusion_matrix_validation[1,1])
TNR_Test= confusion_matrix_validation[0,0]/(confusion_matrix_validation[0,0] +confusion_matrix_validation[0,1])
TPR_Test= confusion_matrix_validation[1,1]/(confusion_matrix_validation[1,0] +confusion_matrix_validation[1,1])

print("Validation TNR: ",TNR_Test)
print("\n")
print("Validation TPR: ",TPR_Test)
print("\n")
print("Validation Accuracy: ",Accuracy_Test)
## Display classification metrics for train data
print(classification_report(y_true=y_train,y_pred=train_pred))
## Display classification metrics for validation data
print(classification_report(y_true=y_test,y_pred=validation_pred))
## Get copy of test data
test_data = test.copy()
## Get the predictions on test data
y_pred = rc.predict(test_data)
y_pred
## Do inverse transform on y_ped so that we will get it's original values(male,female)
test_data['gender'] = le_gender.inverse_transform(y_pred)
## Do inverse transform on session_id so that we will get it's original values
test_data['session_id'] = le_session.inverse_transform(test_data['session_id'])
## Check first 5 records after doing inverse transformation
test_data.head()
## Check the dimensions of the test_data before dropping dupliccate data
test_data.shape
## Drop duplicate session_id values
test_data = test_data.drop_duplicates(subset=['session_id'])
## Check the dimensions of the test_data after dropping dupliccate data
test_data.shape
## Get session_id,gender from test_data and store into to_submit
to_submit = test_data[['session_id', 'gender']]
## Check first 5 records of to_submit
to_submit.head()
## Get gender count value 
to_submit.gender.value_counts()
## store to_submit into csv file with name randomforest_model 
to_submit.to_csv('randomforest_model.csv',index = False)
## Instantiate Bagging model and fit it
clf = BaggingClassifier(n_estimators=10)
clf.fit(X=X_train, y=y_train)
## Get prediction on train and validation data
train_pred = clf.predict(X_train)
validation_pred = clf.predict(X_test)
## Get confusion matrix for train data
confusion_matrix_train = confusion_matrix(y_train, train_pred)
print(confusion_matrix_train)
## Get confusion matrix for validation data
confusion_matrix_validation = confusion_matrix(y_test, validation_pred)
print(confusion_matrix_validation)
Accuracy_Train=(confusion_matrix_train[0,0]+confusion_matrix_train[1,1])/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1]+confusion_matrix_train[1,0]+confusion_matrix_train[1,1])
TNR_Train= confusion_matrix_train[0,0]/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1])
TPR_Train= confusion_matrix_train[1,1]/(confusion_matrix_train[1,0]+confusion_matrix_train[1,1])

print("Train TNR: ",TNR_Train)
print("\n")
print("Train TPR: ",TPR_Train)
print("\n")
print("Train Accuracy: ",Accuracy_Train)
Accuracy_Test=(confusion_matrix_validation[0,0]+confusion_matrix_validation[1,1])/(confusion_matrix_validation[0,0]+confusion_matrix_validation[0,1]+confusion_matrix_validation[1,0]+confusion_matrix_validation[1,1])
TNR_Test= confusion_matrix_validation[0,0]/(confusion_matrix_validation[0,0] +confusion_matrix_validation[0,1])
TPR_Test= confusion_matrix_validation[1,1]/(confusion_matrix_validation[1,0] +confusion_matrix_validation[1,1])

print("Validation TNR: ",TNR_Test)
print("\n")
print("Validation TPR: ",TPR_Test)
print("\n")
print("Validation Accuracy: ",Accuracy_Test)
## Display classification metrics for train data
print(classification_report(y_true=y_train,y_pred=train_pred))
## Display classification metrics for validation data
print(classification_report(y_true=y_test,y_pred=validation_pred))
## Get copy of test data
test_data = test.copy()
## Get the predictions on test data
y_pred = clf.predict(test_data)
y_pred
## Do inverse transform on y_ped so that we will get it's original values(male,female)
test_data['gender'] = le_gender.inverse_transform(y_pred)
## Do inverse transform on session_id so that we will get it's original values
test_data['session_id'] = le_session.inverse_transform(test_data['session_id'])
## Check first 5 records after doing inverse transformation
test_data.head()
## Check the dimensions of the test_data before dropping dupliccate data
test_data.shape
## Drop duplicate session_id values
test_data = test_data.drop_duplicates(subset=['session_id'])
## Check the dimensions of the test_data after dropping dupliccate data
test_data.shape
## Get session_id,gender from test_data and store into to_submit
to_submit = test_data[['session_id', 'gender']]
## Check first 5 records of to_submit
to_submit.head()
## Get gender count value 
to_submit.gender.value_counts()
## store to_submit into csv file with name bagging_model 
to_submit.to_csv('bagging_model.csv',index = False)
## Instantiate AdaBoost model and fit it
Adaboost_model = AdaBoostClassifier(n_estimators=50,learning_rate=1)
%time Adaboost_model.fit(X_train, y_train)
## Get predictions on train and validation data
train_pred = Adaboost_model.predict(X_train)
validation_pred = Adaboost_model.predict(X_test)
## Get confusion matrix for train data
confusion_matrix_train = confusion_matrix(y_train, train_pred)
print(confusion_matrix_train)
## Get confusion matrrix for validation data
confusion_matrix_validation = confusion_matrix(y_test, validation_pred)
print(confusion_matrix_validation)
Accuracy_Train=(confusion_matrix_train[0,0]+confusion_matrix_train[1,1])/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1]+confusion_matrix_train[1,0]+confusion_matrix_train[1,1])
TNR_Train= confusion_matrix_train[0,0]/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1])
TPR_Train= confusion_matrix_train[1,1]/(confusion_matrix_train[1,0]+confusion_matrix_train[1,1])

print("Train TNR: ",TNR_Train)
print("\n")
print("Train TPR: ",TPR_Train)
print("\n")
print("Train Accuracy: ",Accuracy_Train)
Accuracy_Test=(confusion_matrix_validation[0,0]+confusion_matrix_validation[1,1])/(confusion_matrix_validation[0,0]+confusion_matrix_validation[0,1]+confusion_matrix_validation[1,0]+confusion_matrix_validation[1,1])
TNR_Test= confusion_matrix_validation[0,0]/(confusion_matrix_validation[0,0] +confusion_matrix_validation[0,1])
TPR_Test= confusion_matrix_validation[1,1]/(confusion_matrix_validation[1,0] +confusion_matrix_validation[1,1])

print("Validation TNR: ",TNR_Test)
print("\n")
print("Validation TPR: ",TPR_Test)
print("\n")
print("Validation Accuracy: ",Accuracy_Test)
## Display classification metrics for train data
print(classification_report(y_true=y_train,y_pred=train_pred))
## Display classification metrics for validation data
print(classification_report(y_true=y_test,y_pred=validation_pred))
## Get copy of test data
test_data = test.copy()
## Get the predictions on test data
y_pred = Adaboost_model.predict(test_data)
y_pred
## Do inverse transform on y_ped so that we will get it's original values(male,female)
test_data['gender'] = le_gender.inverse_transform(y_pred)
## Do inverse transform on session_id so that we will get it's original values
test_data['session_id'] = le_session.inverse_transform(test_data['session_id'])
## Check first 5 records after doing inverse transformation
test_data.head()
## Check the dimensions of the test_data before dropping dupliccate data
test_data.shape
## Drop duplicate session_id values
test_data = test_data.drop_duplicates(subset=['session_id'])
## Check the dimensions of the test_data after dropping dupliccate data
test_data.shape
## Get session_id,gender from test_data and store into to_submit
to_submit = test_data[['session_id', 'gender']]
## Check first 5 records of to_submit
to_submit.head()
## Get gender count value 
to_submit.gender.value_counts()
## store to_submit into csv file with name adaboost_model 
to_submit.to_csv('adaboost_model.csv',index = False)
## Instantiate GradientBoost model and fit it
gbm = GradientBoostingClassifier(n_estimators=100,learning_rate=0.3)
%time gbm.fit(X=X_train, y=y_train)
## Get predictions on train and validation data
train_pred = gbm.predict(X_train)
validation_pred = gbm.predict(X_test)
## Get confusion matrix for train data
confusion_matrix_train = confusion_matrix(y_train, train_pred)
print(confusion_matrix_train)
## Get confusion matrix for validation data
confusion_matrix_validation = confusion_matrix(y_test, validation_pred)
print(confusion_matrix_validation)
Accuracy_Train=(confusion_matrix_train[0,0]+confusion_matrix_train[1,1])/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1]+confusion_matrix_train[1,0]+confusion_matrix_train[1,1])
TNR_Train= confusion_matrix_train[0,0]/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1])
TPR_Train= confusion_matrix_train[1,1]/(confusion_matrix_train[1,0]+confusion_matrix_train[1,1])

print("Train TNR: ",TNR_Train)
print("\n")
print("Train TPR: ",TPR_Train)
print("\n")
print("Train Accuracy: ",Accuracy_Train)
Accuracy_Test=(confusion_matrix_validation[0,0]+confusion_matrix_validation[1,1])/(confusion_matrix_validation[0,0]+confusion_matrix_validation[0,1]+confusion_matrix_validation[1,0]+confusion_matrix_validation[1,1])
TNR_Test= confusion_matrix_validation[0,0]/(confusion_matrix_validation[0,0] +confusion_matrix_validation[0,1])
TPR_Test= confusion_matrix_validation[1,1]/(confusion_matrix_validation[1,0] +confusion_matrix_validation[1,1])

print("Validation TNR: ",TNR_Test)
print("\n")
print("Validation TPR: ",TPR_Test)
print("\n")
print("Validation Accuracy: ",Accuracy_Test)
## Display classification metrics for train data
print(classification_report(y_true=y_train,y_pred=train_pred))
## Display classification metrics for validation data
print(classification_report(y_true=y_test,y_pred=validation_pred))
## Get copy of test data
test_data = test.copy()
## Get the predictions on test data
y_pred = gbm.predict(test_data)
y_pred
## Do inverse transform on y_ped so that we will get it's original values(male,female)
test_data['gender'] = le_gender.inverse_transform(y_pred)
## Do inverse transform on session_id so that we will get it's original values
test_data['session_id'] = le_session.inverse_transform(test_data['session_id'])
## Check first 5 records after doing inverse transformation
test_data.head()
## Check the dimensions of the test_data before dropping dupliccate data
test_data.shape
## Drop duplicate session_id values
test_data = test_data.drop_duplicates(subset=['session_id'])
## Check the dimensions of the test_data after dropping dupliccate data
test_data.shape
## Get session_id,gender from test_data and store into to_submit
to_submit = test_data[['session_id', 'gender']]
## Check first 5 records of to_submit
to_submit.head()
## Get gender count value 
to_submit.gender.value_counts()
## store to_submit into csv file with name gb_model 
to_submit.to_csv('gb_model.csv',index = False)
## Instantiate SVC model
svc_c10_rbf = SVC(C=10,kernel='rbf')
## Fit a model
%time svc_c10_rbf.fit(X=X_train,y=y_train)
## Get predictions on train and validations data
train_pred = svc_c10_rbf.predict(X_train)
validation_pred = svc_c10_rbf.predict(X_test)
## Get confusion matrix for train data
confusion_matrix_train = confusion_matrix(y_train, train_pred)
print(confusion_matrix_train)
## Get confusion matrix for validation data
confusion_matrix_validation = confusion_matrix(y_test, validation_pred)
print(confusion_matrix_validation)
Accuracy_Train=(confusion_matrix_train[0,0]+confusion_matrix_train[1,1])/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1]+confusion_matrix_train[1,0]+confusion_matrix_train[1,1])
TNR_Train= confusion_matrix_train[0,0]/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1])
TPR_Train= confusion_matrix_train[1,1]/(confusion_matrix_train[1,0]+confusion_matrix_train[1,1])

print("Train TNR: ",TNR_Train)
print("\n")
print("Train TPR: ",TPR_Train)
print("\n")
print("Train Accuracy: ",Accuracy_Train)
Accuracy_Test=(confusion_matrix_validation[0,0]+confusion_matrix_validation[1,1])/(confusion_matrix_validation[0,0]+confusion_matrix_validation[0,1]+confusion_matrix_validation[1,0]+confusion_matrix_validation[1,1])
TNR_Test= confusion_matrix_validation[0,0]/(confusion_matrix_validation[0,0] +confusion_matrix_validation[0,1])
TPR_Test= confusion_matrix_validation[1,1]/(confusion_matrix_validation[1,0] +confusion_matrix_validation[1,1])

print("Validation TNR: ",TNR_Test)
print("\n")
print("Validation TPR: ",TPR_Test)
print("\n")
print("Validation Accuracy: ",Accuracy_Test)
## Display classification metrics for train data
print(classification_report(y_true=y_train,y_pred=train_pred))
## Display classification metrics for validation data
print(classification_report(y_true=y_test,y_pred=validation_pred))
## Get copy of test data
test_data = test.copy()
## Get the predictions on test data
y_pred = svc_c10_rbf.predict(test_data)
y_pred
## Do inverse transform on y_ped so that we will get it's original values(male,female)
test_data['gender'] = le_gender.inverse_transform(y_pred)
## Do inverse transform on session_id so that we will get it's original values
test_data['session_id'] = le_session.inverse_transform(test_data['session_id'])
## Check first 5 records after doing inverse transformation
test_data.head()
## Check the dimensions of the test_data before dropping dupliccate data
test_data.shape
## Drop duplicate session_id values
test_data = test_data.drop_duplicates(subset=['session_id'])
## Check the dimensions of the test_data after dropping dupliccate data
test_data.shape
## Get session_id,gender from test_data and store into to_submit
to_submit = test_data[['session_id', 'gender']]
## Check first 5 records of to_submit
to_submit.head()
## Get gender count value 
to_submit.gender.value_counts()
## store to_submit into csv file with name svc_model 
to_submit.to_csv('svc_model.csv',index = False)
## Instantiate sequential model and fully connected layer to it
perceptron_model = Sequential()

perceptron_model.add(Dense(1, input_dim=X_train.shape[1], activation='sigmoid', kernel_initializer='normal'))
## add compile to seequential model
perceptron_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
## Fit a model
perceptron_model.fit(X_train, y_train, epochs=20)
## Get preidctions on train and validation data
train_pred = perceptron_model.predict_classes(X_train)
validation_pred = perceptron_model.predict_classes(X_test)
## Get confusion matrix for train data
confusion_matrix_train = confusion_matrix(y_train, train_pred)
print(confusion_matrix_train)
## Get confusion matrix for validation data
confusion_matrix_validation = confusion_matrix(y_test, validation_pred)
print(confusion_matrix_validation)
Accuracy_Train=(confusion_matrix_train[0,0]+confusion_matrix_train[1,1])/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1]+confusion_matrix_train[1,0]+confusion_matrix_train[1,1])
TNR_Train= confusion_matrix_train[0,0]/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1])
TPR_Train= confusion_matrix_train[1,1]/(confusion_matrix_train[1,0]+confusion_matrix_train[1,1])

print("Train TNR: ",TNR_Train)
print("\n")
print("Train TPR: ",TPR_Train)
print("\n")
print("Train Accuracy: ",Accuracy_Train)
Accuracy_Test=(confusion_matrix_validation[0,0]+confusion_matrix_validation[1,1])/(confusion_matrix_validation[0,0]+confusion_matrix_validation[0,1]+confusion_matrix_validation[1,0]+confusion_matrix_validation[1,1])
TNR_Test= confusion_matrix_validation[0,0]/(confusion_matrix_validation[0,0] +confusion_matrix_validation[0,1])
TPR_Test= confusion_matrix_validation[1,1]/(confusion_matrix_validation[1,0] +confusion_matrix_validation[1,1])

print("Validation TNR: ",TNR_Test)
print("\n")
print("Validation TPR: ",TPR_Test)
print("\n")
print("Validation Accuracy: ",Accuracy_Test)
## Display classification metrics for train data
print(classification_report(y_true=y_train,y_pred=train_pred))
## Display classification metrics for validation data
print(classification_report(y_true=y_test,y_pred=validation_pred))
## Get copy of test data
test_data = test.copy()
## Get the predictions on test data
y_pred = perceptron_model.predict(test_data)
y_pred
## Copy y_pred values into temp and empty the y_pred
temp = y_pred
y_pred = []
## map y_pred values based on threshold
for i in temp:
    if i>0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)
y_pred[:10]
## Do inverse transform on y_ped so that we will get it's original values(male,female)
test_data['gender'] = le_gender.inverse_transform(y_pred)
## Do inverse transform on session_id so that we will get it's original values
test_data['session_id'] = le_session.inverse_transform(test_data['session_id'])
## Check first 5 records after doing inverse transformation
test_data.head()
## Check the dimensions of the test_data before dropping dupliccate data
test_data.shape
## Drop duplicate session_id values
test_data = test_data.drop_duplicates(subset=['session_id'])
## Check the dimensions of the test_data after dropping dupliccate data
test_data.shape
## Get session_id,gender from test_data and store into to_submit
to_submit = test_data[['session_id', 'gender']]
## Check first 5 records of to_submit
to_submit.head()
## Get gender count value 
to_submit.gender.value_counts()
## store to_submit into csv file with name neuralnetworks_model 
to_submit.to_csv('neuralnetworks_model.csv',index = False)