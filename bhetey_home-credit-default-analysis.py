from __future__ import division
import pandas as pd # this is to import the pandas module
import numpy as np # importing the numpy module 
import os # file system management 
import zipfile # module to read ZIP archive files.
from glob import glob 
import seaborn as sns
import matplotlib.pyplot as plt

# Figures inline and set visualization style
%matplotlib inline
sns.set()

#print(os.listdir("../input/*.csv"))
filenames = glob("../input/*.csv")
filenames
# reading the data with pandas 
def reading_csv_file(filename):
    return pd.read_csv(filename)
app_train = reading_csv_file(filenames[2])
print ('Training data shape :{}'.format(app_train.shape))
app_train.head()

y = app_train.TARGET # y is going to be our target variable
y.head()
app_test = reading_csv_file(filenames[7])
print ('Testing test contains :{}'.format(app_test.shape))
app_test.head(5)
# traing and testing set do not have the same shape. 
app_train.shape == app_test.shape 
# converted all csv in pandas dataframe. 
pos_cash_balance = reading_csv_file(filenames[0])
bureau_balance = reading_csv_file(filenames[1])
previous_application = reading_csv_file(filenames[3])
installments_payments = reading_csv_file(filenames[4])
credit_card_balance = reading_csv_file(filenames[5])
bureau = reading_csv_file(filenames[8])
# this is done for indexing of the joint data later 
data = (os.listdir("../input/"))
data.remove('sample_submission.csv')
data
print (app_train['TARGET'].value_counts())
app_train.head(5)
sns.countplot(x='TARGET',hue='TARGET', data=app_train)
plt.figure(figsize=(10,3))
sns.countplot(x = "NAME_FAMILY_STATUS", hue = "TARGET", data = app_train)
fam_stat_target = pd.crosstab(app_train['NAME_FAMILY_STATUS'], app_train['TARGET'])
kind_of_applicant = ("Repayers", "Defaulters")
fam_stat_target.columns = kind_of_applicant
fam_stat_target
plt.figure(figsize= (10,5))# plot the figure 
plt.show(sns.countplot(x = "NAME_FAMILY_STATUS", 
                       hue = "NAME_INCOME_TYPE" , 
                       # filter the train set by using TARGET column == 1
                       data= app_train.loc[app_train['TARGET'] == 1])) 
# below is a function to check for missing values. 
def check_missing_values(input):
    # checking total missing values
    total_miss_values = input.isnull().sum()
    
    # percentage of missing values. 
    miss_val_percent = total_miss_values/len(input)*100
    
    # table of total_miss_values and it's percentage
    miss_val_percent_tab = pd.concat([total_miss_values, miss_val_percent], axis=1)
    
    # columns renamed
    new_col_names = ('Missing values', 'Total missing values in %')
    miss_val_percent_tab.columns = new_col_names
    renamed_miss_val_percent_tab = miss_val_percent_tab
    
    # descending table sort 
    renamed_miss_val_percent_tab = renamed_miss_val_percent_tab[
        renamed_miss_val_percent_tab.iloc[:,1] != 0
    ].sort_values('Total missing values in %', ascending = False).round(1)
    
    # display information 
    print ('The selected dataframe has {} columns.\n'.format(input.shape[1]))
    print ('There are {} columns missing in the dataset'.format(renamed_miss_val_percent_tab.shape[0]))
    
    return renamed_miss_val_percent_tab
missing_value = check_missing_values(app_train)
missing_value.head(10)
app_train.get_dtype_counts() # Shows the numbers of types of values 
print ('Total numbe of object type is : ', len(app_train.select_dtypes('object')))
app_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
(app_train['DAYS_BIRTH']/-365).describe()
(app_train['DAYS_EMPLOYED']/365).describe()
(app_train['DAYS_EMPLOYED']/365).plot.hist(title = 'DAYS EMPLOYED');
plt.xlabel('NO OF DAYS EMPLOYED BEFORE APPLICATION')
data_corr = app_train.corr()['TARGET'].sort_values()
print ('These are samples of negative correlations : \n',data_corr.head(20))
print ('These are samples of positive correlations : \n', data_corr.tail(20))
app_train.select_dtypes('object').columns
len (app_train.columns) == len(app_train.select_dtypes('object').columns)
from sklearn.preprocessing import OneHotEncoder

one_hot_encoded_app_train = pd.get_dummies(app_train)
one_hot_encoded_app_test = pd.get_dummies(app_test)

print ('Shape of the training set after one hot encoding {}'.format(one_hot_encoded_app_train.shape))
print ('Shape of the test set after one hot encoding {}'.format(one_hot_encoded_app_test.shape))
app_train.shape == one_hot_encoded_app_train.shape
app_train.shape
app_test.shape
#https://pandas.pydata.org/pandas-docs/version/0.21/generated/pandas.DataFrame.align.html
one_hot_encoded_app_train, one_hot_encoded_app_test = one_hot_encoded_app_train.align(one_hot_encoded_app_test,
                                                                                      join='inner', axis=1)

print ('Shape of the training set after alignment {}'.format(one_hot_encoded_app_train.shape))
print ('Shape of the test set after alignment {}'.format(one_hot_encoded_app_test.shape))
one_hot_encoded_app_train['TARGET'] = y # adding it back to the data 
# dropping the target to get our X 
X = one_hot_encoded_app_train.drop(['TARGET'], axis=1)
# function is to drop the missing values 
def dropping_missing_columns(input_set):
    """this function removes the columns with missing values.
    However input_set is the set you will put inside in the function 
    either the training set or the test set 
    """
    to_drop_missing_missing_values = [
        col for col in input_set.columns if X[col].isnull().any()
    ]
    return input_set.drop(to_drop_missing_missing_values, axis = 1)
#assigned a variable to data after dropping the missing values 
after_removing_missing_values = dropping_missing_columns(X)
test_removing_missing_values = dropping_missing_columns(one_hot_encoded_app_test)
print ('The shape of training set after removing missing values : {}'.format(after_removing_missing_values.shape))
print ('The shape of testing set after removing missing values :{}'.format(test_removing_missing_values.shape))
from sklearn.preprocessing import MinMaxScaler, Imputer

# making a copy for the data before imputing 
train_tobe_imputed = one_hot_encoded_app_train.copy() 
test_tobe_imputed = one_hot_encoded_app_test.copy()
new_y = y.copy() # a copy of our target 

# dropping the target columns before imputing 
new_X = train_tobe_imputed.drop(['TARGET'], axis=1)

# calling imputer and transforming the data
imputer = Imputer()
transformed_X = imputer.fit_transform(new_X)
transformed_test_X = imputer.fit_transform(test_tobe_imputed)

print ('Transformed training set :{}'.format(transformed_X.shape))
print ('Transformed testing set :{}'.format(transformed_test_X.shape))
print ('The data is back to the same shape we had during the Hot coding')
# Import statements 
from sklearn.metrics import classification_report
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.cross_validation import train_test_split
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(
    after_removing_missing_values, y, test_size=0.25, random_state=42)

# instantiate a logistic regression model, and fit with X and y
model = LogisticRegression(C = 0.0001)

# evaluate the model by splitting into train and test sets
model.fit(X_train, y_train)
print ('This accuracy seems good {} but need to check further for the prediction and on testing set'.format(model.score(X_train, y_train)))
predicted = model.predict(X_test)
predicting_with_test = model.predict(test_removing_missing_values)
predicting_probability = model.predict_proba(X_test)
matrix = confusion_matrix(y_test, predicted)
scoring = accuracy_score(y_test, predicted)
r_score = r2_score(y_test, predicted)
report = classification_report(y_test, predicted)
print (report)
print (matrix)
print ('Accuracy of the model :{}'.format(scoring))
print ('R2 Score for the prediction :'.format(r_score))
print(metrics.roc_auc_score(y_test, predicting_probability[:, 1]))
# evaluate the model using 10-fold cross-validation
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(LogisticRegression(), 
                         after_removing_missing_values, y, scoring='accuracy', cv=10)
print (scores)
print (scores.mean())
print (predicting_with_test)
my_submission = pd.DataFrame({'SK_ID_CURR': one_hot_encoded_app_test.SK_ID_CURR, 'TARGET': predicting_with_test})
my_submission.to_csv('homecredit.csv', index=False)
my_submission
