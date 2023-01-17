### Understanding the problem and our objective

# In this project,We will behaving as a Loan issuer and will manage the credit risk by using the past data 
# and deciding whom to give the loan in the future. 
# The text files contain complete loan data for all loans issued by XYZ Corp. through 2007-2015. 
# The data contains the indicator of default, payment information, credit history, etc.
# Based on the data that is available during loan application build a model to predict default in the future. 
# This will help the company in deciding whether or not to pass the loan.
#Importing Required packages

import os
get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np                     # For mathematical calculations
import seaborn as sns                  # For data visualization
import matplotlib.pyplot as plt        # For plotting graphs
import math
%matplotlib inline
import warnings                        # To ignore any warnings
warnings.filterwarnings("ignore")
## Reading the XYZ Corporation Lending Data file from .txt file

pd.set_option("display.max_columns",None)
print(os.listdir("../input"))
credit_data = pd.read_table("../input/XYZCorp_LendingData.txt", low_memory = False, delimiter= '\t')

# Making copy of Original data
credit_data_original = credit_data.copy()
# Features in dataset
credit_data.columns
# Printing Datatypes of each variable
credit_data.dtypes
# Shape of dataset
credit_data.shape
# Top 5 rows to look at the data set
credit_data.head()
## Exploratory Data Analyis (EDA)
# 1) Look at the Data and variables
# 2) Data Cleaning 
#      Remove those variables having more than 75% of missing values.
# 3) Missing Values treatment with Mean/ Mode
# 4) Dropping irrelevant variables

credit_data.info()
## To check all NULL values
credit_data_null = credit_data.isnull().sum(axis=0).sort_values( ascending=False)/float(len(credit_data)) 
credit_data_null
# NULL values greater than 75% will be removed
credit_data_null[credit_data_null > 0.75]
credit_data.drop(credit_data_null[credit_data_null>0.75].index, axis = 1, inplace = True)
credit_data.info()
## All NULL values greater than 75% will get removed and now check how many columns are remaining
credit_data.shape
## Few variables are having Unique value which are a bit misleading for Analysis 
## So those variables need to be removed
## Variable - Policy id has same value 1 for all rows
## Title , Emp_Title, Zip Code is not needed 
## Application_Type, acc_now_delinq has same value for almost 99% records
## id and member_id is almost same

credit_data_remove = ['policy_code', 'pymnt_plan', 'id', 'member_id', 'application_type', 
                      'acc_now_delinq','emp_title', 'zip_code','title']

## Drop those variable which are not relevant for Analysis
credit_data.drop( credit_data_remove , axis = 1, inplace = True )
# After dropping , remaining columns
credit_data.columns
## Missing Value treatment
# Categoricals values to be filled with Mode and Interger variable with Mean value

mis_val_per=100*credit_data.isnull().sum()/len(credit_data)
mis_val_per
credit_data.isnull().sum()/len(credit_data)< 0.08
##Fill the variable 'mths_since_last_delinq' with mean value as its an Integer variable
credit_data['mths_since_last_delinq']=credit_data['mths_since_last_delinq'].fillna(credit_data['mths_since_last_delinq'].mean())
# 'next_pymnt_d'is a categorical variable.
# So we check and fill according to MODE
credit_data['next_pymnt_d'].value_counts()
# As we can see tha Feb-2016 has higher number of count so we fill 'next_pymnt_d' missing values with 'Feb-2016' value
nxt = credit_data['next_pymnt_d'].mode()
nxt
credit_data = credit_data.fillna({'next_pymnt_d' : 'Feb-2016'})
mis_val_per=100*credit_data.isnull().sum()/len(credit_data)
mis_val_per
credit_data=credit_data.dropna()
credit_data.isnull().sum()
credit_data.shape
# Variable Transformations
# 
# To make a modelling ready dataset, we have to transform a few variables
# 
# A summary of the operations performed:
# 
# - Strip months from term and make it an integer
# - Extract numbers from emp_length and fill missing values with the median. If emp_length == 10+ years then leave it as 10
# - Transform datetimes to a Period

# Extract months from 'term' and change it to an integer value 

credit_data['term'] = credit_data['term'].str.split(' ').str[1]
# extract numbers from emp_length and fill missing values with the median
credit_data['emp_length'] = credit_data['emp_length'].str.extract('(\d+)').astype(float)
credit_data['emp_length'] = credit_data['emp_length'].fillna(credit_data.emp_length.median())
col_dates = credit_data.dtypes[credit_data.dtypes == 'datetime64[ns]'].index
for d in col_dates:
    credit_data[d] = credit_data[d].dt.to_period('M')
credit_data.head()
credit_data.tail()
credit_data.info()
credit_data['default_ind'].value_counts()
## Data Visualitation
## we plot Categorical variables and integers variables to have a better look of the data
## According to which we do feature engineering i.e. keep only those variables which we use for modelling

# UDF is created to plot Categorical variables

def plot(name,title,value):
    categ_plot=credit_data[value].value_counts(normalize=True)
    plt.figure(figsize=(12,5))
    plt.title(title)
    plt.ylabel(name)
    sns.barplot(x=categ_plot.index, y=categ_plot.values)
    plt.show()
# Now plot the variables

plot('Borrowers','Length of Employment of Borrowers', 'emp_length')
plot('Borrowers','Grade of Loans', 'grade', )    
plot('Borrowers', 'Types of Home', 'home_ownership')
plot('Borrowers', 'Purpose of Loan', 'purpose')
plot('Borrowers','Loan Verification Status', 'verification_status')
plot('Borrowers', 'Loan Status', 'default_ind')
import warnings
warnings.filterwarnings("ignore")
plt.figure(1)
plt.subplot(121)
sns.distplot(credit_data['int_rate']);

plt.subplot(122)
credit_data['int_rate'].plot.box(figsize=(16,5))

plt.show()
plt.figure(1)
plt.subplot(121)
sns.distplot(credit_data['loan_amnt']);

plt.subplot(122)
credit_data['loan_amnt'].plot.box(figsize=(16,5))

plt.show()
plt.figure(1)
plt.subplot(121)
sns.distplot(credit_data['installment']);

plt.subplot(122)
credit_data['installment'].plot.box(figsize=(16,5))

plt.show()
## Feature Engineering 
## Keeping those variables which are relevant for modelling

credit_data['amt_difference'] = 'eq'
credit_data.loc[ ( credit_data['funded_amnt'] - credit_data['funded_amnt_inv']) > 0, 'amt_difference' ] = 'less'
credit_data.head()
# Make categorical

credit_data[ 'delinq_2yrs_cat' ] = 'no'
credit_data.loc[ credit_data [ 'delinq_2yrs' ] > 0, 'delinq_2yrs_cat' ] = 'yes'

credit_data[ 'inq_last_6mths_cat' ] = 'no'
credit_data.loc[ credit_data['inq_last_6mths' ] > 0, 'inq_last_6mths_cat' ] = 'yes'
credit_data[ 'pub_rec_cat' ] = 'no'
credit_data.loc[ credit_data['pub_rec'] > 0,'pub_rec_cat' ] = 'yes'
credit_data.head()
credit_data.tail()
# Create new metric
credit_data['acc_ratio'] = credit_data.open_acc / credit_data.total_acc
credit_data['acc_ratio']
## Feature Selection
## Final Features to be needed for modelling

final_features = [
            'loan_amnt', 'amt_difference', 'term', 
            'installment', 'grade','emp_length',
            'home_ownership', 'annual_inc','verification_status',
            'purpose', 'dti', 'delinq_2yrs_cat', 'inq_last_6mths_cat', 
            'open_acc', 'pub_rec', 'pub_rec_cat', 'acc_ratio', 'initial_list_status',
            'issue_d','default_ind'
           ]
credit_data_final = credit_data[final_features]
# Drop any residual missing values
credit_data_final.dropna( axis=0, how = 'any', inplace = True )
## We will divide our dataset into Train and Test 
## The train ( June 2007 - May 2015 ) and out-of-time test ( June 2015 - Dec 2015 ) data

credit_data_final.shape
credit_data_final.columns
out_of_time = ['Jun-2015', 'Jul-2015', 'Aug-2015', 'Sep-2015', 'Oct-2015', 'Nov-2015', 'Dec-2015']

train_dataset = credit_data_final.loc [ -credit_data_final.issue_d.isin(out_of_time) ]
out_of_time_test = credit_data_final.loc [ credit_data_final.issue_d.isin(out_of_time) ]
train_dataset.shape
out_of_time_test.shape
## To transform Categorical Variables into Dummy Variables

colname_categorical = ['term','amt_difference', 'grade', 'home_ownership', 'verification_status', 
                            'purpose', 'delinq_2yrs_cat', 'inq_last_6mths_cat', 'initial_list_status', 'pub_rec_cat']
colname_categorical
X_train_dataset = pd.get_dummies(train_dataset[train_dataset.columns[:-2]], columns=colname_categorical).astype(float)
y_train_dataset = train_dataset['default_ind']
X_out_of_time_test = pd.get_dummies(out_of_time_test[out_of_time_test.columns[:-2]], columns=colname_categorical).astype(float)
Y_out_of_time_test = out_of_time_test['default_ind']
print(X_train_dataset.shape, X_out_of_time_test.shape)
X_train_dataset.columns
X_out_of_time_test.columns
vars_not_in_test = ['home_ownership_NONE','home_ownership_OTHER','purpose_educational']
X_train_dataset.drop( vars_not_in_test , axis = 1, inplace = True )
print(X_train_dataset.shape)
#Preprocessing 
#Scaling the variables 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train_dataset,X_out_of_time_test )
X_train_dataset=scaler.transform(X_train_dataset)
X_out_of_time_test=scaler.transform(X_out_of_time_test)
print(X_train_dataset)
print(X_out_of_time_test)
y_train_dataset=y_train_dataset.astype(int)
print(y_train_dataset)
## After feature engineering, we can now move to the model building process. 
## So we will start with logistic regression model and then move over to more complex models 
# like Decision Tree, RandomForest and XGBoost.

###  We will build the following models in this section.

 ## Logistic Regression
 ## Decision Tree
 ## Random Forest
 ## XGBoost
## Import packages for modelling and cross validation

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix,recall_score,precision_recall_curve,auc,roc_curve,roc_auc_score,classification_report
model = LogisticRegression(random_state=1)
model.fit(X_train_dataset, y_train_dataset)
pred_test = model.predict(X_out_of_time_test)

score = accuracy_score(Y_out_of_time_test,pred_test)
cnf_matrix=confusion_matrix(Y_out_of_time_test,pred_test)
prec_score = metrics.precision_score(Y_out_of_time_test,pred_test)
recal_score = metrics.recall_score(Y_out_of_time_test,pred_test)
print('Accuracy Score is ', score)
print('Precision score is ',prec_score)
print('Recall score is ',recal_score)
print('Confusion Matrix is ', cnf_matrix)
print(classification_report(Y_out_of_time_test,pred_test))
pred_test = model.predict(X_out_of_time_test)
print(np.unique(pred_test))
pred=model.predict_proba(X_out_of_time_test)[:,1]
# ROC curve

from sklearn import metrics

fpr, tpr, thresholds = metrics.roc_curve(Y_out_of_time_test, pred, drop_intermediate = False, pos_label = 1)
auc = metrics.roc_auc_score(Y_out_of_time_test, pred)
plt.figure(figsize=(12,8))
plt.plot(fpr,tpr,label="validation, auc="+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()
# Lets Fit decision tree model and Check accuracy score 
from sklearn import tree

model = tree.DecisionTreeClassifier(random_state=1)
model.fit(X_train_dataset, y_train_dataset)
pred_test = model.predict(X_out_of_time_test)
score = accuracy_score(Y_out_of_time_test,pred_test)
cnf_matrix=confusion_matrix(Y_out_of_time_test,pred_test)
prec_score = metrics.precision_score(Y_out_of_time_test,pred_test)
recal_score = metrics.recall_score(Y_out_of_time_test,pred_test)
print('Accuracy Score is ', score)
print('Precision score is ',prec_score)
print('Recall score is ',recal_score)
print('Confusion Matrix is ', cnf_matrix)
print(classification_report(Y_out_of_time_test,pred_test))
pred_test = model.predict(X_out_of_time_test)
print(np.unique(pred_test))
pred=model.predict_proba(X_out_of_time_test)[:,1]
# ROC curve

from sklearn import metrics

fpr, tpr, thresholds = metrics.roc_curve(Y_out_of_time_test, pred, drop_intermediate = False, pos_label = 1)
auc = metrics.roc_auc_score(Y_out_of_time_test, pred)
plt.figure(figsize=(12,8))
plt.plot(fpr,tpr,label="validation, auc="+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=1)
model.fit(X_train_dataset, y_train_dataset)
pred_test = model.predict(X_out_of_time_test)
score = accuracy_score(Y_out_of_time_test,pred_test)
cnf_matrix=confusion_matrix(Y_out_of_time_test,pred_test)
prec_score = metrics.precision_score(Y_out_of_time_test,pred_test)
recal_score = metrics.recall_score(Y_out_of_time_test,pred_test)
print('Accuracy Score is ', score)
print('Precision score is ',prec_score)
print('Recall score is ',recal_score)
print('Confusion Matrix is ', cnf_matrix)
print(classification_report(Y_out_of_time_test,pred_test))
pred_test = model.predict(X_out_of_time_test)
pred=model.predict_proba(X_out_of_time_test)[:,1]
# ROC curve

from sklearn import metrics

fpr, tpr, thresholds = metrics.roc_curve(Y_out_of_time_test, pred, drop_intermediate = False, pos_label = 1)
auc = metrics.roc_auc_score(Y_out_of_time_test, pred)
plt.figure(figsize=(12,8))
plt.plot(fpr,tpr,label="validation, auc="+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()
from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=50, max_depth=4)
model.fit(X_train_dataset, y_train_dataset)
pred_test = model.predict(X_out_of_time_test)
score = accuracy_score(Y_out_of_time_test,pred_test)
cnf_matrix=confusion_matrix(Y_out_of_time_test,pred_test)
prec_score = metrics.precision_score(Y_out_of_time_test,pred_test)
recal_score = metrics.recall_score(Y_out_of_time_test,pred_test)
print('Accuracy Score is ', score)
print('Precision score is ',prec_score)
print('Recall score is ',recal_score)
print('Confusion Matrix is ', cnf_matrix)
print(classification_report(Y_out_of_time_test,pred_test))
pred_test = model.predict(X_out_of_time_test)
pred=model.predict_proba(X_out_of_time_test)[:,1]
# ROC curve

from sklearn import metrics

fpr, tpr, thresholds = metrics.roc_curve(Y_out_of_time_test, pred, drop_intermediate = False, pos_label = 1)
auc = metrics.roc_auc_score(Y_out_of_time_test, pred)
plt.figure(figsize=(12,8))
plt.plot(fpr,tpr,label="validation, auc="+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()
## A look at the performance tells us that the techniques are not peforming well as the recall is very poor. An important reason for this can be that the classes are unbalanced. 
## So we will try another iteration but this time by balancing classes.

## Here is where a good business understanding of the practical situation comes in handy, 
## getting good recall is more important than getting good precision, because 
## as a banker I would be more concerned about catching more defaulters 
## to minimize my losses rather than being very right all the time!
## To balance the classes , we use up-sampling the minority class.
## Up-sampling is the process of randomly duplicating observations from the minority class in order to reinforce its signal.

## There are several heuristics for doing so, but the most common way is to simply resample with replacement.

train_dataset['default_ind'].value_counts()
## First, We'll import the resampling module from Scikit-Learn

from sklearn.utils import resample
### Next, we'll create a new DataFrame with an up-sampled minority class. Here are the steps:

## 1) we'll separate observations from each class into different DataFrames.
## 2) we'll resample the minority class with replacement, setting the number of samples to match that of the majority class.
## Finally, we'll combine the up-sampled minority class DataFrame with the original majority class DataFrame.
# Separate majority and minority classes

df_majority = train_dataset[train_dataset.default_ind==0]
df_minority = train_dataset[train_dataset.default_ind==1]
# Upsample minority class

df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=471342,    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class

df_upsampled = pd.concat([df_majority, df_minority_upsampled])
# Display new class counts
df_upsampled.default_ind.value_counts()
## The same process we will do with Test dataset
out_of_time_test['default_ind'].value_counts()
# Separate majority and minority classes
df_test_majority = out_of_time_test[out_of_time_test.default_ind==0]
df_test_minority = out_of_time_test[out_of_time_test.default_ind==1]
# Upsample minority class

from sklearn.utils import resample
df_test_minority_upsampled = resample(df_test_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=233860,    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
df_test_upsampled = pd.concat([df_test_majority, df_test_minority_upsampled])
# Display new class counts
df_test_upsampled.default_ind.value_counts()
X_train_dataset_new = pd.get_dummies(df_upsampled[df_upsampled.columns[:-2]], columns=colname_categorical).astype(float)
y_train_dataset_new = df_upsampled['default_ind']
X_out_of_time_test_new = pd.get_dummies(df_test_upsampled[df_test_upsampled.columns[:-2]], columns=colname_categorical).astype(float)
Y_out_of_time_test_new = df_test_upsampled['default_ind']
print(X_train_dataset_new.shape, X_out_of_time_test_new.shape)
X_train_dataset_new.columns
X_out_of_time_test_new.columns
vars_not_in_test = ['home_ownership_NONE','home_ownership_OTHER','purpose_educational']
X_train_dataset_new.drop( vars_not_in_test , axis = 1, inplace = True )
print(X_train_dataset_new.shape, X_out_of_time_test_new.shape)
#Preprocessing 
#Scaling the variables 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train_dataset_new,X_out_of_time_test_new )
X_train_dataset_new=scaler.transform(X_train_dataset_new)
X_out_of_time_test_new=scaler.transform(X_out_of_time_test_new)
y_train_dataset_new=y_train_dataset_new.astype(int)
## Now we will do Modelling an dvalidation on balanced dataset
from sklearn.model_selection import cross_val_score
def run_models(X_train_dataset_new,y_train_dataset_new,X_out_of_time_test_new,Y_out_of_time_test_new, model_type= "Balanced"):

    classification_models = { 'Logistic Regression': LogisticRegression(random_state=1),
                             'Decision Tree': tree.DecisionTreeClassifier(random_state=1), 
                             'Random Forest': RandomForestClassifier(random_state=1),
                             'Xgboost Classifier':  XGBClassifier(n_estimators=50, max_depth=4)}
    
    cols = ['Model', 'Accuracy Score', 'Precision score', 'Recall score','Confusion Matrix']

    models_report = pd.DataFrame(columns = cols)
    conf_matrix = dict()

    for clf, clf_name in zip(classification_models.values(), classification_models.keys()):

        clf.fit(X_train_dataset_new, y_train_dataset_new)

        y_pred = clf.predict(X_out_of_time_test_new)
        y_score = clf.predict_proba(X_out_of_time_test_new)[:,1]

        print('computing {} - {} '.format(clf_name, model_type))

        tmp = pd.Series({'model_type': model_type,
                         'Model': clf_name,
                         'Accuracy Score' : accuracy_score(Y_out_of_time_test_new,y_pred),
                         'Precision score': metrics.precision_score(Y_out_of_time_test_new, y_pred),
                         'Recall score': metrics.recall_score(Y_out_of_time_test_new, y_pred),
                         'Confusion Matrix': confusion_matrix(Y_out_of_time_test_new,y_pred)})

        models_report = models_report.append(tmp, ignore_index = True)
        conf_matrix[clf_name] = pd.crosstab(Y_out_of_time_test_new, y_pred, rownames=['True'], colnames= ['Predicted'], margins=False)
        fpr, tpr, thresholds = metrics.roc_curve(Y_out_of_time_test_new, y_score, drop_intermediate = False, pos_label = 1)

        plt.figure(1, figsize=(6,6))
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title('ROC curve - {}'.format(model_type))
        plt.plot(fpr, tpr, label = clf_name )
        plt.legend(loc=2, prop={'size':11})
    plt.plot([0,1],[0,1], color = 'black')
    
    return models_report, conf_matrix
from sklearn.model_selection import train_test_split
#X_train_dataset_new,X_out_of_time_test_new, y_train_dataset_new, Y_out_of_time_test_new = train_test_split(X_train_dataset_new,y_train_dataset_new.values, test_size=0.4, random_state=0)

models_report, conf_matrix = run_models(X_train_dataset_new, y_train_dataset_new, X_out_of_time_test_new, Y_out_of_time_test_new, model_type = "Balanced")
models_report
