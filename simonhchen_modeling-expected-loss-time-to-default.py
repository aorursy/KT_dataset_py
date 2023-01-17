# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
raw_data = pd.read_csv("/kaggle/input/lending-club-loan-data/loan.csv",low_memory=False)
raw_data.head()
preprocess_df = raw_data[['emp_length', 'loan_status', 'home_ownership', 'issue_d',
                          'earliest_cr_line', 'purpose', 'term', 'annual_inc', 'dti',
                          'loan_amnt', 'int_rate', 'pub_rec_bankruptcies',]].copy()
preprocess_df = raw_data[['emp_length', 'loan_status', 'home_ownership', 'issue_d',
                          'earliest_cr_line', 'purpose', 'term', 'annual_inc', 'dti',
                          'loan_amnt', 'int_rate', 'pub_rec_bankruptcies',]].copy()
# Create a list of columns that are NOT numeric values
not_numeric_cols = ['emp_length', 'loan_status', 'home_ownership', 'issue_d',
                    'earliest_cr_line', 'purpose', 'term']

# Create list of columns that ARE numeric values and print
numeric_cols = [col for col in preprocess_df.columns if col not in not_numeric_cols]
print(numeric_cols)

# Convert numeric cols into numeric data types
preprocess_df[numeric_cols] = preprocess_df[numeric_cols].apply(pd.to_numeric)
# Create list of datetime columns
datetime_cols = ['earliest_cr_line', 'issue_d']

# Convert to datetime
preprocess_df[datetime_cols] = preprocess_df[datetime_cols].apply(pd.to_datetime)
preprocess_df.isnull().sum()
character_df = preprocess_df[['pub_rec_bankruptcies', 'earliest_cr_line', 'issue_d']].copy()
print(character_df.head())
print(character_df.describe())
print(character_df.dtypes)
# fill the missing value for earliest_cr_line with most frequently occuring
character_df['earliest_cr_line'].fillna(character_df['earliest_cr_line'].value_counts().index[0], inplace=True)
# count months between now and 'earliest_cr_line'
character_df['credit_hist_in_months'] = ((character_df['issue_d'] - character_df['earliest_cr_line'])/np.timedelta64(1, 'M'))
character_df['credit_hist_in_months'] = character_df['credit_hist_in_months'].astype(int)

character_df.head()
# Create a new binary feature of whether or not there is a bankruptcy on file in customers credit history
character_df['cb_person_bk_on_file_Y'] = character_df['pub_rec_bankruptcies'].apply(lambda x: 1 if x >= 1 else 0)
character_df.head()
# drop the old features from the character_df
character_df.drop(['pub_rec_bankruptcies', 'earliest_cr_line', 'issue_d'], axis=1, inplace=True)
character_df.head()
capacity_df = preprocess_df[['annual_inc', 'dti']].copy()
capacity_df.head()
# fill missing values for annual income with the mean
capacity_df['annual_inc'] = capacity_df['annual_inc'].fillna(capacity_df['annual_inc'].mean())

# fill missing values for dti with the mean
capacity_df['dti'] = capacity_df['dti'].fillna(capacity_df['dti'].mean())
# describe the capacity/cash flow proxy features
print(capacity_df[['annual_inc', 'dti']].describe())
conditions_df = preprocess_df[['loan_amnt', 'int_rate', 'term']].copy()
conditions_df.term.value_counts()
# Convert values of term to 0, 1 where 0 = 36 months and 1 = 60 months
conditions_df['term'] = conditions_df['term'].replace({' 36 months': '0',
                                                       ' 60 months': '1'})

# convert term into an integer data type 
conditions_df['term'] = conditions_df['term'].astype(int)

# Rename term column
conditions_df = conditions_df.rename(columns={'term': 'term_60'})

print(conditions_df.term_60.value_counts())
conditions_df.head()
collateral_df = preprocess_df[['home_ownership']].copy()
collateral_df.head()
# create emp_length dummy data frame
home_ownership = pd.DataFrame(pd.get_dummies(collateral_df['home_ownership'], prefix='home_ownership'))

# join the loan_amnt dummy dataframe to conditions_df
collateral_df = pd.concat([collateral_df, home_ownership], axis=1, sort=False)

# drop original emp_length feature
collateral_df.drop(['home_ownership'], axis=1, inplace=True)

collateral_df.head()
loan_status_df = preprocess_df[['loan_status']].copy()
loan_status_df.loan_status.value_counts()
loan_status_df['loan_status'] = loan_status_df['loan_status'].replace({'Fully Paid': 0, 'Current': 0, 'Charged Off': 1,
                                                                       'Late (31-120 days)': 1, 'In Grace Period': 1,
                                                                       'Late (16-30 days)':1,
                                                                       'Does not meet the credit policy. Status:Fully Paid': 0,
                                                                       'Does not meet the credit policy. Status:Charged Off': 1,
                                                                       'Default':1})
loan_status_df.loan_status.value_counts()
# Concatenate all the processed dataframes into a single one
processed_df = pd.concat([character_df,
                          capacity_df,
                          conditions_df,
                          collateral_df,
                          loan_status_df,
                          raw_data['grade']], axis=1, sort=False)
processed_df.head()
print(processed_df.loan_status.value_counts())
ax = processed_df.loan_status.value_counts().plot(kind='bar')
labels = ['Non-Default', 'Default']
ax.set_xticklabels(labels, rotation='horizontal')
ax.set_ylabel('Number of Loans')
grade_default = pd.crosstab(processed_df['grade'], processed_df['loan_status'])

fig, ax = plt.subplots()

grade_default.plot.bar(legend=True, alpha=0.7, ax=ax)
plt.title("Non-Default vs Default by Grade")
ax.legend(["Non-Default", "Default"])
# Sample figsize in inches
fig, ax = plt.subplots(figsize=(20,10))

# Imbalanced DataFrame Correlation
corr = processed_df.corr()
sns.heatmap(corr, cmap='YlGnBu', annot_kws={'size':30}, ax=ax)
ax.set_title("Imbalanced Correlation Matrix", fontsize=14)
plt.show()
# Shuffle the Dataset.
random_data = processed_df.sample(frac=1,random_state=4)

# Put all the fraud class in a separate dataset.
default_df = random_data.loc[random_data['loan_status'] == 1]

#Randomly select 297033 observations from the non-fraud (majority class)
non_default_df = random_data.loc[random_data['loan_status'] == 0].sample(n=297033, random_state=42)
# Concatenate both dataframes again
US_df = pd.concat([default_df, non_default_df])

#plot the dataset after the undersampling
plt.figure(figsize=(8, 8))
sns.countplot('loan_status', data=US_df)
plt.title('Undersampled Balanced Loan Status')
plt.xticks(ticks=(0,1), labels=('Non-Default', 'Default'))
plt.show()
# Create X and y using undersampled dataframe
X = US_df.drop(['loan_status', 'grade'], axis=1)
y = US_df['loan_status']

X_train_US, X_test_US, y_train_US, y_test_US = train_test_split(X, y, test_size=.4, random_state=123)
# Create, train, and fit a logistic regression model
from sklearn.linear_model import LogisticRegression
clf_logistic = LogisticRegression(solver='lbfgs').fit(X_train_US, np.ravel(y_train_US))

# Create predictions of probability for loan status using test data
# .predict_proba creates an array of probabilities of default: [[non-defualt, default]]
lr_preds = clf_logistic.predict_proba(X_test_US)

# # Create dataframes of predictions and true labels
lr_preds_df = pd.DataFrame(lr_preds[:,1][0:], columns = ['lr_pred_PD'])
true_df = y_test_US

# Concatenate and print the two data frames for comparison
print(pd.concat([true_df.reset_index(drop = True), lr_preds_df], axis = 1))
# Reassign loan status based on the threshold and print the predictions
lr_preds_df['lr_pred_loan_status_60'] = lr_preds_df['lr_pred_PD'].apply(lambda x: 1 if x > 0.60 else 0)
print("Non-Default / Default predictions at 60% Threshhold: ")
print(lr_preds_df['lr_pred_loan_status_60'].value_counts())

# Print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion Matrix at 60% Threshhold: ")
print(confusion_matrix(y_test_US, lr_preds_df['lr_pred_loan_status_60']))

# Print the classification report
from sklearn.metrics import classification_report
target_names = ['Non-Default', 'Default']
print(classification_report(y_test_US, lr_preds_df['lr_pred_loan_status_60'], target_names=target_names))
# Reassign loan status based on the threshold and print the predictions
lr_preds_df['lr_pred_loan_status_50'] = lr_preds_df['lr_pred_PD'].apply(lambda x: 1 if x > 0.50 else 0)
print("Non-Default / Default t predictions at 50% Threshhold: ")
print(lr_preds_df['lr_pred_loan_status_50'].value_counts())

# Print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion Matrix at 50% Threshhold: ")
print(confusion_matrix(y_test_US, lr_preds_df['lr_pred_loan_status_50']))

# Print the classification report
from sklearn.metrics import classification_report
target_names = ['Non-Default', 'Default']
print(classification_report(y_test_US, lr_preds_df['lr_pred_loan_status_50'], target_names=target_names))
# Print the accuracy score the model
print(clf_logistic.score(X_test_US, y_test_US))

# Plot the ROC curve of the probabilities of default
from sklearn.metrics import roc_curve

lr_prob_default = lr_preds[:, 1]
fallout, sensitivity, thresholds = roc_curve(y_test_US, lr_prob_default)
plt.plot(fallout, sensitivity, color = 'darkorange')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title("ROC Chart for LR on PD")
plt.xlabel("Fall-out")
plt.ylabel("Sensitivity")
plt.show()

# Compute the AUC and store it in a variable
from sklearn.metrics import roc_auc_score

lr_auc = roc_auc_score(y_test_US, lr_prob_default)
# Create X and y using processed_df

X = processed_df.drop(['loan_status', 'grade'], axis=1)
y = processed_df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=123)
from imblearn.over_sampling import SMOTE

# Resample the minority class. You can change the strategy to 'auto' if you are not sure.
sm = SMOTE(sampling_strategy='minority', random_state=7)

# Fit the model to generate the data.
oversampled_trainX, oversampled_trainY = sm.fit_sample(X_train, y_train)
oversampled_train = pd.concat([pd.DataFrame(oversampled_trainY), pd.DataFrame(oversampled_trainX)], axis=1)
oversampled_train.columns = processed_df.drop(['grade'], axis=1).columns
# Sample figsize in inches
fig, ax = plt.subplots(figsize=(20,10))

# Imbalanced DataFrame Correlation
corr = oversampled_train.corr()
sns.heatmap(corr, cmap='YlGnBu', annot_kws={'size':30}, ax=ax)
ax.set_title("Imbalanced Correlation Matrix after oversampling", fontsize=14)
plt.show()
# Create, train, and fit a logistic regression model
from sklearn.linear_model import LogisticRegression
clf_logistic = LogisticRegression(solver='lbfgs').fit(oversampled_trainX, np.ravel(oversampled_trainY))

# Create predictions of probability for loan status using test data
# .predict_proba creates an array of probabilities of default: [[non-defualt, default]]
lr_preds = clf_logistic.predict_proba(X_test)

# # Create dataframes of predictions and true labels
lr_preds_df = pd.DataFrame(lr_preds[:,1][0:], columns = ['lr_pred_PD'])
true_df = y_test

# Concatenate and print the two data frames for comparison
print(pd.concat([true_df.reset_index(drop = True), lr_preds_df], axis = 1))
# Reassign loan status based on the threshold and print the predictions
lr_preds_df['lr_pred_loan_status_60'] = lr_preds_df['lr_pred_PD'].apply(lambda x: 1 if x > 0.60 else 0)
print("Non-Default / Default predictions at 60% Threshhold: ")
print(lr_preds_df['lr_pred_loan_status_60'].value_counts())

# Print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion Matrix at 60% Threshhold: ")
print(confusion_matrix(y_test, lr_preds_df['lr_pred_loan_status_60']))

# Print the classification report
from sklearn.metrics import classification_report
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, lr_preds_df['lr_pred_loan_status_60'], target_names=target_names))
# Reassign loan status based on the threshold and print the predictions
lr_preds_df['lr_pred_loan_status_50'] = lr_preds_df['lr_pred_PD'].apply(lambda x: 1 if x > 0.50 else 0)
print("Non-Default / Default t predictions at 50% Threshhold: ")
print(lr_preds_df['lr_pred_loan_status_50'].value_counts())

# Print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion Matrix at 50% Threshhold: ")
print(confusion_matrix(y_test, lr_preds_df['lr_pred_loan_status_50']))

# Print the classification report
from sklearn.metrics import classification_report
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, lr_preds_df['lr_pred_loan_status_50'], target_names=target_names))
# Print the accuracy score the model
print(clf_logistic.score(X_test, y_test))

# Plot the ROC curve of the probabilities of default
from sklearn.metrics import roc_curve

lr_prob_default = lr_preds[:, 1]
fallout, sensitivity, thresholds = roc_curve(y_test, lr_prob_default)
plt.plot(fallout, sensitivity, color = 'darkorange')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title("ROC Chart for LR on PD")
plt.xlabel("Fall-out")
plt.ylabel("Sensitivity")
plt.show()

# Compute the AUC and store it in a variable
from sklearn.metrics import roc_auc_score

lr_auc = roc_auc_score(y_test, lr_prob_default)
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Create an object of the classifier and fit oversampled training data to the object
bbc = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                sampling_strategy='auto',
                                replacement=False,
                                random_state=0).fit(oversampled_trainX, np.ravel(oversampled_trainY))

# Create predictions of probability for loan status using test data
bbc_preds = bbc.predict_proba(X_test)
# Create dataframes of predictions and true labels
bbc_preds_df = pd.DataFrame(bbc_preds[:,1][0:], columns = ['bbc_pred_PD'])
true_df = y_test

# Concatenate and print the two data frames for comparison
print(pd.concat([true_df.reset_index(drop = True), bbc_preds_df], axis = 1))
# Reassign loan status based on the threshold and print the predictions
bbc_preds_df['bbc_pred_loan_status_60'] = bbc_preds_df['bbc_pred_PD'].apply(lambda x: 1 if x > 0.60 else 0)
print("Non-Default / Default predictions at 60% Threshhold: ")
print(bbc_preds_df['bbc_pred_loan_status_60'].value_counts())

# Print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion Matrix at 60% Threshhold: ")
print(confusion_matrix(y_test, bbc_preds_df['bbc_pred_loan_status_60']))

# Print the classification report
from sklearn.metrics import classification_report
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, bbc_preds_df['bbc_pred_loan_status_60'], target_names=target_names))

# Reassign loan status based on the threshold and print the predictions
bbc_preds_df['bbc_pred_loan_status_50'] = bbc_preds_df['bbc_pred_PD'].apply(lambda x: 1 if x > 0.50 else 0)
print("Non-Default / Default predictions at 50% Threshhold: ")
print(bbc_preds_df['bbc_pred_loan_status_50'].value_counts())

# Print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion Matrix at 50% Threshhold: ")
print(confusion_matrix(y_test, bbc_preds_df['bbc_pred_loan_status_50']))

# Print the classification report
from sklearn.metrics import classification_report
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, bbc_preds_df['bbc_pred_loan_status_50'], target_names=target_names))

# Print the accuracy score the model
print(bbc.score(X_test, y_test))

# Plot the ROC curve of the probabilities of default
from sklearn.metrics import roc_curve

bbc_prob_default = bbc_preds[:, 1]
fallout, sensitivity, thresholds = roc_curve(y_test, bbc_prob_default)
plt.plot(fallout, sensitivity, color = 'darkorange')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title("ROC Chart for BBC on PD")
plt.xlabel("Fall-out")
plt.ylabel("Sensitivity")
plt.show()

# Compute the AUC and store it in a variable
from sklearn.metrics import roc_auc_score

bbc_auc = roc_auc_score(y_test, bbc_prob_default)
# Train a model
import xgboost as xgb
clf_gbt = xgb.XGBClassifier().fit(oversampled_trainX, np.ravel(oversampled_trainY))

# Predict with a model
# .predict_proba creates an array of probabilities of default: [[non-default, default]]
gbt_preds = clf_gbt.predict_proba(X_test)

# Create dataframes of first five predictions, and first five true labels
gbt_preds_df = pd.DataFrame(gbt_preds[:,1][0:], columns = ['gbt_pred_PD'])
true_df = y_test

# Concatenate and print the two data frames for comparison
print(pd.concat([true_df.reset_index(drop = True), gbt_preds_df], axis = 1))
# Reassign loan status based on the threshold and print the predictions
gbt_preds_df['gbt_pred_loan_status_60'] = gbt_preds_df['gbt_pred_PD'].apply(lambda x: 1 if x > 0.60 else 0)
print("Non-Default / Default  predictions at 60% Threshhold: ")
print(gbt_preds_df['gbt_pred_loan_status_60'].value_counts())

# Print the confusion matrix
print("Confusion Matrix at 60% Threshhold: ")
print(confusion_matrix(y_test, gbt_preds_df['gbt_pred_loan_status_60']))

# Print the classification report
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, gbt_preds_df['gbt_pred_loan_status_60'], target_names=target_names))
# Reassign loan status based on the threshold and print the predictions
gbt_preds_df['gbt_pred_loan_status_50'] = gbt_preds_df['gbt_pred_PD'].apply(lambda x: 1 if x > 0.50 else 0)
print("Non-Default / Default predictions at 50% Threshhold: ")
print(gbt_preds_df['gbt_pred_loan_status_50'].value_counts())

# Print the confusion matrix
print("Confusion Matrix at 50% Threshhold: ")
print(confusion_matrix(y_test, gbt_preds_df['gbt_pred_loan_status_50']))

# Print the classification report
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, gbt_preds_df['gbt_pred_loan_status_50'], target_names=target_names))
# Print the accuracy score the model
print(clf_gbt.score(X_test, y_test))

# Plot the ROC curve of the probabilities of default
from sklearn.metrics import roc_curve

xgb_prob_default = gbt_preds[:, 1]
fallout, sensitivity, thresholds = roc_curve(y_test, xgb_prob_default)
plt.plot(fallout, sensitivity, color = 'darkorange')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title("ROC Chart for XGB on PD")
plt.xlabel("Fall-out")
plt.ylabel("Sensitivity")
plt.show()

# Compute the AUC and store it in a variable
from sklearn.metrics import roc_auc_score

xgb_auc = roc_auc_score(y_test, xgb_prob_default)

# Creating a portfolio datafram
portfolio_5c = pd.DataFrame(gbt_preds[:,1][0:], columns = ['gbt_prob_default'])
portfolio_5c.index = X_test.index
portfolio_5c['bbc_prob_default'] = bbc_preds[:,1][0:]
portfolio_5c['lr_prob_default'] = lr_preds[:,1][0:]
portfolio_5c['lgd'] = 1 # Assumes that given a default, entire loan is a loss with no recoverable amounts
portfolio_5c['loan_amnt'] = X_test.loan_amnt
portfolio_5c.head()
# Create expected loss columns for each model using the formula
portfolio_5c['gbt_expected_loss'] = portfolio_5c['gbt_prob_default'] * portfolio_5c['lgd'] * portfolio_5c['loan_amnt']
portfolio_5c['bbc_expected_loss'] = portfolio_5c['bbc_prob_default'] * portfolio_5c['lgd'] * portfolio_5c['loan_amnt']
portfolio_5c['lr_expected_loss'] = portfolio_5c['lr_prob_default'] * portfolio_5c['lgd'] * portfolio_5c['loan_amnt']

# Print the total portfolio size
print('Portfolio size: $' + "{:,.2f}".format(np.sum(portfolio_5c['loan_amnt'])))

# Print the sum of the expected loss for bbc
print('BBC expected loss: $' + "{:,.2f}".format(np.sum(portfolio_5c['bbc_expected_loss'])))

# Print the sum of the expected loss for gbt
print('GBT expected loss: $' + "{:,.2f}".format(np.sum(portfolio_5c['gbt_expected_loss'])))

# Print the sum of the expected loss for lr 
print('LR expected loss: $' + "{:,.2f}".format(np.sum(portfolio_5c['lr_expected_loss'])))

# Print portfolio first five rows
portfolio_5c.head()
preprocess_TTD_df = raw_data[['emp_length', 'chargeoff_within_12_mths',
                              'mths_since_last_delinq', 'mths_since_last_record',
                              'mths_since_last_major_derog', 'mths_since_recent_revol_delinq']].copy()
preprocess_TTD_df.head()
preprocess_TTD_df.isnull().sum()
preprocess_TTD_df.emp_length.value_counts()
# fill in missing values with the most frequently occuring employment length
preprocess_TTD_df['emp_length'].fillna(preprocess_TTD_df['emp_length'].value_counts().index[0], inplace=True)

# create a dataframe of employment length for customers in dataset
emp_length_df = pd.DataFrame(pd.get_dummies(preprocess_TTD_df.emp_length, prefix='emp_length'))
emp_length_df.head()

# concat the emp_length dataframe with preprocess_TTD_df, dropping original columns
preprocess_TTD_df = pd.concat([preprocess_TTD_df, emp_length_df], axis=1, sort=False)
preprocess_TTD_df.drop(['emp_length'], axis=1, inplace=True)
preprocess_TTD_df.head()
# fill in missing values with the most frequently occuring chargeoff
preprocess_TTD_df['chargeoff_within_12_mths'].fillna(preprocess_TTD_df['chargeoff_within_12_mths'].value_counts().index[0], inplace=True)
preprocess_TTD_df.mths_since_last_delinq
print(preprocess_TTD_df.mths_since_last_delinq.describe())
preprocess_TTD_df.mths_since_last_record 
print(preprocess_TTD_df.mths_since_last_record .describe())
preprocess_TTD_df.mths_since_last_major_derog
print(preprocess_TTD_df.mths_since_last_major_derog.describe())
preprocess_TTD_df.mths_since_recent_revol_delinq
print(preprocess_TTD_df.mths_since_recent_revol_delinq.describe())
preprocess_TTD_df.fillna(0, inplace=True)
preprocess_TTD_df
preprocess_TTD_df.isnull().sum()
process_TTD_df = preprocess_TTD_df.copy()
process_TTD_df
# create a dataframe of employment length for customers in dataset
grade_df = pd.DataFrame(pd.get_dummies(processed_df.grade, prefix='grade'))
grade_df.head()

# concat the emp_length dataframe with preprocess_TTD_df, dropping original columns
processed_df = pd.concat([processed_df, grade_df], axis=1, sort=False)
processed_df.drop(['grade'], axis=1, inplace=True)
processed_df.head()
final_processed = processed_df.join(process_TTD_df)
final_processed.columns
# Go to settings tab and turn internet on to downloand lifelines and instantiate KMF 

!pip install lifelines
from lifelines import KaplanMeierFitter
raw_data.grade.value_counts()
kmf1 = KaplanMeierFitter() ## instantiate the class to create an object

tenure = final_processed['credit_hist_in_months']
event = final_processed['loan_status']
## Two Cohorts are compared. Cohort 1: Not Grade A Customers, and Cohort  2: Grade A CUstomers
grade_A = final_processed['grade_A']    
not_grade_A = (grade_A == 0)      ## Cohort Not Grade A Customers, having the pandas series  for the 1st cohort
yes_grade_A = (grade_A == 1)     ## Cohort Grade A Customers, having the pandas series  for the 2nd cohort


## fit the model for 1st cohort
kmf1.fit(tenure[not_grade_A], event[not_grade_A], label='Not Grade A')
a1 = kmf1.plot()

## fit the model for 2nd cohort
kmf1.fit(tenure[yes_grade_A], event[yes_grade_A], label='Grade A')
kmf1.plot(ax=a1)
# a1.set_xlim(0, 100)
type(a1)
## Two Cohorts are compared. Cohort 1: Not Grade B Customers, and Cohort  2: Grade B CUstomers
grade_B = final_processed['grade_B']    
not_grade_B = (grade_B == 0)      ## Cohort Not Grade B Customers, having the pandas series  for the 1st cohort
yes_grade_B = (grade_B == 1)     ## Cohort Grade B Customers, having the pandas series  for the 2nd cohort


## fit the model for 1st cohort
kmf1.fit(tenure[not_grade_B], event[not_grade_B], label='Not Grade B')
a1 = kmf1.plot()

## fit the model for 2nd cohort
kmf1.fit(tenure[yes_grade_B], event[yes_grade_B], label='Grade B')
kmf1.plot(ax=a1)
## Two Cohorts are compared. Cohort 1: Not Grade C Customers, and Cohort  2: Grade C CUstomers
grade_C = final_processed['grade_C']    
not_grade_C = (grade_C == 0)      ## Cohort Not Grade C Customers, having the pandas series  for the 1st cohort
yes_grade_C = (grade_C == 1)     ## Cohort Grade C Customers, having the pandas series  for the 2nd cohort


## fit the model for 1st cohort
kmf1.fit(tenure[not_grade_C], event[not_grade_C], label='Not Grade C')
a1 = kmf1.plot()

## fit the model for 2nd cohort
kmf1.fit(tenure[yes_grade_C], event[yes_grade_C], label='Grade C')
kmf1.plot(ax=a1)
## Two Cohorts are compared. Cohort 1: Not Grade D Customers, and Cohort  2: Grade D CUstomers
grade_D = final_processed['grade_D']    
not_grade_D = (grade_D == 0)      ## Cohort Not Grade D Customers, having the pandas series  for the 1st cohort
yes_grade_D = (grade_D == 1)     ## Cohort Grade D Customers, having the pandas series  for the 2nd cohort


## fit the model for 1st cohort
kmf1.fit(tenure[not_grade_D], event[not_grade_D], label='Not Grade D')
a1 = kmf1.plot()

## fit the model for 2nd cohort
kmf1.fit(tenure[yes_grade_D], event[yes_grade_D], label='Grade D')
kmf1.plot(ax=a1)
## Two Cohorts are compared. Cohort 1: Not Grade E Customers, and Cohort  2: Grade E CUstomers
grade_E = final_processed['grade_E']    
not_grade_E = (grade_E == 0)      ## Cohort Not Grade E Customers, having the pandas series  for the 1st cohort
yes_grade_E = (grade_E == 1)     ## Cohort Grade E Customers, having the pandas series  for the 2nd cohort


## fit the model for 1st cohort
kmf1.fit(tenure[not_grade_E], event[not_grade_E], label='Not Grade D')
a1 = kmf1.plot()

## fit the model for 2nd cohort
kmf1.fit(tenure[yes_grade_E], event[yes_grade_E], label='Grade D')
kmf1.plot(ax=a1)
## Two Cohorts are compared. Cohort 1: Not Grade F Customers, and Cohort  2: Grade F CUstomers
grade_F = final_processed['grade_F']    
not_grade_F = (grade_F == 0)      ## Cohort Not Grade F Customers, having the pandas series  for the 1st cohort
yes_grade_F = (grade_F == 1)     ## Cohort Grade F Customers, having the pandas series  for the 2nd cohort


## fit the model for 1st cohort
kmf1.fit(tenure[not_grade_F], event[not_grade_F], label='Not Grade F')
a1 = kmf1.plot()

## fit the model for 2nd cohort
kmf1.fit(tenure[yes_grade_F], event[yes_grade_F], label='Grade F')
kmf1.plot(ax=a1)
## Two Cohorts are compared. Cohort 1: Not Grade G Customers, and Cohort  2: Grade G CUstomers
grade_G = final_processed['grade_G']    
not_grade_G = (grade_G == 0)      ## Cohort Not Grade G Customers, having the pandas series  for the 1st cohort
yes_grade_G = (grade_G == 1)     ## Cohort Grade G Customers, having the pandas series  for the 2nd cohort


## fit the model for 1st cohort
kmf1.fit(tenure[not_grade_G], event[not_grade_G], label='Not Grade G')
a1 = kmf1.plot()

## fit the model for 2nd cohort
kmf1.fit(tenure[yes_grade_G], event[yes_grade_G], label='Grade G')
kmf1.plot(ax=a1)