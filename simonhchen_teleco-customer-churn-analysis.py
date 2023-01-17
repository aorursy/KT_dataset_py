# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


%matplotlib inline        
        
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


raw_data = pd.read_csv("/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv",low_memory=False)
raw_data.head()
raw_data.dtypes.value_counts()
raw_data.isnull().sum()
missing_perc = raw_data.isnull().mean()
missing_perc = missing_perc[missing_perc > 0]

missing_perc
preprocess_df = raw_data.copy()
preprocess_df.shape

# Delete raw data.
# del raw_data
preprocess_df.head()
preprocess_df.columns
y_n_cols = ['Partner', 'Dependents', 'PhoneService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'PaperlessBilling', 'MultipleLines']
yes_no_df = preprocess_df[y_n_cols].copy()
yes_no_df.head()
# Show the most frequently occuring observations in each columns

cols = list(yes_no_df.columns)
for col in cols:
    print("Column Name: " + col)
    print(preprocess_df[col].value_counts().head())
replacement_dict = {'Yes': 1,
                    'No': 0,
                    'No internet service':0,
                    'No phone service':0}

yes_no_df.replace(replacement_dict, inplace=True)
yes_no_df.head()
cols = list(yes_no_df.columns)
for col in cols:
    print("Column Name: " + col)
    print(yes_no_df[col].value_counts().head())
preprocess_df.drop(y_n_cols, axis=1, inplace=True)
print(preprocess_df.shape)
preprocess_df.head()
preprocess_df = pd.concat([preprocess_df, yes_no_df], axis=1, sort=False)
print(preprocess_df.shape)
preprocess_df.head()
preprocess_df['InternetServiceType'] = preprocess_df['InternetService']
preprocess_df[['InternetService', 'InternetServiceType']].head()
IS_replacement_dict = {'Fiber optic': 1,
                       'DSL': 1,
                       'No': 0}
preprocess_df['InternetService'].replace(IS_replacement_dict, inplace=True)
preprocess_df['InternetServiceType'].replace('No', 'None', inplace=True)
preprocess_df['InternetService'].value_counts()
make_dummy = ['gender', 'Contract', 'PaymentMethod', 'InternetServiceType']
make_dummy_df = preprocess_df[make_dummy].copy()
make_dummy_df.head()
cols = list(make_dummy_df.columns)
for col in cols:
    print("Column Name: " + col)
    print(make_dummy_df[col].value_counts().head())
make_dummy_df = pd.get_dummies(make_dummy_df)
make_dummy_df.drop(['gender_Male'], axis=1, inplace=True)
make_dummy_df
preprocess_df.drop(make_dummy, axis=1, inplace=True)
print(preprocess_df.shape)
preprocess_df.head()
preprocess_df = pd.concat([preprocess_df, make_dummy_df], axis=1, sort=False)
print(preprocess_df.shape)
preprocess_df.head()
churn_dict = {'No': 0,
              'Yes': 1}
preprocess_df['Churn'].replace(churn_dict, inplace=True)
preprocess_df.head()
preprocess_df['tenure'].describe()
# Create bins for customer tenure with teleco
bins = [0, 6, 12, 18, 24, 30, 36, 48, 60, 72]
labels = ['tenure_0_to_6', 'tenure_6_to_12', 'tenure_12_to_18', 'tenure_18_to_24', 'tenure_24_to_30',
          'tenure_30_to_36', 'tenure_36_to_48', 'tenure_48_to_60', 'tenure_60_to_72']

# create new feature indicating the customer tenure in months binnned
preprocess_df['tenure_binned'] = pd.cut(preprocess_df['tenure'], bins, labels=labels)

# print by bin count
preprocess_df['tenure_binned'].value_counts()
# create dataframe of number of customer tenure
tenure_number = pd.DataFrame(pd.get_dummies(preprocess_df['tenure_binned']))

# add the tenure_binned dummy variables to the dataframe
preprocess_df = pd.concat([preprocess_df, tenure_number], axis=1, sort=False)

# drop the previously created tenure_binned feature with the get_dummies created
preprocess_df.drop(['tenure_binned'], axis=1, inplace=True)
preprocess_df.head()
preprocess_df.dtypes
preprocess_df.TotalCharges = pd.to_numeric(preprocess_df.TotalCharges, errors='coerce')
process_df = preprocess_df.copy()
# Sample figsize in inches
fig, ax = plt.subplots(figsize=(20,10))

# Imbalanced DataFrame Correlation
corr = process_df.corr()
sns.heatmap(corr, cmap='YlGnBu', annot_kws={'size':30}, ax=ax)
ax.set_title("Correlation Matrix", fontsize=14)
plt.show()
# See correlation of 'Churn' with other variables
plt.figure(figsize=(15,8))
process_df.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')
ax = process_df.Churn.value_counts().plot(kind='bar')
labels = ['No', 'Yes']
ax.set_xticklabels(labels, rotation='horizontal')
ax.set_ylabel('Number of Customers')
contract_m2m = pd.crosstab(process_df['Contract_Month-to-month'], process_df['Churn'])
ax = contract_m2m.plot.bar(alpha=0.7)
ax.set_xlabel('Month to Month Contracts')
labels = ['NO', 'YES']
ax.set_xticklabels(labels, rotation='horizontal')
ax.set_title('Customer Churn for Customers on Month to Month Contracts')
ax.legend(["No", "Yes"], title="Customer Churn")
IST_fiber = pd.crosstab(process_df['InternetServiceType_Fiber optic'], process_df['Churn'])
ax = IST_fiber.plot.bar(alpha=0.7)
ax.set_xlabel('Fiber Optics')
labels = ['NO', 'YES']
ax.set_xticklabels(labels, rotation='horizontal')
ax.set_title('Customer Churn for Customers on Fiber Optics')
ax.legend(["No", "Yes"], title="Customer Churn")
pmt_type_EC = pd.crosstab(process_df['PaymentMethod_Electronic check'], process_df['Churn'])
ax = pmt_type_EC.plot.bar(alpha=0.7)
ax.set_xlabel('Payment Type Electronic Check')
labels = ['NO', 'YES']
ax.set_xticklabels(labels, rotation='horizontal')
ax.set_title('Customer Churn for Customers on Electronic Check Payments')
ax.legend(["No", "Yes"], title="Customer Churn")
process_df['tenure_0_to_6']
## Bin Customer Tenure

customer_tenure = process_df[['tenure_0_to_6', 'tenure_6_to_12', 'tenure_12_to_18', 'tenure_18_to_24',
                              'tenure_24_to_30', 'tenure_30_to_36', 'tenure_36_to_48', 'tenure_48_to_60',
                              'tenure_60_to_72', 'Churn']]

# See correlation of 'Churn' with Customer Tenure
plt.figure(figsize=(15,8))
customer_tenure.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')
two_year_contract = pd.crosstab(process_df['Contract_Two year'], process_df['Churn'])
ax = two_year_contract.plot.bar(alpha=0.7)
ax.set_xlabel('Two Year Contract')
labels = ['NO', 'YES']
ax.set_xticklabels(labels, rotation='horizontal')
ax.set_title('Customer Churn for Customers on Two year Contracts')
ax.legend(["No", "Yes"], title="Customer Churn")
one_year_contract = pd.crosstab(process_df['Contract_One year'], process_df['Churn'])
ax = one_year_contract.plot.bar(alpha=0.7)
ax.set_xlabel('One Year Contract')
labels = ['NO', 'YES']
ax.set_xticklabels(labels, rotation='horizontal')
ax.set_title('Customer Churn for Customers on One Year Contracts')
ax.legend(["No", "Yes"], title="Customer Churn")
predictive_features = ['Contract_One year', 'Contract_Two year','tenure',
                       'PaymentMethod_Electronic check','InternetServiceType_Fiber optic',
                       'Contract_Month-to-month']
model_features = process_df[predictive_features]
model_features = pd.concat([model_features, process_df['Churn']], axis=1, sort=False)
model_features.head()
X = model_features.drop(['Churn'], axis=1)
y = model_features['Churn']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=123)

# Create, train, and fit a logistic regression model
from sklearn.linear_model import LogisticRegression
clf_logistic = LogisticRegression(solver='lbfgs').fit(X_train, np.ravel(y_train))

# Create predictions of probability for loan status using test data
# .predict_proba creates an array of probabilities of default: [[non-defualt, default]]
lr_preds = clf_logistic.predict_proba(X_test)

# # Create dataframes of predictions and true labels
lr_preds_df = pd.DataFrame(lr_preds[:,1][0:], columns = ['lr_pred'])
true_df = y_test

# Concatenate and print the two data frames for comparison
print(pd.concat([true_df.reset_index(drop = True), lr_preds_df], axis = 1))
# Reassign loan status based on the threshold and print the predictions
lr_preds_df['lr_pred_churn_status_50'] = lr_preds_df['lr_pred'].apply(lambda x: 1 if x > 0.50 else 0)
print("Churn: Yes / Not Churn predictions at 50% Threshhold: ")
print(lr_preds_df['lr_pred_churn_status_50'].value_counts())

# Print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion Matrix at 50% Threshhold: ")
print(confusion_matrix(y_test, lr_preds_df['lr_pred_churn_status_50']))

# Print the classification report
from sklearn.metrics import classification_report
target_names = ['No Churn', 'Yes Churn']
print(classification_report(y_test, lr_preds_df['lr_pred_churn_status_50'], target_names=target_names))
# Print the accuracy score the model
print(clf_logistic.score(X_test, y_test))

# Plot the ROC curve of the probabilities of default
from sklearn.metrics import roc_curve

lr_pred_churn = lr_preds[:, 1]
fallout, sensitivity, thresholds = roc_curve(y_test, lr_pred_churn)
plt.plot(fallout, sensitivity, color = 'darkorange')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title("ROC Chart for LR on Churn")
plt.xlabel("Fall-out")
plt.ylabel("Sensitivity")
plt.show()

# Compute the AUC and store it in a variable
from sklearn.metrics import roc_auc_score

lr_auc = roc_auc_score(y_test, lr_pred_churn)
# Train a model
import xgboost as xgb
clf_gbt = xgb.XGBClassifier().fit(X_train, np.ravel(y_train))

# Predict with a model
# .predict_proba creates an array of probabilities of default: [['No Churn', 'Churn']]
gbt_preds = clf_gbt.predict_proba(X_test)

# Create dataframes of first five predictions, and first five true labels
gbt_preds_df = pd.DataFrame(gbt_preds[:,1][0:], columns = ['gbt_pred_churn'])
true_df = y_test

# Concatenate and print the two data frames for comparison
print(pd.concat([true_df.reset_index(drop = True), gbt_preds_df], axis = 1))
# Reassign loan status based on the threshold and print the predictions
gbt_preds_df['gbt_pred_churn_status_50'] = gbt_preds_df['gbt_pred_churn'].apply(lambda x: 1 if x > 0.50 else 0)
print("No Churn / Churn at 50% Threshhold: ")
print(gbt_preds_df['gbt_pred_churn_status_50'].value_counts())

# Print the confusion matrix
print("Confusion Matrix at 50% Threshhold: ")
print(confusion_matrix(y_test, gbt_preds_df['gbt_pred_churn_status_50']))

# Print the classification report
target_names = ['No Churn', 'Churn']
print(classification_report(y_test, gbt_preds_df['gbt_pred_churn_status_50'], target_names=target_names))
# Print the accuracy score the model
print(clf_gbt.score(X_test, y_test))

# Plot the ROC curve of the probabilities of default
from sklearn.metrics import roc_curve

xgb_pred_churn = gbt_preds[:, 1]
fallout, sensitivity, thresholds = roc_curve(y_test, xgb_pred_churn)
plt.plot(fallout, sensitivity, color = 'darkorange')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title("ROC Chart for XGB on Churn")
plt.xlabel("Fall-out")
plt.ylabel("Sensitivity")
plt.show()

# Compute the AUC and store it in a variable
from sklearn.metrics import roc_auc_score

xgb_auc = roc_auc_score(y_test, xgb_pred_churn)
!pip install lifelines
from lifelines import KaplanMeierFitter
process_df.head()
kmf1 = KaplanMeierFitter() ## instantiate the class to create an object

tenure = process_df['tenure']
event = process_df['Churn']

## Two Cohorts are compared. Cohort 1. Streaming TV Not Subscribed by users, and Cohort  2. Streaming TV subscribed by the users.
streaming_cohorts = process_df['StreamingTV']    
no_stream_TV = (streaming_cohorts == 0)      ## Cohort WITHOUT streaming TV, having the pandas series  for the 1st cohort
yes_stream_TV = (streaming_cohorts == 1)     ## Cohort WITH streaming TV, having the pandas series  for the 2nd cohort


## fit the model for 1st cohort
kmf1.fit(tenure[no_stream_TV], event[no_stream_TV], label='Not Subscribed StreamingTV')
a1 = kmf1.plot()

## fit the model for 2nd cohort
kmf1.fit(tenure[yes_stream_TV], event[yes_stream_TV], label='Subscribed StreamingTV')
kmf1.plot(ax=a1)
tenure[no_stream_TV]
survival_pred_df = process_df[['tenure', 'Churn', 'MonthlyCharges', 'SeniorCitizen', 'InternetService',
                               'Partner', 'Dependents', 'PhoneService', 'StreamingTV']]
survival_pred_df.head()
from lifelines import CoxPHFitter

cph = CoxPHFitter()

cph.fit(survival_pred_df, 'tenure', event_col='Churn')

cph.print_summary()
plt.figure(figsize=(16, 8))
cph.plot()
plt.show()
## We want to see the Survival curve at the customer level. Therefore, we have selected 6 customers (rows 5 till 9).

tr_rows = survival_pred_df.iloc[5:10, 2:]
tr_rows
## Lets predict the survival curve for the selected customers. 
## Customers can be identified with the help of the number mentioned against each curve.
sns.set(rc={'figure.figsize':(18,10)})
cph.predict_survival_function(tr_rows).plot(figsize=(16,8))