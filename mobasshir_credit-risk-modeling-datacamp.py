import operator as opt

import numpy as np 

import pandas as pd 

import os

import gc

from contextlib import contextmanager

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import matplotlib

warnings.filterwarnings('ignore')
## DONE: Create files dictionary for any file in the input directory

files = {}

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        files[filename] = os.path.join(dirname, filename)

        print(os.path.join(dirname, filename))
## DONE: Loading files in pandas dataframe [1.1]

def load_file_into_dataframe(file_name):

    df = pd.read_csv(files[file_name])

    return df
cr_loan = load_file_into_dataframe('cr_loan2.csv')

cr_loan_clean = load_file_into_dataframe('cr_loan_nout_nmiss.csv')

cr_loan_prep = load_file_into_dataframe('cr_loan_w2.csv')
### Explore the credit data



# Check the structure of the data

print(cr_loan.dtypes)



# Check the first five rows of the data

print(cr_loan.head())



# Look at the distribution of loan amounts with a histogram

n, bins, patches = plt.hist(x=cr_loan['loan_amnt'], bins='auto', color='blue',alpha=0.7, rwidth=0.85)

plt.xlabel("Loan Amount")

plt.show()



print("There are 32 000 rows of data so the scatter plot may take a little while to plot.")



# Plot a scatter plot of income against age

plt.scatter(cr_loan['person_income'], cr_loan['person_age'],c='blue', alpha=0.5)

plt.xlabel('Personal Income')

plt.ylabel('Persone Age')

plt.show()
### Crosstab and pivot tables



# Create a cross table of the loan intent and loan status

print(pd.crosstab(cr_loan['loan_intent'], cr_loan['loan_status'], margins = True))



# Create a cross table of home ownership, loan status, and grade

print(pd.crosstab(cr_loan['person_home_ownership'],[cr_loan['loan_status'],cr_loan['loan_grade']]))



# Create a cross table of home ownership, loan status, and average percent income

print(pd.crosstab(cr_loan['person_home_ownership'], cr_loan['loan_status'],

              values=cr_loan['loan_percent_income'], aggfunc='mean'))



# Create a box plot of percentage income by loan status

cr_loan.boxplot(column = ['loan_percent_income'], by = 'loan_status')

plt.title('Average Percent Income by Loan Status')

plt.suptitle('')

plt.show()
### Finding outliers with cross tables



# Create the cross table for loan status, home ownership, and the max employment length

print(pd.crosstab(cr_loan['loan_status'],cr_loan['person_home_ownership'],

        values=cr_loan['person_emp_length'], aggfunc='max'))



# Create the cross table for loan status, home ownership, and the max employment length

print(pd.crosstab(cr_loan['loan_status'],cr_loan['person_home_ownership'],

                  values=cr_loan['person_emp_length'], aggfunc='max'))



# Create an array of indices where employment length is greater than 60

indices = cr_loan[cr_loan['person_emp_length'] > 60].index



# Create the cross table for loan status, home ownership, and the max employment length

print(pd.crosstab(cr_loan['loan_status'],cr_loan['person_home_ownership'],

                  values=cr_loan['person_emp_length'], aggfunc='max'))



# Create an array of indices where employment length is greater than 60

indices = cr_loan[cr_loan['person_emp_length'] > 60].index



# Drop the records from the data based on the indices and create a new dataframe

cr_loan_new = cr_loan.drop(indices)



# Create the cross table for loan status, home ownership, and the max employment length

print(pd.crosstab(cr_loan['loan_status'],cr_loan['person_home_ownership'],

                  values=cr_loan['person_emp_length'], aggfunc='max'))



# Create an array of indices where employment length is greater than 60

indices = cr_loan[cr_loan['person_emp_length'] > 60].index



# Drop the records from the data based on the indices and create a new dataframe

cr_loan_new = cr_loan.drop(indices)



# Create the cross table from earlier and include minimum employment length

print(pd.crosstab(cr_loan_new['loan_status'],cr_loan_new['person_home_ownership'],

            values=cr_loan_new['person_emp_length'], aggfunc=['min','max']))
### Visualizing credit outliers



# Create the scatter plot for age and amount

plt.scatter(cr_loan['person_age'], cr_loan['loan_amnt'], c='blue', alpha=0.5)

plt.xlabel("Person Age")

plt.ylabel("Loan Amount")

plt.show()



# Use Pandas to drop the record from the data frame and create a new one

cr_loan_new = cr_loan.drop(cr_loan[cr_loan['person_age'] > 100].index)



# Create a scatter plot of age and interest rate

colors = ["blue","red"]

plt.scatter(cr_loan_new['person_age'], cr_loan_new['loan_int_rate'],

            c = cr_loan_new['loan_status'],

            cmap = matplotlib.colors.ListedColormap(colors),

            alpha=0.5)

plt.xlabel("Person Age")

plt.ylabel("Loan Interest Rate")

plt.show()
### Replacing missing credit data



# Print a null value column array

print(cr_loan.columns[cr_loan.isnull().any()])



# Print the top five rows with nulls for employment length

print(cr_loan[cr_loan['person_emp_length'].isnull()].head())



# Impute the null values with the median value for all employment lengths

cr_loan['person_emp_length'].fillna((cr_loan['person_emp_length'].median()), inplace=True)



# Create a histogram of employment length

n, bins, patches = plt.hist(cr_loan['person_emp_length'], bins='auto', color='blue')

plt.xlabel("Person Employment Length")

plt.show()
### Removing missing data



# Print the number of nulls

print(cr_loan['loan_int_rate'].isnull().sum())



# Store the array on indices

indices = cr_loan[cr_loan['loan_int_rate'].isnull()].index



# Save the new data without missing data

cr_loan_clean = cr_loan.drop(indices)
### Missing data intuition
### Logistic regression basics

from sklearn.linear_model import LogisticRegression

# Create the X and y data sets

X = cr_loan_clean[['loan_int_rate']]

y = cr_loan_clean[['loan_status']]



# Create and fit a logistic regression model

clf_logistic_single = LogisticRegression()

clf_logistic_single.fit(X, np.ravel(y))



# Print the parameters of the model

print(clf_logistic_single.get_params())



# Print the intercept of the model

print(clf_logistic_single.intercept_)
### Multivariate logistic regression



# Create X data for the model

X_multi = cr_loan_clean[['loan_int_rate','person_emp_length']]



# Create a set of y data for training

y = y[['loan_status']]



# Create and train a new logistic regression

clf_logistic_multi = LogisticRegression(solver='lbfgs').fit(X_multi, np.ravel(y))



# Print the intercept of the model

print(clf_logistic_multi.intercept_)
### Creating training and test sets

from sklearn.model_selection import train_test_split

# Create the X and y data sets

X = cr_loan_clean[['loan_int_rate','person_emp_length','person_income']]

y = cr_loan_clean[['loan_status']]



# Use test_train_split to create the training and test sets



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=123)



# Create and fit the logistic regression model

clf_logistic = LogisticRegression(solver='lbfgs').fit(X_train, np.ravel(y_train))



# Print the models coefficients

print(clf_logistic.coef_)
### Changing coefficients

X1_train = cr_loan_clean[['person_income', 'person_emp_length', 'loan_amnt']]

X2_train = cr_loan_clean[['person_income', 'loan_percent_income', 'cb_person_cred_hist_length']]

y_train = cr_loan_clean[['loan_status']]

# Print the first five rows of each training set

print(X1_train.head())

print(X2_train.head())



# Create and train a model on the first training data

clf_logistic1 = LogisticRegression(solver='lbfgs').fit(X1_train, np.ravel(y_train))



# Create and train a model on the second training data

clf_logistic2 = LogisticRegression(solver='lbfgs').fit(X2_train, np.ravel(y_train))



# Print the coefficients of each model

print(clf_logistic1.coef_)

print(clf_logistic2.coef_)
### One-hot encoding credit data



# Create two data sets for numeric and non-numeric data

cred_num = cr_loan_clean.select_dtypes(exclude=['object'])

cred_str = cr_loan_clean.select_dtypes(include=['object'])



# One-hot encode the non-numeric columns

cred_str_onehot = pd.get_dummies(cred_str)



# Union the one-hot encoded columns to the numeric ones

cr_loan_prep = pd.concat([cred_num, cred_str_onehot], axis=1)



# Print the columns in the new data set

print(cr_loan_prep.columns)
### Predicting probability of default



X = cr_loan_prep[['loan_int_rate','person_emp_length','person_income']]

y = cr_loan_prep[['loan_status']]



# Use test_train_split to create the training and test sets



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=123)



# Train the logistic regression model on the training data

clf_logistic = LogisticRegression(solver='lbfgs').fit(X_train, np.ravel(y_train))



# Create predictions of probability for loan status using test data

preds = clf_logistic.predict_proba(X_test)



# Create dataframes of first five predictions, and first five true labels

preds_df = pd.DataFrame(preds[:,1][0:5], columns = ['prob_default'])

true_df = y_test.head()



# Concatenate and print the two data frames for comparison

print(pd.concat([true_df.reset_index(drop = True), preds_df], axis = 1))
### Default classification reporting

from sklearn.metrics import classification_report, precision_recall_fscore_support



# Create a dataframe for the probabilities of default

preds_df = pd.DataFrame(preds[:,1], columns = ['prob_default'])



# Reassign loan status based on the threshold

preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.50 else 0)



# Print the row counts for each loan status

print(preds_df['loan_status'].value_counts())



# Print the classification report

target_names = ['Non-Default', 'Default']

print(classification_report(y_test, preds_df['loan_status'], target_names=target_names))
### Selecting report metrics



# Print the classification report

target_names = ['Non-Default', 'Default']

print(classification_report(y_test, preds_df['loan_status'], target_names=target_names))



# Print all the non-average values from the report

print(precision_recall_fscore_support(y_test,preds_df['loan_status']))



# Print the first two numbers from the report

print(precision_recall_fscore_support(y_test,preds_df['loan_status'])[:2])
### Visually scoring credit models

from sklearn.metrics import roc_curve, roc_auc_score



# Create predictions and store them in a variable

preds = clf_logistic.predict_proba(X_test)



# Print the accuracy score the model

print(clf_logistic.score(X_test, y_test))



# Plot the ROC curve of the probabilities of default

prob_default = preds[:, 1]

fallout, sensitivity, thresholds = roc_curve(y_test, prob_default)

plt.plot(fallout, sensitivity, color = 'darkorange')

plt.plot([0, 1], [0, 1], linestyle='--')

plt.show()



# Compute the AUC and store it in a variable

auc = roc_auc_score(y_test, prob_default)
### Thresholds and confusion matrices.

from sklearn.metrics import confusion_matrix

# Set the threshold for defaults to 0.5

preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.5 else 0)



# Print the confusion matrix

print(confusion_matrix(y_test,preds_df['loan_status']))



# Set the threshold for defaults to 0.4

preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.4 else 0)



# Print the confusion matrix

print(confusion_matrix(y_test,preds_df['loan_status']))



cr_loan_prep.columns
### How thresholds affect performance

avg_loan_amnt = cr_loan_prep['loan_amnt'].sum() / cr_loan_prep['loan_amnt'].count()

# Reassign the values of loan status based on the new threshold

preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.4 else 0)



# Store the number of loan defaults from the prediction data

num_defaults = preds_df['loan_status'].value_counts()[1]



# Store the default recall from the classification report

default_recall = precision_recall_fscore_support(y_test,preds_df['loan_status'])[1][1]



# Calculate the estimated impact of the new default recall rate

print(num_defaults * avg_loan_amnt * (1 - default_recall))
### Threshold selection

thresh = [0.2,0.225,0.25,0.275,0.3,0.325,0.35,0.375,0.4,0.425,0.45, 0.475,0.5,0.525,0.55,0.575,0.6,0.625,00.65]

def_recalls = [0.7981438515081206,

 0.7583139984532096,

 0.7157772621809745,

 0.6759474091260634,

 0.6349574632637278,

 0.594354215003867,

 0.5467904098994586,

 0.5054137664346481,

 0.46403712296983757,

 0.39984532095901004,

 0.32211910286156226,

 0.2354988399071926,

 0.16782675947409126,

 0.1148491879350348,

 0.07733952049497293,

 0.05529775715390565,

 0.03750966744006187,

 0.026295436968290797,

 0.017788089713843776]

nondef_recalls = [0.5342465753424658,

 0.5973037616873234,

 0.6552511415525114,

 0.708306153511633,

 0.756468797564688,

 0.8052837573385518,

 0.8482278756251359,

 0.8864970645792564,

 0.9215046749293324,

 0.9492280930637095,

 0.9646662317895195,

 0.9733637747336378,

 0.9809741248097412,

 0.9857577734290063,

 0.9902152641878669,

 0.992280930637095,

 0.9948901935203305,

 0.9966297021091541,

 0.997499456403566]

accs =[0.5921588594704684,

 0.6326374745417516,

 0.6685336048879837,

 0.7012050237610319,

 0.7298031228784793,

 0.7589952477936185,

 0.7820773930753564,

 0.8028682959945689,

 0.8211133740665308,

 0.8286659877800407,

 0.8236591989137814,

 0.811439239646979,

 0.8025288526816021,

 0.7946367956551256,

 0.7898845892735913,

 0.7866598778004074,

 0.7847929395790902,

 0.7836897488119484,

 0.7825016972165648]

ticks = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]

plt.plot(thresh,def_recalls)

plt.plot(thresh,nondef_recalls)

plt.plot(thresh,accs)

plt.xlabel("Probability Threshold")

plt.xticks(ticks)

plt.legend(["Default Recall","Non-default Recall","Model Accuracy"])

plt.show()
### Trees for defaults



# Train a model

import xgboost as xgb

clf_gbt = xgb.XGBClassifier().fit(X_train, np.ravel(y_train))



# Predict with a model

gbt_preds = clf_gbt.predict_proba(X_test)



# Create dataframes of first five predictions, and first five true labels

preds_df = pd.DataFrame(gbt_preds[:,1][0:5], columns = ['prob_default'])

true_df = y_test.head()



# Concatenate and print the two data frames for comparison

print(pd.concat([true_df.reset_index(drop = True), preds_df], axis = 1))
### Gradient boosted portfolio performance



# Print the first five rows of the portfolio data frame

print(portfolio.head())



# Create expected loss columns for each model using the formula

portfolio['gbt_expected_loss'] = portfolio['gbt_prob_default'] * portfolio['lgd'] * portfolio['loan_amnt']

portfolio['lr_expected_loss'] = portfolio['lr_prob_default'] * portfolio['lgd'] * portfolio['loan_amnt']



# Print the sum of the expected loss for lr

print('LR expected loss: ', np.sum(portfolio['lr_expected_loss']))



# Print the sum of the expected loss for gbt

print('GBT expected loss: ', np.sum(portfolio['gbt_expected_loss']))