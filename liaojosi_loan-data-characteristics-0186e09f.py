# Imports and setup

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt



%matplotlib inline

plt.style.use('ggplot')



# Kaggle note: Any results written to the current directory are saved as output
# Read loan data

date_cols = ['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d']

loans = pd.read_csv("../input/loan.csv", low_memory=False, index_col='id',

    parse_dates=date_cols, infer_datetime_format=True)

print("Dataset size: {}".format(loans.shape))

print(loans.head())
# What columns do we have?

print("{} columns: {}".format(len(loans.columns), loans.columns))
# Let's take a look at the different columns and what data they contain

#cols = loans.columns[0:10]  # cycle through 0:10, 10:20, ...

cols = ['loan_amnt', 'term', 'int_rate', 'installment', 'emp_length']  # or pick specific columns

print(cols)

for col in cols:

    print(loans[col].describe())  # describe one by one in case of mixed types
# Parse term durations: ' 36 months' -> 36 (numeric)

print("term before:-")

print(loans.term.head())

loans.term = pd.to_numeric(loans.term.str[:3])

print("term after:-")

print(loans.term.head())
# Parse emp_length: '< 1 year' -> 1.0, '1 year' -> 1.0, '7 year' -> 7.0, etc. (numeric)

print("emp_length before:-")

print(loans.emp_length.head())

loans.emp_length = loans.emp_length.str.extract("(\d+)", expand=False).map(float)

print("emp_length after:-")

print(loans.emp_length.head())
# What is the distribution of loans by status?

loans_by_status = loans.groupby('loan_status')

print(loans_by_status['loan_status'].count())

loans_by_status['loan_status'].count().plot(kind='bar')
# What is the distribution of loans by purpose?

loans_by_purpose = loans.groupby('purpose')

print(loans_by_purpose['purpose'].count())

loans_by_purpose['purpose'].count().plot(kind='bar')
# What is the distribution of loans by term?

loans_by_term = loans.groupby('term')

print(loans_by_term['term'].count())

loans_by_term['term'].count().plot(kind='bar')
# Select loans issued within desired date range

#loans.issue_d.describe()  # dataset range: 2007-06-01 to 2015-12-01

range_selected = ('2007-06-01', '2010-12-31')

loans_selected = loans.loc[(range_selected[0] <= loans.issue_d) & (loans.issue_d <= range_selected[1])]

print("{num} loans were issued from {range[0]} to {range[1]}".format(num=len(loans_selected), range=range_selected))



# What is their distribution by status?

print(loans_selected.groupby('loan_status')['loan_status'].count())
# Let's setup a binary classification target 'default': 0 => Fully Paid, 1 => Charged Off

loans_subset = loans_selected.copy()

loans_subset['default'] = None

loans_subset.loc[(loans_subset.loan_status == 'Fully Paid') | (loans_subset.loan_status == 'Does not meet the credit policy. Status:Fully Paid'), 'default'] = 0

loans_subset.loc[(loans_subset.loan_status == 'Charged Off') | (loans_subset.loan_status == 'Does not meet the credit policy. Status:Charged Off'), 'default'] = 1



# Drop loans that haven't been terminated yet (we don't know what their final status will be)

loans_subset = loans_subset[~loans_subset.default.isnull()]

print("Data subset size: {}".format(loans_subset.shape))



# Re-encode 'default' column as numeric (0 or 1)

loans_subset['default'] = pd.to_numeric(loans_subset['default'])
# Drop columns that are unimportant, superfluous or leak target information

# Note: We only want to keep information that is available at loan *application* stage

application_cols = [

    # Identifiers and dates

    #'id',  # used as index column

    'member_id',

    'issue_d',

    

    # Loan application details

    #'application_type',  # all 'INDIVIDUAL'

    'loan_amnt',  # $ applied for

    'term',  # 36 or 60 months

    'int_rate',  # % annual (?) interest rate

    'installment',  # $ monthly payment

    'emp_title',  # employee/employer title

    'emp_length',  # 0-10+ years

    'home_ownership',  # RENT, OWN, MORTGAGE, etc.

    'verification_status',  # mostly 'Not Verified'

    #'verification_status_joint',  # all 0

    'purpose',  # 'debt_consolidation', 'small_business', etc.

    'title',  # text

    #'desc',  # text, too verbose, may contain updates after application stage

    'zip_code',  # 100XX

    'addr_state',  # covered by zip_code?

    

    # Additional loan listing details

    #'initial_list_status',  # all 'f'

    #'policy_code',  # all 1

    #'url',  # unqiue per loan



    # Borrower's creditworthiness

    'annual_inc', #'annual_inc_joint',  # income ($; individual only, no joint loans)

    'dti', #'dti_joint',  # debt-to-income ratio (%; individual only, no joint loans)

    'revol_bal', 'revol_util',  # revolving accounts: balance ($), utilization (%)

    #'tot_cur_bal', 'max_bal_bc',  # overall balance: total current, max; all null

    'earliest_cr_line', 'total_acc', 'open_acc',  # credit accounts

    'inq_last_6mths', #'inq_last_12m', 'inq_fi',  # credit inquiries (only 6 mths available)

    'delinq_2yrs', 'mths_since_last_delinq', #'acc_now_delinq',  # delinquency (acc_now_delinq is mostly 0)

    #'tot_coll_amt', 'collections_12_mths_ex_med',  # collections; all null or 0

    #'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util',  # installment accounts; all null

    #'open_acc_6m', 'open_rv_12m', 'open_rv_24m', 'total_rev_hi_lim', 'total_cu_tl', 'all_util', # revolving trading accounts; all null

    

    # Public records

    'pub_rec', 'mths_since_last_record',

    #'mths_since_last_major_derog',  # all null



    # Loan rating as determined by lender (potential multi-class targets to predict?)

    #'grade',

    #'sub_grade',



    # Desired binary target to predict

    'default'

]



loans_small = loans_subset[application_cols]



# Check selected data subset

print("Small dataset has {} rows, {} columns:".format(len(loans_small), len(loans_small.columns)))

print(loans_small.head())

print("Class distribution:")

print(loans_small.groupby('default')['default'].count())
# Write dataset to disk (if you want to save it)

loans_small.to_csv("loans_small.csv")

print("Dataset saved!")
# Read back from disk (to skip all previous steps if you've saved it already)

loans_small = pd.read_csv("loans_small.csv", index_col=0, parse_dates=True)

print("Loaded data has {} rows, {} columns:".format(len(loans_small), len(loans_small.columns)))

print(loans_small.head())

print("Class distribution:")

print(loans_small.groupby('default')['default'].count())
# Specify a subset of feature columns and a target to predict ('default')

feature_cols = [

    'loan_amnt', 'term', 'int_rate', 'installment', 'purpose',

    #'emp_title', # free text

    'emp_length', 'home_ownership',

    #'zip_code', 'addr_state',  # categorical, but too many levels

    'annual_inc', 'dti',

    'revol_bal', 'revol_util',

    'verification_status'

]



target_col = 'default'



# Create the final dataset we'll use for classification

keep_cols = feature_cols + [target_col]

loans_final = loans_small[keep_cols]



# Drop samples with null values (few enough that we can ignore)

loans_final.dropna(inplace=True)



print("Final dataset: {} features, {} samples".format(len(loans_final.columns), len(loans_final)))

print(loans_final.head())

print("Final class distribution (after dropping nulls):")

class_counts = loans_final.groupby(target_col)[target_col].agg({

    'count': len,

    'ratio': lambda x: float(len(x)) / len(loans_final)

})

print(class_counts)



# Extract desired features and target column

X = loans_final[feature_cols]

y = loans_final[target_col]

print("{} features: {}".format(len(X.columns), X.columns))

print("Target: {}".format(y.name))
# Encode categorical variables among features

categorical_vars = ['home_ownership', 'purpose', 'verification_status']

X = pd.get_dummies(X, columns=categorical_vars, drop_first=True)

print("{} features after encoding categorical variables: {}".format(len(X.columns), X.columns))
# Split into training and test sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print("Training set: {} samples, test set: {} samples".format(len(X_train), len(X_test)))
# Common sklearn imports

from sklearn.metrics import classification_report



# Define a simple train-predict utility function

def train_predict(clf, X_train, X_test, y_train, y_test):

    """Train clf on <X_train, y_train>, predict <X_test, y_test>; return y_pred."""

    print("Training a {}...".format(clf.__class__.__name__))

    %time clf.fit(X_train, y_train)

    print(clf)

    

    print("Predicting test labels...")

    y_pred = clf.predict(X_test)

    return y_pred
# Classify using a Decision Tree

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=42)

y_pred = train_predict(clf, X_train, X_test, y_train, y_test)

print(classification_report(y_test, y_pred))



# Analyze feature importance

feature_imps = pd.DataFrame({'feature': X_train.columns, 'importance': clf.feature_importances_})

feature_imps.sort_values(by='importance', ascending=False, inplace=True)

print("Top 10 important features:")

print(feature_imps[:10])
# Classify using a Random Forest

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10, random_state=42)

y_pred = train_predict(clf, X_train, X_test, y_train, y_test)

print(classification_report(y_test, y_pred))
# Classify using a Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=100, max_depth=1, learning_rate=1.0, random_state=42)

y_pred = train_predict(clf, X_train, X_test, y_train, y_test)

print(classification_report(y_test, y_pred))
# Note: The performance on the interesting class (default=1) is very low!

# TODO: Try subsampling the other class (default=0) or other methods to mitigate class imbalance.