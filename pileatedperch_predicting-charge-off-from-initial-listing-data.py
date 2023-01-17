import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Pandas options
pd.set_option('display.max_colwidth', 1000, 'display.max_rows', None, 'display.max_columns', None)

# Plotting options
%matplotlib inline
mpl.style.use('ggplot')
sns.set(style='whitegrid')
loans = pd.read_csv('../input/accepted_2007_to_2017Q3.csv.gz', compression='gzip', low_memory=True)
loans.info()
loans.sample(5)
loans['loan_status'].value_counts(dropna=False)
loans = loans.loc[loans['loan_status'].isin(['Fully Paid', 'Charged Off'])]
loans.shape
loans['loan_status'].value_counts(dropna=False)
loans['loan_status'].value_counts(normalize=True, dropna=False)
missing_fractions = loans.isnull().mean().sort_values(ascending=False)
missing_fractions.head(10)
plt.figure(figsize=(6,3), dpi=90)
missing_fractions.plot.hist(bins=20)
plt.title('Histogram of Feature Incompleteness')
plt.xlabel('Fraction of data missing')
plt.ylabel('Feature count')
drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))
print(drop_list)
len(drop_list)
loans.drop(labels=drop_list, axis=1, inplace=True)
loans.shape
print(sorted(loans.columns))
keep_list = ['addr_state', 'annual_inc', 'application_type', 'dti', 'earliest_cr_line', 'emp_length', 'emp_title', 'fico_range_high', 'fico_range_low', 'grade', 'home_ownership', 'id', 'initial_list_status', 'installment', 'int_rate', 'issue_d', 'loan_amnt', 'loan_status', 'mort_acc', 'open_acc', 'pub_rec', 'pub_rec_bankruptcies', 'purpose', 'revol_bal', 'revol_util', 'sub_grade', 'term', 'title', 'total_acc', 'verification_status', 'zip_code']
len(keep_list)
drop_list = [col for col in loans.columns if col not in keep_list]
print(drop_list)
len(drop_list)
loans.drop(labels=drop_list, axis=1, inplace=True)
loans.shape
def plot_var(col_name, full_name, continuous):
    """
    Visualize a variable with and without faceting on the loan status.
    - col_name is the variable name in the dataframe
    - full_name is the full variable name
    - continuous is True if the variable is continuous, False otherwise
    """
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,3), dpi=90)
    
    # Plot without loan status
    if continuous:
        sns.distplot(loans.loc[loans[col_name].notnull(), col_name], kde=False, ax=ax1)
    else:
        sns.countplot(loans[col_name], order=sorted(loans[col_name].unique()), color='#5975A4', saturation=1, ax=ax1)
    ax1.set_xlabel(full_name)
    ax1.set_ylabel('Count')
    ax1.set_title(full_name)

    # Plot with loan status
    if continuous:
        sns.boxplot(x=col_name, y='loan_status', data=loans, ax=ax2)
        ax2.set_ylabel('')
        ax2.set_title(full_name + ' by Loan Status')
    else:
        charge_off_rates = loans.groupby(col_name)['loan_status'].value_counts(normalize=True).loc[:,'Charged Off']
        sns.barplot(x=charge_off_rates.index, y=charge_off_rates.values, color='#5975A4', saturation=1, ax=ax2)
        ax2.set_ylabel('Fraction of Loans Charged-off')
        ax2.set_title('Charge-off Rate by ' + full_name)
    ax2.set_xlabel(full_name)
    
    plt.tight_layout()
print(list(loans.columns))
loans['id'].sample(5)
loans['id'].describe()
loans.drop('id', axis=1, inplace=True)
loans['loan_amnt'].describe()
plot_var('loan_amnt', 'Loan Amount', continuous=True)
loans.groupby('loan_status')['loan_amnt'].describe()
loans['term'].value_counts(dropna=False)
loans['term'] = loans['term'].apply(lambda s: np.int8(s.split()[0]))
loans['term'].value_counts(normalize=True)
loans.groupby('term')['loan_status'].value_counts(normalize=True).loc[:,'Charged Off']
loans['int_rate'].describe()
plot_var('int_rate', 'Interest Rate', continuous=True)
loans.groupby('loan_status')['int_rate'].describe()
loans['installment'].describe()
plot_var('installment', 'Installment', continuous=True)
loans.groupby('loan_status')['installment'].describe()
print(sorted(loans['grade'].unique()))
print(sorted(loans['sub_grade'].unique()))
loans.drop('grade', axis=1, inplace=True)
plot_var('sub_grade', 'Subgrade', continuous=False)
loans['emp_title'].describe()
loans.drop(labels='emp_title', axis=1, inplace=True)
loans['emp_length'].value_counts(dropna=False).sort_index()
loans['emp_length'].replace(to_replace='10+ years', value='10 years', inplace=True)
loans['emp_length'].replace('< 1 year', '0 years', inplace=True)
def emp_length_to_int(s):
    if pd.isnull(s):
        return s
    else:
        return np.int8(s.split()[0])
loans['emp_length'] = loans['emp_length'].apply(emp_length_to_int)
loans['emp_length'].value_counts(dropna=False).sort_index()
plot_var('emp_length', 'Employment Length', continuous=False)
loans['home_ownership'].value_counts(dropna=False)
loans['home_ownership'].replace(['NONE', 'ANY'], 'OTHER', inplace=True)
loans['home_ownership'].value_counts(dropna=False)
plot_var('home_ownership', 'Home Ownership', continuous=False)
loans.groupby('home_ownership')['loan_status'].value_counts(normalize=True).loc[:,'Charged Off']
loans['annual_inc'].describe()
loans['log_annual_inc'] = loans['annual_inc'].apply(lambda x: np.log10(x+1))
loans.drop('annual_inc', axis=1, inplace=True)
loans['log_annual_inc'].describe()
plot_var('log_annual_inc', 'Log Annual Income', continuous=True)
loans.groupby('loan_status')['log_annual_inc'].describe()
plot_var('verification_status', 'Verification Status', continuous=False)
loans['purpose'].value_counts()
loans.groupby('purpose')['loan_status'].value_counts(normalize=True).loc[:,'Charged Off'].sort_values()
loans['title'].describe()
loans['title'].value_counts().head(10)
loans.drop('title', axis=1, inplace=True)
loans['zip_code'].sample(5)
loans['zip_code'].nunique()
loans['addr_state'].sample(5)
loans['addr_state'].nunique()
loans.drop(labels='zip_code', axis=1, inplace=True)
loans.groupby('addr_state')['loan_status'].value_counts(normalize=True).loc[:,'Charged Off'].sort_values()
loans['dti'].describe()
plt.figure(figsize=(8,3), dpi=90)
sns.distplot(loans.loc[loans['dti'].notnull() & (loans['dti']<60), 'dti'], kde=False)
plt.xlabel('Debt-to-income Ratio')
plt.ylabel('Count')
plt.title('Debt-to-income Ratio')
(loans['dti']>=60).sum()
loans.groupby('loan_status')['dti'].describe()
loans['earliest_cr_line'].sample(5)
loans['earliest_cr_line'].isnull().any()
loans['earliest_cr_line'] = loans['earliest_cr_line'].apply(lambda s: int(s[-4:]))
loans['earliest_cr_line'].describe()
plot_var('earliest_cr_line', 'Year of Earliest Credit Line', continuous=True)
loans[['fico_range_low', 'fico_range_high']].describe()
loans[['fico_range_low','fico_range_high']].corr()
loans['fico_score'] = 0.5*loans['fico_range_low'] + 0.5*loans['fico_range_high']
loans.drop(['fico_range_high', 'fico_range_low'], axis=1, inplace=True)
plot_var('fico_score', 'FICO Score', continuous=True)
loans.groupby('loan_status')['fico_score'].describe()
plt.figure(figsize=(10,3), dpi=90)
sns.countplot(loans['open_acc'], order=sorted(loans['open_acc'].unique()), color='#5975A4', saturation=1)
_, _ = plt.xticks(np.arange(0, 90, 5), np.arange(0, 90, 5))
plt.title('Number of Open Credit Lines')
loans.groupby('loan_status')['open_acc'].describe()
loans['pub_rec'].value_counts().sort_index()
loans.groupby('loan_status')['pub_rec'].describe()
loans['revol_bal'].describe()
loans['log_revol_bal'] = loans['revol_bal'].apply(lambda x: np.log10(x+1))
loans.drop('revol_bal', axis=1, inplace=True)
plot_var('log_revol_bal', 'Log Revolving Credit Balance', continuous=True)
loans.groupby('loan_status')['log_revol_bal'].describe()
loans['revol_util'].describe()
plot_var('revol_util', 'Revolving Line Utilization', continuous=True)
loans.groupby('loan_status')['revol_util'].describe()
plt.figure(figsize=(12,3), dpi=90)
sns.countplot(loans['total_acc'], order=sorted(loans['total_acc'].unique()), color='#5975A4', saturation=1)
_, _ = plt.xticks(np.arange(0, 176, 10), np.arange(0, 176, 10))
plt.title('Total Number of Credit Lines')
loans.groupby('loan_status')['total_acc'].describe()
plot_var('initial_list_status', 'Initial List Status', continuous=False)
loans['application_type'].value_counts()
loans.groupby('application_type')['loan_status'].value_counts(normalize=True).loc[:,'Charged Off']
loans['mort_acc'].describe()
loans['mort_acc'].value_counts().head(10)
loans.groupby('loan_status')['mort_acc'].describe()
loans['pub_rec_bankruptcies'].value_counts().sort_index()
plot_var('pub_rec_bankruptcies', 'Public Record Bankruptcies', continuous=False)
loans['charged_off'] = (loans['loan_status'] == 'Charged Off').apply(np.uint8)
loans.drop('loan_status', axis=1, inplace=True)
loans.shape
missing_fractions = loans.isnull().mean().sort_values(ascending=False) # Fraction of data missing for each variable
print(missing_fractions[missing_fractions > 0]) # Print variables that are missing data
print(loans.columns)
loans = pd.get_dummies(loans, columns=['sub_grade', 'home_ownership', 'verification_status', 'purpose', 'addr_state', 'initial_list_status', 'application_type'], drop_first=True)
loans.shape
loans.sample(5)
loans['issue_d'].sample(5)
loans['issue_d'].isnull().any()
loans['issue_d'] = pd.to_datetime(loans['issue_d'])
loans['issue_d'].sample(5)
loans['issue_d'].describe()
plt.figure(figsize=(6,3), dpi=90)
loans['issue_d'].dt.year.value_counts().sort_index().plot.bar(color='darkblue')
plt.xlabel('Year')
plt.ylabel('Number of Loans Funded')
plt.title('Loans Funded per Year')
loans_train = loans.loc[loans['issue_d'] <  loans['issue_d'].quantile(0.9)]
loans_test =  loans.loc[loans['issue_d'] >= loans['issue_d'].quantile(0.9)]
print('Number of loans in the partition:   ', loans_train.shape[0] + loans_test.shape[0])
print('Number of loans in the full dataset:', loans.shape[0])
loans_test.shape[0] / loans.shape[0]
del loans
loans_train['issue_d'].describe()
loans_test['issue_d'].describe()
loans_train.drop('issue_d', axis=1, inplace=True)
loans_test.drop('issue_d', axis=1, inplace=True)
y_train = loans_train['charged_off']
y_test = loans_test['charged_off']
X_train = loans_train.drop('charged_off', axis=1)
X_test = loans_test.drop('charged_off', axis=1)
del loans_train, loans_test
linear_dep = pd.DataFrame()
for col in X_train.columns:
    linear_dep.loc[col, 'pearson_corr'] = X_train[col].corr(y_train)
linear_dep['abs_pearson_corr'] = abs(linear_dep['pearson_corr'])
from sklearn.feature_selection import f_classif
for col in X_train.columns:
    mask = X_train[col].notnull()
    (linear_dep.loc[col, 'F'], linear_dep.loc[col, 'p_value']) = f_classif(pd.DataFrame(X_train.loc[mask, col]), y_train.loc[mask])
linear_dep.sort_values('abs_pearson_corr', ascending=False, inplace=True)
linear_dep.drop('abs_pearson_corr', axis=1, inplace=True)
linear_dep.reset_index(inplace=True)
linear_dep.rename(columns={'index':'variable'}, inplace=True)
linear_dep.head(20)
linear_dep.tail(20)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
pipeline_sgdlogreg = Pipeline([
    ('imputer', SimpleImputer(copy=False)), # Mean imputation by default
    ('scaler', StandardScaler(copy=False)),
    ('model', SGDClassifier(loss='log', max_iter=1000, tol=1e-3, random_state=1, warm_start=True))
])
param_grid_sgdlogreg = {
    'model__alpha': [10**-5, 10**-2, 10**1],
    'model__penalty': ['l1', 'l2']
}
grid_sgdlogreg = GridSearchCV(estimator=pipeline_sgdlogreg, param_grid=param_grid_sgdlogreg, scoring='roc_auc', n_jobs=1, pre_dispatch=1, cv=5, verbose=1, return_train_score=False)
grid_sgdlogreg.fit(X_train, y_train)
grid_sgdlogreg.best_score_
grid_sgdlogreg.best_params_
from sklearn.ensemble import RandomForestClassifier
pipeline_rfc = Pipeline([
    ('imputer', SimpleImputer(copy=False)),
    ('model', RandomForestClassifier(n_jobs=-1, random_state=1))
])
param_grid_rfc = {
    'model__n_estimators': [50] # The number of randomized trees to build
}
grid_rfc = GridSearchCV(estimator=pipeline_rfc, param_grid=param_grid_rfc, scoring='roc_auc', n_jobs=1, pre_dispatch=1, cv=5, verbose=1, return_train_score=False)
grid_rfc.fit(X_train, y_train)
grid_rfc.best_score_
from sklearn.neighbors import KNeighborsClassifier
pipeline_knn = Pipeline([
    ('imputer', SimpleImputer(copy=False)),
    ('scaler', StandardScaler(copy=False)),
    ('lda', LinearDiscriminantAnalysis()),
    ('model', KNeighborsClassifier(n_jobs=-1))
])
param_grid_knn = {
    'lda__n_components': [3, 9], # Number of LDA components to keep
    'model__n_neighbors': [5, 25, 125] # The 'k' in k-nearest neighbors
}
grid_knn = GridSearchCV(estimator=pipeline_knn, param_grid=param_grid_knn, scoring='roc_auc', n_jobs=1, pre_dispatch=1, cv=5, verbose=1, return_train_score=False)
grid_knn.fit(X_train, y_train)
grid_knn.best_score_
grid_knn.best_params_
print('Cross-validated AUROC scores')
print(grid_sgdlogreg.best_score_, '- Logistic regression')
print(grid_rfc.best_score_, '- Random forest')
print(grid_knn.best_score_, '- k-nearest neighbors')
param_grid_sgdlogreg = {
    'model__alpha': np.logspace(-4.5, 0.5, 11), # Fills in the gaps between 10^-5 and 10^1
    'model__penalty': ['l1', 'l2']
}

print(param_grid_sgdlogreg)
grid_sgdlogreg = GridSearchCV(estimator=pipeline_sgdlogreg, param_grid=param_grid_sgdlogreg, scoring='roc_auc', n_jobs=1, pre_dispatch=1, cv=5, verbose=1, return_train_score=False)
grid_sgdlogreg.fit(X_train, y_train)
grid_sgdlogreg.best_score_
grid_sgdlogreg.best_params_
from sklearn.metrics import roc_auc_score
y_score = grid_sgdlogreg.predict_proba(X_test)[:,1]
roc_auc_score(y_test, y_score)