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
import os

import pandas as pd

from pandas import Series, DataFrame

#import pandas_profiling



pd.set_option('display.max_rows', None,'display.max_columns', None)



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



from collections import Counter



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, cross_validate

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, make_scorer, roc_auc_score,accuracy_score, roc_curve

from scipy.stats import ks_2samp

#from treeinterpreter import treeinterpreter as ti



from sklearn.preprocessing import Imputer, StandardScaler

#from sklearn import cross_validation

from sklearn import metrics

"""

from sklearn import metrics

from sklearn import linear_model



from sklearn import tree

from sklearn import svm

from sklearn import ensemble

from sklearn import neighbors

from sklearn import preprocessing

"""



# ignore Deprecation Warning

import warnings

#warnings.filterwarnings("ignore", category=DeprecationWarning,RuntimeWarning) 

warnings.filterwarnings("ignore") 



#plt.style.use('fivethirtyeight') # Good looking plots

pd.set_option('display.max_columns', None) # Display any number of columns



df = pd.read_csv('../input/LoanStats3d-Modified.csv')
df.info()
df.sample(3)
df['loan_status'].value_counts()

def missing_values_table(df):

     # Total missing values

    mis_val = df.isnull().sum()

    # Percentage of missing values

    mis_val_percent = 100 * df.isnull().sum() / len(df)

    mis_val_type = df.dtypes

    # Make a table with the results

    mis_val_table = pd.concat([mis_val, mis_val_percent, mis_val_type], axis=1)

        

     # Rename the columns

    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values', 2: 'type'})

        

    # Sort the table by percentage of missing descending

    mis_val_table_ren_columns = mis_val_table_ren_columns[ mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)

        

    # Print some summary information

    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n" "There are " + str(mis_val_table_ren_columns.shape[0]) + " columns that have missing values.")

        

    # Return the dataframe with missing information

    return mis_val_table_ren_columns
missing_values_table(df)

missing_frac = df.isnull().mean()

drop_list = sorted(missing_frac[missing_frac > 0.50].index)
print(drop_list)

len(drop_list)

def drop_cols(cols):

    df.drop(labels=cols, axis=1, inplace=True)
drop_cols(drop_list)

df.shape

drop_list = ['acc_now_delinq', 'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util', 'chargeoff_within_12_mths', 'collection_recovery_fee', 'collections_12_mths_ex_med', 'debt_settlement_flag', 'delinq_2yrs', 'delinq_amnt',  'funded_amnt', 'funded_amnt_inv', 'hardship_flag', 'inq_last_6mths', 'last_credit_pull_d',  'last_pymnt_amnt', 'last_pymnt_d', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mths_since_recent_bc', 'mths_since_recent_inq', 'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m',  'out_prncp', 'out_prncp_inv', 'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'pymnt_plan', 'recoveries', 'tax_liens', 'tot_coll_amt', 'tot_cur_bal', 'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit', 'total_pymnt', 'total_pymnt_inv', 'total_rec_int', 'total_rec_late_fee', 'total_rec_prncp', 'total_rev_hi_lim']
drop_cols(drop_list)

len(df.columns)
def plot_var(col_name, full_name, continuous):

    """

    Visualize a variable with/without faceting on the loan status.

    - col_name is the variable name in the dataframe

    - full_name is the full variable name

    - continuous is True for continuous variables

    """

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=False, figsize=(15,3))

    # plot1: counts distribution of the variable

    

    if continuous:  

        sns.distplot(df.loc[df[col_name].notnull(), col_name], kde=False, ax=ax1)

    else:

        sns.countplot(df[col_name], order=sorted(df[col_name].unique()), color='#5975A4', saturation=1, ax=ax1)

    ax1.set_xlabel(full_name)

    ax1.set_ylabel('Count')

    ax1.set_title(full_name)



          

    # plot2: bar plot of the variable grouped by loan_status

    if continuous:

        sns.boxplot(x=col_name, y='loan_status', data=df, ax=ax2)

        ax2.set_ylabel('')

        ax2.set_title(full_name + ' by Loan Status')

    else:

        Charged_Off_rates = df.groupby(col_name)['loan_status'].value_counts(normalize=True)[:,'Charged Off']

        sns.barplot(x=Charged_Off_rates.index, y=Charged_Off_rates.values, color='#5975A4', saturation=1, ax=ax2)

        ax2.set_ylabel('Fraction of Loans Charged Off')

        ax2.set_title('Charged Off Rate by ' + full_name)

        ax2.set_xlabel(full_name)

    

    # plot3: kde plot of the variable gropued by loan_status

    if continuous:  

        facet = sns.FacetGrid(df, hue = 'loan_status', size=3, aspect=4)

        facet.map(sns.kdeplot, col_name, shade=True)

        #facet.set(xlim=(df[col_name].min(), df[col_name].max()))

        facet.add_legend()  

    else:

        fig = plt.figure(figsize=(12,3))

        sns.countplot(x=col_name, hue='loan_status', data=df, order=sorted(df[col_name].unique()) )

        

plt.tight_layout()    
df['loan_amnt'].describe()

plot_var('loan_amnt', 'Loan Amount', continuous=True)

df['term'] = df['term'].apply(lambda s: np.int8(s.split()[0]))
plot_var('term', 'Term', continuous=False)
df['term'].value_counts(normalize=True)
df.groupby('term')['loan_status'].value_counts(normalize=True).loc[:,'Charged Off']
df['int_round'] = df['int_rate'].replace("%","", regex=True).astype(float)
plot_var('int_round', 'Interest Rate', continuous=True)
def outliers_modified_z_score(dataframe, n, features):

    """

    Takes a dataframe df of features and returns a list of the indices corresponding to the observations containing more than n outliers according to the modified z-score Method

    """

    threshold = 3.5

    outlier_indices = []

    for col in features:

        median_y = np.median(dataframe[col])

        median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in dataframe[col]])

        modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y for y in dataframe[col]]

        outlier_list_col = dataframe[np.abs(modified_z_scores) > threshold].index

       # append the found outlier indices for col to the list of outlier indices 

        outlier_indices.extend(outlier_list_col)

        # select observations containing more than 2 outliers

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    return multiple_outliers
def outliers_iqr(dataframe, n, features):

    """

    Takes a dataframe df of features and returns a list of the indices

    corresponding to the observations containing more than n outliers according

    to the Tukey method.

    """

    outlier_indices = []

    for col in features:

        # 1st quartile (25%) & # 3rd quartile (75%)

        quartile_1, quartile_3 = np.percentile(dataframe[col], [25,75])

        #quartile_3 = np.percentile(dataframe[col], 75)

      

        iqr = quartile_3 - quartile_1

        lower_bound = quartile_1 - (iqr * 1.5)

        upper_bound = quartile_3 + (iqr * 1.5)

        # Determine a list of indices of outliers for feature col

        outlier_list_col = dataframe[(dataframe[col] < lower_bound) | (dataframe[col] > upper_bound)].index

       # append the found outlier indices for col to the list of outlier indices 

        outlier_indices.extend(outlier_list_col)

        # select observations containing more than 2 outliers

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    return multiple_outliers
df.groupby('loan_status')['int_rate'].describe()
df.loc[(df.int_round > 15.61) & (df.loan_status == 'Fully Paid')].shape[0]

(df.loc[(df.int_round > 15.61) & (df.loan_status == 'Fully Paid')].shape[0])/df['loan_status'].value_counts(normalize=False, dropna=False)[0]



df.loc[(df.int_round >18.55) & (df.loan_status == 'Charged Off')].shape[0]/df['loan_status'].value_counts(normalize=False, dropna=False)[1]
sorted(df['grade'].unique())
print(sorted(df['sub_grade'].unique()))
plot_var('sub_grade','Subgrade',continuous=False)
plot_var('grade','Grade',continuous=False)
df['emp_length'].value_counts(dropna=False).sort_index()

df['emp_length'].replace('10+ years', '10 years', inplace=True)

df['emp_length'].replace('< 1 year', '0 years', inplace=True)

df['emp_length'].value_counts(dropna=False).sort_index()

df.emp_length.map( lambda x: str(x).split()[0]).value_counts(dropna=True).sort_index()

df['emp_length'] = df.emp_length.map( lambda x: float(str(x).split()[0]))

df['emp_length'].sample(5)
plot_var('emp_length', 'Employment length', continuous=False)
df['home_ownership'].replace(['NONE','ANY'],'OTHER', inplace=True)

df['annual_inc'] = df['annual_inc'].apply(lambda x:np.log10(x+1))

plot_var('annual_inc', 'Log10 Annual income', continuous=True)
drop_cols('title')

drop_cols('zip_code')

f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,3), dpi=90)

sns.distplot(df.loc[df['dti'].notnull() & (df['dti'] < 60), 'dti'], kde=False, ax=ax1)

ax1.set_xlabel('dti')

ax1.set_ylabel('Count')

ax1.set_title('debt to income')

sns.boxplot(x=df.loc[df['dti'].notnull() & (df['dti'] < 60), 'dti'], y='loan_status', data=df, ax=ax2)

ax2.set_xlabel('DTI')

ax2.set_ylabel('Fraction of Loans fully paid')

ax2.set_title('Fully paid rate by debt to income')

ax2.set_title('DTI by loan status')
from datetime import datetime



df.earliest_cr_line = pd.to_datetime(df.earliest_cr_line, errors = 'coerce')

df = df.dropna(subset=['earliest_cr_line'])

dttoday = datetime.now().strftime('%Y-%m-%d')

df.earliest_cr_line = df.earliest_cr_line.apply(lambda x:(np.timedelta64((x - pd.Timestamp(dttoday)),'D').astype(int))/-365)



df.earliest_cr_line.shape

plot_var('earliest_cr_line', 'Length of of the earliest Credit Line (Months to today)', continuous=True)
df.pub_rec = df.pub_rec.map(lambda x: 3 if x >2.0 else x)

df['revol_bal'] = df['revol_bal'].apply(lambda x:np.log10(x+1))

drop_cols('policy_code')

df.mort_acc = df.mort_acc.map(lambda x: 6.0 if x > 6.0 else x)

# Next, I will convert the "loan_status" column to a 0/1 "charged off" column. Fully Paid:0 Charged Off: 1

df['Charged_Off'] = df['loan_status'].apply(lambda s: np.float(s == 'Charged Off'))

list_float = df.select_dtypes(exclude=['object']).columns
def run_KS_test(feature):

    dist1 = df.loc[df.Charged_Off == 0,feature]

    dist2 = df.loc[df.Charged_Off == 1,feature]

    print(feature+':')

    print(ks_2samp(dist1,dist2),'\n')
from statsmodels.stats.proportion import proportions_ztest

def run_proportion_Z_test(feature):

    dist1 = df.loc[df.Charged_Off == 0, feature]

    dist2 = df.loc[df.Charged_Off == 1, feature]

    n1 = len(dist1)

    p1 = dist1.sum()

    n2 = len(dist2)

    p2 = dist2.sum()

    z_score, p_value = proportions_ztest([p1, p2], [n1, n2])

    print(feature+':')

    print('z-score = {}; p-value = {}'.format(z_score, p_value),'\n')
from scipy.stats import chi2_contingency

def run_chi2_test(df, feature):



    dist1 = df.loc[df.loan_status == 'Fully Paid',feature].value_counts().sort_index().tolist()

    dist2 = df.loc[df.loan_status == 'Charged Off',feature].value_counts().sort_index().tolist()

    chi2, p, dof, expctd = chi2_contingency([dist1,dist2])

    print(feature+':')

    print("chi-square test statistic:", chi2)

    print("p-value", p, '\n')
for i in list_float:

    run_KS_test(i)
df.info()

list_float = df.select_dtypes(exclude=['object']).columns
list_float
fig, ax = plt.subplots(figsize=(15,10))         # Sample figsize in inches

cm_df = sns.heatmap(df[list_float].corr(),annot=True, fmt = ".2f", cmap = "coolwarm", ax=ax)
cor = df[list_float].corr()

cor.loc[:,:] = np.tril(cor, k=-1) # below main lower triangle of an array

cor = cor.stack()

cor[(cor > 0.1) | (cor < -0.1)]
df[["installment","loan_amnt","mo_sin_old_rev_tl_op","earliest_cr_line","total_acc","open_acc", "pub_rec_bankruptcies", "pub_rec"]].isnull().any()
list_linear = ['installment', 'mo_sin_old_rev_tl_op','total_acc','pub_rec_bankruptcies']
linear_corr = pd.DataFrame()
# Pearson coefficients

for col in df[list_float].columns:

    linear_corr.loc[col, 'pearson_corr'] = df[col].corr(df['Charged_Off'])

linear_corr['abs_pearson_corr'] = abs(linear_corr['pearson_corr'])
linear_corr.reset_index(inplace=True)

#linear_corr.rename(columns={'index':'variable'}, inplace=True)
linear_corr
df.shape
missing_values_table(df)
dummy_list =['sub_grade','home_ownership','verification_status','purpose','addr_state','initial_list_status','application_type']
df[dummy_list].isnull().any()
df = pd.get_dummies(df, columns=dummy_list, drop_first=True)
drop_cols('revol_util')

drop_cols('grade')

df['int_round'].describe()
X_train = df.drop(['Charged_Off'], axis=1)

y_train = df.loc[:, 'Charged_Off']



X_test = df.drop(['Charged_Off'], axis=1)

y_test = df['Charged_Off']


X_all = df.drop(['Charged_Off'], axis=1)

Y_all = df.loc[:, 'Charged_Off']
from sklearn.linear_model import SGDClassifier

from sklearn.pipeline import Pipeline


# CV model with Kfold stratified cross val

kfold = 3

random_state = 42


pipeline_sgdlr = Pipeline([

    ('model', SGDClassifier(loss='log', max_iter=1000, tol=1e-3, random_state=random_state, warm_start=False))

])


param_grid_sgdlr  = {

    'model__alpha': [10**-5, 10**-1, 10**2],

    'model__penalty': ['l1', 'l2']

}
grid_sgdlr = GridSearchCV(estimator=pipeline_sgdlr, param_grid=param_grid_sgdlr, scoring='roc_auc', n_jobs=-1, pre_dispatch='2*n_jobs', cv=kfold, verbose=1, return_train_score=False)
grid_sgdlr.fit(X_train, y_train)

sgdlr_estimator = grid_sgdlr.best_estimator_

print('Best score: ', grid_sgdlr.best_score_)

print('Best parameters set: \n', grid_sgdlr.best_params_)

y_pred_sgdlr = sgdlr_estimator.predict(X_test)

y_prob_sgdlr = sgdlr_estimator.predict_proba(X_test)[:,1]

y_train_pred_sgdlr = sgdlr_estimator.predict(X_train)

y_train_prob_sgdlr = sgdlr_estimator.predict_proba(X_train)[:,1]
LRmodel_l2 = SGDClassifier(loss='log', max_iter=1000, tol=1e-3, random_state=random_state, warm_start=False, alpha=0.1, penalty='l2')

LRmodel_l2.fit(X_train, y_train)

temp = sorted(zip(np.round(LRmodel_l2.coef_.reshape(-1),3), X_train.columns.values), key=lambda x: -abs(x[0]))

weight = [x for x, _ in temp]

feature = [x for _, x in temp]

print("Logistic Regression (L2) Coefficients: Top 10")

pd.DataFrame({'weight': weight}, index = feature).head(10)

from sklearn.feature_selection import RFE

rfe_l2 = RFE(LRmodel_l2, n_features_to_select=1) # If None, half of the features are selected.

rfe_l2.fit(X_train, y_train)

temp = sorted(zip(map(lambda x: round(x, 4), rfe_l2.ranking_), X_train.columns))

rank = [x for x, _ in temp]

feature = [x for _, x in temp]

print("Logistic Regression (L2) RFE Result: Top 10")

pd.DataFrame({'rank': rank}, index = feature).head(10)

from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_jobs=-1, n_estimators=100, random_state=random_state,max_features='sqrt') 

param_grid_rf = {

    #'n_estimators': [50, 100], 

    'class_weight': [{0:1, 1:1}], #'model__class_weight': [{0:1, 1:1}, {0:1,1:2}, {0:1, 1:3}, {0:1, 1:4}]

    #'model__min_samples_split':[2,3]

    #'model__max_features':[2,3,4,5],

    #"model__max_depth":range(8,13)

}

grid_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, scoring='roc_auc',n_jobs=-1,pre_dispatch='2*n_jobs', cv=kfold, verbose=1, return_train_score=False)

grid_rf.fit(X_train, y_train)

rf_estimator = grid_rf.best_estimator_

print('Best score: ', grid_rf.best_score_)

print('Best parameters set: \n', grid_rf.best_params_)

y_pred_rf = rf_estimator.predict(X_test)

y_prob_rf = rf_estimator.predict_proba(X_test)[:,1]

y_train_pred_rf = rf_estimator.predict(X_train)

y_train_prob_rf = rf_estimator.predict_proba(X_train)[:,1]


from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_selection import RFECV

from sklearn import decomposition

#chaining a PCA and a knn

pipeline_knn = Pipeline([

    ('pca', decomposition.PCA()),

    ('model', KNeighborsClassifier(n_jobs=-1))   

])



pipeline_knn2 = Pipeline([

    ('lda', LinearDiscriminantAnalysis()),

    ('model', KNeighborsClassifier(n_jobs=-1))   

])



param_grid_knn = {

    'pca__n_components': range(3,6),

    'model__n_neighbors': [5, 25, 125]

}

param_grid_knn2 = {

    'lda__n_components': range(3,6),

    'model__n_neighbors': [5, 25, 125]

}

grid_knn = GridSearchCV(estimator=pipeline_knn, param_grid=param_grid_knn, scoring='roc_auc', n_jobs=-1, pre_dispatch='2*n_jobs', cv=kfold, verbose=1, return_train_score=False)

grid_knn2 = GridSearchCV(estimator=pipeline_knn2, param_grid=param_grid_knn2, scoring='roc_auc', n_jobs=-1, pre_dispatch='2*n_jobs', cv=kfold, verbose=1, return_train_score=False)



%%time

grid_knn2.fit(X_train, y_train)



knn_estimator2 = grid_knn2.best_estimator_

print('Best score: ', grid_knn2.best_score_)

print('Best parameters set: \n', grid_knn2.best_params_)



y_pred_knn = knn_estimator2.predict(X_test)

y_prob_knn = knn_estimator2.predict_proba(X_test)[:,1]





y_train_pred_knn = knn_estimator2.predict(X_train)

y_train_prob_knn = knn_estimator2.predict_proba(X_train)[:,1]



# plot the AUROC scores on the training dataset. 

auroc_means = [grid_sgdlr.best_score_, grid_rf.best_score_,grid_knn2.best_score_]

auroc_res = pd.DataFrame({"AUROC":auroc_means,"Algorithm":["SGD Logistic Regression",

"RandomForest","KNeighboors"]})



g = sns.barplot("AUROC","Algorithm", data = auroc_res, palette="Set3",orient = "h")

g.set_xlabel("ROC_AUC score")

g = g.set_title("Cross validation AUROC scores")

auroc_res
def evaluation(X_train, X_test, Y_train, Y_test, Y_train_pred, Y_train_prob, Y_pred, Y_prob):

    print("--- ROC AUC ---")

    print("Training Set:", roc_auc_score(Y_train, Y_train_prob))

    print("Test Set:", roc_auc_score(Y_test, Y_prob))

    

    print("\n--- Accuracy ---")

    print("Training Set:", accuracy_score(Y_train, Y_train_pred))

    print("Test Set:", accuracy_score(Y_test, Y_pred))



    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()

    print("\n--- Confusion Matrix ---")

    print("True Positive:", tp)

    print("False Negative:", fn)

    print("True Negative:", tn)

    print("False Positive:", fp)



    print("\n--- Precision ---")

    print("Training Set:", precision_score(Y_train, Y_train_pred))

    print("Test Set:", precision_score(Y_test, Y_pred))



    print("\n--- Recall ---")

    print("Training Set:", recall_score(Y_train, Y_train_pred))

    print("Test Set:", recall_score(Y_test, Y_pred))



    print("\n--- F1 Score ---")

    print("Training Set:", f1_score(Y_train, Y_train_pred))

    print("Test Set:", f1_score(Y_test, Y_pred))

      

def plot_ROC(X_test, Y_test, Y_prob):

    

    #Y_prob = model.predict_proba(X_test)[:,1]

    fpr, tpr, thresh = roc_curve(Y_test, Y_prob, pos_label=1)

    roc_auc = roc_auc_score(Y_test, Y_prob)

    # These are the points at threshold = 0.1~0.5

    x1 = fpr[(thresh <= 0.5) & (thresh >= 0.1)] 

    x2 = tpr[(thresh <= 0.5) & (thresh >= 0.1)]

    

    

    fig = plt.figure()

    plt.plot(fpr, tpr, color='r', lw=2)

    plt.plot([0, 1], [0, 1], color='b', lw=2, linestyle='--')

    plt.plot(x1, x2, color='k', lw=3, label='threshold = 0.1 ~ 0.5')

    plt.xlim([-0.05, 1.05])

    plt.ylim([-0.05, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('ROC Curve (Area = {:.2f})'.format(roc_auc))

    plt.legend(loc="lower right")

    plt.show()
# bank

score_bank = sum(y_test == 0) - sum(y_test == 1)



# my Logistic regression model

tn, fp, fn, tp = confusion_matrix(y_test, grid_sgdlr.predict(X_test)).ravel()

score_lr = tn - fn



print("The bank scores {} points".format(score_bank))

print("The Logistic regression model scores {} points".format(score_lr))