import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



loan = pd.read_csv("../input/credit_assessment_data.csv", sep=",")

loan.info()
# let's look at the first few rows of the df

loan.head()
# Looking at all the column names

loan.columns
# summarising number of missing values in each column

loan.isnull().sum()
# percentage of missing values in each column

round(loan.isnull().sum()/len(loan.index), 2)*100
# removing the columns having more than 90% missing values

missing_columns = loan.columns[100*(loan.isnull().sum()/len(loan.index)) > 90]

print(missing_columns)
loan = loan.drop(missing_columns, axis=1)

print(loan.shape)



# summarise number of missing values again

100*(loan.isnull().sum()/len(loan.index))
# There are now 2 columns having approx 32 and 64% missing values - 

# description and months since last delinquent



# let's have a look at a few entries in the columns

loan.loc[:, ['desc', 'mths_since_last_delinq']].head()
# dropping the two columns

loan = loan.drop(['desc', 'mths_since_last_delinq'], axis=1)
# summarise number of missing values again

100*(loan.isnull().sum()/len(loan.index))
# missing values in rows

loan.isnull().sum(axis=1)
# checking whether some rows have more than 5 missing values

len(loan[loan.isnull().sum(axis=1) > 5].index)
loan.info()
# The column int_rate is character type, let's convert it to float

loan['int_rate'] = loan['int_rate'].apply(lambda x: pd.to_numeric(x.split("%")[0]))
# checking the data types

loan.info()
# also, lets extract the numeric part from the variable employment length



# first, let's drop the missing values from the column (otherwise the regex code below throws error)

loan = loan[~loan['emp_length'].isnull()]



# using regular expression to extract numeric values from the string

import re

loan['emp_length'] = loan['emp_length'].apply(lambda x: re.findall('\d+', str(x))[0])



# convert to numeric

loan['emp_length'] = loan['emp_length'].apply(lambda x: pd.to_numeric(x))
# looking at type of the columns again

loan.info()
behaviour_var =  [

  "delinq_2yrs",

  "earliest_cr_line",

  "inq_last_6mths",

  "open_acc",

  "pub_rec",

  "revol_bal",

  "revol_util",

  "total_acc",

  "out_prncp",

  "out_prncp_inv",

  "total_pymnt",

  "total_pymnt_inv",

  "total_rec_prncp",

  "total_rec_int",

  "total_rec_late_fee",

  "recoveries",

  "collection_recovery_fee",

  "last_pymnt_d",

  "last_pymnt_amnt",

  "last_credit_pull_d",

  "application_type"]

behaviour_var
# let's now remove the behaviour variables from analysis

df = loan.drop(behaviour_var, axis=1)

df.info()
# also, we will not be able to use the variables zip code, address, state etc.

# the variable 'title' is derived from the variable 'purpose'

# thus let get rid of all these variables as well



df = df.drop(['title', 'url', 'zip_code', 'addr_state'], axis=1)
df['loan_status'] = df['loan_status'].astype('category')

df['loan_status'].value_counts()
# filtering only fully paid or charged-off

df = df[df['loan_status'] != 'Current']

df['loan_status'] = df['loan_status'].apply(lambda x: 0 if x=='Fully Paid' else 1)



# converting loan_status to integer type

df['loan_status'] = df['loan_status'].apply(lambda x: pd.to_numeric(x))



# summarising the values

df['loan_status'].value_counts()
# default rate

round(np.mean(df['loan_status']), 2)
# plotting default rates across grade of the loan

sns.barplot(x='grade', y='loan_status', data=df)

plt.show()
# lets define a function to plot loan_status across categorical variables

def plot_cat(cat_var):

    sns.barplot(x=cat_var, y='loan_status', data=df)

    plt.show()

    
# compare default rates across grade of loan

plot_cat('grade')
# term: 60 months loans default more than 36 months loans

plot_cat('term')
# sub-grade: as expected - A1 is better than A2 better than A3 and so on 

plt.figure(figsize=(16, 6))

plot_cat('sub_grade')
# home ownership: not a great discriminator

plot_cat('home_ownership')
# verification_status: surprisingly, verified loans default more than not verifiedb

plot_cat('verification_status')
# purpose: small business loans defualt the most, then renewable energy and education

plt.figure(figsize=(16, 6))

plot_cat('purpose')
# let's also observe the distribution of loans across years

# first lets convert the year column into datetime and then extract year and month from it

df['issue_d'].head()
from datetime import datetime

df['issue_d'] = df['issue_d'].apply(lambda x: datetime.strptime(x, '%b-%y'))

# extracting month and year from issue_date

df['month'] = df['issue_d'].apply(lambda x: x.month)

df['year'] = df['issue_d'].apply(lambda x: x.year)





# let's first observe the number of loans granted across years

df.groupby('year').year.count()
# number of loans across months

df.groupby('month').month.count()
# lets compare the default rates across years

# the default rate had suddenly increased in 2011, inspite of reducing from 2008 till 2010

plot_cat('year')
# comparing default rates across months: not much variation across months

plt.figure(figsize=(16, 6))

plot_cat('month')
# loan amount: the median loan amount is around 10,000

sns.distplot(df['loan_amnt'])

plt.show()
# binning loan amount

def loan_amount(n):

    if n < 5000:

        return 'low'

    elif n >=5000 and n < 15000:

        return 'medium'

    elif n >= 15000 and n < 25000:

        return 'high'

    else:

        return 'very high'

        

df['loan_amnt'] = df['loan_amnt'].apply(lambda x: loan_amount(x))

df['loan_amnt'].value_counts()
# let's compare the default rates across loan amount type

# higher the loan amount, higher the default rate

plot_cat('loan_amnt')
# let's also convert funded amount invested to bins

df['funded_amnt_inv'] = df['funded_amnt_inv'].apply(lambda x: loan_amount(x))
# funded amount invested

plot_cat('funded_amnt_inv')
# lets also convert interest rate to low, medium, high

# binning loan amount

def int_rate(n):

    if n <= 10:

        return 'low'

    elif n > 10 and n <=15:

        return 'medium'

    else:

        return 'high'

    

    

df['int_rate'] = df['int_rate'].apply(lambda x: int_rate(x))
# comparing default rates across rates of interest

# high interest rates default more, as expected

plot_cat('int_rate')
# debt to income ratio

def dti(n):

    if n <= 10:

        return 'low'

    elif n > 10 and n <=20:

        return 'medium'

    else:

        return 'high'

    



df['dti'] = df['dti'].apply(lambda x: dti(x))
# comparing default rates across debt to income ratio

# high dti translates into higher default rates, as expected

plot_cat('dti')
# funded amount

def funded_amount(n):

    if n <= 5000:

        return 'low'

    elif n > 5000 and n <=15000:

        return 'medium'

    else:

        return 'high'

    

df['funded_amnt'] = df['funded_amnt'].apply(lambda x: funded_amount(x))
plot_cat('funded_amnt')

# installment

def installment(n):

    if n <= 200:

        return 'low'

    elif n > 200 and n <=400:

        return 'medium'

    elif n > 400 and n <=600:

        return 'high'

    else:

        return 'very high'

    

df['installment'] = df['installment'].apply(lambda x: installment(x))
# comparing default rates across installment

# the higher the installment amount, the higher the default rate

plot_cat('installment')
# annual income

def annual_income(n):

    if n <= 50000:

        return 'low'

    elif n > 50000 and n <=100000:

        return 'medium'

    elif n > 100000 and n <=150000:

        return 'high'

    else:

        return 'very high'



df['annual_inc'] = df['annual_inc'].apply(lambda x: annual_income(x))
# annual income and default rate

# lower the annual income, higher the default rate

plot_cat('annual_inc')
# employment length

# first, let's drop the missing value observations in emp length

df = df[~df['emp_length'].isnull()]



# binning the variable

def emp_length(n):

    if n <= 1:

        return 'fresher'

    elif n > 1 and n <=3:

        return 'junior'

    elif n > 3 and n <=7:

        return 'senior'

    else:

        return 'expert'



df['emp_length'] = df['emp_length'].apply(lambda x: emp_length(x))
# emp_length and default rate

# not much of a predictor of default

plot_cat('emp_length')
# purpose: small business loans defualt the most, then renewable energy and education

plt.figure(figsize=(16, 6))

plot_cat('purpose')
# lets first look at the number of loans for each type (purpose) of the loan

# most loans are debt consolidation (to repay otehr debts), then credit card, major purchase etc.

plt.figure(figsize=(16, 6))

sns.countplot(x='purpose', data=df)

plt.show()
# filtering the df for the 4 types of loans mentioned above

main_purposes = ["credit_card","debt_consolidation","home_improvement","major_purchase"]

df = df[df['purpose'].isin(main_purposes)]

df['purpose'].value_counts()
# plotting number of loans by purpose 

sns.countplot(x=df['purpose'])

plt.show()
# let's now compare the default rates across two types of categorical variables

# purpose of loan (constant) and another categorical variable (which changes)



plt.figure(figsize=[10, 6])

sns.barplot(x='term', y="loan_status", hue='purpose', data=df)

plt.show()

# lets write a function which takes a categorical variable and plots the default rate

# segmented by purpose 



def plot_segmented(cat_var):

    plt.figure(figsize=(10, 6))

    sns.barplot(x=cat_var, y='loan_status', hue='purpose', data=df)

    plt.show()



    

plot_segmented('term')
# grade of loan

plot_segmented('grade')
# home ownership

plot_segmented('home_ownership')
# year

plot_segmented('year')
# emp_length

plot_segmented('emp_length')
# loan_amnt: same trend across loan purposes

plot_segmented('loan_amnt')
# interest rate

plot_segmented('int_rate')
# installment

plot_segmented('installment')
# debt to income ratio

plot_segmented('dti')
# annual income

plot_segmented('annual_inc')
# variation of default rate across annual_inc

df.groupby('annual_inc').loan_status.mean().sort_values(ascending=False)
# one can write a function which takes in a categorical variable and computed the average 

# default rate across the categories

# It can also compute the 'difference between the highest and the lowest default rate' across the 

# categories, which is a decent metric indicating the effect of the varaible on default rate



def diff_rate(cat_var):

    default_rates = df.groupby(cat_var).loan_status.mean().sort_values(ascending=False)

    return (round(default_rates, 2), round(default_rates[0] - default_rates[-1], 2))



default_rates, diff = diff_rate('annual_inc')

print(default_rates) 

print(diff)

# filtering all the object type variables

df_categorical = df.loc[:, df.dtypes == object]

df_categorical['loan_status'] = df['loan_status']



# Now, for each variable, we can compute the incremental diff in default rates

print([i for i in df.columns])
# storing the diff of default rates for each column in a dict

d = {key: diff_rate(key)[1]*100 for key in df_categorical.columns if key != 'loan_status'}

print(d)
df.shape
df['loan_status'].value_counts()
df = df.dropna()
df.isnull().sum()
df.head()
df.info()
# drop id,member_id,emp_title

df.drop(['id','member_id','emp_title'],axis=1,inplace=True)
# import required libraries

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import FeatureUnion

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score

from imblearn.metrics import sensitivity_specificity_support

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC
le = preprocessing.LabelEncoder()
df = df.apply(le.fit_transform)
df.head()
X = df.drop('loan_status',axis=1)

y = df.loan_status

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25, random_state = 4, stratify = y)
# print shapes of train and test sets

X_train.shape

y_train.shape

X_test.shape

y_test.shape
import statsmodels.api as sm
# Logistic regression model

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())

logm1.fit().summary()
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
from sklearn.feature_selection import RFE

rfe = RFE(logreg, 15)             # running RFE with 15 variables as output

rfe = rfe.fit(X_train, y_train)
rfe.support_
list(zip(X_train.columns, rfe.support_, rfe.ranking_))
col = X_train.columns[rfe.support_]
X_train.columns[~rfe.support_]
X_train_sm = sm.add_constant(X_train[col])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
# Getting the predicted values on the train set

y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]
y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'loan_status':y_train.values, 'loan_status_prob':y_train_pred})

y_train_pred_final['CustID'] = y_train.index

y_train_pred_final.head()
y_train_pred_final['predicted'] = y_train_pred_final.loan_status_prob.map(lambda x: 1 if x > 0.3 else 0)



# Let's see the head

y_train_pred_final.head()
from sklearn import metrics
# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.loan_status, y_train_pred_final.predicted )

print(confusion)
# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final.loan_status, y_train_pred_final.predicted))
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
col = col.drop('issue_d',1)

col
# Let's re-run the model using the selected variables

X_train_sm = sm.add_constant(X_train[col])

logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm3.fit()

res.summary()
y_train_pred = res.predict(X_train_sm).values.reshape(-1)
y_train_pred[:10]
y_train_pred_final['loan_status_prob'] = y_train_pred
# Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0

y_train_pred_final['predicted'] = y_train_pred_final.loan_status_prob.map(lambda x: 1 if x > 0.5 else 0)

y_train_pred_final.head()
# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final.loan_status, y_train_pred_final.predicted))
vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Let's drop TotalCharges since it has a high VIF

col = col.drop('sub_grade')

col
# Let's re-run the model using the selected variables

X_train_sm = sm.add_constant(X_train[col])

logm4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm4.fit()

res.summary()
y_train_pred[:10]
y_train_pred_final['loan_status_prob'] = y_train_pred
# Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0

y_train_pred_final['predicted'] = y_train_pred_final.loan_status_prob.map(lambda x: 1 if x > 0.5 else 0)

y_train_pred_final.head()
# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final.loan_status, y_train_pred_final.predicted))
vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Let's drop TotalCharges since it has a high VIF

col = col.drop('year')

col
# Let's re-run the model using the selected variables

X_train_sm = sm.add_constant(X_train[col])

logm4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm4.fit()

res.summary()
y_train_pred = res.predict(X_train_sm).values.reshape(-1)
y_train_pred[:10]
y_train_pred_final['loan_status_prob'] = y_train_pred
# Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0

y_train_pred_final['predicted'] = y_train_pred_final.loan_status_prob.map(lambda x: 1 if x > 0.5 else 0)

y_train_pred_final.head()
# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final.loan_status,y_train_pred_final.predicted))
TP = confusion[1,1] # true positive 

TN = confusion[0,0] # true negatives

FP = confusion[0,1] # false positives

FN = confusion[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model

TP / float(TP+FN)
# Let's see the sensitivity of our logistic regression model

TP / float(TP+FN)
# Let us calculate specificity

TN / float(TN+FP)
# Calculate false postive rate - predicting churn when customer does not have churned

print(FP/ float(TN+FP))
# positive predictive value 

print (TP / float(TP+FP))
# Negative predictive value

print (TN / float(TN+ FN))
def draw_roc( actual, probs ):

    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,

                                              drop_intermediate = False )

    auc_score = metrics.roc_auc_score( actual, probs )

    plt.figure(figsize=(5, 5))

    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()



    return None
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.loan_status, y_train_pred_final.loan_status_prob, drop_intermediate = False )
draw_roc(y_train_pred_final.loan_status, y_train_pred_final.loan_status_prob)
# Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_train_pred_final[i]= y_train_pred_final.loan_status_prob.map(lambda x: 1 if x > i else 0)

y_train_pred_final.head()
# apply pca to train data

pca = Pipeline([('scaler', StandardScaler()), ('pca', PCA())])
pca.fit(X_train)

churn_pca = pca.fit_transform(X_train)
# extract pca model from pipeline

pca = pca.named_steps['pca']



# look at explainded variance of PCA components

print(pd.Series(np.round(pca.explained_variance_ratio_.cumsum(), 4)*100))
# plot feature variance

features = range(pca.n_components_)

cumulative_variance = np.round(np.cumsum(pca.explained_variance_ratio_)*100, decimals=4)

plt.figure(figsize=(175/20,100/20)) # 100 elements on y-axis; 175 elements on x-axis; 20 is normalising factor

plt.plot(cumulative_variance)
# create pipeline

PCA_VARS = 10

steps = [('scaler', StandardScaler()),

         ("pca", PCA(n_components=PCA_VARS)),

         ("logistic", LogisticRegression(class_weight='balanced'))

        ]

pipeline = Pipeline(steps)
# fit model

pipeline.fit(X_train, y_train)



# check score on train data

pipeline.score(X_train, y_train)
# predict churn on test data

y_pred = pipeline.predict(X_test)



# create onfusion matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)



# check sensitivity and specificity

sensitivity, specificity, _ = sensitivity_specificity_support(y_test, y_pred, average='binary')

print("Sensitivity: \t", round(sensitivity, 2), "\n", "Specificity: \t", round(specificity, 2), sep='')



# check area under curve

y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

print("AUC:    \t", round(roc_auc_score(y_test, y_pred_prob),2))
# class imbalance

y_train.value_counts()/y_train.shape
# PCA

pca = PCA()



# logistic regression - the class weight is used to handle class imbalance - it adjusts the cost function

logistic = LogisticRegression(class_weight={0:0.1, 1: 0.9})



# create pipeline

steps = [("scaler", StandardScaler()), 

         ("pca", pca),

         ("logistic", logistic)

        ]



# compile pipeline

pca_logistic = Pipeline(steps)



# hyperparameter space

params = {'pca__n_components': [10, 15], 'logistic__C': [0.1, 0.5, 1, 2, 3, 4, 5, 10], 'logistic__penalty': ['l1', 'l2']}



# create 5 folds

folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 4)



# create gridsearch object

model = GridSearchCV(estimator=pca_logistic, cv=folds, param_grid=params, scoring='roc_auc', n_jobs=-1, verbose=1)
# fit model

model.fit(X_train, y_train)
# cross validation results

pd.DataFrame(model.cv_results_)
# print best hyperparameters

print("Best AUC: ", model.best_score_)

print("Best hyperparameters: ", model.best_params_)
# predict churn on test data

y_pred = model.predict(X_test)



# create onfusion matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)



# check sensitivity and specificity

sensitivity, specificity, _ = sensitivity_specificity_support(y_test, y_pred, average='binary')

print("Sensitivity: \t", round(sensitivity, 2), "\n", "Specificity: \t", round(specificity, 2), sep='')



# check area under curve

y_pred_prob = model.predict_proba(X_test)[:, 1]

print("AUC:    \t", round(roc_auc_score(y_test, y_pred_prob),2))
# random forest - the class weight is used to handle class imbalance - it adjusts the cost function

forest = RandomForestClassifier(class_weight={0:0.1, 1: 0.9}, n_jobs = -1)



# hyperparameter space

params = {"criterion": ['gini', 'entropy'], "max_features": ['auto', 0.4]}



# create 5 folds

folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 4)



# create gridsearch object

model = GridSearchCV(estimator=forest, cv=folds, param_grid=params, scoring='roc_auc', n_jobs=-1, verbose=1)
# fit model

model.fit(X_train, y_train)
# print best hyperparameters

print("Best AUC: ", model.best_score_)

print("Best hyperparameters: ", model.best_params_)
# predict churn on test data

y_pred = model.predict(X_test)



# create onfusion matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)



# check sensitivity and specificity

sensitivity, specificity, _ = sensitivity_specificity_support(y_test, y_pred, average='binary')

print("Sensitivity: \t", round(sensitivity, 2), "\n", "Specificity: \t", round(specificity, 2), sep='')



# check area under curve

y_pred_prob = model.predict_proba(X_test)[:, 1]

print("AUC:    \t", round(roc_auc_score(y_test, y_pred_prob),2))
# run a random forest model on train data

max_features = int(round(np.sqrt(X_train.shape[1])))    # number of variables to consider to split each node

print(max_features)



rf_model = RandomForestClassifier(n_estimators=100, max_features=max_features, class_weight={0:0.1, 1: 0.9}, oob_score=True, random_state=4, verbose=1)
# fit model

rf_model.fit(X_train, y_train)
# OOB score

rf_model.oob_score_
# predict churn on test data

y_pred = rf_model.predict(X_test)



# create onfusion matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)



# check sensitivity and specificity

sensitivity, specificity, _ = sensitivity_specificity_support(y_test, y_pred, average='binary')

print("Sensitivity: \t", round(sensitivity, 2), "\n", "Specificity: \t", round(specificity, 2), sep='')



# check area under curve

y_pred_prob = rf_model.predict_proba(X_test)[:, 1]

print("ROC:    \t", round(roc_auc_score(y_test, y_pred_prob),2))
# predictors

features = df.drop('loan_status', axis=1).columns



# feature_importance

importance = rf_model.feature_importances_



# create dataframe

feature_importance = pd.DataFrame({'variables': features, 'importance_percentage': importance*100})

feature_importance = feature_importance[['variables', 'importance_percentage']]



# sort features

feature_importance = feature_importance.sort_values('importance_percentage', ascending=False).reset_index(drop=True)

print("Sum of importance=", feature_importance.importance_percentage.sum())

feature_importance