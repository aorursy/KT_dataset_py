import pandas as pd

import seaborn as sns

from scipy.stats import chi2_contingency

from statsmodels.stats import weightstats as stests

import matplotlib.pyplot as plt

df = pd.read_csv('../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')

df.head()
# dataset comprises of 614 observations and 13 charateristics

df.shape
df.info()
freq_tbl1 = df.groupby(['Gender','Married','Education','Self_Employed'])

freq_tbl1.size()
freq_tbl2 = df.groupby(['Credit_History','Property_Area','Loan_Amount_Term'])

freq_tbl2.size()
df.Loan_Status.value_counts()
df[['ApplicantIncome','CoapplicantIncome','LoanAmount']].describe()
# indeed we found there are quite a few outliers

sns.boxplot(df['ApplicantIncome'])

sns.despine()
IQR_app = df.ApplicantIncome.quantile(0.75) - df.ApplicantIncome.quantile(0.25)

upper_limit_app = df.ApplicantIncome.quantile(0.75) + (IQR_app*1.5)

upper_limit_extreme_app = df.ApplicantIncome.quantile(0.75) + (IQR_app*2)

upper_limit_app, upper_limit_extreme_app
outlier_count_app = len(df[(df['ApplicantIncome'] > upper_limit_app)])

outlier_count_app
# the ApplicantIncome distribution is right skewed 

fig=plt.figure()

ax=fig.add_subplot(1,1,1)

ax.hist(df['ApplicantIncome'], bins=30)

plt.title('ApplicantIncome Distribution')

plt.xlabel('Applicant Income')

plt.ylabel('Frequency')

plt.show()
sns.boxplot(df['CoapplicantIncome'])

sns.despine()
IQR_coapp = df.CoapplicantIncome.quantile(0.75) - df.CoapplicantIncome.quantile(0.25)

upper_limit_coapp = df.CoapplicantIncome.quantile(0.75) + (IQR_coapp*1.5)

upper_limit_extreme_coapp = df.CoapplicantIncome.quantile(0.75) + (IQR_coapp*2)

upper_limit_coapp, upper_limit_extreme_coapp
outlier_count_coapp = len(df[(df['CoapplicantIncome'] > upper_limit_coapp)])

outlier_count_coapp
sns.boxplot(df['LoanAmount'])

sns.despine()
IQR_loanAmt = df.LoanAmount.quantile(0.75) - df.LoanAmount.quantile(0.25)

upper_limit_loanAmt = df.LoanAmount.quantile(0.75) + (IQR_loanAmt*1.5)

upper_limit_extreme_loanAmt = df.LoanAmount.quantile(0.75) + (IQR_loanAmt*2)

upper_limit_loanAmt, upper_limit_extreme_loanAmt
outlier_count_loanAmt = len(df[(df['LoanAmount'] > upper_limit_loanAmt)])

outlier_count_loanAmt
# the LoanAmount distribution is right skewed 

fig=plt.figure()

ax=fig.add_subplot(1,1,1)

ax.hist(df['LoanAmount'], bins=100)

plt.title('Loan Amount Distribution')

plt.xlabel('Loan Amount')

plt.ylabel('Frequency')

plt.show()
tbl = pd.crosstab(index = df['Loan_Status'], columns = df['Gender'])

tbl.index = ['Not Approved','Approved']

tbl
# performing the test using Python

# frequency table without missing values

stat, p, dof, expected = chi2_contingency(tbl)

alpha = 0.05

print("p value is " + str(p))

if p <= alpha:

    print('Dependent (reject H_0)')

else:

    print('Independent (H_0 holds true)')
# How would the table look like if we include counts of missing-value obersvations?

tbl2 = tbl

df_new = df[df['Gender'].isnull()]

tbl_tp = df_new.groupby('Loan_Status')

tbl_tp.size()
tbl2['NaN_val'] = [5,8]

tbl2
tbl2.plot.bar(xlabel = 'Loan Status', rot = 0)
# performing the test using Python

# frequency table with missing values

stat, p, dof, expected = chi2_contingency(tbl2)

alpha = 0.05

print("p value is " + str(p))

if p <= alpha:

    print('Dependent (reject H_0)')

else:

    print('Independent (H_0 holds true)')
tbl_marrd = pd.crosstab(index = df['Loan_Status'], columns = df['Married'])

tbl_marrd.index = ['Not Approved','Approved']

tbl_marrd.columns = ['Not Married','Married']

tbl_marrd
stat, p, dof, exptected = chi2_contingency(tbl_marrd)

alpha = 0.05

print("p value is " + str(p))

if p <= alpha:

    print('Dependent (reject H_0)')

else:

    print('Independent (H_0 holds true)')
tbl_ed = pd.crosstab(df['Loan_Status'],df['Education'])

tbl_ed.index = ['Not Approved','Approved']

tbl_ed
tbl_ed.plot.bar(xlabel = 'Loan Status', rot = 0)
stat, p, dof, exptected = chi2_contingency(tbl_ed)

alpha = 0.05

print("p value is " + str(p))

if p <= alpha:

    print('Dependent (reject H_0)')

else:

    print('Independent (H_0 holds true)')
tbl_emp = pd.crosstab(index = df['Loan_Status'], columns = df['Self_Employed'])

tbl_emp.index = ['Not Approved','Approved']

tbl_emp
tbl_emp.plot.bar(xlabel = 'Loan Status',rot = 0)
stat, p, dof, expected = chi2_contingency(tbl_emp)

alpha = 0.05

print("p value is " + str(p))

if p <= alpha:

    print('Dependent (reject H_0)')

else:

    print('Independent (H_0 holds true)')
tbl_term = pd.crosstab(index = df['Loan_Status'], columns = df['Loan_Amount_Term'])

tbl_term.index = ['Not Approved','Approved']

tbl_term
stat, p, dof, exptected = chi2_contingency(tbl_term)

alpha = 0.05

print("p value is " + str(p))

if p <= alpha:

    print('Dependent (reject H_0)')

else:

    print('Independent (H_0 holds true)')
tbl_crt = pd.crosstab(index = df['Loan_Status'], columns = df['Credit_History'])

tbl_crt.index = ['Not Approved','Approved']

tbl_crt.columns = ['Guidelines Not Met', 'Guidelines Met'] 

tbl_crt
tbl_crt.plot.bar(xlabel = 'Loan_Status', rot = 0)
stat, p, dof, expected = chi2_contingency(tbl_crt)

alpha = 0.05

print("p value is " + str(p))

if p <= alpha:

    print('Dependent (reject H_0)')

else:

    print('Independent (H_0 holds true)')
tbl_area = pd.crosstab(index = df['Loan_Status'], columns = df['Property_Area'])

tbl_area.index = ['Not Approved','Approved']

tbl_area
tbl_area.plot.bar(xlabel = 'Loan Status', rot = 0)
stat, p, dof, exptected = chi2_contingency(tbl_area)

alpha = 0.05

print("p value is " + str(p))

if p <= alpha:

    print('Dependent (reject H_0)')

else:

    print('Independent (H_0 holds true)')
df['ApplicantIncome'].groupby(df['Loan_Status']).describe()
# data set of Loan_Status Not Approved

tbl_appN = df[(df['Loan_Status'] == 'N')]

tbl_appN = tbl_appN['ApplicantIncome']

tbl_appN 
# data set of Loan_Status Approved

tbl_appY = df[(df['Loan_Status'] == 'Y')]

tbl_appY = tbl_appY['ApplicantIncome']

tbl_appY
ztest, pval = stests.ztest(tbl_appN,tbl_appY,value = 0, alternative = 'two-sided')

alpha = 0.05

print("p value is " + str(pval))

if pval <= alpha:

    print('Two population means are not equal (reject H_0)')

else:

    print('Two population means are equal (H_0 holds true)')

fig, ax = plt.subplots(figsize = (6,5))

df.boxplot(column = 'ApplicantIncome', by = 'Loan_Status', ax = ax, grid = False)

plt.show()
df['LoanAmount'].groupby(df['Loan_Status']).describe()
tbl_amtN = df[(df['Loan_Status'] == 'N')]

tbl_amtN = tbl_amtN['LoanAmount'].dropna() # we know there are missing values

tbl_amtN 
tbl_amtY = df[(df['Loan_Status'] == 'Y')]

tbl_amtY = tbl_amtY['LoanAmount'].dropna() # missing values dropped

tbl_amtY 
ztest, pval = stests.ztest(tbl_amtN,tbl_amtY,value = 0, alternative = 'two-sided')

alpha = 0.05

print("p value is " + str(pval))

if pval <= alpha:

    print('Two population means are not equal (reject H_0)')

else:

    print('Two population means are equal (H_0 holds true)')
fig, ax = plt.subplots(figsize = (6,5))

df.boxplot(column = 'LoanAmount', by = 'Loan_Status', ax = ax, grid = False)

plt.show()
fig = plt.figure()

ax = fig.add_subplot(1,1,1)

ax.scatter(df['ApplicantIncome'],df['LoanAmount'])

plt.title('ApplicantIncome and LoanAmount Distribution')

plt.xlabel('Applicant Income')

plt.ylabel('Loan Amount')

plt.show()
fig = plt.figure()

ax = fig.add_subplot(1,1,1)

ax.scatter(df['CoapplicantIncome'],df['LoanAmount'])

plt.title('CoapplicantIncome and LoanAmount Distribution')

plt.xlabel('Coapplicant Income')

plt.ylabel('Loan Amount')

plt.show()
tbl_cGender = pd.crosstab(df['Credit_History'],df['Gender'])

stat, p, dof, exptected = chi2_contingency(tbl_cGender)

alpha = 0.05

print("p value is " + str(p))

if p <= alpha:

    print('Dependent (reject H_0)')

else:

    print('Independent (H_0 holds true)')
tbl_cMarried = pd.crosstab(df['Credit_History'],df['Married'])

stat, p, dof, exptected = chi2_contingency(tbl_cMarried)

alpha = 0.05

print("p value is " + str(p))

if p <= alpha:

    print('Dependent (reject H_0)')

else:

    print('Independent (H_0 holds true)')
tbl_cEd = pd.crosstab(df['Credit_History'],df['Education'])

stat, p, dof, exptected = chi2_contingency(tbl_cEd)

alpha = 0.05

print("p value is " + str(p))

if p <= alpha:

    print('Dependent (reject H_0)')

else:

    print('Independent (H_0 holds true)')
tbl_cEmp = pd.crosstab(df['Credit_History'],df['Self_Employed'])

stat, p, dof, exptected = chi2_contingency(tbl_cEmp)

alpha = 0.05

print("p value is " + str(p))

if p <= alpha:

    print('Dependent (reject H_0)')

else:

    print('Independent (H_0 holds true)')
tbl_cArea = pd.crosstab(df['Credit_History'],df['Property_Area'])

stat, p, dof, exptected = chi2_contingency(tbl_cArea)

alpha = 0.05

print("p value is " + str(p))

if p <= alpha:

    print('Dependent (reject H_0)')

else:

    print('Independent (H_0 holds true)')
df['ApplicantIncome'].groupby(df['Credit_History']).describe()
tbl_incomeN = df[(df['Credit_History'] == 0.0)]

tbl_incomeN = tbl_incomeN['ApplicantIncome']

tbl_incomeN 
tbl_incomeY = df[(df['Credit_History'] == 1.0)]

tbl_incomeY = tbl_incomeY['ApplicantIncome']

tbl_incomeY 
ztest, pval = stests.ztest(tbl_incomeN,tbl_incomeY,value = 0, alternative = 'two-sided')

alpha = 0.05

print("p value is " + str(pval))

if pval <= alpha:

    print('Two population means are not equal (reject H_0)')

else:

    print('Two population means are equal (H_0 holds true)')
import numpy as np

from sklearn.impute import SimpleImputer

imr = SimpleImputer(missing_values = np.nan,strategy = 'most_frequent')

imr = imr.fit(df[['Credit_History']])

df['Credit_History'] = imr.transform(df[['Credit_History']]).ravel()

df.info()
# if proceed with dropping NaN before imputation

# the sample size will reduce down to 480

# yet if we drop observations where missing values are found after imputation

# the sample size is reduced to 523 rather

df = df.dropna()

df.info()
df[['ApplicantIncome','CoapplicantIncome','LoanAmount']].describe()
sns.boxplot(df['ApplicantIncome'])

sns.despine()
IQR_1 = df.ApplicantIncome.quantile(0.75) - df.ApplicantIncome.quantile(0.25)

upper_limit_1 = df.ApplicantIncome.quantile(0.75) + (IQR_1*1.5)

upper_limit_extreme_1 = df.ApplicantIncome.quantile(0.75) + (IQR_1*2)

upper_limit_1, upper_limit_extreme_1
outlier_count_1 = len(df[(df['ApplicantIncome'] > upper_limit_1)])

outlier_count_1 

# notice the number of the outliers did not change after missing-value treatment
pd.set_option('mode.chained_assignment', None)

index_1 = df[(df['ApplicantIncome'] >= upper_limit_1)].index

#index_1

df.drop(index_1, inplace = True)

outlier_ct_1 = len(df[(df['ApplicantIncome'] > upper_limit_1)])

outlier_ct_1
sns.boxplot(df['CoapplicantIncome'])

sns.despine()
IQR_2 = df.CoapplicantIncome.quantile(0.75) - df.CoapplicantIncome.quantile(0.25)

upper_limit_2 = df.CoapplicantIncome.quantile(0.75) + (IQR_2*1.5)

upper_limit_extreme_2 = df.CoapplicantIncome.quantile(0.75) + (IQR_2*2)

upper_limit_2, upper_limit_extreme_2
outlier_count_2 = len(df[(df['CoapplicantIncome'] > upper_limit_2)])

outlier_count_2 

# number of outliers reduced from 17 to 16 after missing-value treatment
#pd.set_option('mode.chained_assignment', None)

index_2 = df[(df['CoapplicantIncome'] >= upper_limit_2)].index

#index_2

df.drop(index_2, inplace = True)

outlier_ct_2 = len(df[(df['CoapplicantIncome'] > upper_limit_2)])

outlier_ct_2
sns.boxplot(df['LoanAmount'])

sns.despine()
IQR_3 = df.LoanAmount.quantile(0.75) - df.LoanAmount.quantile(0.25)

upper_limit_3 = df.LoanAmount.quantile(0.75) + (IQR_3*1.5)

upper_limit_extreme_3 = df.LoanAmount.quantile(0.75) + (IQR_3*2)

upper_limit_3, upper_limit_extreme_3
outlier_count_3 = len(df[(df['LoanAmount'] > upper_limit_3)])

outlier_count_3 

# number of outliers reduced from 30 to 23 after missing-value treatment
#pd.set_option('mode.chained_assignment', None)

index_3 = df[(df['LoanAmount'] >= upper_limit_3)].index

#index_3

df.drop(index_3, inplace = True)

outlier_ct_3 = len(df[(df['LoanAmount'] > upper_limit_3)])

outlier_ct_3
#encoding to numeric data type

code_numeric = {'Male':1, 'Female':2,

               'Yes': 1, 'No':2,

                'Graduate':1, 'Not Graduate':2,

                'Urban':1, 'Semiurban':2, 'Rural':3,

                'Y':1, 'N':0,

                '3+':3 }

df = df.applymap(lambda i: code_numeric.get(i) if i in code_numeric else i)

df['Dependents'] = pd.to_numeric(df.Dependents)

df.info()
matrix = np.triu(df.corr())

fig, ax = plt.subplots(figsize = (10,10))

sns.heatmap(df.corr(), annot = True, mask = matrix, linewidths = .5, ax = ax)
m = matrix[:,11]

m = pd.DataFrame(m)

m1 = np.transpose(m)
m1.columns = (df.columns[1:])

m2 = np.transpose(m1)

new_col = ['corr_to_Loan_Status']

m2.columns =new_col
# sort predictor variables by their correlation strength to the Target variable

m2['corr_to_Loan_Status'] = m2['corr_to_Loan_Status'].abs()

m2.sort_values(by = new_col, ascending = False)