import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sbs
df = pd.read_csv('../input/loan.csv')
df.head()
df.shape
NaColumn = df.isnull().sum()

#print(NaColumn)

NaColumn = NaColumn[NaColumn.values>(0.3*len(df))]

#print(NaColumn)

plt.figure(figsize=(20,4))

NaColumn.plot(kind='bar')

plt.title('List of Columns & NA counts where NA values are more than 30%')

plt.show()
def removeNulls(dataframe, axis = 1, percent = 0.3):

    df = dataframe.copy()

    ishape=df.shape

    if axis==0:

        rownames = df.transpose().isnull.sum()

        print(rownnames)

        rownames =list(rownames[rownames.value>(percent*len(df))].index)

        df.drop(df.index[rownames],inplace=True)

        print('Number Of Rows drop',len(rownames))

    else:

        colnames = (df.isnull().sum()/len(df))

        colnames = list(colnames[colnames.values>=percent].index)

        df.drop(labels = colnames,axis =1,inplace=True)        

        print("Number of Columns dropped\t: ",len(colnames))

    print('\n Old Data Shape',ishape,'\n New Data shape',df.shape)

    return df

loan = removeNulls(df,axis=1,percent=0.3)
unique = loan.nunique()

unique = unique[unique.values == 1]
loan.drop(labels = list(unique.index), axis =1, inplace=True)

print("So now left column will be ",loan.shape)
loan.emp_length.unique()
loan.emp_length.fillna('0',inplace=True)

loan.emp_length.replace(['n/a'],'Self-Employed',inplace = True)

print(loan.emp_length.unique())
NoRequireColumns = ['url','id','zip_code']

loan.drop(labels=NoRequireColumns, axis =1, inplace= True)

print(loan.shape)
numeric_columns = ['loan_amnt', 'funded_amnt','funded_amnt_inv','installment','int_rate','annual_inc','dti']

loan[numeric_columns] = loan[numeric_columns].apply(pd.to_numeric)
loan.head(5)
loan.purpose.value_counts()
del_loan_purpose = (loan.purpose.value_counts()*100)/len(loan)

print(del_loan_purpose)

del_loan_purpose = del_loan_purpose[(del_loan_purpose<0.75) | (del_loan_purpose.index == 'other')]

loan.drop(labels= loan[loan.purpose.isin(del_loan_purpose.index)].index, inplace = True)

print(loan.shape)

print(loan.purpose.unique())
loan.loan_status.unique()
loanStatus = sbs.countplot(x='loan_status', data = loan)
loan['loan_income_ratio'] = loan['loan_amnt']/loan['annual_inc']

#print(loan['loan_income_ratio'])
loan['issue_month'],loan['issue_year'] = loan['issue_d'].str.split('-',1).str

loan[['isse_d'],['issue_month'],['issue_year']].head()
month_order = ['jan','feb','march','may','june','july','aug','sep','oct','nov','dec']

loan['issue_month'] = pd.Categorical(loan['issue_month'], categories = month_order, ordered=True)

groups = [0,5000,10000,20000,30000,40000]

slots = ['0-5000', '5000-10000', '10000-20000', '20000-30000', '30000 and above']

loan['loan_amnt_range'] = pd.cut(loan['loan_amnt'],groups, labels = slots)

print(loan['loan_amnt_range'].unique())