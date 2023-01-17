import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

pd.set_option('display.max_columns', None)
%matplotlib inline
df=pd.read_excel('default-payment.xlsx',skiprows=1, index_col=0)
#renaming abnormal columns
df=df.rename(columns = {'default payment next month':'def_payment_nm','PAY_0':'PAY_1'})
# edit unknown attributes
def to_unknown(df, column, range_to_be):
    df[column] = np.where(df[column].isin(range_to_be), df[column], 0)
    
#add square roots, square, and log tranformations to bill, pay, and credit limit
def add_square_root_log(df, bill, pay):
    for i in (bill+pay+["LIMIT_BAL"]):
        df[i+'_sq']=df[i]**2
        df[i+'_rt']=np.where(df[i]==0, 0, df[i]**0.5)
        df[i+"_log"] = np.where(df[i]==0, 0, np.log(df[i]))

# add monthly expense, normalized monthly expense, average expense, average normalized expense
def diff (df, last_status, next_status, last_pay, abs_diff_name, scaled_diff_name):
    for i, j, k, a, b in zip(last_status, next_status, last_pay,abs_diff_name, scaled_diff_name):
        df[a] = df[j] - df[i] + df[k]
        df[b] = df[a]/df['LIMIT_BAL']
    df["avg_abs_diff"] = df[abs_diff_name].mean(axis = 1)
    df['avg_scaled_diff'] = df[scaled_diff_name].mean(axis = 1)

# add monthly leftover credits
def closeness (df, bill, name):
    for i, j in zip(bill, name):
        df[j] = (df['LIMIT_BAL'] - df[i])/df['LIMIT_BAL']

# add average bill, and average payment
def pay_status_avg (df, bill, pay):
    df['avg_bill'] = df[bill].mean(axis = 1)
    df['avg_pay'] = df[pay].mean(axis = 1)

# add sum payment, and sum expenses
def sum_pay_exp (df, pay, exp):
    df['sum_pay'] = df[pay].sum(axis = 1)
    df['sum_exp'] = df[exp].sum(axis = 1)
    
# add monthly pay over monthly status
def pay_over_status (df, bill, pay, name):
    for i, j, k in zip(bill, pay, name):
        df[k] = np.where(df[i]==0, 0, df[j]/df[i])
# add variance to pay amount      
def var_of_pay_amount(df, pay):
    df["pay_amount_var"] = df[pay].var(axis = 1)
    
# add PCA result for bills, pays, expenses, scaled expenses
def PCA_it(df, cols, name):
    x = StandardScaler().fit_transform(df[cols])
    pca = PCA(n_components = 1)
    output = pca.fit_transform(x)
    df['PCA'+ name] = pd.DataFrame(output)

#Resampling dataset: undersampling approach
def resample(df,col):
    rus = RandomUnderSampler(random_state=42)
    y=df[col]
    X=df.drop(col,axis=1)
    X, y = rus.fit_resample(X, y)
    df=pd.concat([X,pd.DataFrame(y)], axis = 1)
    return df

# remove multicollinearity
def correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
    print(col_corr)
# target education range
edu_range=list(range(1,5))
#target marriage range
mar_range=list(range(1,4))
# bill status from Apr to Aug
last_status = ['BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']
# bill status from May to Sep
next_status = ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5']
# pay amount from May to Sep
last_pay = ['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5']
# new column names for monthly expenses from May to Sep
abs_diff_name = ["Sep_abs_expense", "Aug_abs_expense","Jul_abs_expense","Jun_abs_expense","May_abs_expense"]
# new column names for scaled/normalized expenses from May to Sep
scaled_diff_name = ["Sep_scaled_expense", "Aug_scaled_expense","Jul_scaled_expense","Jun_scaled_expense","May_scaled_expense"]
# new column names for monthly leftover credits
closeness_name = ['Sep_leftover_credit','Aug_leftover_credit','Jul_leftover_credit','Jun_leftover_credit',
                  'May_leftover_credit','Apr_leftover_credit' ]
# bill status from Apr to Sep
bill_status = ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']
# pay amount from Apr to Sep
pay_amount = ['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']

# monthly pay over bill name
pay_over_bill_name = ['Sep_pay_over_bill','Aug_pay_over_bill','Jul_pay_over_bill','Jun_pay_over_bill','May_pay_over_bill','Apr_pay_over_bill']

to_unknown(df, "MARRIAGE", mar_range)
to_unknown(df, "EDUCATION", edu_range)

add_square_root_log(df,bill_status, pay_amount)

diff(df,last_status, next_status, last_pay, abs_diff_name, scaled_diff_name)

closeness(df, bill_status,closeness_name)

pay_status_avg (df, bill_status, pay_amount)

sum_pay_exp (df, pay_amount, abs_diff_name)

pay_over_status (df, bill_status, pay_amount, pay_over_bill_name)

var_of_pay_amount(df, pay_amount)

PCA_it(df, bill_status, "bill statements")

PCA_it(df, pay_amount, "pay amount")

PCA_it(df, abs_diff_name, "expenses")

PCA_it(df, scaled_diff_name, "scaled expenses")

df = resample(df,'def_payment_nm')

correlation(df, 0.9)
df.shape
df.to_csv("pre_processed_data_v2.csv")
