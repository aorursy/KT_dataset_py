print(os.listdir("../input/accepted_2007_to_2018q4.csv/"))


import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))
subset_cols = [

 'loan_amnt',

 'term',

 'int_rate',

 'installment',

 'grade',

 'emp_title',

 'emp_length',

 'home_ownership',

 'annual_inc',

 'verification_status',

 'issue_d',

 'loan_status',

 'pymnt_plan',

 'desc',

 'purpose',

 'title',

 'zip_code',

 'addr_state',

 'dti',

 'delinq_2yrs',

 'earliest_cr_line',

 'fico_range_low',

 'fico_range_high',

 'inq_last_6mths',

    

 'open_acc',

 'pub_rec',

 'revol_bal',

 'revol_util',

 'total_acc',

 'initial_list_status',

 'out_prncp',

 'total_pymnt',

 'total_pymnt_inv',

 'last_fico_range_high',

 'last_fico_range_low',

 'mths_since_last_major_derog',

 'policy_code',

 'application_type',

 'annual_inc_joint',

 'dti_joint',

 'verification_status_joint',

 'tot_cur_bal',

 'open_acc_6m',

 'max_bal_bc',

 'all_util',

 'total_rev_hi_lim',

 'inq_last_12m',

 'acc_open_past_24mths',

 'avg_cur_bal',

 'mort_acc',

 'percent_bc_gt_75',

 'pub_rec_bankruptcies',

 'tax_liens',

 'tot_hi_cred_lim',

 'total_bal_ex_mort',

 'total_bc_limit',

 'revol_bal_joint',

  "disbursement_method"]
loans = pd.read_csv('../input/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv', usecols=subset_cols)#,nrows=123456



loans = loans.loc[loans['loan_status'].isin(['Fully Paid', 'Charged Off'])].drop_duplicates() #,"Default"



loans.issue_d = pd.to_datetime(loans.issue_d,infer_datetime_format=True)

loans.zip_code = loans.zip_code.str.replace("xx","")



print(loans.shape)

print(loans.columns)

loans.tail()
# loans.dropna(how="any",thresh=)
100*loans.isna().mean()
print(loans.shape[0])

print(loans.loc[(loans.desc!="")|(loans.title!="")].shape[0])

print(loans.loc[(~loans.desc.isna())|(~loans.title.isna())].shape[0])
## drop columns with many missing values

missing_fractions = loans.isnull().mean().sort_values(ascending=False)

## https://www.kaggle.com/pileatedperch/predicting-charge-off-from-initial-listing-data

# drop rows with more than 50% missing

drop_list = sorted(list(missing_fractions[missing_fractions > 0.4].index))

print(len(drop_list))

print(drop_list)
loans.drop(drop_list,axis=1,inplace=True)
list(set(['addr_state', 'annual_inc', 'application_type', 'dti',

             'earliest_cr_line', 'emp_length', 'emp_title', 'fico_range_high', 

             'fico_range_low', 'grade', 'home_ownership', 'initial_list_status',

             'installment', 'int_rate', 'issue_d', 'loan_amnt', 'loan_status', 'mort_acc', 

             'open_acc', 'pub_rec', 'pub_rec_bankruptcies', 'purpose', 'revol_bal', 'revol_util', 

             'term', 'title', 'total_acc', 'verification_status', 'zip_code',

 'inq_last_6mths',

 'open_acc',

 'pub_rec',

 'revol_bal',

 'revol_util',

 'total_acc',

               'pymnt_plan',

 'open_acc',

 'pub_rec',

 'revol_bal',

 'revol_util',

 'total_acc',

 'initial_list_status',

 'delinq_2yrs', 'inq_last_6mths',

 'earliest_cr_line',]))
### ALT keep list:

### https://www.kaggle.com/pileatedperch/predicting-charge-off-from-initial-listing-data

keep_list = ['application_type',

 'delinq_2yrs',

 'issue_d',

 'open_acc',

 'revol_util',

 'zip_code',

 'loan_amnt',

 'mort_acc',

 'initial_list_status',

 'loan_status',

 'pub_rec_bankruptcies',

 'term',

 'revol_bal',

 'emp_title',

 'installment',

 'pymnt_plan',

 'dti',

 'purpose',

 'addr_state',

 'grade',

 'inq_last_6mths',

 'annual_inc',

 'verification_status',

 'fico_range_low',

 'home_ownership',

 'earliest_cr_line',

 'emp_length',

 'pub_rec',

 'total_acc',

 'fico_range_high',

 'title',

 'int_rate']

#'sub_grade', 
## https://www.kaggle.com/pileatedperch/predicting-charge-off-from-initial-listing-data

### The list of features to drop is any feature not in keep_list:



drop_list = [col for col in loans.columns if col not in keep_list]

print(drop_list)

print(loans.shape)

loans.drop(labels=drop_list, axis=1, inplace=True)

print(loans.shape)
loans['earliest_cr_line'] = pd.to_datetime(loans['earliest_cr_line'],infer_datetime_format=True)
## Convert emp_length to integers:



loans['emp_length'].replace(to_replace='10+ years', value='10 years', inplace=True)

loans['emp_length'].replace('< 1 year', '0 years', inplace=True)



def emp_length_to_int(s):

    if pd.isnull(s):

        return s

    else:

        return np.int8(s.split()[0])

    

loans['emp_length'] = loans['emp_length'].apply(emp_length_to_int)
loans.to_csv("loanDefaults_2018.csv.gz",index=False,compression="gzip")