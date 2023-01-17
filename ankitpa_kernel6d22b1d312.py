# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
loanstats = '../input/lending-club-loan-data/loan.csv'

try:
    loan_df = pd.read_csv(loanstats, skipinitialspace=True, low_memory=False)
except Exception as e:
    print(e)
loan_data = loan_df[loan_df['issue_d'] == 'Jan-07']
loan_data['loan_status'].value_counts()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
loan_df1 = loan_df[loan_df['loan_status'] == 'Fully Paid']
loan_df1[['loan_status','loan_amnt','out_prncp','out_prncp_inv','funded_amnt','funded_amnt_inv','term','int_rate','total_pymnt_inv','total_rec_prncp','total_rec_int','total_rec_late_fee','recoveries','collection_recovery_fee','issue_d','last_pymnt_d','last_pymnt_amnt','next_pymnt_d','last_credit_pull_d']].head(500)
loan_df2 = loan_df[loan_df['loan_status'] == 'Charged Off']
loan_df2[['loan_status','loan_amnt','out_prncp','out_prncp_inv','funded_amnt','funded_amnt_inv','term','int_rate','total_pymnt_inv','total_rec_prncp','total_rec_int','total_rec_late_fee','recoveries','collection_recovery_fee','issue_d','last_pymnt_d','last_pymnt_amnt','next_pymnt_d','last_credit_pull_d']].head(500)
loan_default = loan_df[loan_df['loan_status'] == 'Default']
loan_default[['loan_status','loan_amnt','out_prncp','out_prncp_inv','funded_amnt','funded_amnt_inv','term','int_rate','total_pymnt_inv','total_rec_prncp','total_rec_int','total_rec_late_fee','recoveries','collection_recovery_fee','issue_d','last_pymnt_d','last_pymnt_amnt','next_pymnt_d','last_credit_pull_d']].head(500)
loan_current = loan_df[loan_df['loan_status'] == 'Current']
loan_current[['loan_status','loan_amnt','out_prncp','out_prncp_inv','funded_amnt','funded_amnt_inv','term','int_rate','total_pymnt_inv','total_rec_prncp','total_rec_int','total_rec_late_fee','recoveries','collection_recovery_fee','last_pymnt_d','last_pymnt_amnt','issue_d','next_pymnt_d','last_credit_pull_d']].head(500)
loan_default['nb_months'] = ((pd.to_datetime(loan_default.last_credit_pull_d) - pd.to_datetime(loan_default.issue_d))/np.timedelta64(1, 'M'))
loan_default['nb_months'] = pd.to_numeric(loan_default['nb_months'])
loan_default['nb_months']
loan_default['term'] = loan_default['term'].replace('months','')
loan_default['term'] = pd.to_numeric(loan_default['term'])