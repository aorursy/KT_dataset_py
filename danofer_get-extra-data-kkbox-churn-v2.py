# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
transactions = pd.read_csv('../input/transactions.csv')

transactions = pd.concat((transactions, pd.read_csv('../input/transactions_v2.csv')), axis=0, ignore_index=True).reset_index(drop=True)





transactions.membership_expire_date = pd.to_datetime(transactions.membership_expire_date,format="%Y%m%d")

transactions.transaction_date = pd.to_datetime(transactions.transaction_date,format="%Y%m%d")



print(transactions.shape)

transactions.head()
transactions["price_paid_diff"]  = transactions.plan_list_price -  transactions.actual_amount_paid

transactions["transactions_expiry_transaction_days_diff"] = ((transactions.membership_expire_date - transactions.transaction_date).dt.days) 



# # transactions["transactions_expiry_transaction_days_diff_div_plan"] = transactions.payment_plan_days/transactions.transactions_expiry_transaction_days_diff

transactions.to_csv("kkbox_churn_transactions_v3.csv.gz",index=False,compression="gzip")
members = pd.read_csv('../input/members_v3.csv')

members.registration_init_time = pd.to_datetime(transactions.registration_init_time,format="%Y%m%d")

members.head()
members.to_csv("kkbox_churn_members_v3.csv.gz",index=False,compression="gzip")