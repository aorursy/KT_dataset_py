import numpy as np

import pandas as pd 

import matplotlib.mlab as mlab

import matplotlib.pyplot as plt



%matplotlib inline



from matplotlib import style

plt.style.use('ggplot')
data = pd.read_csv('../input/loan.csv', low_memory=False, parse_dates=['issue_d'], infer_datetime_format=True)
state_count = data.addr_state.value_counts()



state_count.plot(kind = 'bar',figsize=(16,8), title = 'Loans per State')
tn_data = data.loc[data.addr_state == 'TN']



tn_x = range(1, 12888)



tn_loan_amnt = tn_data.loan_amnt
plt.figure(figsize=(16, 10))

plt.scatter(tn_x, tn_loan_amnt)



plt.xlim(1,12888)

plt.ylim(0, 37500)



plt.ylabel("Loan Amount")

plt.title("Loan Size in Tennessee")



plt.show()
plt.figure(figsize=(16,8))



mu = tn_loan_amnt.mean()

sigma = tn_loan_amnt.std()

num_bins = 100



n, bins, patches = plt.hist(tn_loan_amnt, num_bins, normed=1, facecolor='blue', alpha=0.7)

y = mlab.normpdf(bins, mu, sigma)

plt.plot(bins, y, 'r--')



plt.xlabel("Loan Amount")

plt.title("Loan Amount Distribution in Tennessee")

plt.show()
tloan_tn_df = tn_data['issue_d'].value_counts().sort_index()

tloan_tn_df = tloan_tn_df.cumsum()
tloan_tn_df.plot(figsize=(16,8), title='Number of Loans Issued in Tennessee')