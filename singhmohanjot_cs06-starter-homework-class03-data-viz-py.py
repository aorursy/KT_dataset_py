# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/lending-club-loan-data/loan.csv')
data.shape
data.head()
ca_data=data[data.addr_state == 'CA']
ca_data.shape
ny_data=data[data.addr_state == 'NY']
ny_data.shape
il_data=data[data.addr_state == 'IL']
il_data.shape
frames = [ca_data, ny_data, il_data]
result_data = pd.concat(frames)
result_data.shape
result_data.head()
plt.figure(figsize=(10, 8))
sns.scatterplot(x='loan_amnt', y='funded_amnt', data=result_data, hue= 'addr_state')
plt.title('Sample scatterplot')
plt.show()
ca_r_sqrd = ca_data[['dti', 'funded_amnt']].corr()['dti'][1] **2
ca_r_sqrd
ny_r_sqrd = ny_data[['dti', 'funded_amnt']].corr()['dti'][1] **2
ny_r_sqrd
il_r_sqrd = il_data[['dti', 'funded_amnt']].corr()['dti'][1] **2
il_r_sqrd
plt.figure(figsize=(10, 8))
sns.set(color_codes=True)
comp_diag = sns.lmplot(x='dti', y='funded_amnt', col='addr_state', hue='addr_state', legend=True, data=result_data, markers=["o", "x", "+"], line_kws={'color':'black'})
comp_diag.set_axis_labels("Debt-to-income Ratio", "Amount Funded in USD")
comp_diag.set(xscale='log', yscale='linear')
#plt.title('Debt-to-income Ratio vs. Amount Funded\n CA R^2 = ' + str(ca_r_sqrd) + ' NY R^2 = ' + str(ny_r_sqrd))
plt.xlim((10**(-2), None))
plt.ylim((1000,50000))
#plt.xlabel('Debt-to-income Ratio')
#plt.ylabel('Amount Funded')
#plt.legend()
plt.show()
plt.figure(figsize=(12, 9))
sns.distplot(ca_data.loan_amnt,rug=True, kde_kws={"color": "b", "lw": 1}, hist_kws={"histtype": "step", "linewidth": 1, "color": "k"},label='CA')
sns.distplot(ny_data.loan_amnt,rug=True, kde_kws={"color": "r", "lw": 1}, label='NY')
sns.distplot(il_data.loan_amnt,rug=True, kde_kws={"color": "r", "lw": 1}, label='IL')
plt.title('Distribution for Loan Amounts in CA , NY, IL')
plt.legend()
plt.show()