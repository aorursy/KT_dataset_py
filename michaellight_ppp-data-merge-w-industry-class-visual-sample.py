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
df = pd.read_csv('../input/ppp-loan-data/foia_less_than_150k.csv')
df2 = pd.read_csv('../input/naics-soc/naics_codes.csv',engine='python')

df2['naics_code'] = pd.to_numeric(df2['naics_code'], errors='coerce')
df2['naics_code'] = df2.rename(columns={'naics_code':'NAICSCode'}, inplace=True)
df = pd.merge(df,df2[['NAICSCode','naics_title']],on='NAICSCode', how='left')
df['naics_title'] = df.rename(columns={'naics_title':'IndustryClassification'}, inplace=True)

import seaborn as sns
from matplotlib import pyplot as plt

df = df.groupby(df['IndustryClassification']).LoanAmount.sum().reset_index()
df = df.sort_values(by='LoanAmount', ascending=False)
print(df.head(11))
f, ax = plt.subplots()
sns.set_palette("pastel")
ax = sns.barplot(x='IndustryClassification',y='LoanAmount',data=df)
ax.set(xlabel="Industry Classification", ylabel="Total Loans Received (Ten-Billion USD)")
ax.set(xlim=(-0.5,9.5))
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.title('PPP Loan Distribution by Industry')
plt.tight_layout()
ttl = ax.title
ttl.set_position([.5, 10])
fig = ax.get_figure()
fig.savefig('bad_ppp_plot_2.png') 