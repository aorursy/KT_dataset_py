get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

Data = pd.read_csv('../input/US_90_08_wk3.csv')
EPS_rt = Data[['cusip', 'conm', 't', 'return','eps']]

EPS_rt.head()
EPS_rt['lag_eps'] = EPS_rt['eps'].shift(1).where(EPS_rt['cusip'] == EPS_rt['cusip'].shift(1), np.nan)
EPS_rt.head()
EPS_rt.count()
EPS_rt2 = EPS_rt.dropna()

EPS_rt2.head()
data = EPS_rt2.sample(frac=0.1473, replace=False, random_state = 480391988);data.count()
data.head()
data['return_positive'] = np.where(data['return'] > 0, 1, 0)

data['lag_eps_positive'] = np.where(data['lag_eps'] > 0, 1, 0)

data.head()
data1 = data.iloc[:,6:8]

data1.describe()
data1.sum()
ax = sns.countplot(x='return_positive',data=data)

ax.set_title('Number of Positive Asset Returns')

ax.set_ylabel('Number of Observations')

ax.set_xlabel('Sign of the Observation (1 = Positive, 0 = Negative)')

plt.show()
ax = sns.countplot(x='lag_eps_positive',data=data)

ax.set_title('Number of Positive Lagged Earnings per Share ')

ax.set_ylabel('Number of Observations')

ax.set_xlabel('Sign of the Observation (1 = Positive, 0 = Negative)')

plt.show()
pd.crosstab(data['lag_eps_positive'],data['return_positive'])
ax = sns.countplot(x="lag_eps_positive", hue="return_positive", data=data)

ax.set_title('Sign of Return by Sign of LEPS')

ax.set_xlabel('Sign of Lagged Earnings Per Share (1 = Positive, 0 = Negative)')

ax.set_ylabel('Number of Observations')

ax.legend(title='Sign of Asset Return')

plt.show()
from scipy.stats import chi2_contingency

tab=pd.crosstab(data['lag_eps_positive'],data['return_positive']);tab
chi2_contingency(tab)
from scipy.stats import fisher_exact

fisher_exact(tab)