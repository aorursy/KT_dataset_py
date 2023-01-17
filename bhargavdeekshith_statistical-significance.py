import numpy as np
import pandas as pd
pd.set_option('display.max_columns',100)
df = pd.read_table('../input/human-resources/HR_1.txt')
df.head()
df.shape
ct = pd.crosstab(df['Attrition'],df['Gender'])
ct
x = np.array([150,87])
n = np.array([882,588]) #Total records for 1 and 2
from statsmodels.stats.proportion import proportions_ztest
proportions_ztest(x,n)
df.Department.value_counts()
t = pd.crosstab(df['Attrition'],df['Department'])
t
from scipy.stats import chi2_contingency
chi2_contingency(t)
df.Gender.dtype
ym = df[df['Attrition'] == 'Yes']['MonthlyIncome']
nm = df[df['Attrition'] == 'No']['MonthlyIncome']
from scipy.stats import mannwhitneyu,ttest_ind,f_oneway
ttest_ind(ym,nm)
mannwhitneyu(ym,nm)
d1 = df[df['Department'] == 1]['MonthlyIncome']
d2 = df[df['Department'] == 2]['MonthlyIncome']
d3 = df[df['Department'] == 3]['MonthlyIncome']
f_oneway(d1,d2,d3)