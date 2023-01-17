import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

from sklearn import linear_model

from scipy.stats import linregress, pearsonr, ttest_ind



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/refined/CapstoneRefined.csv', index_col = 0)

df.head()
df['AVG_INS'] = (df['INS1'] + df['INS2'] + df['INS3'] + df['INS4'] + df['INS5']) / 5

df['AVG_STUD'] = (df['STUD1'] + df['STUD2'] + df['STUD3'] + df['STUD4'] + df['STUD5']) / 5

df['AVG_TECH'] = (df['TECH1'] + df['TECH2'] + df['TECH3'] + df['TECH4'] + df['TECH5']) / 5

df['AVG_SUP'] = (df['SUP1'] + df['SUP2'] + df['SUP3'] + df['SUP4']) / 5

df['AVG_ELU'] = (df['ELU1'] + df['ELU2'] + df['ELU3'] + df['ELU4']) / 4



df_avg = df[['Gender', 'Device', 'AVG_INS', 'AVG_STUD', 'AVG_TECH', 'AVG_SUP', 'AVG_ELU']]

df_avg
x = df_avg[['AVG_INS', 'AVG_STUD', 'AVG_TECH', 'AVG_SUP']]

y = df_avg['AVG_ELU']

result = linear_model.LinearRegression()

result.fit(x, y)

print('Intercept (a) = %0.4f' % result.intercept_)

print('Slope (b) = ', result.coef_)
x = sm.add_constant(x)

model = sm.OLS(y, x).fit()

predictions = model.predict(x)

print(model.summary())
plt.figure(figsize=(10,10))

plt.title('Correlation Map')

ax=sns.heatmap(df_avg.corr(),

               linewidth=3.1,

               annot=True,

               center=1)
laptop = df_avg[df_avg['Device'] == 'Laptop']

laptop.head()
mobile = df_avg[df_avg['Device'] == 'HP']

mobile.head()
tstat, pval = ttest_ind(laptop['AVG_ELU'], mobile['AVG_ELU'])

print("T Score = %0.4f" % tstat)

print('P-Value = %0.4f' % pval)

print("Null hypothesis rejected") if pval <= 0.05 else print ("Fail to reject null hypothesis")
sns.countplot(x="Device", data=df_avg, hue="Gender")

plt.show()