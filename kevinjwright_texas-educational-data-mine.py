import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#Read csv and find columns
df = pd.read_csv('../input/elsect_summary.csv')
texas = df[df['STATE']=='Texas']
texas.columns
#format data for pairplot
pplot_texas = texas[['TOTAL_REVENUE', 'FEDERAL_REVENUE','STATE_REVENUE', 'LOCAL_REVENUE', 'TOTAL_EXPENDITURE']]
sns.pairplot(pplot_texas)
#Visualization of TEA income and spending over the years
plt.scatter(x=texas['YEAR'],y=texas['STATE_REVENUE'],c='red')

plt.scatter(x=texas['YEAR'],y=texas['LOCAL_REVENUE'],c='green')

plt.scatter(x=texas['YEAR'],y=texas['FEDERAL_REVENUE'],c='blue')

plt.scatter(x=texas['YEAR'],y=texas['TOTAL_REVENUE'],c='black')

plt.scatter(x=texas['YEAR'],y=texas['TOTAL_EXPENDITURE'],c='orange',marker='x')

plt.legend()

plt.title('Education Spending in 10 Millions')