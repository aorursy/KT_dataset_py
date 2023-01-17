import pandas as pd

pd.options.display.max_rows=15
# Download data directly online
df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv')
df.head()
## Pivot out sex columns and index by day and time
new_df = df.pivot(columns='sex',
         values=['tip'])

## Reset the columns for easier selection
new_df.columns = ['Female','Male']
## Graph histograms for each sex

%matplotlib inline

new_df['Male'].plot(kind='hist',edgecolor='black')
new_df['Female'].plot(kind='hist',edgecolor='black')
# Import data from a research experiment

control = [21,19.5,22.5,21.5,20.5,21]
t1 = [32,30.5,25,27.5,28,28.6]
t2 = [22.5,26,28,27,26.5,25.2]
t3 = [28,27.5,31,29.5,30,29.2]

df = pd.DataFrame(dict(control=control,
                       treatment1=t1,
                       treatment2=t2,
                       treatment3=t3))
df
## Gather all the data across columns into one column using melt

mdf = df.melt(value_vars=['control','treatment1','treatment2','treatment3'],
              var_name='treatment',
              value_name='result')
mdf
## Run an ANOVA test to find any statistical differences that exist
## between groups:

import statsmodels.api as sm
from statsmodels.formula.api import ols

lm = ols('result ~ treatment',data = mdf).fit()
table = sm.stats.anova_lm(lm,type=1)
print(table)