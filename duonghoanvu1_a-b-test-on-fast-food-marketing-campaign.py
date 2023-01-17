# Data Processing
import numpy as np
import pandas as pd

# Data Visualizing
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from IPython.display import display, HTML

# Math
from scipy import stats  # Computing the t and p values using scipy 
from statsmodels.stats import weightstats 

# Warning Removal
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
df = pd.read_csv('../input/WA_Fn-UseC_-Marketing-Campaign-Eff-UseC_-FastF.csv')
df
df.describe()
df.info()
pd.DataFrame(df.groupby(['MarketID', 'MarketSize'])['SalesInThousands'].mean().sort_values(ascending=False))
pd.DataFrame(df.groupby(['MarketSize'])['SalesInThousands'].mean().sort_values(ascending=False))
fig = plt.figure(constrained_layout=True, figsize=(20, 6))
grid = gridspec.GridSpec(nrows=1, ncols=3,  figure=fig)

ax1 = fig.add_subplot(grid[0, 0])
df.groupby(['MarketSize'])['SalesInThousands'].mean().plot.pie(autopct='%1.0f%%', ax=ax1)
plt.ylabel('')
plt.title('Average Sales In Thousands')

ax1 = fig.add_subplot(grid[0, 1])
df.groupby(['MarketSize'])['SalesInThousands'].sum().plot.pie(autopct='%1.0f%%', ax=ax1)
plt.ylabel('')
plt.title('Sum of Sales In Thousands')

ax2 = fig.add_subplot(grid[0, 2])
df.groupby(['MarketSize'])['MarketID'].count().plot.pie(autopct='%1.0f%%', ax=ax2)
plt.ylabel('')
plt.title('Percentage of a number of campaigns')
plt.show()
df.groupby(['MarketSize'])['AgeOfStore'].agg(['mean','median']).sort_values(by=['mean'], ascending=False)
pd.DataFrame(df.groupby(['Promotion'])['SalesInThousands'].mean().sort_values(ascending=False))
pd.DataFrame(df.groupby(['MarketSize', 'Promotion'])['SalesInThousands'].mean().sort_values(ascending=False))
# Now let's view the marketsize for each promotion
pd.DataFrame(df.groupby(['MarketSize', 'Promotion'])['MarketID'].count())
df.groupby(['MarketSize', 'Promotion']).count()['MarketID'].unstack('MarketSize')
df.groupby(['MarketSize', 'Promotion']).count()['MarketID']
plt.figure(constrained_layout=True, figsize=(10, 8))
sns.countplot(x=df['Promotion'], hue=df['MarketSize'])
df.groupby(['MarketSize', 'Promotion']).count()['MarketID'].unstack('MarketSize').plot(
                                                                                    kind='bar',
                                                                                    figsize=(12,10),
                                                                                    grid=True,
                                                                                    stacked=True)
sns.distplot(df['AgeOfStore'])
sns.countplot(df['AgeOfStore'])
pd.DataFrame(df.groupby(['week'])['SalesInThousands'].mean().sort_values(ascending=False))
pd.DataFrame(df.groupby(['Promotion', 'week'])['SalesInThousands'].mean().sort_values(ascending=False))
pd.DataFrame(df.groupby(['MarketSize', 'week'])['SalesInThousands'].mean().sort_values(ascending=False))
fig = plt.figure()
sns.distplot(df.loc[df['Promotion'] == 1, 'SalesInThousands'])
plt.title('Promotion 1')
plt.xlabel('SaleInThousand')


fig = plt.figure()
sns.distplot(df.loc[df['Promotion'] == 2, 'SalesInThousands'])
plt.title('Promotion 2')
plt.xlabel('SaleInThousand')

fig = plt.figure()
sns.distplot(df.loc[df['Promotion'] == 3, 'SalesInThousands'])
plt.title('Promotion 3')
plt.xlabel('SaleInThousand')
# a group of Promotions
PromotionNumber = df['Promotion'].unique()
# pd.unique(df['Promotion'])

# a list of sales are categorized according to Promotion group, using dict
d_data = {promotion:df[df['Promotion'] == promotion]['SalesInThousands'] for promotion in PromotionNumber}

# apply Anova to 3 groups
F, p = stats.f_oneway(d_data[1], d_data[2], d_data[3])
print("p-value: {}, thus rejecting the null hypothesis".format(p))
t, p = stats.ttest_ind(df.loc[df['Promotion'] == 1, 'SalesInThousands'],
                       df.loc[df['Promotion'] == 2, 'SalesInThousands'], 
                       equal_var=False)
print("p-value = {:.4f}, thus Promotion 1 and 2 are statistically similar".format(p))

t, p = stats.ttest_ind(df.loc[df['Promotion'] == 1, 'SalesInThousands'],
                       df.loc[df['Promotion'] == 3, 'SalesInThousands'], 
                       equal_var=False)
print("p-value = {:.4f}, thus Promotion 1 and 3 are statistically different".format(p))

t, p = stats.ttest_ind(df.loc[df['Promotion'] == 2, 'SalesInThousands'],
                       df.loc[df['Promotion'] == 3, 'SalesInThousands'], 
                       equal_var=False)
print("p-value = {:.4f}, thus Promotion 2 and 3 are statistically similar".format(p))
from sklearn.linear_model import LinearRegression
df['MarketID'] = df['MarketID'].astype(int)
df['Promotion'] = df['Promotion'].astype(str)
all_data = pd.get_dummies(df.drop('LocationID', axis=1), drop_first=True)
x = all_data.drop('SalesInThousands', axis=1)
y = all_data['SalesInThousands']
reg = LinearRegression().fit(x, y)
coef = pd.Series(reg.coef_, index = x.columns)

imp_coef = coef.sort_values()
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Linear Model")
