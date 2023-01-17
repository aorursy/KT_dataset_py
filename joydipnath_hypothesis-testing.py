import os

print(os.listdir("../input"))

# !pip install researchpy
import pandas as pd

# import researchpy as rp

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from scipy.stats import chisquare

import statsmodels.api         as     sm

from   statsmodels.formula.api import ols

sns.set(style="darkgrid")

# df = pd.read_csv('insurance.csv')

df = pd.read_csv('../input/insurance.csv')

df.head()
df.dtypes
df.shape
df.isnull().sum()
df.describe()
# ax = sns.boxplot(x=df["sex"], y=df['charges'], hue="smoker", data=df)

# ax = sns.boxplot(x="bmi", y="age", data=df)

ax = sns.boxplot(df["bmi"])
bx = sns.boxplot(df["age"])
cx = sns.boxplot(df["charges"])
sns.distplot(df['bmi'], kde=False, fit=stats.gamma)
sns.distplot(df['age'], kde=False, fit=stats.gamma)
sns.distplot(df['charges'], kde=False, fit=stats.gamma)
sns.pairplot(df)
sns.boxplot(x=df['smoker'], y=df["charges"], data=df)
smoker_charges = pd.crosstab(index=df["charges"], columns=df["smoker"])

# print(smoker_charges)
mod_a = ols('charges ~ smoker', data = df).fit()

aov_table_a = sm.stats.anova_lm(mod_a, typ=2)

print(aov_table_a)
sns.boxplot(x=df['sex'], y=df["bmi"], data=df)

# df.boxplot("bmi", by='sex')
# gender_bmi = pd.crosstab(index=df["bmi"], columns=df["sex"])

# print(gender_bmi)



mod = ols('bmi ~ sex', data = df).fit()

aov_table = sm.stats.anova_lm(mod, typ=2)

print(aov_table)
# Two-way frequency tables, also called contingency tables,

# are tables of counts with two dimensions where each dimension is a different variable. 

# Two-way tables can give you insight into the relationship between two variables

gender_smoke = pd.crosstab(index=df["sex"], columns=df["smoker"])

print(gender_smoke)

# grouped = gender_smoke.groupby(['smoker','sex'])

# grouped.size().unstack()
chi_sq_Stat, p_value, deg_freedom, exp_freq = stats.chi2_contingency(gender_smoke)

print('Chi-square statistic %3.5f P value %1.6f Degrees of freedom %d' %(chi_sq_Stat, p_value,deg_freedom))
# df[df["sex"] == 'female']['sex']

# sns.scatter(df['age'], kde=False, fit=stats.gamma)

ax = sns.scatterplot(x=df['children'], y=df['bmi'], hue=df[df['sex'] == 'female']['sex'], data=df)
# mod_a = ols('charges ~ smoker', data = df).fit()

# aov_table_a = sm.stats.anova_lm(mod_a, typ=2)

# print(aov_table_a)

# df.describe(df.groupby(['sex']))['bmi']

# df_2 = df['sex'].dropna()

sns.jointplot(y=df["bmi"], x=df['children'], data=df);
# import scipy.stats as ss

# for name_group in df.groupby('bmi'):

#     samples = [condition[1] for condition in name_group[1].groupby('sex')['children']]

#     f_val, p_val = ss.f_oneway(*samples)

#     print('Name: {}, F value: {:.3f}, p value: {:.3f}'.format(name_group[0], f_val, p_val))

# from statsmodels.stats.multicomp import pairwise_tukeyhsd

# for bmi, grouped_df in df.groupby('bmi'):

#     print('Name {}'.format(bmi), pairwise_tukeyhsd(grouped_df['sex'], grouped_df['children']))
gender_smoke = pd.crosstab(index=df["sex"], columns=[df["bmi"], df['children']])

print(gender_smoke)