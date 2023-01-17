import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('../input/treatment-of-migraine-headaches/KosteckiDillon.csv',index_col=0)
df.head()
df['headache'].value_counts()
df_headache_yes = df[df['headache'] == 'yes']['age']

df_headache_no = df[df['headache'] == 'no']['age']
from scipy.stats import shapiro,bartlett,mannwhitneyu

print(shapiro(df_headache_yes))

print(shapiro(df_headache_no))
df_headache_no.plot(kind = 'density')
df_headache_yes.plot(kind = 'density')
bartlett(df_headache_no,df_headache_yes)
mannwhitneyu(df_headache_no,df_headache_yes)
df['sex'].value_counts()
ct = pd.crosstab(df['headache'],df['sex'])

print(ct)
prop_female_headache = 2279/3545

prop_male_headache = 387/607

print(prop_female_headache,prop_male_headache)
from statsmodels.stats.proportion import proportions_ztest

x = np.array([2279,387])

n=  np.array([3545,607])

proportions_ztest(x,n)
ct = pd.crosstab(df['hatype'],df['sex'])

print(ct)
from scipy.stats import chi2_contingency

chi2_contingency(ct)
print('Aura : Female - ',1593/3545,", Male - ",117/607)

print('Mixed : Female - ',291/3545,", Male - ",166/607)

print('No Aura : Female - ',1661/3545,", Male - ",324/607)
df = pd.read_csv('../input/ibm-data/ibm.csv')
df.drop('Over18',axis = 1,inplace = True)
df.head()
df['Attrition'].value_counts()
df['Gender'].value_counts()
pd.crosstab(df['Attrition'],df['Gender'])
prop_attrition_female = 87/588

prop_attrition_male = 150/882
from statsmodels.stats.proportion import proportions_ztest

count = np.array([87,150])

nobs = np.array([588,882])

proportions_ztest(count,nobs)
df['Department'].value_counts()
ct = pd.crosstab(df['Attrition'],df['Department'])

ct
from scipy.stats import chi2_contingency

chi2_contingency(ct)
prop_attrition_yes_HR = 12/63

prop_attrition_yes_RnD = 133/961

prop_attrition_yes_Sales = 92/446

prop_attrition_no_HR = 51/63

prop_attrition_no_RnD = 828/961

prop_attrition_no_Sales = 354/446

print('prop_attrition_yes_HR : ',prop_attrition_yes_HR)

print('prop_attrition_yes_RnD : ',prop_attrition_yes_RnD)

print('prop_attrition_yes_Sales : ',prop_attrition_yes_Sales)

print('prop_attrition_no_HR : ',prop_attrition_no_HR)

print('prop_attrition_no_RnD : ',prop_attrition_no_RnD)

print('prop_attrition_no_Sales : ',prop_attrition_no_Sales)
df_male_income = df[df['Gender'] == 'Male']['MonthlyIncome']

df_female_income = df[df['Gender'] == 'Female']['MonthlyIncome']
from scipy.stats import shapiro

print(shapiro(df_male_income))

print(shapiro(df_female_income))
from scipy.stats import bartlett

print(bartlett(df_male_income,df_female_income))
from scipy.stats import mannwhitneyu

print(mannwhitneyu(df_male_income,df_female_income))
print(df_male_income.mean(),df_female_income.mean())
df_HR_income = df[df['Department'] == 'Human Resources']['MonthlyIncome']

df_RnD_income = df[df['Department'] == 'Research & Development']['MonthlyIncome']

df_Sales_income = df[df['Department'] == 'Sales']['MonthlyIncome']
from scipy.stats import f_oneway

f_oneway(df_HR_income,df_RnD_income,df_Sales_income)
print('Avg monthly income of HR Department : ',df_HR_income.mean())

print('Avg monthly income of R&D Department : ',df_RnD_income.mean())

print('Avg monthly income of Sales Department : ',df_Sales_income.mean())
df['Education'].value_counts()
df_mnthInc_EduLvl1 = df[df['Education'] == 1]['MonthlyIncome']

df_mnthInc_EduLvl2 = df[df['Education'] == 2]['MonthlyIncome']

df_mnthInc_EduLvl3 = df[df['Education'] == 3]['MonthlyIncome']

df_mnthInc_EduLvl4 = df[df['Education'] == 4]['MonthlyIncome']

df_mnthInc_EduLvl5 = df[df['Education'] == 5]['MonthlyIncome']
f_oneway(df_mnthInc_EduLvl1,df_mnthInc_EduLvl2,df_mnthInc_EduLvl3,df_mnthInc_EduLvl4,df_mnthInc_EduLvl5)
print('Avg monthly income for employees with Education Level 1 : ',df_mnthInc_EduLvl1.mean())

print('Avg monthly income for employees with Education Level 2 : ',df_mnthInc_EduLvl2.mean())

print('Avg monthly income for employees with Education Level 3 : ',df_mnthInc_EduLvl3.mean())

print('Avg monthly income for employees with Education Level 4 : ',df_mnthInc_EduLvl4.mean())

print('Avg monthly income for employees with Education Level 5 : ',df_mnthInc_EduLvl5.mean())
df.boxplot(column='MonthlyIncome',by = 'Education')
df.boxplot(column='MonthlyIncome',by = 'Department')
df.boxplot(column='MonthlyIncome',by = 'Gender')