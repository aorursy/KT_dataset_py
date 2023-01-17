#1. Import the necessary libraries 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

from PIL import Image 

from scipy.stats import ttest_1samp, ttest_ind, mannwhitneyu, levene, shapiro

from statsmodels.stats.power import ttest_power

import scipy.stats as stats

from scipy.stats import f_oneway
#2.Read the data as a data frame

insurance_data = pd.read_csv('../input/insurance/insurance.csv')
#top 5 rows

insurance_data.head()
#a. Shape of the data

insurance_data.shape
#Insurance dataframe contains 1338 rows and 7 columns
#b. Data type of each attribute

insurance_data.dtypes
#There are 5 integer and 2 float columns

#There are 3 object columns
#c. Checking the presence of missing values 

insurance_data.isnull().sum()
#There are no null-values present in this dataframe
#d. 5 point summary of numerical attributes 

insurance_data.describe()
#Mean of age,bmi and children is close to their median respectively

#Average charge is 13270.42

#Most of the users have 1 children
# Distribution of age 

sns.set()

sns.distplot(insurance_data['age'],kde=True)
#bmi distribution

insurance_data['bmi'].hist(color='seagreen',bins=40,figsize=(8,4))
#charges distribution

sns.set()

insurance_data['charges'].hist(color='red',bins=40,figsize=(18,4))
#f. Measure of skewness of ‘bmi’, ‘age’ and ‘charges’ columns 

insurance_data.skew()
#visualization via Box Plot



fig, axes = plt.subplots(2, 2, figsize=(12, 6))

sns.boxplot(data=insurance_data['age'],orient='h',palette=None,ax=axes[0,0])

sns.boxplot(data=insurance_data['bmi'],orient='h',palette='Set2',ax=axes[0,1])

sns.boxplot(data=insurance_data['children'],orient='h',palette='Set3',ax=axes[1,0])

sns.boxplot(data=insurance_data['charges'],orient='h',palette='Set1',ax=axes[1,1])

#lets visualize via count plot

fig, axes = plt.subplots(2, 2, figsize=(12, 6))

sns.countplot(x=insurance_data['smoker'],ax=axes[0,0])

sns.countplot(x=insurance_data['region'],ax=axes[0,1])

sns.countplot(x=insurance_data['sex'],ax=axes[1,0])

sns.countplot(x=insurance_data['children'],orient='h',palette='Set1',ax=axes[1,1])

axes[0,0].set_title('smoker')

axes[0,1].set_title('region')

axes[1,0].set_title('sex')

axes[1,1].set_title('children')

fig.tight_layout()
#lets visualize bi-variate analysis via bar plot

fig, axes = plt.subplots(2, 2, figsize=(12, 6))

sns.barplot(x='smoker',y='charges',data=insurance_data,ax=axes[0,0])

sns.barplot(x='sex',y='charges',data=insurance_data,ax=axes[0,1])

sns.barplot(x='region',y='charges',data=insurance_data,ax=axes[1,0])

sns.barplot(x='children',y='charges',data=insurance_data,ax=axes[1,1])
#i. Pair plot that includes all the columns of the data frame

#As these are categorical columns,encoding is required (as pair plot considers only numerical columns)

insurance_data['region_code']=pd.factorize(insurance_data.region)[0]

insurance_data['sex_code']=pd.factorize(insurance_data.sex)[0]

insurance_data['smoker_code']=pd.factorize(insurance_data.smoker)[0]
insurance_data.head()
sns.set()

sns.pairplot(insurance_data)
sns.heatmap(insurance_data.corr(),cmap='coolwarm',annot=True)
#Do charges of people who smoke differ significantly from the people who don't? 

plt.figure(figsize=(8,4))

ax = sns.barplot(x="smoker", y="charges", data=insurance_data)
charge_smokers=np.array(insurance_data[insurance_data['smoker']=='yes']['charges'])

charge_non_smokers=np.array(insurance_data[insurance_data['smoker']=='no']['charges'])
t_statistic, p_value = ttest_ind(charge_smokers,charge_non_smokers)

print(t_statistic, p_value)
# p_value < 0.05 => alternative hypothesis:

# they don't have the same mean at the 5% significance level

print ("two-sample t-test p-value=", p_value)
#b. Does bmi of males differ significantly from that of females? 

plt.figure(figsize=(8,4))

ax = sns.barplot(x="sex", y="bmi", data=insurance_data)
bmi_male=np.array(insurance_data[insurance_data['sex']=='male']['bmi'])

bmi_female=np.array(insurance_data[insurance_data['sex']=='female']['bmi'])
t_statistic, p_value = ttest_ind(bmi_male,bmi_female)

print(t_statistic, p_value)
# p_value > 0.05 => null hypothesis:

print ("two-sample t-test p-value=", p_value)
#Is the proportion of smokers significantly different in different genders? 
plt.figure(figsize=(8,4))

ax = sns.barplot(x="sex", y="smoker_code", data=insurance_data)
female_smokers = insurance_data[insurance_data['sex'] == 'female'].smoker.value_counts()[1]  # number of female smokers

male_smokers = insurance_data[insurance_data['sex'] == 'male'].smoker.value_counts()[1] # number of male smokers

tot_females = insurance_data['sex'].value_counts()[1] # number of females in the data

tot_males = insurance_data['sex'].value_counts()[0] #number of males in the data
#proportion of smokers in different genders

print([female_smokers, male_smokers] , [tot_females, tot_males])



print(f' Proportion of smokers in females, males = {round(115/662,2)}%, {round(159/676,2)}% respectively')
#crosstables of sex and smokers

pd.crosstab(insurance_data['sex'],insurance_data['smoker'])
#Calculate p value or chi-square statistic value

chi_sq_Stat, p_value, deg_freedom, exp_freq = stats.chi2_contingency(pd.crosstab(insurance_data['sex'],insurance_data['smoker']))

print('Chi-square statistic %3.5f P value %1.6f Degrees of freedom %d' %(chi_sq_Stat, p_value,deg_freedom))
#d. Is the distribution of bmi across women with no children, one child and two children, the same? 
no_child=insurance_data[(insurance_data['sex']=='female') & (insurance_data['children'] == 0)]['bmi']#bmi of women with 0 child

one_child=insurance_data[(insurance_data['sex']=='female') & (insurance_data['children'] == 1)]['bmi']#bmi of women with 1 child

two_child=insurance_data[(insurance_data['sex']=='female') & (insurance_data['children'] == 2)]['bmi']#bmi of women with 2 child
no_child.mean() 
one_child.mean()
two_child.mean()
#Mean of bmi across above 3 groups seems same.
f_oneway(no_child,one_child,two_child) #One-way ANOVA test