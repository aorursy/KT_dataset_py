import numpy as np

import pandas as pd

import seaborn as sns

import scipy.stats as stats

import scipy

import matplotlib.pyplot as plt

%matplotlib inline

from scipy.stats import ttest_1samp, ttest_ind,wilcoxon,mannwhitneyu

from statsmodels.stats.power import ttest_power
#Using pandas to read csv file

data = pd.read_csv('../input/insurance.csv')
mydata = pd.DataFrame(data)

mydata
mydata.head() 

#displays the top 5 rows of the dataset
mydata.shape
mydata.info() 

#can also use mydata.dtypes to find datatype of each attribute
mydata.isnull().sum()
#5 point summary will give Xmin, 25th percentile, Median, 75th percentile,Xmax values

mydata.describe()
#Additional information of all attributes

mydata.describe(include = 'all')
#Since these columns are continuous variables hence using pairplot

#Pairplot plots frequency distribution (histogram) & scatter plots

sns.pairplot(mydata[['bmi','age','charges']])
sns.distplot(data['bmi'])
sns.distplot(data['age'])
sns.distplot(data['charges'])
sns.countplot(data['children'], hue = data['sex'])
#Skewness is a measure of attribute's symmetry.

mydata.skew(axis = None)
#Outliers are exceptions which are undesirable. Boxplots depict outliers as *

sns.boxplot(data = mydata, orient = 'h')
sns.boxplot(mydata['bmi'])
sns.boxplot(mydata['age'])
sns.boxplot(mydata['charges'])
sns.catplot(x = 'region', y = 'children', data = mydata, hue = 'sex', kind = 'violin', col = 'smoker')
sns.pairplot(data = mydata, hue = 'region')
sns.pairplot(data = mydata, hue = 'smoker')
sns.pairplot(data = mydata, hue = 'sex')
#Null hypothosis: H0 = Smoking does not affect Insurance Charges

#Alternate hypothesis: Ha = Smoking does affect Insurance Charges



Yes = np.array(mydata[mydata['smoker'] == 'yes']['charges'])

No = np. array(mydata[mydata['smoker'] == 'no']['charges'])

fig = plt.figure(figsize = (10,6))

sns.distplot(Yes)

sns.distplot(No)

fig.legend(labels = ["Yes","No"])
#Using 2 sided T test for independent samples



t_statistic, p_value = stats.ttest_ind(Yes,No)

t_statistic, p_value
if p_value < 0.05:

    print("Reject Null hypothesis")

else:

    print("Fail to reject Null hypothesis")
#Using mannwhitneyu test

u_statistic, p_value = mannwhitneyu(Yes, No)

u_statistic, p_value

#u_statistic, p_value leads us to reject Null hypothesis
#Calculate Power of test - This is the probability of rejecting the Null hypothesis

#To show how statistically significant is the mannwhitneyu test

(np.mean(Yes) - np.mean(No))/ np.sqrt(((len(Yes) - 1)*np.var(Yes) + (len(No) - 1)*np.var(No))/ len(Yes) + len(No)-2)
print(ttest_power(1.4333, nobs = (len(Yes) + len(No)), alpha = 0.05, alternative = 'two-sided'))
#Null Hypothesis: H0 = Bmi of Males do not differ significantly from that of Females

#Alternate Hypothesis: Ha = Bmi of Males differs significantly from that of Females

#2 sided T test for independent samples

bmi_male = np.array(mydata[mydata['sex'] == 'male']['bmi'])

bmi_female = np.array(mydata[mydata['sex'] == 'female']['bmi'])



fig = plt.figure(figsize = (10,6))

sns.distplot(bmi_male)

sns.distplot(bmi_female)

fig.legend(labels = ["BMI_Male","BMI_Female"])
#Using 2 sided T test for independent samples

t_statistic,p_value = stats.ttest_ind(bmi_male,bmi_female)

t_statistic,p_value
if p_value < 0.05:

    print("Reject Null hypothesis")

else:

    print("Fail to Reject Null hypothesis")
#Using mannwhitneyu test

u_statistic, p_value = mannwhitneyu(bmi_male, bmi_female)

u_statistic, p_value

#u_statistic, p_value leads us to Fail to reject Null hypothesis
#Calculate Power of test - This is the probability of rejecting the Null hypothesis

#To show how statistically significant is the mannwhitneyu test

(np.mean(bmi_male) - np.mean(bmi_female))/ np.sqrt(((len(bmi_male) - 1)*np.var(bmi_male) + (len(bmi_female) - 1)*np.var(bmi_female))/ len(bmi_male) + len(bmi_female)-2)
print(ttest_power(0.020, nobs = (len(bmi_male) + len(bmi_female)), alpha = 0.05, alternative = 'two-sided'))
#As Smokers and Gender are categorical variables hence we use Proportions test

#Null hypothesis: H0 = Proportion of smokers do not differ significantly in different genders

#Alternate hypothesis: Ha = Proportion of smokers differs significantly in different genders



pd.crosstab(mydata['sex'], mydata['smoker'], margins = True)
sns.countplot(mydata['sex'], hue = mydata['smoker'])
#Ex11 = Expected value of Smoker = No & Sex = Female

#Ex12 = Expected value of Smoker = No & Sex = Male

#Ex21 = Expected value of Smoker = Yes & Sex = Female

#Ex22 = Expected value of Smoker = Yes & Sex = Male

Ex11 = (1064 * 662) / 1338

Ex12 = (1064 * 676) / 1338

Ex21 = (274 * 662) / 1338

Ex22 = (274 * 676) / 1338
from statsmodels.stats.proportion import proportions_ztest



z_stats, p_val = proportions_ztest([115,159], [662,676])

z_stats, p_val
if p_val < 0.05:

    print('Reject Null Hypothesis')

else:

    print('Fail to Reject Null Hypothesis')
#Using Chi square test sum(Obs - Exp)^2/ Exp

observed_values = scipy.array([547,517,115,159])

n = observed_values.sum()

expected_values = scipy.array([Ex11,Ex12,Ex21,Ex22])

chi_square_stat, p_value = stats.chisquare(observed_values, f_exp=expected_values)

chi_square_stat, p_value
#Degree of freedom for chi square test = (row - 1)(col -1)

dof = (2-1)*(2-1)

dof
#Using Chi-square distribution table, we should check if chisquare stat of 7.765 exceeds 

#critical value of chisquare distribution. Critical value of alpha is 0.05 for 95% confidence 

#which is 3.84. As 7.765 > 3.84, we can reject Null Hypothesis



if chi_square_stat > 3.84:

    print("Reject Null Hypothesis")

else:

    print("Fail to Reject Null Hypothesis")
#Null Hypothesis: H0 = Distribution of BMI for women with 0,1,2 children is same

#Alternate Hypothesis: Ha = Distribution of BMI for women with 0,1,2 children is not same



bmidata = mydata[(mydata['children'] <= 2) & (mydata['sex'] == 'female')][['sex','bmi', 'children']]

bmidata.head()
#Grouping into 3 groups, 0,1,2 children

zero_ch = np.array(bmidata[bmidata['children'] == 0]['bmi'])

one_ch = np.array(bmidata[bmidata['children'] == 1]['bmi'])

two_ch = np.array(bmidata[bmidata['children'] == 2]['bmi'])
#Relationship between Bmi and children for women

bmigraph = sns.jointplot(bmidata['bmi'],bmidata['children'])

bmigraph = bmigraph.annotate(stats.pearsonr, fontsize=10, loc=(0.2, 0.8))
sns.boxplot(x = 'children', y = 'bmi', data = bmidata)
#Use One Way ANOVA for 3 sample groups 

#Null Hypothesis: H0: mean(zero_ch) = mean(one_ch) = mean(two_ch)

#Alternate Hypothesis: Ha: One of the means would differ
import statsmodels.api as sm

from statsmodels.formula.api import ols

 

mod = ols('bmi ~ children', data = bmidata).fit()

aov_table = sm.stats.anova_lm(mod, typ=2)

print(aov_table)
# Here p_value is 0.79376 > 0.05 hence we Fail to Reject Null Hypothesis therefore 



if 0.79 > 0.05:

    print("Fail to Reject Null Hypothesis")

else:

    print("Reject Null Hypothesis")