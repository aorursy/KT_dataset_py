# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

%matplotlib inline

sns.set(color_codes = True)

from scipy.stats import ttest_1samp, ttest_ind, chi2_contingency

from statsmodels.stats.power import ttest_power

from statsmodels.formula.api import ols      

from statsmodels.stats.anova import _get_covariance,anova_lm
insurance_dataframe = pd.read_csv("../input/insurance-premium-prediction/insurance.csv")

insurance_dataframe.head()
insurance_dataframe.shape
insurance_dataframe.dtypes
insurance_dataframe.info()

insurance_dataframe.isnull().values.any()
insurance_dataframe.describe().T
sns.distplot(insurance_dataframe['bmi']) #Distribution of BMI
sns.distplot(insurance_dataframe['age']) #Distribution of Age
sns.distplot(insurance_dataframe['expenses']) #Distribution of Charges
sns.boxplot(insurance_dataframe['bmi']) 
sns.boxplot(insurance_dataframe['age'])     
sns.boxplot(insurance_dataframe['expenses'])
sns.swarmplot(insurance_dataframe['sex']) #Distribution of sex
sns.swarmplot(insurance_dataframe['children']) #Distribution of children
sns.swarmplot(insurance_dataframe['smoker']) #Distribution of smoker
sns.swarmplot(insurance_dataframe['region']) #Distribution of region
#Distribution of sex

sex_plot = pd.crosstab(index = insurance_dataframe["sex"], columns="count")     

sex_plot.plot.bar()
#Distribution of children

children_plot = pd.crosstab(index = insurance_dataframe["children"], columns="count")     

children_plot.plot.bar()
#Distribution of smoker

smoker_plot = pd.crosstab(index = insurance_dataframe["smoker"], columns="count")     

smoker_plot.plot.bar()
#Distribution of region

region_plot = pd.crosstab(index = insurance_dataframe["region"], columns="count")     

region_plot.plot.bar()
sns.pairplot(insurance_dataframe, hue='sex')
sns.pairplot(insurance_dataframe, hue='smoker')
sns.pairplot(insurance_dataframe, hue='region')
smoker_dataframe = insurance_dataframe[insurance_dataframe['smoker']=='yes'] #Creating a sub-dataframe for smokers

smoker_dataframe.head() 
nonsmoker_dataframe = insurance_dataframe[insurance_dataframe['smoker']=='no'] #Creating a sub-dataframe for non-smokers

nonsmoker_dataframe.head()
# finding the ttest independence between the charges of smokers and non smokers

t_statistic, p_value = ttest_ind(smoker_dataframe['expenses'], nonsmoker_dataframe['expenses'])

print(t_statistic, p_value)
# When P value < 0.05 reject the null hypothesis, otherwise accept

if p_value < 0.05:

    print("Reject H0, the charges of smokers differ significantly from the charges of non smokers.")

else:

    print("Accept H0, the charges of smokers do not differ significantly from the charges of non smokers.")
(smoker_dataframe['expenses'].mean()-nonsmoker_dataframe['expenses'].mean())/np.sqrt(((smoker_dataframe['expenses'].size-1)*np.var(smoker_dataframe['expenses']) + (nonsmoker_dataframe['expenses'].size-1)*np.var(nonsmoker_dataframe['expenses']))/(smoker_dataframe['expenses'].size + nonsmoker_dataframe['expenses'].size-2))
print(ttest_power(3.165, nobs = 1338, alpha = 0.05, alternative = 'two-sided'))

male_bmi_dataframe = insurance_dataframe[insurance_dataframe['sex']=='male'] #Creating a sub-dataframe for all males

male_bmi_dataframe.head()
female_bmi_dataframe = insurance_dataframe[insurance_dataframe['sex']=='female'] #Creating a sub-dataframe for all females

female_bmi_dataframe.head()
# finding the ttest independence between the bmi of male and female

t_statistic, p_value = ttest_ind(male_bmi_dataframe['bmi'], female_bmi_dataframe['bmi'])

print(t_statistic, p_value)
# When P value < 0.05 reject the null hypothesis, otherwise accept

if p_value < 0.05:

    print("Reject H0, the bmi of males differ significantly from bmi of females")

else:

    print("Accept Ho, the bmi of males does not differ significantly from bmi of females")
(male_bmi_dataframe['bmi'].mean()-female_bmi_dataframe['bmi'].mean())/np.sqrt(((male_bmi_dataframe['bmi'].size-1)*np.var(male_bmi_dataframe['bmi']) + (female_bmi_dataframe['bmi'].size-1)*np.var(female_bmi_dataframe['bmi']))/(male_bmi_dataframe['bmi'].size + female_bmi_dataframe['bmi'].size-2))
print(ttest_power(0.093, nobs = 1338, alpha = 0.05, alternative = 'two-sided'))
# Preparing the crosstab function

cont = pd.crosstab(insurance_dataframe['sex'],insurance_dataframe['smoker'])
cont #Printing the contingency table
# Applying Chi-square contigency to find the P value

chi2_contingency(cont)
p_value = 0.006548143503580696 #From previous output





# When P value < 0.05 reject the null hypothesis, otherwise accept

if p_value < 0.05:

    print("Reject H0, the proportion of smokers is not independent on genders")

else:

    print("Accept Ho, the proportion of smokers is independent on genders")
#We are construction a new dataframe with conditions where sex is female and number of children 0,1 or 2

distribution_dataframe = insurance_dataframe[(insurance_dataframe['sex']=='female') & (insurance_dataframe['children'] < 3)]

distribution_dataframe.head(20)
distribution_dataframe.info()
# Converting children into categorical variables

distribution_dataframe.children = pd.Categorical(distribution_dataframe.children)
# Converting sex and children into categorical variables

distribution_dataframe.sex = pd.Categorical(distribution_dataframe.sex)
distribution_dataframe.info()
# Applying anova function where BMI is a function of children

formula = 'bmi ~ C(children)'

model = ols(formula, distribution_dataframe).fit()

aov_table = anova_lm(model)

print(aov_table)
p_value = 0.715858 #From the above output



# When P value < 0.05 reject the null hypothesis, otherwise accept

if p_value < 0.05:

    print("Reject H0, the means of different number of children are significantly different")

else:

    print("Accept Ho, the means of different number of children are equal")