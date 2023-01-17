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
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from numpy import percentile

import seaborn as sns

from scipy import stats

from statsmodels.stats import weightstats

from statsmodels.stats.proportion import proportions_ztest

from statsmodels.stats.multicomp import pairwise_tukeyhsd
# df is a name given to the dataframe 

    

df = pd.read_csv('/kaggle/input/sample-insurance-claim-prediction-dataset/insurance2.csv') 

df.head()
rows_count, columns_count = df.shape

rows_count
columns_count
len(df.index)
len(df.columns)
df.dtypes
df.info() 
df.isnull().sum() 
df.isnull().values.any()
df_transpose = df.describe().T
df_transpose
# we have four numerical attributes age, bmi, children, chanrges

#calculate quartiles

quartiles_age      = percentile(df['age'], [25,50,75])  # calculate the quartiles of age 

quartiles_bmi      = percentile(df['bmi'], [25,50,75])  # calculate the quartiles of bmi 

quartiles_children = percentile(df['children'], [25,50,75]) # calculate the quartiles of children 

quartiles_charges  = percentile(df['charges'], [25,50,75]) # calculate the quartiles of charges 



#calculate min/max

df_min_age, df_max_age           = df['age'].min(), df['age'].max() # calculate the min/max of age 

df_min_bmi, df_max_bmi           = df['bmi'].min(), df['bmi'].max() # calculate the min/max of bmi

df_min_children, df_max_children = df['children'].min(), df['children'].max() # ca|lculate the min/max of children

df_min_charges, df_max_charges   = df['charges'].min(), df['charges'].max() # calculate the min/max of charges



#display five point summary



print('5 point summary of age:')

print('Min: %3.f' %  df_min_age)

print('Q1: %3.f' %  quartiles_age[0])

print('Median: %3.f' %  quartiles_age[1])

print('Q3: %3.f' %  quartiles_age[2])

print('Max: %3.f' %  df_max_age)

print('\n')

print('5 point summary of bmi:')

print('Min: %3f' %  df_min_bmi)

print('Q1: %3f' %  quartiles_bmi[0])

print('Median: %3f' %  quartiles_bmi[1])

print('Q3: %3f' %  quartiles_bmi[2])

print('Max: %3f' %  df_max_bmi)

print('\n')

print('5 point summary of children:')

print('Min: %3.f' %  df_min_children)

print('Q1: %3.f' %  quartiles_children[0])

print('Median: %3.f' %  quartiles_children[1])

print('Q3: %3.f' %  quartiles_children[2])

print('Max: %3.f' %  df_max_children)

print('\n')

print('5 point summary of charges:')

print('Min: %3f' %  df_min_charges)

print('Q1: %3f' %  quartiles_charges[0])

print('Median: %3f' %  quartiles_charges[1])

print('Q3: %3f' %  quartiles_charges[2])

print('Max: %3f' %  df_max_charges)
sns.distplot(df['bmi'])
sns.distplot(df['age'])  # Distribution of age
sns.distplot(df['charges'])  # Distribution of charges
df['bmi'].skew()
df['bmi'].skew()
df['age'].skew()
df['charges'].skew()
sns.boxplot(data = df[['bmi', 'age', 'charges']])
sns.boxplot(df['bmi']) 
sns.boxplot(df['age'])
sns.boxplot(df['charges']) 
sns.catplot(x="sex", y="children", hue="region", kind="violin", data=df, col="smoker");
sns.catplot(x="sex", hue="children", kind= 'count', data=df)
sns.catplot(x="smoker", hue="children", kind= 'count', data=df)
ax = sns.catplot(x="region", hue="children", kind= 'count', data=df)
sns.pairplot(df)
α = 0.05 

smokers_charges = df[df['smoker']==1].charges

non_smokers_charges = df[df['smoker']==0].charges
smokers_charges
non_smokers_charges
statistic, p_value = weightstats.ztest(smokers_charges, x2=non_smokers_charges, value=0,alternative='two-sided')

p_value
if p_value < α:

    print('Null Hypothesis (H0) is Rejected')

else :

    print('Null Hypothesis (H0) is failed to Reject')
sns.boxplot(df['smoker'], df['charges'])
α = 0.05 

male_bmi = df[df['sex']==1].bmi

female_bmi = df[df['sex']==0].bmi
male_bmi
statistic, p_value = weightstats.ztest(male_bmi, x2=female_bmi, value=0, alternative='two-sided')

p_value
if p_value < α:

    print('Null Hypothesis (H0) is Rejected')

else :

    print('Null Hypothesis (H0) is failed to Reject')
sns.boxplot(df['sex'],df['bmi'])
α = 0.05

female_smokers = df[df['sex'] ==  0].smoker.value_counts()

male_smokers = df[df['sex'] == 1].smoker.value_counts()
female_smokers = df[df['sex'] == 0].smoker.value_counts()[1]  # number of female smokers

male_smokers = df[df['sex'] == 1].smoker.value_counts()[1] # number of male smokers

females_count = df.sex.value_counts()[1] # number of females in the data

males_count = df.sex.value_counts()[0] #number of males in the data

female_smokers
females_count
statistic, p_value = proportions_ztest([female_smokers, male_smokers] , [females_count, males_count])

p_value
if p_value < α:

    print(f'With a p-value of {round(p_value,4)} the difference is significant. Hence We reject the null')

else:

    print(f'With a p-value of {round(p_value,4)} the difference is not significant. Hence We fail to reject the null hypothesis')
α = 0.05

female_bmi_no_child = df[(df['sex']==0) & (df['children'] == 0)].bmi

female_bmi_one_child = df[(df['sex']==1) & (df['children'] == 1)].bmi

female_bmi_two_child = df[(df['sex']==1) & (df['children'] == 2)].bmi
female_bmi_no_child_mu = df[(df['sex']== 0 ) & (df['children'] == 0)].bmi.mean()

female_bmi_one_child_mu = df[(df['sex']== 0) & (df['children'] == 1)].bmi.mean()

female_bmi_two_child_mu = df[(df['sex']==0) & (df['children'] == 2)].bmi.mean()



female_bmi_no_child_std = df[(df['sex']==0) & (df['children'] == 0)].bmi.std()

female_bmi_one_child_std = df[(df['sex']==0) & (df['children'] == 1)].bmi.std()

female_bmi_two_child_std = df[(df['sex']==0) & (df['children'] == 2)].bmi.std()



female_bmi_two_child_std

statistic,p_value = stats.f_oneway(female_bmi_no_child, female_bmi_one_child, female_bmi_two_child)

p_value
if(p_value < α):

    print('Null Hypothesis (H0) is Rejected')

else :

    print('Null Hypothesis (H0) is failed to Reject')
mean_bmi_df = pd.DataFrame()



df1            = pd.DataFrame({'Bmi_Type': 'No_Child', 'Mean_Bmi':female_bmi_no_child})

df2            = pd.DataFrame({'Bmi_Type': 'One_Child', 'Mean_Bmi':female_bmi_one_child})

df3            = pd.DataFrame({'Bmi_Type': 'Two_Child', 'Mean_Bmi':female_bmi_two_child})



mean_bmi_df = mean_bmi_df.append(df1) 

mean_bmi_df = mean_bmi_df.append(df2) 

mean_bmi_df = mean_bmi_df.append(df3) 
tukey = pairwise_tukeyhsd(endog=mean_bmi_df['Mean_Bmi'],     # Data

                          groups=mean_bmi_df['Bmi_Type'],   # Groups

                          alpha=0.05)          # Significance level



tukey.plot_simultaneous()    # Plot group confidence intervals

plt.vlines(x=49.57,ymin=-0.5,ymax=4.5, color="red")



tukey.summary()              # See test summary
sns.kdeplot(df1['Mean_Bmi'], label="female with no child")

sns.kdeplot(df2['Mean_Bmi'], label="female with one child")

sns.kdeplot(df3['Mean_Bmi'], label="female with two child")
sns.boxplot(x = "Bmi_Type", y = "Mean_Bmi", data = mean_bmi_df)

plt.title('Mean BMI exerted by BMI types')

plt.show()