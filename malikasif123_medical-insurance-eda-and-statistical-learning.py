#working with data
import pandas as pd
import numpy as np

#visualization
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#statistics
import scipy.stats as stats
data = pd.read_csv('../input/insurance/insurance.csv')
data.head() #checking top five records
shape_data=data.shape
print('Data set contains "{x}" number of rows and "{y}" number of columns columns'.format(x=shape_data[0],y=shape_data[1]))
# b - type of each attributes
count = 1
for item in data.columns:
    print('"{x}" : "{y}", Data Type = "{z}" '.format(x=count,y=item,z=type(data[item].iloc[0])))
    count = count+1
#or we can use another method as well that is .dtypes
data.dtypes
data.isnull().sum()
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
data.info()
data.describe().iloc[[3,4,5,6,7]]
#5 point summary include

# MIN
#Q1-25%
#Q2-50%
#Q3-75%
# MAX
#visualization via Box Plot

fig, axes = plt.subplots(2, 2, figsize=(12, 6))
sns.boxplot(data=data['age'],orient='h',palette=None,ax=axes[0,0])
sns.boxplot(data=data['bmi'],orient='h',palette='Set2',ax=axes[0,1])
sns.boxplot(data=data['children'],orient='h',palette='Set3',ax=axes[1,0])
sns.boxplot(data=data['charges'],orient='h',palette='Set1',ax=axes[1,1])

#--------------------------------------
axes[0,0].set_title('Age Distribution')
#5 point summary include

# MIN    =  18
#Q1-25%  =  27
#Q2-50%  =  39
#Q3-75%  =  51
# MAX    =  64

#--------------------------------------
axes[0,1].set_title('BMI Distribution')
#5 point summary include

# MIN    = 15.96
#Q1-25%  = 26.29
#Q2-50%  = 30.40
#Q3-75%  = 34.69
# MAX    = 53.13
#we can see presense of outliers

#--------------------------------------
axes[1,0].set_title('Children Distribution')
#5 point summary include

# MIN    = 0
#Q1-25%  = 0
#Q2-50%  = 1
#Q3-75%  = 2
# MAX    = 5

#--------------------------------------
axes[1,1].set_title('charges Distribution')
#5 point summary include

# MIN    = 1121.87
#Q1-25%  = 4740.28
#Q2-50%  = 9382.03
#Q3-75%  = 16639.91
# MAX    = 63770.42
#we can see presense of outliers
#The distplot shows the distribution of a univariate set of observations.
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
sns.distplot(data['age'],bins=30,ax=axes[0])
sns.distplot(data['bmi'],bins=30,ax=axes[1])
sns.distplot(data['charges'],bins=30,ax=axes[2])
axes[0].set_title('Age Distribution')
axes[1].set_title('BMI Distribution')
axes[2].set_title('charges Distribution')
#A Skewness value of 0 in the output denotes a symmetrical distribution
#A negative Skewness value in the output denotes tail is larger towrds left hand side of data so we can say left skewed
#A Positive Skewness value in the output denotes tail is larger towrds Right hand side of data so we can say Right skewed
data['age bmi charges'.split()].skew()
#lets simply do via box plot

fig, axes = plt.subplots(1, 3, figsize=(14, 3))
sns.boxplot(data=data['age'],orient='h',palette='Set1',ax=axes[0])
sns.boxplot(data=data['bmi'],orient='h',palette='Set2',ax=axes[1])
sns.boxplot(data=data['charges'],orient='h',palette='Set3',ax=axes[2])
axes[0].set_title('Age Distribution')
axes[1].set_title('BMI Distribution')
axes[2].set_title('charges Distribution')

#lets firstly go for value_counts
data['sex'].value_counts()
data['children'].value_counts()
data['smoker'].value_counts()
data['region'].value_counts()
#lets visualize via count plot
fig, axes = plt.subplots(2, 2, figsize=(12, 6))
sns.countplot(x=data['sex'],ax=axes[0,0])
sns.countplot(x=data['children'],ax=axes[0,1])
sns.countplot(x=data['smoker'],ax=axes[1,0])
sns.countplot(x=data['region'],orient='h',palette='Set1',ax=axes[1,1])
axes[0,0].set_title('Sex Distribution')
axes[0,1].set_title('Children Distribution')
axes[1,0].set_title('smoker Distribution')
axes[1,1].set_title('region Distribution')
fig.tight_layout()
#thus we can conclude following things
# 1. form Sex distribution we can state that there are almost equal number of males and females 
# 2. from children Distribution states that maximum number of people in data has no children, 
#    and there are quite few to have five childre 
# 3.from smoker Distribution we can state that maximum number of people are non-smoker
#pair plor includes all the colums of data frame and show us the scatter plot or
# how these coloumns are related to each other
sns.pairplot(data)
#let's check out some catagorical data effect on each as well
#1.How smoker and Non smoker are spread out
sns.pairplot(data,hue='smoker')
#2.How Male and Female are spread out
sns.pairplot(data,hue='sex')
#3. How Regional data is spread out
sns.pairplot(data,hue='region')

#lets try correlation metrix as well to get more insight of the pairing of all data
#and visualise it via heatmap
fig,ax= plt.subplots(figsize=(8, 4))
sns.heatmap(data.corr(),annot=True)
fig.tight_layout()
#thus now we can clearly see the correlation values, highest correlation value is 
# .3 for age and charges 
# followed by .2 for bmi and charges
#data generation
smoker_data = data[data['smoker']=='yes']['charges']
non_smoker_data = data[data['smoker']=='no']['charges']
#Using the seaborn python library to generate a histogram of our 2 samples outputs the following.
sns.kdeplot(smoker_data, shade=False)
sns.kdeplot(non_smoker_data, shade=True)
plt.title("Independent Sample T-Test")
#check of mean and SD
data.groupby('smoker').describe()['charges']
sm_arr = np.array(smoker_data)
nsm_arr = np.array(non_smoker_data)
#performing an independent T-test
t,p_value = stats.ttest_ind(sm_arr,nsm_arr,axis =0)
print("t = ",t, ", p_twosided = ", p_value, ", p_onesided =", p_value/2)
#doing Two Tail testing since p_value < (alpha/2)
if p_value < (0.05)/2:
   print('We Reject the Null Hypothesis!')
else:
    print('We Fail to Reject the Null Hypothesis!')
    
#data generation
male_data = data[data['sex']=='male']['bmi']
female_data = data[data['sex']=='female']['bmi']
#Using the seaborn python library to generate a histogram of our 2 samples outputs the following.
sns.kdeplot(male_data, shade=False)
sns.kdeplot(female_data, shade=True)
plt.title("Independent Sample T-Test")
m_arr = np.array(male_data)
f_arr = np.array(female_data)
#performing an independent T-test
t,p_value = stats.ttest_ind(m_arr,f_arr,axis =0)
print("t = ",t, ", p_twosided = ", p_value, ", p_onesided =", p_value/2)
#doing Two Tail testing 
if p_value < (0.05)/2:
   print('We Reject the Null Hypothesis!')
else:
    print('We Fail to Reject the Null Hypothesis!')
#doing Two Tail testing at confidence interval 90% 
if p_value < (0.10)/2:
   print('We Reject the Null Hypothesis!')
else:
    print('We Fail to Reject the Null Hypothesis!')
#extracting Data
cross_table=pd.crosstab(data['smoker'],data['sex'])

#how the data look like
cross_table

#performing chi-square test
stats.chi2_contingency(cross_table)
#data extraction
bmi_of_female_with_no_child = data[(data['sex']=='female') & (data['children']==0)]['bmi']
bmi_of_female_with_one_child = data[(data['sex']=='female') & (data['children']==1)]['bmi']
bmi_of_female_with_two_child = data[(data['sex']=='female') & (data['children']==2)]['bmi']
sns.kdeplot(bmi_of_female_with_no_child, shade=False)
sns.kdeplot(bmi_of_female_with_one_child, shade=True)
sns.kdeplot(bmi_of_female_with_two_child, shade=True)

plt.title("Independent Sample T-Test")
#The analysis of variance (ANOVA) can be thought of as an extension to the t-test. 
#The independent t-test is used to compare the means of a condition between 2 groups. 
#ANOVA is used when one wants to compare the means of a condition between 2+ groups. 
#ANOVA is an omnibus test, meaning it tests the data as a whole. 
#Another way to say that is this,
#ANOVA tests if there is a difference in the mean somewhere in the model (testing if there was an overall effect),
#but it does not tell one where the difference is if the there is one.

F_stats,p_value = stats.f_oneway(bmi_of_female_with_no_child, 
            bmi_of_female_with_one_child,
             bmi_of_female_with_two_child)
print('F-statistic = {x}, p_value = {y}'.format(x=F_stats,y=p_value))
if p_value<.05:
    print('we Reject the Null Hypothesis!')
else:
    print('we fail to Reject Null Hypothesis')