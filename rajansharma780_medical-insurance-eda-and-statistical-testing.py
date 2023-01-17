# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Load the csv file and make the data frame

insurance_df = pd.read_csv("../input/insurance.csv")
#display the data frame

insurance_df
#display the first 5 rows of data frame

insurance_df.head()
#display the last 5 rows of data frame

insurance_df.tail()
#display how many rows and columns are there in data frame

print("The data frame has {} rows and {} columns".format(insurance_df.shape[0],insurance_df.shape[1]))
#display the data type of each column

insurance_df.dtypes
#display the information of data frame

insurance_df.info()
#display the null values column wise

insurance_df.apply(lambda x:sum(x.isnull()))
#display the size of data frame

insurance_df.size
#display how many males and females are there

insurance_df['sex'].value_counts()
#display graphically how many males and females are there

sns.countplot(insurance_df['sex'])

plt.show()
#display how many are smoker and not smoker

insurance_df['smoker'].value_counts()
#display graphically how many are smoker and not smoker

sns.countplot(insurance_df['smoker'])

plt.show()
#display how many children are there

insurance_df['children'].unique()
#display the count of children

insurance_df['children'].value_counts()
#display grapically the count of children

sns.countplot(insurance_df['children'])

plt.show()
#display how many unique regions are there

insurance_df['region'].unique()
#display the count of region

insurance_df['region'].value_counts()
#display graphically the count of region

sns.countplot(insurance_df['region'])

plt.show()
#display the count of people with how many children they are having and they are smoker or not

pd.crosstab(insurance_df['children'],insurance_df['smoker'])
#display graphically the count of people with how many children they are having and they are smoker or not

sns.countplot(insurance_df['children'],hue=insurance_df['smoker'])

plt.show()
#description of data frame or 5 point summary of data frame

insurance_df.describe()
#display graphically the distribution of bmi column

sns.distplot(insurance_df['bmi'])

plt.show()
#display graphically the distribution of age column

sns.distplot(insurance_df['age'])

plt.show()
#display graphically the distribution of charges column

sns.distplot(insurance_df['charges'])

plt.show()
#display graphically the distribution of children column

sns.distplot(insurance_df['children'])

plt.show()
#display the skewness of each column

pd.DataFrame([stats.skew(insurance_df['age']),stats.skew(insurance_df['bmi']),stats.skew(insurance_df['charges'])],index=['age','bmi','charges'],columns=['skewness'])
#check any outliers are there in bmi column

sns.boxplot(x='bmi',data=insurance_df)

plt.show()
#check any outliers are there in age column

sns.boxplot(x='age',data=insurance_df)

plt.show()
#check any outliers are there in charges column

sns.boxplot(x='charges',data=insurance_df)

plt.show()
#display graphically the relation between all features

sns.pairplot(insurance_df,hue='smoker')

plt.show()
#display graphically the relation between all features

sns.pairplot(insurance_df,hue='sex')

plt.show()
#display graphically the relation between all features

sns.pairplot(insurance_df,hue='region')

plt.show()
#Null Hypothesis--> H0 = "charges has no effect on smoking"

#Alternate hypothesis--> H1 = "charges has effect on smoking" 



x = np.array(insurance_df[insurance_df['smoker'] == 'yes']['charges'])#selecting charges values corresponding to smoker as an array

y = np.array(insurance_df[insurance_df['smoker'] == 'no']['charges'])#selecting charges values corresponding to not smoker as an array

t,p_value = stats.ttest_ind(x,y,axis =0)#performing an independent T-test

if p_value <0.05:

    print("charges has effect on smoking(reject H0)")

else:

    print("charges has no effect on smoking(accept H0)")
#Null Hypothesis--> H0 = "bmi has no effect on gender"

#Alternate hypothesis--> H1 = "bmi has effect on gender" 



x = np.array(insurance_df[insurance_df['sex'] == 'male']['bmi'])#selecting bmi values corresponding to male as an array

y = np.array(insurance_df[insurance_df['sex'] == 'female']['bmi'])#selecting bmi values corresponding to female as an array

t,p_value = stats.ttest_ind(x,y,axis =0)#performing an independent T-test

if p_value <0.05:

    print("bmi has effect on gender(reject H0)")

else:

    print("bmi has no effect on gender(accept H0)")
pd.crosstab(insurance_df['smoker'],insurance_df['sex'])
#Null Hypothesis--> H0 = "there is no difference in proportion of smokers in different genders"

#Alternate hypothesis--> H1 = "there is difference in proportion of smokers in different genders" 



# so from above we can see that 547 females are not smoker and 115 females are smoker

#like wise 517 males are not smoker and 159 males are smoker



#E11= expected value of smoker=no and sex=female

#E12 = expected value of smoker=no and sex= male

#E21 = expected value of smoker=yes and sex= female

#E22 = expected value of smoker=yes and sex= male



E11 = 1064*(662/1338)

print("expected value of smoker=no and sex=female is {}".format(E11))

E12 = 1064*(676/1338)

print("expected value of smoker=no and sex= male is {}".format(E12))

E21 = 274*(662/1338)

print("expected value of smoker=yes and sex= female is {}".format(E21))

E22 = 274*(676/1338)

print("expected value of smoker=yes and sex= male is {}".format(E22))



#chi-square = summation((observed-expected)^2/expected)

chiE11 = np.square(547-E11)/E11

print("chiE11 is {}".format(chiE11))

chiE12 = np.square(517-E12)/E12

print("chiE12 is {}".format(chiE12))

chiE21 = np.square(115-E21)/E21

print("chiE21 is {}".format(chiE21))

chiE22 = np.square(159-E22)/E22

print("chiE22 is {}".format(chiE22))



chisquare = chiE11+chiE12+chiE21+chiE22

print("chi-square is {}".format(chisquare))



degree_of_freedom = (2-1)*(2-1)

print("degree of freedom is {}".format(degree_of_freedom))



#having calculated the chi square value and degree of freedom ,we consult a chi-square

#table to check whether the chi square statistics of 7.76 exceeds the critical value

#of the chi square distribution, The critical value for alpha of 0.05(95% confidence) is 3.84

#since the statistics is much larger than 3.84, we have sufficient evidence to reject the H0(null hypothesis)



if chisquare<3.84:

    print("there is no difference in proportion of smokers in different genders(accept H0)")

else:

    print("there is difference in proportion of smokers in different genders(reject H0)")
#Null Hypothesis--> H0 = "The distribution of bmi across women with no children,one child and two children is not same"

#Alternate hypothesis--> H1 = "The distribution of bmi across women with no children,one child and two children is same"



#make the data frame of females having children less than or equal to 2



df = insurance_df[(insurance_df['children']<=2) & (insurance_df['sex']=='female')]

df
#plot the graph between bmi and children

jp = sns.jointplot(df['bmi'],df['children'])

jp = jp.annotate(stats.pearsonr, fontsize=10, loc=(0.1, 0.8))

plt.show()