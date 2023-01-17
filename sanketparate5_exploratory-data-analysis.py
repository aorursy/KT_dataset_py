import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline

import statsmodels.api as sm

import scipy.stats as stats

from sklearn.preprocessing import LabelEncoder

import copy
sns.set() #setting the default seaborn style for our plots
import os

os.chdir("../input/insurance")
df = pd.read_csv('insurance.csv') # read the data as a data frame
df.head()  #checking the head of the data frame
df.info()  #info about the data
df.isna().apply(pd.value_counts)   #null value check
df.describe().T   # five point summary of the continuous attributes
#Plots to see the distribution of the continuous features individually



plt.figure(figsize= (20,15))

plt.subplot(3,3,1)

plt.hist(df.bmi, color='lightblue', edgecolor = 'black', alpha = 0.7)

plt.xlabel('bmi')



plt.subplot(3,3,2)

plt.hist(df.age, color='lightblue', edgecolor = 'black', alpha = 0.7)

plt.xlabel('age')



plt.subplot(3,3,3)

plt.hist(df.charges, color='lightblue', edgecolor = 'black', alpha = 0.7)

plt.xlabel('charges')



plt.show()
Skewness = pd.DataFrame({'Skewness' : [stats.skew(df.bmi),stats.skew(df.age),stats.skew(df.charges)]},

                        index=['bmi','age','charges'])  # Measure the skeweness of the required columns

Skewness
plt.figure(figsize= (20,15))

plt.subplot(3,1,1)

sns.boxplot(x= df.bmi, color='lightblue')



plt.subplot(3,1,2)

sns.boxplot(x= df.age, color='lightblue')



plt.subplot(3,1,3)

sns.boxplot(x= df.charges, color='lightblue')



plt.show()
plt.figure(figsize=(20,25))





x = df.smoker.value_counts().index    #Values for x-axis

y = [df['smoker'].value_counts()[i] for i in x]   # Count of each class on y-axis



plt.subplot(4,2,1)

plt.bar(x,y, align='center',color = 'lightblue',edgecolor = 'black',alpha = 0.7)  #plot a bar chart

plt.xlabel('Smoker?')

plt.ylabel('Count ')

plt.title('Smoker distribution')



x1 = df.sex.value_counts().index    #Values for x-axis

y1 = [df['sex'].value_counts()[j] for j in x1]   # Count of each class on y-axis



plt.subplot(4,2,2)

plt.bar(x1,y1, align='center',color = 'lightblue',edgecolor = 'black',alpha = 0.7)  #plot a bar chart

plt.xlabel('Gender')

plt.ylabel('Count')

plt.title('Gender distribution')



x2 = df.region.value_counts().index    #Values for x-axis

y2 = [df['region'].value_counts()[k] for k in x2]   # Count of each class on y-axis



plt.subplot(4,2,3)

plt.bar(x2,y2, align='center',color = 'lightblue',edgecolor = 'black',alpha = 0.7)  #plot a bar chart

plt.xlabel('Region')

plt.ylabel('Count ')

plt.title("Regions' distribution")



x3 = df.children.value_counts().index    #Values for x-axis

y3 = [df['children'].value_counts()[l] for l in x3]   # Count of each class on y-axis



plt.subplot(4,2,4)

plt.bar(x3,y3, align='center',color = 'lightblue',edgecolor = 'black',alpha = 0.7)  #plot a bar chart

plt.xlabel('No. of children')

plt.ylabel('Count ')

plt.title("Children distribution")



plt.show()

#Label encoding the variables before doing a pairplot because pairplot ignores strings

df_encoded = copy.deepcopy(df)

df_encoded.loc[:,['sex', 'smoker', 'region']] = df_encoded.loc[:,['sex', 'smoker', 'region']].apply(LabelEncoder().fit_transform) 



sns.pairplot(df_encoded)  #pairplot

plt.show()
df.smoker.value_counts()
#Scatter plot to look for visual evidence of dependency between attributes smoker and charges accross different ages

plt.figure(figsize=(8,6))

sns.scatterplot(df.age, df.charges,hue=df.smoker,palette= ['red','green'] ,alpha=0.6)

plt.show()
# T-test to check dependency of smoking on charges

Ho = "Charges of smoker and non-smoker are same"   # Stating the Null Hypothesis

Ha = "Charges of smoker and non-smoker are not the same"   # Stating the Alternate Hypothesis



x = np.array(df[df.smoker == 'yes'].charges)  # Selecting charges corresponding to smokers as an array

y = np.array(df[df.smoker == 'no'].charges) # Selecting charges corresponding to non-smokers as an array



t, p_value  = stats.ttest_ind(x,y, axis = 0)  #Performing an Independent t-test



if p_value < 0.05:  # Setting our significance level at 5%

    print(f'{Ha} as the p_value ({p_value}) < 0.05')

else:

    print(f'{Ho} as the p_value ({p_value}) > 0.05')
df.sex.value_counts()   #Checking the distribution of males and females
plt.figure(figsize=(8,6))

sns.scatterplot(df.age, df.charges,hue=df.sex,palette= ['pink','lightblue'] )

plt.show()
# T-test to check dependency of bmi on gender

Ho = "Gender has no effect on bmi"   # Stating the Null Hypothesis

Ha = "Gender has an effect on bmi"   # Stating the Alternate Hypothesis



x = np.array(df[df.sex == 'male'].bmi)  # Selecting bmi values corresponding to males as an array

y = np.array(df[df.sex == 'female'].bmi) # Selecting bmi values corresponding to females as an array



t, p_value  = stats.ttest_ind(x,y, axis = 0)  #Performing an Independent t-test



if p_value < 0.05:  # Setting our significance level at 5%

    print(f'{Ha} as the p_value ({p_value.round()}) < 0.05')

else:

    print(f'{Ho} as the p_value ({p_value.round(3)}) > 0.05')
# Chi_square test to check if smoking habits are different for different genders

Ho = "Gender has no effect on smoking habits"   # Stating the Null Hypothesis

Ha = "Gender has an effect on smoking habits"   # Stating the Alternate Hypothesis



crosstab = pd.crosstab(df['sex'],df['smoker'])  # Contingency table of sex and smoker attributes



chi, p_value, dof, expected =  stats.chi2_contingency(crosstab)



if p_value < 0.05:  # Setting our significance level at 5%

    print(f'{Ha} as the p_value ({p_value.round(3)}) < 0.05')

else:

    print(f'{Ho} as the p_value ({p_value.round(3)}) > 0.05')

crosstab
# Chi_square test to check if smoking habits are different for people of different regions

Ho = "Region has no effect on smoking habits"   # Stating the Null Hypothesis

Ha = "Region has an effect on smoking habits"   # Stating the Alternate Hypothesis



crosstab = pd.crosstab(df['smoker'], df['region'])  # Contingency table of sex and smoker attributes



chi, p_value, dof, expected =  stats.chi2_contingency(crosstab)



if p_value < 0.05:  # Setting our significance level at 5%

    print(f'{Ha} as the p_value ({p_value.round(3)}) < 0.05')

else:

    print(f'{Ho} as the p_value ({p_value.round(3)}) > 0.05')

crosstab
# Test to see if the distributions of bmi values for females having different number of children, are significantly different



Ho = "No. of children has no effect on bmi"   # Stating the Null Hypothesis

Ha = "No. of children has an effect on bmi"   # Stating the Alternate Hypothesis





female_df = copy.deepcopy(df[df['sex'] == 'female'])



zero = female_df[female_df.children == 0]['bmi']

one = female_df[female_df.children == 1]['bmi']

two = female_df[female_df.children == 2]['bmi']





f_stat, p_value = stats.f_oneway(zero,one,two)





if p_value < 0.05:  # Setting our significance level at 5%

    print(f'{Ha} as the p_value ({p_value.round(3)}) < 0.05')

else:

    print(f'{Ho} as the p_value ({p_value.round(3)}) > 0.05')