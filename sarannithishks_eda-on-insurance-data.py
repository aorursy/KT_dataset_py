# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import seaborn as sns

import scipy.stats as stat

import matplotlib.pyplot as plt


ins_df = pd.read_csv("/kaggle/input/insurance.csv")

ins_df.shape




ins_df.dtypes


print(pd.isna(ins_df["age"]).value_counts())

print(pd.isna(ins_df["bmi"]).value_counts())

print(pd.isna(ins_df["sex"]).value_counts())

print(pd.isna(ins_df["children"]).value_counts())

print(pd.isna(ins_df["smoker"]).value_counts())

print(pd.isna(ins_df["region"]).value_counts())

print(pd.isna(ins_df["charges"]).value_counts())





#Since all columns have false values equal to the total number of rows, we can conclude that there are no missing values


ins_df.describe().drop(["children"],axis=1)



#The describe function gives the five point summary - min,25%,median(50%) ,75% and the max value of each of the columns

#Column "children" has been dropped here because it is not a continuous valued column and calculations in fractions will not be suitable for this column




#Distribution of BMI

sns.distplot(ins_df["bmi"])
#Additional plot showing the BMI distribution across genders



g = sns.FacetGrid(ins_df,col="sex")

g.map(sns.distplot, "bmi");





#Distribution of age



sns.distplot(ins_df["age"])
#Distribution of charges



sns.distplot(ins_df["charges"])
#3f. Measure of skewness of ‘bmi’, ‘age’ and ‘charges’ columns (2marks)





ins_df.drop(["children"],axis=1).skew(axis=0)



#Positive skewness values indicate that the data is right skewed 

#Age and Bmi columns are almost symmetrical with very minimal skew (0.055 and 0.2 respectively)

#Charges column is heavily skewed to the right with a value of 1.5



#Outliers in BMI columns.

#The BMI column has considerable outliers to the right side as shown in the box plot below

sns.boxplot(ins_df["bmi"])
#Outliers in Age columns.

#The Age column has no outliers and is symmetrically distributed

sns.boxplot(ins_df["age"])

#Outliers in Charged columns.

#The Charges column has lot of outliers to the right side as shown in the box plot below.

#This also shows heavy right skew in the distribution

sns.boxplot(ins_df["charges"])
#Finding the exact values of outliers

#Outliers are values which are below Q1-1.5IQR or above Q3+1.5IQR



fq, tq = ins_df["age"].quantile([0.25,0.75])

def findoutlier(column):

    fq, tq = ins_df[column].quantile([0.25,0.75]);

    iqr = tq - fq;

    for value in ins_df[column]:

        if ((value > (tq + (1.5*iqr))) or (value < (fq - (1.5*iqr)))):

            print(value);

            



print("---------------Bmi------------")            

print(findoutlier("bmi"));

print("---------------Age------------")

print(findoutlier("age"));

print("---------------Charges------------")

print(findoutlier("charges"));

    

    
#Count of people in various genders





xvalue = ins_df.sex.unique()

yvalue  = [ins_df["sex"].value_counts()[i] for i in ins_df.sex.unique()]







figure = sns.barplot(x = xvalue, y = yvalue)

figure.set(xlabel='Gender', ylabel='Count')

plt.show()

#Count of people having different number of children





xvalue = ins_df.children.unique()

yvalue  = [ins_df["children"].value_counts()[i] for i in ins_df.children.unique()]







figure = sns.barplot(x = xvalue, y = yvalue)

figure.set(xlabel='No of Children', ylabel='Count')

plt.show()
#Count of people who are smokers and non-smokers



xvalue = ins_df.smoker.unique()

yvalue  = [ins_df["smoker"].value_counts()[i] for i in ins_df.smoker.unique()]





#



figure = sns.barplot(x = xvalue, y = yvalue)

figure.set(xlabel='Smoker ?', ylabel='Count')

plt.show()
#Count of people across different regions

xvalue = ins_df.region.unique()

yvalue  = [ins_df["region"].value_counts()[i] for i in ins_df.region.unique()]



#sns.barplot(x =xvalue, y=yvalue)



figure = sns.barplot(x = xvalue, y = yvalue)

figure.set(xlabel='Regions', ylabel='Count')

plt.show()
#Since the categorical columns cannot be plotted ,we are assigning arbitrary numbers to denote them.



sex_coded= ins_df["sex"].astype('category').cat.codes

region_coded= ins_df["region"].astype('category').cat.codes

smoker_coded= ins_df["smoker"].astype('category').cat.codes



ins_df_coded = ins_df.drop(["sex","region","smoker"],axis=1)

ins_df_coded.insert(0,"sex",sex_coded)

ins_df_coded.insert(0,"region",region_coded)

ins_df_coded.insert(0,"smoker",smoker_coded)

                            

sns.pairplot(ins_df_coded)



                            
#4A Do charges of people who smoke differ significantly from the people who don't? (



#H0 : Charges do not differ whether they are smokers or not 

#Ha : Charges differ considerably based on smoker or not



#Choosing t test since it is a comparison of means between two independent samples.

#choosing the confidence interval as 95%



#alpha = 0.05



#The threshold pvalue is 0.05

#Critical value considering two tailed test : -2.064 to 2.064



#Considering the dataset as population

smokers_population = ins_df[ins_df["smoker"] == "yes"]

non_smokers_population = ins_df[ins_df["smoker"] == "no"]



#taking equal samples from each population



smokers_sample = smokers_population.sample(n=25,random_state=1)

non_smokers_sample = non_smokers_population.sample(n=25,random_state=1)



tstat,pvalue = stat.ttest_ind(smokers_sample["charges"],non_smokers_sample["charges"])



tstat,pvalue



#Since the tstat 8.5 is not within the critical value range, we reject the null hypothesis

#The pvalue is also less than the alpha value (0.05) .



#Hence we can clearly reject the null hypothesis

#We conculde that charges of people who smoke differ significantly from the people who don't.
#4B Does bmi of males differ significantly from that of females



#H0 : BMI does not differ between genders

#Ha : BMI differs between genders



#Choosing t test since it is a comparison of means between two independent samples.

#choosing the confidence interval as 95%



#alpha = 0.05



#The threshold pvalue is 0.05

#Critical value considering two tailed test : -2.064 to 2.064



#Considering the dataset as population

males_population = ins_df[ins_df["sex"] == "male"]

females_population = ins_df[ins_df["sex"] == "female"]



#taking equal samples from each population



males_sample = males_population.sample(n=25,random_state=1)

females_sample = females_population.sample(n=25,random_state=1)



tstat,pvalue = stat.ttest_ind(males_sample["bmi"],females_sample["bmi"])



tstat,pvalue



#Since the tstat -0.6 is well within the critical value range (-2.064 to 2.064), we fail to reject the null hypothesis

#The pvalue 0.5 is also greater than the alpha value (0.05) .



#Hence we  clearly fail to reject the null hypothesis.

# There is no significant difference in BMI between male and female

#4C Is the proportion of smokers significantly different in different genders



#H0 : The proportion of smokers is same across genders

#Ha : The proportion of smokers is different across genders



#Choosing chi-squared test since it is a comparison of proportion among categorical variables

#choosing the confidence interval as 95%



#alpha = 0.05



#The threshold pvalue is 0.05

#Critical value considering two tailed test and degree of freedom 1 : 0.001 on the left and  5.024 on the right



contingency_table = pd.crosstab(ins_df['sex'],ins_df['smoker']) 



chisq_value, p_value,dof,expected =  stat.chi2_contingency(contingency_table)



chisq_value,p_value



#Since the chisq_value 7.3 is not within the critical value range (0.001 to  5.024), we reject the null hypothesis

#The pvalue 0.006 is also lesser than the alpha value (0.05) .



#Hence we clearly reject the null hypothesis.

# There is significant difference in the proportion of smokers between male and female

#4D Is the distribution of bmi across women with no children, one child and two children, the same?







#H0 : The distribution of BMI across women is same irrespective of the number of children

#Ha : The distribution of BMI across women is different for different number of children



#Choosing ANOVA test since it is a comparison of variances across multiple independent samples

#choosing the confidence interval as 95%



#alpha = 0.05



#The threshold pvalue is 0.05

#Critical F value considering two tailed test  :0.025 on the left and  3.735 on the right







bmi_no = ins_df.loc[(ins_df["sex"] == "female") & (ins_df["children"] == 0)]["bmi"].sample(n=100,random_state=1)

bmi_one = ins_df.loc[(ins_df["sex"] == "female") & (ins_df["children"] == 1)]["bmi"].sample(n=100,random_state=1)

bmi_two = ins_df.loc[(ins_df["sex"] == "female") & (ins_df["children"] == 2)]["bmi"].sample(n=100,random_state=1)







f_stat,pvalue = stat.f_oneway(bmi_no,bmi_one,bmi_two)



f_stat,pvalue



#Since the f_stat 0.34 is well within the critical value range (0.025 to  3.735), we fail to reject the null hypothesis

#The pvalue 0.7 is also greater than the alpha value (0.05) .



#Hence we clearly fail to reject the null hypothesis.

#Hence the distribution of BMI across women is same irrespective of the number of children






