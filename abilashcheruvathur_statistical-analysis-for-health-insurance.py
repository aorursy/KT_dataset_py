



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import numpy as np

import pandas as pd
#Reading the csv file "DATA-SET" from the home directory using pandas and assigning to 'data_frame'

data_frame=pd.read_csv("/kaggle/input/health-insurance/insurance.csv")
#To view the first 10 rows of the dataframe

data_frame.head(10)
shape=data_frame.shape

print("The Shape of the data set is"+str(shape))
#to find the information of the dataframe

info=data_frame.info()

info
d_types=data_frame.dtypes

d_types
null_table=data_frame.isnull().values.any()

null_table

if(null_table):

    print("There are missing values in the data set")

else:

    print("There are no missing values in the data set")

#five point summary of age

Five_point_summary_age=data_frame['age'].describe()

Five_point_summary_age
#Five point summary of bmi

Five_point_summary_bmi=data_frame['bmi'].describe()

Five_point_summary_bmi
#Five point summary of children

Five_point_summary_children=data_frame['children'].describe()

Five_point_summary_children
#Five point summary of charges

Five_point_summary_charges=data_frame['charges'].describe()

Five_point_summary_charges
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#Visualization of 5 point summary of age

sns.boxplot(data_frame['age'])
#Visualization of 5 point summary of bmi

sns.boxplot(data_frame['bmi'])
#Visualization of 5 point summary of children

sns.boxplot(data_frame['children'])
#Visualization of 5 point summary of charges

sns.boxplot(data_frame['charges'])
#Visualization of distribution of 'bmi'- Seems to follow normal distribution

sns.distplot(data_frame['bmi'])
#visulaization of distribution of 'charges'- Seems to be exponential or gamma distribution

sns.distplot(data_frame['charges'])
#Visulation of distribution of 'age'- Seems to be uniform distribution

sns.distplot(data_frame['age'])
#To measure the skew of 'age'

skew_age=data_frame['age'].skew(axis=0)

skew_age_string=str(skew_age)

print("The skew for 'age' is "+skew_age_string)
#To measure the skew of 'bmi'

skew_bmi=data_frame['bmi'].skew(axis=0)

skew_bmi_string=str(skew_bmi)

print("The skew for 'bmi' is "+skew_bmi_string)
#To measure the skew of 'charges'

skew_charges=data_frame['charges'].skew(axis=0)

skew_charges_string=str(skew_charges)

print("The skew for 'charges' is "+skew_charges_string)
from scipy.stats import skew
#Using scipy.stats.skew to identify skew

Skew_age=str(skew(data_frame['age']))

Skew_bmi=str(skew(data_frame['bmi']))

Skew_charges=str(skew(data_frame['charges']))



print("The skew values for age, bmi and charges are "+Skew_age,Skew_bmi,Skew_charges)
# Function to detect outliers

outlier=[]

def outlier_data(data):

    mean= np.mean(data)

    standard_deviation=np.std(data)

    threshold=3

    outlier=[]



    

    for x in data:

        z_score=(x-mean)/standard_deviation

        if np.abs(z_score)>threshold:

            outlier.append(x)

            

    return outlier
#The outliers of BMI

outliers_bmi=outlier_data(data_frame['bmi'])

print("The outliers of bmi are "+str(outliers_bmi))

print("The # of outliers of bmi are "+str(len(outliers_bmi)))
#The outliers of Age

outliers_age=outlier_data(data_frame['age'])

print("The outliers of age are "+str(len(outliers_age)))
#The outliers of charges

outliers_charges=outlier_data(data_frame['charges'])

print("The outliers of charges are "+str(outliers_charges))

print("The # of outliers of charges are "+str(len(outliers_charges)))
#Categorical visualization of sex

sns.countplot(data_frame['sex'])
#Categorical visualization of smoker

sns.countplot(data_frame['smoker'])
#Categorical visualization of region

sns.countplot(data_frame['region'])
#Categorical visualization of 'children'

sns.countplot(data_frame['children'])
#pair plot to decribe the relations between different attributes

sns.pairplot(data_frame)
#To visualize relation between sex and charges

sns.catplot(x='sex',y='charges', data=data_frame)
#To visualize relation between smoker and charges

sns.catplot(x='smoker',y='charges', data=data_frame)
#To visualize relation between smoker and region

sns.catplot(x='region',y='charges', data=data_frame)
# Grouping the data by 'smoker' and viewing the charges

groupby_data_smoker=data_frame.groupby('smoker')['charges']

groupby_data_smoker.describe().transpose()
#Creating a new dataframe with just smoker and charges

data_frame_2=data_frame[['smoker','charges']]

#data_frame_2
#Grouping data based on Smokers='Yes'

grouped_Data_smokers=data_frame_2[data_frame_2['smoker']=='yes']

#grouped_Data_smokers
#Grouping data based on Smokers='No'

grouped_Data_non_smokers=data_frame_2[data_frame_2['smoker']=='no']
#Taking up charges for smokers

charges_of_smokers=grouped_Data_smokers['charges']
#Taking up charges for non-smokers

charges_of_non_smokers=grouped_Data_non_smokers['charges']
#Converting to numpy array of charges of smokers and taking a sample

array_of_smokers=np.array(charges_of_smokers)

len(array_of_smokers)

sample_array_smokers=np.random.choice(a=array_of_smokers,size=100)
#Converting to numpy array of charges of non-smokers and taking a sample

array_of_non_smokers=np.array(charges_of_non_smokers)

len(array_of_non_smokers)

sample_array_non_smokers=np.random.choice(a=array_of_non_smokers,size=100)
#mean of array of smokers

mean_smokers=sample_array_smokers.mean()

mean_smokers
#mean of array of non-smokers

mean_non_smokers=sample_array_non_smokers.mean()

mean_non_smokers
from scipy import stats

alpha=0.01
#calculating the t-statistic and p-value

tstatistic,p_value=stats.ttest_ind(sample_array_smokers,sample_array_non_smokers)
print("T-statistic is "+str(tstatistic))
print("P-value is "+str(p_value))
if(p_value<alpha):

    print("The p-value "+str(p_value)+" is less than alpha, Hence reject the null hypothesis and we infer charges for smokers are not equal to charges for non-smokers")

else:

    print("The p-value "+str(p_value)+" is more than alpha, Hence failing to reject the null hypothesis  and we infer charges for smokers equal to charges for non-smokers")
#Creating a new dataframe with just bmi and sex

data_frame_3=data_frame[['sex','bmi']]
#Grouping data based on sex='male'

grouped_Data_male=data_frame[data_frame_3['sex']=='male']
#Grouping data based on sex='female'

grouped_Data_female=data_frame[data_frame_3['sex']=='female']
#Taking up bmi for male

male_bmi=grouped_Data_male['bmi']
#Taking up bmi for female

female_bmi=grouped_Data_female['bmi']
#Converting to numpy array of bmi of male

array_of_male_bmi=np.array(male_bmi)

len(array_of_male_bmi)

sample_array_male_bmis=np.random.choice(a=array_of_male_bmi,size=100)
#Converting to numpy array of bmi of female

array_of_female_bmi=np.array(female_bmi)

len(array_of_female_bmi)

sample_array_female_bmis=np.random.choice(a=array_of_female_bmi,size=100)
mean_male_bmi= sample_array_male_bmis.mean()

mean_male_bmi
mean_female_bmi=sample_array_female_bmis.mean()

mean_female_bmi
alpha1=0.01
tstatistic_1,p_value_1= stats.ttest_ind(sample_array_male_bmis,sample_array_female_bmis)
print("T-statistic is "+str(tstatistic_1))
print("P-value is "+str(p_value_1))
if p_value_1<alpha1:

    print("The p-value "+str(p_value_1)+" is less than alpha, Hence reject the null hypothesis  and we infer bmi for males are not equal to bmi for females ")

else:

    print("The p-value "+str(p_value_1)+" is more than alpha, Hence failing to reject the null hypothesis  and we infer bmi for males equal to bmi for females")
#Creating a new data_frame with just 'sex' and 'smoker'

data_frame_4=data_frame[['sex','smoker']]

#data_frame_4
#Grouping data based on sex='male' and sex='female'

grouped_male=data_frame_4[data_frame_4['sex']=='male'].smoker.value_counts()

print(grouped_male)

grouped_female=data_frame_4[data_frame_4['sex']=='female'].smoker.value_counts()

print(grouped_female)
#Taking only the smokers in both male and female in consideration

[male_smokers,female_smokers]=[grouped_male[1],grouped_female[1]]

print([male_smokers,female_smokers])
male_female=data_frame_4.sex.value_counts()
#taking in the number of males and females 

number_of_males=male_female[0]

number_of_females=male_female[1]

print([number_of_males,number_of_females])
propotion_of_male_smoker=159/676

print("The propotion of male smoker is "+str(propotion_of_male_smoker))
propotion_of_female_smoker=115/662

print("The propotion of female smoker is "+str(propotion_of_female_smoker))


from statsmodels.stats.proportion import proportions_ztest



alpha2=0.01

z_stat, p_value2=proportions_ztest([male_smokers,female_smokers],[number_of_males,number_of_females])

print("The value of the z-statistic is "+str(z_stat))

print("The p-value associated is "+str(p_value2))
if p_value2<alpha2:

    print("The p-value "+str(p_value2)+" is less than alpha, Hence reject the null hypothesis and we infer that the propotions are not equal")

else:

    print("The p-value "+str(p_value2)+" is more than alpha, Hence failing to reject the null hypothesis and we infer that the propotions are equal")
data_frame_5=data_frame[['sex','children','bmi']]
data_frame_5.head()
#Filtering out the data-frame for only female that we are interested in

female_data_frame=data_frame_5[data_frame_5['sex']=='female']

female_data_frame.head()
#Further down filtering down only females with no children

female_no_children=female_data_frame[female_data_frame['children']==0]

female_no_children=female_no_children.drop(['children'],axis=1)

female_no_children=female_no_children.reset_index()

female_no_children=female_no_children.drop(['index'],axis=1)

female_no_children.head()
#taking a sample of 100 females with no children and converting to numpy array

bmi_female_no_children=female_no_children['bmi']

sample_array_female_no_children_bmis=np.random.choice(a=bmi_female_no_children,size=100)

sample_array_female_no_children_bmis
#Further down filtering down only females with 1 children

female_1_children=female_data_frame[female_data_frame['children']==1]

female_1_children=female_1_children.drop(['children'],axis=1)

female_1_children=female_1_children.reset_index()

female_1_children=female_1_children.drop(['index'],axis=1)

female_1_children.head()
#taking a sample of 100 females with 1 children and converting to numpy array

bmi_female_1_children=female_1_children['bmi']

sample_array_female_1_children_bmis=np.random.choice(a=bmi_female_1_children,size=100)

sample_array_female_1_children_bmis
#Further down filtering down only females with 2 children

female_2_children=female_data_frame[female_data_frame['children']==2]

female_2_children=female_2_children.drop(['children'],axis=1)

female_2_children=female_2_children.reset_index()

female_2_children=female_2_children.drop(['index'],axis=1)

female_2_children.head()
#taking a sample of 100 females with 2 children and converting to numpy array

bmi_female_2_children=female_2_children['bmi']

sample_array_female_2_children_bmis=np.random.choice(a=bmi_female_1_children,size=100)

sample_array_female_2_children_bmis
print("The sample mean for female with no children is "+str(sample_array_female_no_children_bmis.mean()))
print("The sample mean for female with 1 child is "+str(sample_array_female_1_children_bmis.mean()))
print("The sample mean for female with 2 children is "+str(sample_array_female_2_children_bmis.mean()))
#Let us visualize the mean bmi graphically

mean_bmi_df = pd.DataFrame()



dframe1            = pd.DataFrame({'Children': '0', 'BMI':sample_array_female_no_children_bmis})

dframe2            = pd.DataFrame({'Children': '1', 'BMI':sample_array_female_1_children_bmis})

dframe3            = pd.DataFrame({'Children': '2', 'BMI':sample_array_female_2_children_bmis})



mean_bmi_df = mean_bmi_df.append(dframe1) 

mean_bmi_df = mean_bmi_df.append(dframe2) 

mean_bmi_df = mean_bmi_df.append(dframe3) 

sns.boxplot(x = "Children", y = "BMI", data = mean_bmi_df)

plt.show()  
import statsmodels.api         as     sm

from   statsmodels.formula.api import ols

 

anova = ols('BMI ~ Children', data = mean_bmi_df).fit()

anova_table = sm.stats.anova_lm(anova, typ=2)

print(anova_table)