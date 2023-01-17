import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

import warnings

warnings.filterwarnings('ignore')



df = pd.read_csv("/kaggle/input/framingham-heart-study-dataset/framingham.csv")

df.head(5)
#------------------------------------#

#### About the Framingham Study  #####

#------------------------------------#



# The objectives of the Framingham Study are to study the incidence and prevalence of cardiovascular disease (CVD) 

# and its risk factors over time. The study began in 1948 under the U.S. Public Health Service. Participants were 

# sampled from Framingham, Massachusetts, including both men and women. This was the first prospective study of 

# cardiovascular disease and identified the concept of risk factors and their joint effects. 



# The study has continued to examine participants every two years and is currently supported by a contract to Boston University.

# As of February 28, 1999, there are 993 surviving participants. Examination of participants has taken place every two years 

# and the cohort has been followed for morbidity and mortality over that time period.



# The cardiovascular disease conditions under investigation include coronary heart disease (angina pectoris, myocardial infarction, 

# coronary insufficiency and sudden and non-sudden death), stroke, hypertension, peripheral arterial disease and congestive heart failure. 
print('#---------------------------#')

print('####   Target Variable  #####')

print('#---------------------------#\n')



print('10 year risk of coronary heart disease CHD')

print('(0 = “No”, 1 = “Yes”, )\n')

print(df['TenYearCHD'].value_counts())
print('#--------------------------------#')

print('####  Correlation Analysis   #####')

print('#--------------------------------#')



CHD_corr = df.corr()['TenYearCHD']

corr_CHD = CHD_corr.abs().sort_values(ascending=False)[1:]; round(corr_CHD,2)
print('Reduce the variables to only those with a correlation of 10% or more')



df = df[['TenYearCHD','age','sysBP','prevalentHyp','diaBP','glucose',

         'diabetes']]; df.describe()  
print('######   Identify missing values   #######')



df.isnull().sum().sort_values(ascending = False)

# null values as a percentage

# round((df.isnull().sum()/df.isnull().count()),2).sort_values(ascending = False)
print('####   Visuaize the nulls with a heatmap   #####')



sns.heatmap(df.isnull(), yticklabels = False,

           cbar = False,  cmap= 'Greys')

# College is missing the most, then Salary,

# we can also see one row is missing all values
print('#####    Replace missing glucose values with average glucose level  #######\n')

print('average glusoce level = {:,.1f}'.format(df.glucose.mean()))

plt.hist(df.glucose, bins = 15, color = '#6f828a') # visual check



# replace with the mean

df.glucose.fillna(df.glucose.mean(), inplace = True)

# check to see if any missing values now

#print('missing values from glucose = {:,.0f}'.format(df.glucose.isna().sum()))
print('check for any missing values')

df.isnull().sum().sort_values(ascending = False)
col_list = ['steel grey']

col_list_palette = sns.xkcd_palette(col_list)

sns.set_palette(col_list_palette)



print('Create a pairplot to look identify data types and get a summary of the distributions')

sns.pairplot(data=df, markers="+", palette = sns.set_palette(col_list_palette))
print('#-----------------------#')

print('####   Data Types   #####')

print('#-----------------------#\n')

print(df.dtypes)
print('#----------------------------#')

print('####   Categorical Data   ####')

print('#----------------------------#\n')



print('Change numeric data to categorical')

#df.TenYearCHD.value_counts()  

#df.prevalentHyp.value_counts()

#df.diabetes.value_counts() 



#df[['TenYearCHD','prevalentHyp','diabetes']].dtypes

# change to categorical data types

df[['TenYearCHD','prevalentHyp','diabetes']] = df[['TenYearCHD','prevalentHyp','diabetes']].astype('category')

print(df.dtypes)



####   Continuous Data   #####



#df.age.value_counts()  

#df.sysBP.value_counts()  

#df.diaBP.value_counts()  

#df.glucose.value_counts() 
print('####   Rename columns for legibility    ####') 



#df.columns

df.rename(columns = {'TenYearCHD' : 'At Risk',

                     'age' : 'Age',

                     'sysBP' : 'Systolic Blood Pressure',

                     'prevalentHyp' : 'Hypertensive',

                     'diaBP' : 'Diastolic Blood Pressure',

                     'glucose' : 'Glucose Levels',

                     'diabetes' : 'Diabetes',

                     'male': 'Gender',

                     'BPMeds' : 'Blood Pressure Medication',

                     'totChol' : 'Total Cholesterol Level',

                     'BMI' : 'Body Mass Index'}, inplace = True); df.dtypes
    #----------------------------------------------#

    #####    assign categorical data types    ######

    #####  change numbers to English labels   ######

    #----------------------------------------------#



#----  'At Risk of Coronary Heart Disease'  ----#



#df['At Risk of Coronary Heart Disease'].value_counts()

atrisk = {1 : "YES", 0 : "NO"}

df['At Risk'] = [atrisk[item] for item in df['At Risk']]



# check to make sure it worked

#df['At Risk'].value_counts()



#----  'Hypertensive'  ----#



hyper = {1 : "YES", 0 : "NO"}

df['Hypertensive'] = [hyper[item] for item in df['Hypertensive']]



# check to make sure it worked

#df['Hypertensive'].value_counts()



#----  'Diabetes'  ----#



diab = {1 : "YES", 0 : "NO"}

df['Diabetes'] = [diab[item] for item in df['Diabetes']]



# check to make sure it worked

#df['Diabetes'].value_counts()
        #-----------------------------------#

        #####   Data Visualizations    ######

        #-----------------------------------#



#col_list = ['rosa', 'blood']

col_list = ['pink', 'crimson']

col_list_palette = sns.xkcd_palette(col_list)

sns.set_palette(col_list_palette)



sns.set(rc={"figure.figsize": (10,6)},

            palette = sns.set_palette(col_list_palette),

            context="talk",

            style="ticks")
#-------------------------------------------#

####  Count Plots for categorical data  #####

#-------------------------------------------#



#----  At Risk of Coronary Heart Disease  ----#



# df['At Risk of Coronary Heart Disease'].value_counts()

sns.countplot(x = 'At Risk', data=df, edgecolor = "black",

              palette = sns.set_palette(col_list_palette))

              #palette = sns.color_palette("PiYG", 3))

plt.suptitle('At Risk of Coronary Heart Disease', fontsize = 20)

plt.xlabel('At Risk', fontsize = 20)
#----  'Hypertensive'  ----#



#df['Hypertensive'].value_counts()

# how many people are in the study male and female?

sns.countplot(x = 'Hypertensive', data=df, edgecolor = "black")

plt.suptitle('Study Participants by Hypertensive', fontsize = 20)

plt.xlabel('Hypertensive', fontsize = 20)
#----------------------------------------#

####      At Risk & Hypertensive      ####

#----------------------------------------#



g = sns.catplot(x='Hypertensive', hue='At Risk', aspect = 2, # col="Sex"

                data=df, kind="count", height=5,

                #hue_order = ["No", "Yes"], 

                edgecolor = "black")

g.set_axis_labels('Hypertensive', "Counts")

plt.suptitle('At Risk & Hypertensive', fontsize = 20)
#----  'Diabetes'  ----#



# df['Diabetes'].value_counts()

sns.countplot(x = 'Diabetes', data=df, edgecolor = "black",

              palette = sns.set_palette(col_list_palette))

plt.suptitle('Diabetes', fontsize = 20)

plt.xlabel('Diabetes', fontsize = 20)
#------------------------------------#

####      At Risk & Diabetes      ####

#------------------------------------#



g = sns.catplot(x='Diabetes', hue='At Risk', aspect = 2, # col="Sex"

                data=df, kind="count", height=5,

                #hue_order = ["No", "Yes"], 

                edgecolor = "black")

g.set_axis_labels('At Risk of Coronary Heart Disease', "Counts")

plt.suptitle('At Risk & Diabetes', fontsize = 20)
#-----------------------------------------#

#####   Distribution Visualizations   #####

#-----------------------------------------#



#--------------------------------#

######    At Risk by Age    ######

#--------------------------------#



g = sns.FacetGrid(data = df, col='At Risk',

                  hue = 'At Risk',height = 6)

g.map(plt.hist, 'Age', edgecolor = "black", bins = 20)

g.set_axis_labels('Age', 'Counts')

plt.suptitle('At Risk & Age')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])       
#----------------------------------------------#

####   At Risk by Systolic Blood Pressure   ####

#----------------------------------------------#



g = sns.FacetGrid(data = df, col='At Risk', hue = 'At Risk',height = 6)

g.map(plt.hist, 'Systolic Blood Pressure', edgecolor = "black", bins = 20)

g.set_axis_labels('Systolic Blood Pressure', 'Counts')

plt.suptitle('At Risk & Systolic Blood Pressure')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
#-----------------------------------------------#

####   At Risk by Diastolic Blood Pressure   ####

#-----------------------------------------------#



g = sns.FacetGrid(data = df, col='At Risk',hue = 'At Risk',height = 6)

g.map(plt.hist, 'Diastolic Blood Pressure', edgecolor = "black", bins = 20)

g.set_axis_labels('Diastolic Blood Pressure', 'Counts')

plt.suptitle('At Risk & Diastolic Blood Pressure')

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
#------------------------------------#

####   At Risk by Glucose Levels  ####

#------------------------------------#



# change the bin height to reduce or enhance detail

g = sns.FacetGrid(data = df, col='At Risk',hue = 'At Risk',height = 6)

g.map(plt.hist, 'Glucose Levels', edgecolor = "black", bins = 20)

g.set_axis_labels('Glucose Levels', 'Counts')

plt.suptitle('At Risk & Glucose Levels')

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
#----------------------------------------#

####  Boxplots for categorical data  #####

#----------------------------------------#



boxprops = {'edgecolor': 'k', 'linewidth': 2}

lineprops = {'color': 'k', 'linewidth': 2}

kwargs = {'hue_order': ["At Risk", "Healthy"]}

boxplot_kwargs = dict({'boxprops': boxprops, 'medianprops': lineprops,

                       'whiskerprops': lineprops, 'capprops': lineprops,

                       'width': 0.75}, **kwargs)



#--------------------------------#

######    At Risk by Age    ######

#--------------------------------#



sns.boxplot(x = 'At Risk', y = 'Age', data=df, **boxplot_kwargs)

plt.suptitle('At Risk & Age', fontsize = 20)

plt.xlabel('At Risk of Coronary Heart Disease', fontsize = 16)
#----------------------------------------------------#

######    At Risk by Systolic Blood Pressure    ######

#----------------------------------------------------#



sns.boxplot(x = 'At Risk',y = 'Systolic Blood Pressure', data=df, **boxplot_kwargs)

plt.suptitle('At Risk & Systolic Blood Pressure', fontsize = 20)

plt.xlabel('At Risk of Coronary Heart Disease', fontsize = 16)
#-----------------------------------------------------#

######    At Risk by Diastolic Blood Pressure    ######

#-----------------------------------------------------#



sns.boxplot(x = 'At Risk',y = 'Diastolic Blood Pressure', data=df, **boxplot_kwargs)

plt.suptitle('At Risk & Diastolic Blood Pressure', fontsize = 20)

plt.xlabel('At Risk of Coronary Heart Disease', fontsize = 16)
#-----------------------------------------#

######    At Risk by Glucose Levels  ######

#-----------------------------------------#



sns.boxplot(x = 'At Risk',y = 'Glucose Levels', data=df, **boxplot_kwargs)

plt.suptitle('At Risk & Glucose Levels', fontsize = 20)

plt.xlabel('At Risk of Coronary Heart Disease', fontsize = 16)
#------------------------------#

#####     scatterplots     #####

#------------------------------#



#-----------------------------------------------------#

####   At Risk by Age & Systolic Blood Pressure'   ####

#-----------------------------------------------------#



sns.lmplot(x="Age", y='Systolic Blood Pressure', hue='At Risk',data=df, height = 9) 

plt.suptitle('At Risk by Systolic Blood Pressure')
#----------------------------------------------------#

####   At Risk by Age & Systolic Blood Pressure   ####

#----------------------------------------------------#



sns.lmplot(x='Age', y='Diastolic Blood Pressure', hue='At Risk',data=df, height = 9) 

plt.suptitle('At Risk by Age & Diastolic Blood Pressure')
#----------------------------------------------------------------#

####   At Risk by Diastolic Blood Pressure & Glucose Levels   ####

#----------------------------------------------------------------#



sns.lmplot(x='Age', y='Glucose Levels', hue='At Risk',data=df, height = 9) 

plt.suptitle('At Risk by Age & Glucose Levels')
#------------------------------------------------------------#

#####    Statistical Comparison of at Risk and Healthy   #####

#####      compare the means of at Risk and Healthy      #####

#------------------------------------------------------------#



from scipy import stats

from statistics import variance



#-------------------------------------------------#

####   Healthy and At Risk of Heart Disease   #####

#-------------------------------------------------#



atRisk = df[df['At Risk'] == 'YES']

healthy = df[df['At Risk'] == 'NO']



#--------------------------#

####  At Risk by Age   #####

#--------------------------#



atRisk_age = atRisk['Age']

healthy_age = healthy['Age']



# mean, variance, and standard deviation

(atRisk_age.mean(), healthy_age.mean())

(atRisk_age.std(), healthy_age.std())
#-------------------------------------#

####   Equal Variance and T Test  #####

#-------------------------------------#



####   are the variances equal?   #####

stats.levene(atRisk_age, healthy_age)

# tests the null hypothesis that all input samples are from

# populations with equal variances



# p-value is just over 0.05 so I ran the test for both equal variance TRUE & FALSE

# results ot t-test were not affected either way. P-value of t-test was below 0.05

# so we must reject the NULL hypothesis that the means are equal



####    are the means the same?   #####

stats.ttest_ind(atRisk_age, healthy_age, equal_var = True)

# two-sided test for the null hypothesis that 2 independent samples

# have identical average (expected) values.
#---------------------------------------------------------------#

####    Visualization of Health and At Risk Distributions    ####

#---------------------------------------------------------------#



plt.figure(figsize=(11,6))

sns.kdeplot(healthy_age ,lw=8, shade=True, label='Healthy', alpha = 0.7)

plt.axvline(np.mean(healthy_age), linestyle='--', linewidth = 5,color = '#fe86a4')

sns.kdeplot(atRisk_age,lw=8, shade=True,label='At Risk', alpha = 0.7)

plt.axvline(np.mean(atRisk_age), linestyle='--', linewidth = 5, color='darkred')

plt.suptitle('At Risk by Age', fontsize = 20)

plt.xlabel('Age');plt.ylabel('Frequency');plt.legend()
#---------------------------------#

####   binning the age data   #####

#---------------------------------#



amn = min(df.Age)-2

amx = max(df.Age)

b = np.linspace(amn, amx, num=9).astype(int)

df['age_binned'] = pd.cut(df['Age'], bins = b); df.head(10)



# Group by the bin and calculate averages

avg_age  = df.groupby('age_binned').mean()
#---------------------------------------------#

#####   Age and Associated Medical Cost   #####

#---------------------------------------------#



plt.bar(avg_age.index.astype(str), avg_age['Systolic Blood Pressure'], edgecolor = "black")

plt.xticks(rotation = 75); plt.xlabel('Age'); plt.ylabel('Systolic Blood Pressure')

plt.suptitle('Age and Systolic Blood Pressure');
#--------------------------------------------------#

####  At Risk by Age & Systolic Blood Pressure  ####

#--------------------------------------------------#



plt.subplots(figsize=(10,8))

g = sns.boxplot(x = 'age_binned', y = "Systolic Blood Pressure", hue = "At Risk",data=df)

plt.suptitle('At Risk by Age & Systolic Blood Pressure', fontsize = 20)

plt.xlabel('Age', fontsize = 20)

plt.xticks(rotation = 70)

plt.legend(loc='upper left', title = "At Risk")
#---------------------------------------------------#

####  At Risk by Age & Diastolic Blood Pressure  ####

#---------------------------------------------------#

df.columns

plt.subplots(figsize=(10,8))

g = sns.boxplot(x = 'age_binned', y = 'Diastolic Blood Pressure', hue = "At Risk",data=df)

plt.suptitle('At Risk by Age & Diastolic Blood Pressure', fontsize = 20)

plt.xlabel('Age', fontsize = 20)

plt.xticks(rotation = 70)

plt.legend(loc='upper left', title = "At Risk")
#-----------------------------------------#

####  At Risk by Age & Glucose Levels  ####

#-----------------------------------------#



plt.subplots(figsize=(10,8))

g = sns.boxplot(x = 'age_binned', y = 'Glucose Levels', hue = "At Risk",data=df)

plt.suptitle('At Risk by Age & Glucose Levels', fontsize = 20)

plt.xlabel('Age', fontsize = 20)

plt.xticks(rotation = 70)

plt.ylim(40,200)

plt.legend(loc='upper left', title = "At Risk")