# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt   

import seaborn as sns

import scipy.stats as st

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Load the dataset



pima_df = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")
#Top 5 rows

pima_df.head(5)
#Shape of the data

pima_df.shape
#Checking for the info of the dataset



pima_df.info()
#Lets analysze the distribution of the various attributes

pima_df.describe().transpose()
# Let us check whether any of the columns has any value other than numeric i.e. data is not corrupted such as a "?" instead of 

# a number.



# we use np.isreal a numpy function which checks each column for each row and returns a bool array, 

# where True if input element is real.

# applymap is pandas dataframe function that applies the np.isreal function columnwise

# Following line selects those rows which have some non-numeric value in any of the columns hence the  ~ symbol



pima_df[~pima_df.applymap(np.isreal).all(1)]
#There are 0 values in the dataset in the Glucose,BloodPressure,SkinThickness, Insulin and BMI, we need to replace them with the NAN 



pima_df[["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]]=pima_df[["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]].replace(0, np.NaN)
pima_df.isnull().any()
#Checking for the missing values in the dataset



pima_df.isna().sum()
#Replacing the null values with the mean and median respectively



pima_df['Glucose'].fillna(pima_df['Glucose'].mean(), inplace = True)

pima_df['BloodPressure'].fillna(pima_df['BloodPressure'].mean(),inplace=True)

pima_df['SkinThickness'].fillna(pima_df['SkinThickness'].median(),inplace=True)

pima_df['Insulin'].fillna(pima_df['Insulin'].median(),inplace=True)

pima_df['BMI'].fillna(pima_df['BMI'].median(),inplace=True)
#Convert the target column to a categorical variable

pima_df['Outcome']=pima_df['Outcome'].astype('category')
#Distribution of the target class



sns.countplot(pima_df['Outcome'])

#pima_df['Outcome'].value_counts(normalize=True).plt()
pima_df['Outcome'].value_counts(normalize=True)
# Let us look at the target column which is 'class' to understand how the data is distributed amongst the various values

pima_df.groupby(["Outcome"]).mean() 
# Pairplot using seaborn



sns.pairplot(pima_df, hue='Outcome')
from scipy.stats import zscore





numeric_cols = pima_df.drop('Outcome', axis=1)



# Copy the 'class' column alone into the y dataframe. This is the dependent variable

class_values = pd.DataFrame(pima_df[['Outcome']])



numeric_cols = numeric_cols.apply(zscore)

pima_df_z = numeric_cols.join(class_values)   



pima_df_z.head()
corr = pima_df[pima_df.columns].corr()

sns.heatmap(corr, annot = True,cmap='Blues')
import matplotlib.pylab as plt



pima_df_z.boxplot(by = 'Outcome',  layout=(3,4), figsize=(15, 20))
pima_df_z.hist('Age')
pima_df_z["log_age"] = np.log(pima_df_z['Age'])

pima_df_z["log_test"] = np.log(pima_df_z["Insulin"])

pima_df_z["log_preg"] = np.log(pima_df_z["Pregnancies"])

pima_df_z.hist('log_age')
pima_df_z.hist("log_test")
pima_df_z.hist("log_preg")
plt.scatter(pima_df_z['log_test'] , pima_df_z["Outcome"])
pima_df.describe().transpose()
# H0 - The difference in mean between sample BP column and population mean for BP is a statistical fluctuation. The given data represents the population distribution on the BP column



# H1 - The difference in mean between sample BP column and population mean is significant. The difference is too high to be result of statistical fluctuation



# If statistical tests result in rejecting H0, then building a model on the given sample data and expecting it to generalize may be a mistake
# Used to compare mean of single sample with that of the population / production

# Requisites -  Number of samples >= 30, the mean and standard deviation of population should be known



# Application of NDZT  on blood pressure column 

# Population Avg and Standard Deviation for  diastolic blood pressure = 71.3 with standard deviation of 7.2 

    

# Required population parameters and sample statistic

import scipy.stats as st

Mu = 72.4  

Std = 12.09



sample_avg_bp = np.average(pima_df['BloodPressure'])

std_error_bp = Std / np.sqrt(pima_df.size) # Standard dev of the sampling mean distribution... estimated from population

print("Sample Avg BP : " , sample_avg_bp)

print("Standard Error: " , std_error_bp)



# Z_norm_deviate =  sample_mean - population_mean / std_error_bp



Z_norm_deviate = (sample_avg_bp - Mu) / std_error_bp

print("Normal Deviate Z value :" , Z_norm_deviate)



p_value = st.norm.sf(abs(Z_norm_deviate))*2 #twosided using sf - Survival Function

print('p values' , p_value)



if p_value > 0.05:

	print('Samples are likely drawn from the same distributions (fail to reject H0)')

else:

	print('Samples are likely drawn from different distributions (reject H0)')
# Z score magnitude is much lower than 1.96 cutoff in normal distribution for 95% CL

# This indicates that the H0 cannot be  rejected. Which means this BP sample data is from the population whose mean is 72.4 and 

# std = 12.09
# used when the two requirements of normal deviate Z test cannot be met i.e. when the population mean and standard deviation

# is unknown



Mu = 72.4   

# Std = ?  Population standard deviatin is unknown



x = pima_df['BloodPressure']  # Storing values in a list to avoid long names

est_pop_std = np.sqrt(np.sum(abs(x - x.mean())**2) / (pima_df.size - 1))     #  sqrt(sum(xi - Xbar)^2 / (n -1))



sample_avg_bp =(pima_df['BloodPressure']).mean()



std_error_bp = est_pop_std / np.sqrt(pima_df.size) # Standard dev of the sampling mean distribution... estimated from population



T_Statistic = (( sample_avg_bp - Mu) / std_error_bp)



pvalue = st.t.sf(np.abs(T_Statistic), pima_df.size-1)*2

print("Estimated Pop Stand Dev" , est_pop_std)

print("Sample Avg BP : " , sample_avg_bp)

print("Standard Error: " , std_error_bp)

print("T Statistic" , T_Statistic)

print("Pval" , pvalue)



if pvalue > 0.05:

	print('Samples are likely drawn from the same distributions (fail to reject H0)')

else:

	print('Samples are likely drawn from different distributions (reject H0)')
#T-Statistic magnitude is very large compared to Z  score of 1.96.

# P value is almost 1, much greater than 0.05 

# Reject H0 at 95% confidence

# That memans the given data column of BP has is from the population BP distribution whose mean is 72.4 and est std dev 4.02
# Tests whether the means of two independent samples are significantly different.



# Pima Indians Dataset has many missing values in multiple columns. Let us replace the missing values with median. Does this

# step of handling missing values modify the distribution so much that statistically it is no more equivalent of original data?



pima_df_mod = pima_df.copy()





pima_df_mod['BloodPressure'] = pima_df_mod['BloodPressure'].mask(pima_df['BloodPressure'] == 0,pima_df['BloodPressure'].median())

from scipy.stats import ttest_ind



stat, pvalue = ttest_ind(pima_df_mod['BloodPressure'] , pima_df['BloodPressure'])

print("compare means", pima_df_mod['BloodPressure'].mean() , pima_df['BloodPressure'].mean())

print("Tstatistic , Pvalue", stat, pvalue)



if pvalue > 0.05:

	print('Samples are likely drawn from the same distributions (fail to reject H0)')

else:

	print('Samples are likely drawn from different distributions (reject H0)')