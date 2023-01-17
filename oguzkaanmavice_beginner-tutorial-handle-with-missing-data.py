# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



#import data



import pandas as pd



df=pd.read_csv('/kaggle/input/pima-indians_diabetes.csv')



df.info()
# EDA : describe data quickly by pandas-profiling



from pandas_profiling import ProfileReport



profile=ProfileReport(df,title='Descriptive Analysis of Diabetes',html={'style':{'full_width':True}})
profile.to_widgets()
# TYPE OF MISSINGNESS



"""We can use pandas profiling report for that, but i imlement 'missingno' package as being different tool"""



import missingno as msno

import matplotlib.pyplot as plt



msno.matrix(df)



plt.show()
"""Analyzing the missingness of a variable against another variable helps 

you determine any relationships between missing and non-missing values. """



import numpy as np

from numpy.random import rand



def dummy(df, scaling_factor=0.075):

    df_dummy = df.copy(deep=True)

    for col_name in df_dummy:

        

        col = df_dummy[col_name]

        col_null = col.isnull()    

    # Calculate number of missing values in column 

        num_nulls = col_null.sum()

    # Calculate column range

        col_range = col.max() - col.min()

    # Scale the random values to scaling_factor times col_range

        dummy_values = (rand(num_nulls) - 2) * scaling_factor * col_range + col.min()

        col[col_null] = dummy_values

    return df_dummy





# Fill dummy values in diabetes_dummy

diabetes_dummy = dummy(df)



# Sum the nullity of Skin_Fold and BMI

nullity0 = df['Skin_Fold'].isnull()+df['BMI'].isnull()



# Create a scatter plot of Skin Fold and BMI 

diabetes_dummy.plot(x='Skin_Fold', y='BMI', kind='scatter', alpha=0.5,

                    

                    # Set color to nullity of BMI and Skin_Fold

                    c=nullity0, 

                    cmap='rainbow')





plt.show()
# Sum the nullity of Skin_Fold and Serum_Insulin

nullity1 = df['Skin_Fold'].isnull()+df['Serum_Insulin'].isnull()



# Create a scatter plot of Skin Fold and BMI 

diabetes_dummy.plot(x='Skin_Fold', y='Serum_Insulin', kind='scatter', alpha=0.5,

                    

                    # Set color to nullity of BMI and Skin_Fold

                    c=nullity1, 

                    cmap='rainbow')
# Correlations among Missingness



# We can use heatmap or dendrogram which you have already seen at the profile-report. But in this code-line, i will missingno package.





# Plot missingness heatmap of diabetes

msno.heatmap(df)



# Plot missingness dendrogram of diabetes

msno.dendrogram(df)



# Show plot

plt.show()
# I will delete MAR type of missingness. I implement both 'all' and 'any' strategies to give example.Furthermore: 

"""https://www.w3resource.com/pandas/dataframe/dataframe-dropna.php"""



# Print the number of missing values in MAR types

print(df['Glucose'].isnull().sum())

print(df['BMI'].isnull().sum())



df_2 = df.copy(deep=True)



# Drop rows where 'Glucose' has a missing value

df_2.dropna(subset=['Glucose'], how='any', inplace=True)



# Drop rows where 'BMI' has a missing value

df_2.dropna(subset=['BMI'], how='all', inplace=True)



df_2.info()



df_drop=df.copy(deep=True)

df_drop.dropna(how='any',inplace=True)
# I will create dummy dataframe for all techniques which i implement







from sklearn.impute import SimpleImputer



## Mean Imputation



# Make a copy of diabetes

diabetes_mean = df_2.copy(deep=True)



# Create mean imputer object

mean_imputer = SimpleImputer(strategy='mean')



# Impute mean values in the DataFrame diabetes_mean

diabetes_mean.iloc[:, :] = mean_imputer.fit_transform(diabetes_mean)



## Median Imputation



# Make a copy of diabetes

diabetes_median = df_2.copy(deep=True)



# Create median imputer object

median_imputer = SimpleImputer(strategy='median')



# Impute median values in the DataFrame diabetes_median

diabetes_median.iloc[:, :] = median_imputer.fit_transform(diabetes_median)



## Mode Imputation



# Make a copy of diabetes

diabetes_mode = df_2.copy(deep=True)



# Create mode imputer object

mode_imputer = SimpleImputer(strategy='most_frequent')



# Impute using most frequent value in the DataFrame mode_imputer

diabetes_mode.iloc[:, :] = mode_imputer.fit_transform(diabetes_mode)



## Constant Imputation



# Make a copy of diabetes

diabetes_constant = df_2.copy(deep=True)



# Create median imputer object

constant_imputer = SimpleImputer(strategy='constant', fill_value=0)



# Impute missing values to 0 in diabetes_constant

diabetes_constant.iloc[:, :] = constant_imputer.fit_transform(diabetes_constant)

# Import KNN from fancyimpute

from fancyimpute import KNN



# Copy diabetes to diabetes_knn_imputed

diabetes_knn_imputed = df_2.copy(deep=True)



# Initialize KNN

knn_imputer = KNN()



# Impute using fit_tranform on diabetes_knn_imputed

diabetes_knn_imputed.iloc[:, :] = knn_imputer.fit_transform(diabetes_knn_imputed)





# Import IterativeImputer from fancyimpute

from fancyimpute import IterativeImputer



# Copy diabetes to diabetes_mice_imputed

diabetes_mice_imputed = df_2.copy(deep=True)



# Initialize IterativeImputer

mice_imputer = IterativeImputer()



# Impute using fit_tranform on diabetes

diabetes_mice_imputed.iloc[:, :] = mice_imputer.fit_transform(diabetes_mice_imputed)
# Basic Graphics to demonstrate bias



df['Skin_Fold'].plot(kind='kde', c='red', linewidth=3)

df_drop['Skin_Fold'].plot(kind='kde')

diabetes_mean['Skin_Fold'].plot(kind='kde')

#diabetes_median['Skin_Fold'].plot(kind='kde')

#diabetes_mode['Skin_Fold'].plot(kind='kde')

#diabetes_constant['Skin_Fold'].plot(kind='kde')

diabetes_knn_imputed['Skin_Fold'].plot(kind='kde')

diabetes_mice_imputed['Skin_Fold'].plot(kind='kde')

labels = ['First_Df','Baseline (Drop Any)', 'Mean Imputation', 'KNN Imputation',

'MICE Imputation']

plt.legend(labels)

plt.xlabel('Skin Fold')



#'Median_Imputation','Mode_Imputation','Constant_Imputation'