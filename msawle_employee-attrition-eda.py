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



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Importing the dataset



raw_data = pd.read_csv(r'/kaggle/input/hr-attrition/HR_Employee_Attrition_Data.csv')
# Description of DataFrame

raw_data.describe().T
raw_data.info() # Information regarding each feature (Length & data type)
print("The dataset has {} rows and {} columns".format(raw_data.shape[0], raw_data.shape[1]))
# generating a list of columns containing na values using List comprehension



cols_na = [var for var in raw_data.columns if raw_data[var].isna().sum() >= 1]



print(cols_na)
# Seggregating columns based on numerical & categorical data

num_cols = [var for var in raw_data.columns if raw_data[var].dtype != 'O']

obj_cols = [var for var in raw_data.columns if raw_data[var].dtype == 'O']



print("The list of columns with numeric data is: \n {}".format(num_cols))

print()

print("There are {} columns with numerical data".format(len(num_cols)))

print()



print('#'*75)



print("The list of columns with categorical data is: \n {}".format(obj_cols))

print()

print("There are {} columns with categorical data".format(len(obj_cols)))
#Checking for the blank values



np.where(raw_data.applymap(lambda x: x == ''))
# Identifying columns that are discrete in nature

# Criteria - less than 10 Unique values in a column 



dis_vars = [var for var in raw_data.columns if len(raw_data[var].unique()) <=10]



print('Number of discrete variables: ', len(dis_vars))

print(dis_vars)
raw_data[dis_vars].head()
def analyse_discrete(df, var):

    df = df.copy()

    df.groupby(var)['MonthlyIncome'].median().plot.bar()

    plt.title(var)

    plt.ylabel('MonthlyIncome')

    plt.show()

for var in dis_vars:

    analyse_discrete(raw_data, var)
# Describing continuous variables 

cont_vars = [var for var in raw_data.columns if len(raw_data[var].unique()) > 10]



print('Number of continuous variables: ', len(cont_vars))

print(cont_vars)
def attrition_check(df, var):

    df = df.copy()

    df.groupby('Attrition')[var].median().plot.bar()

    print(df.groupby('Attrition')[var].median())

    plt.title(var)

    plt.ylabel('Attrition')

    plt.show()

    

for var in cont_vars:

    attrition_check(raw_data, var)
for var in cont_vars:

    print("The mean of {} is {}".format(var, np.round(raw_data[var].mean(), 3)))

    print("The median of {} is {}".format(var, np.round(raw_data[var].median(),3)))

    print("The length of {} is {}".format(var, len(raw_data[var])))



print(cont_vars)
Mean = [np.round(raw_data[var].mean(), 3) for var in cont_vars]

Median = [np.round(raw_data[var].median(), 3) for var in cont_vars]

Length = [len(raw_data[var]) for var in cont_vars]

Range = [raw_data[var].max() - raw_data[var].min() for var in cont_vars]
Num_desc = pd.DataFrame(index = cont_vars)

Num_desc['Mean'] = Mean

Num_desc['Median'] = Median

Num_desc['Length'] = Length

Num_desc['Range'] = Range



Num_desc
def Quantile_calc(df, vars1):

    df = df.copy()

    for var in vars1:

        sorted_values = df[var].sort_values(ascending = True)

        len_var = len(df[var])

    

        if len_var % 2 == 0:

            x = sorted_values[0:int((len_var/2))]

#             print(len(x))

            Q1 = np.round(x.mean(),2)

            Q3 = np.round(sorted_values[int((len_var/2)):].mean(), 2)

            print("The 1st Quartile for {} is {}".format(var,Q1))

            print("The 3rd Quartile for {} is {}".format(var,Q3))

        else:

            x = sorted_values[0:int(((len_var-1)/2))]

#             print(len(x))

            Q1 = np.round(x.mean(),2)

            Q3 = np.round(sorted_values[int(((len_var-1)/2)):].mean(),2)

#     return Q1, Q3
Quantile_calc(raw_data, cont_vars)
def Quantile_calc1(df, vars1):

    df = df.copy()

    

    len_var = len(df[var])

    print(len_var)

    

    if len_var % 2 == 0:

        Q1 = [np.ceil(df[var].sort_values()[0:int((len_var/2))].mean()) for var in vars1]

        Q3 = [np.ceil(df[var].sort_values()[int((len_var/2)):].mean()) for var in vars1]

        IQR = [(np.ceil(df[var].sort_values()[int((len_var/2)):].mean())) - (np.ceil(df[var].sort_values()[0:int((len_var/2))].mean())) for var in vars1]

        Lower_limit = [(np.ceil(df[var].sort_values()[0:int((len_var/2))].mean())) - 1.5 *((np.ceil(df[var].sort_values()[int((len_var/2)):].mean())) - (np.ceil(df[var].sort_values()[0:int((len_var/2))].mean()))) for var in vars1]

        Upper_limit = [(np.ceil(df[var].sort_values()[int((len_var/2)):].mean())) + 1.5 * ((np.ceil(df[var].sort_values()[int((len_var/2)):].mean())) - (np.ceil(df[var].sort_values()[0:int((len_var/2))].mean()))) for var in vars1]

       

        

    else:

        

        Q1 = [np.ceil(df[var].sort_values()[0:int((len_var/2))].mean()) for var in vars1]

        Q3 = [np.ceil(df[var].sort_values()[int((len_var/2)):].mean()) for var in vars1]

        IQR = [(np.ceil(df[var].sort_values()[int((len_var/2)):].mean())) - (np.ceil(df[var].sort_values()[0:int((len_var/2))].mean())) for var in vars1]

        Lower_limit = [(np.ceil(df[var].sort_values()[0:int((len_var/2))].mean())) - 1.5 *((np.ceil(df[var].sort_values()[int((len_var/2)):].mean())) - (np.ceil(df[var].sort_values()[0:int((len_var/2))].mean()))) for var in vars1]

        Upper_limit = [(np.ceil(df[var].sort_values()[int((len_var/2)):].mean())) + 1.5 * ((np.ceil(df[var].sort_values()[int((len_var/2)):].mean())) - (np.ceil(df[var].sort_values()[0:int((len_var/2))].mean()))) for var in vars1]

        

    return Q1, Q3, IQR, Lower_limit, Upper_limit
X = Quantile_calc1(raw_data, cont_vars)
Num_desc['1st Quartile'] = X[0]

Num_desc['3rd Quartile'] = X[1]

Num_desc['IQR'] = X[2]

Num_desc['Lower_limit'] = X[3]

Num_desc['Upper_limit'] = X[4]
Num_desc.T
# X = Num_desc.T

# Testing block DataFrame slicing - used below in calculating outliers

Num_desc.T.loc[["Lower_limit"],['DailyRate']]
# Index position for the outliers that are found in continuous features

Index_pos_upper_outliers = {var : np.where(raw_data[var] > Num_desc.T.loc[["Upper_limit"],[var]].values[0][0]) for var in cont_vars}



Index_pos_lower_outliers = {var : np.where(raw_data[var] < Num_desc.T.loc[["Upper_limit"],[var]].values[0][0]) for var in cont_vars}

   



    
print(Index_pos_upper_outliers)

print(Index_pos_lower_outliers)
def test_normal(df, var):

    df = df.copy()

    sns.distplot(df[var], bins =20, kde = True)

    plt.xlim(0)

    plt.xlabel(var)

    plt.ylabel('Number of Employees')

    plt.title('Spread of Variables')

    plt.show()

    



for var in cont_vars:

    test_normal(raw_data, var)
# Boxplot to visualize each numeric feature



def box_plot_ft(df, var):

    df = df.copy()

    sns.boxplot(df[var])

    plt.title('Box plot for {}'.format(var))

    plt.show()
for var in num_cols:

    box_plot_ft(raw_data, var)