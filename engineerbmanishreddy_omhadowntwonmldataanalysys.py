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
# to handle datasets

import pandas as pd

import numpy as np



# for plotting

import matplotlib.pyplot as plt

%matplotlib inline



# to divide train and test set

from sklearn.model_selection import train_test_split



# feature scaling

from sklearn.preprocessing import MinMaxScaler



# to visualise al the columns in the dataframe

pd.pandas.set_option('display.max_columns', None)
data = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')
print(data.shape)

print(data.head())
##Counting the missing values  All those values that have na values 

vars_with_na = [var for var in data.columns if data[var].isnull().sum()>1]



for var in vars_with_na:

    print(var,np.round(data[var].isnull().mean(),3))
#we are looking at how much infuluential is the sale price on missing values 

def analyse_na_values(df,var):

    df = df.copy()

    

    #indicate 1 if var is missing else indicate 0 

    df[var] = np.where(df[var].isnull(),1,0)

    

    #lets caliculate the mean sale price where the information is missing or present 

    df.groupby(var)['SalePrice'].median().plot.bar(color = ['b','g'])

    plt.title(var)

    plt.show()

for var in vars_with_na:

    analyse_na_values(data,var)
#We can see in most of the cases when the missing values are more the house price is high 
#Get all the values with numerical varibles 

num_vars = [var for var in data.columns if data[var].dtypes != 'O']

data[num_vars].head()
#there are some year variables 

#Lets get all those year variables 

year_vars = [var for var in num_vars if 'Yr' in var or 'Year' in var]

year_vars
for var in year_vars:

    print(var,data[var].unique())

    print()
data.groupby('YrSold')['SalePrice'].median().plot()

plt.ylabel("Median house price")

plt.title("Change house price ")
#Relation between year varible and sale price 

def analyse_year_vars(df,var):

    df = df.copy()

    df[var] = df['YrSold'] - df[var]

    plt.scatter(df[var],df['SalePrice'])

    plt.ylabel('Sale Price')

    plt.xlabel(var)

    plt.show()

for var in year_vars:

    if var != 'YrSold':

        analyse_year_vars(data,var)

    
#We can see house price decreased over time 
#Identify descrete varibles 

descrete_vars = [var for var in num_vars if len(data[var].unique())<20 and var not in year_vars+['Id']]

print("Number of descrete varibles  = ",len(descrete_vars))
data[descrete_vars].head()
def analyse_descretevars(df,var):

    df = df.copy()

    df.groupby(var)['SalePrice'].median().plot.bar(color = ['r','y'])

    plt.title(var)

    plt.ylabel('SalePrice')

    plt.show()

    

for var in descrete_vars:

    analyse_descretevars(data,var)
#we can see .. as over all quaility of the house increases the cost of the house also increases 

#
#lets get the list of coninious varibles 

cont_vars = [var for var in num_vars if var not in descrete_vars+year_vars+['Id']]

print("number of cont vars = ",len(cont_vars))
data[cont_vars].head()
#lets plot the data and see ... continous data 

def analyse_cont_data(df,var):

    df = df.copy()

    df[var].hist(bins = 20)

    plt.ylabel('No of houses')

    plt.xlabel(var)

    plt.title(var)

    plt.show()

for var in cont_vars:

    analyse_cont_data(data,var)
#distrubiton is not gaussian 

#lets normalize the data using a log transform 

def analyse_data_using_log(df,var):

    df = df.copy()

    if 0 in data[var].unique():

        pass

    else:

        df[var] = np.log(df[var])

        df[var].hist(bins = 20)

        plt.ylabel('No of houses ')

        plt.xlabel(var)

        plt.title(var)

        plt.show()

for var in cont_vars:

    analyse_data_using_log(data,var)
#lets log transform the data  and plot a grapth and ee if has a linear relation 

def analyse_log_transform(df,var):

    df = df.copy()

    if 0 in data[var].unique():

        pass

    else:

        df[var] = np.log(df[var])

        df['SalePrice'] = np.log(df['SalePrice'])

        plt.scatter(df[var],df['SalePrice'])

        plt.ylabel('Number of houses ')

        plt.xlabel(var)

        plt.title(var)

        plt.show()

for var in cont_vars:

    if var != 'SalePrice':

        analyse_log_transform(data,var)

    
#checking for  outliar varibles 

def find_outliers(df,var):

    df = df.copy()

    if 0 in data[var].unique():

        pass

    else:

        df[var] = np.log(df[var])

        df.boxplot(column= var)

        plt.title(var)

        plt.ylabel(var)

        plt.show()

for var in cont_vars:

    find_outliers(data,var)
#lets analyse cateogorical Variables 

cat_vars = [var for var in data.columns if data[var].dtypes == 'O']

print("cat vars = " ,len(cat_vars))
data[cat_vars].head()
#cardinality of varibles 

for var in cat_vars:

    print(var,len(data[var].unique()))
def analyse_rare_labels(df,var,rere_perc):

    df = df.copy()

    tmp = df.groupby(var)['SalePrice'].count()/len(df)

    return tmp[tmp<rere_perc]

for var in cat_vars:

    print(analyse_rare_labels(data,var,0.01))

    print()
for var in cat_vars:

    analyse_descretevars(data, var)