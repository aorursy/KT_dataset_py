# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#load csv file

#data = pd.read_csv("../input/dit-loan-train.txt")

##add an index colume to our data file - for the ID-Number as reference instead of 0,1,2,3,4,

data = pd.read_csv("../input/dit-loan-train.txt", index_col="LOAN_ID")



#display head + first rows

data.head()







#for more data information

data.describe()

#see let last rows - how many rows in the file

data.tail()

data.loc[(data["Gender"]== "Female"), ["Gender", "Education", "Loan_Status"]]

#show only date with gender female & you want to see gender, edu, loan,

data.loc[(data["Gender"]== "Female") & (data["Education"] == "Not Graduate"), ["Gender", "Education", "Loan_Status"]]

data.loc[(data["Gender"]== "Female") & (data["Education"] == "Not Graduate") & (data["Loan_Status"] == "Y"), ["Gender", "Education", "Loan_Status"]]

# working with custom functions #def = define sum function #summerize wgich data are missing



def num_missing(x):

    return sum(x.isnull())



data.apply(num_missing, axis=0)
data.apply(num_missing, axis=1).head() #head option for show only the first few lines

data["Gender"].value_counts() #count all destinc options in the column

data["Gender"].fillna("Male", inplace=True)   #fillna = fill not available

data["Gender"].value_counts() #now you have all null values filled with "male"



data.apply(num_missing, axis=0) # the gender columne has no null-values now
print(data["Married"].value_counts())

print(data["Self_Employed"].value_counts())



data["Married"].fillna("Yes", inplace=True)

data["Self_Employed"].fillna("No", inplace=True)



data.apply(num_missing, axis=0)

## working with pivot tables 



impute_grps = data.pivot_table(values=["LoanAmount"], index=["Gender", "Married", "Self_Employed"], aggfunc=np.mean)



print(impute_grps)
#interate over rows with missing LoanAmount and fill in the Loan Amount of the Gender, Status & Employment



#we look up in the impute_groups (Pivot Table above) und looking for the lines thats true 



for i,row in data.loc[data["LoanAmount"].isnull()].iterrows():

    ind = tuple([row["Gender"], row["Married"], row["Self_Employed"]])

    print(impute_grps.loc[ind])

    data.loc[i,"LoanAmount"] = impute_grps.loc[ind].values[0]

    

#LoanAmount is totaly filled now

data.apply(num_missing, axis=0) 
pd.crosstab(data["Credit_History"], data["Loan_Status"], margins = True)
### percent Convertion 



def percentConvert(v):

    return v/float(v[-1])



pd.crosstab(data["Credit_History"], data["Loan_Status"], margins = True).apply(percentConvert, axis=1)



# margins = True stands for the total SUM

82+378
460/564

print("{:.1%}".format(460/564)) #for rounding the format
prop_rates = pd.DataFrame([1000, 5000, 12000], index=['Rural', 'Semiurban', 'Urban'], columns=['rates'])

prop_rates

data.head(5)

data_merged = data.merge(right=prop_rates, how="inner", left_on="Property_Area", right_index=True, sort=False)

data_merged.head(5)
data_merged.pivot_table(values='Credit_History', index=['Property_Area', 'rates'], aggfunc=len)
data_sort1 = data.sort_values(['ApplicantIncome', 'CoapplicantIncome'], ascending=False)

data_sort1[['ApplicantIncome', 'CoapplicantIncome']].head(10)
import matplotlib.pyplot as plt

%matplotlib inline

#importing typs for plotting 



data.boxplot(column='ApplicantIncome', figsize=(15,10))



#is a median information, in the box itself 

#the box is inbetween the interquartal range 

#everything above the box are the extrem values 
data.boxplot(column='ApplicantIncome',by="Loan_Status", figsize=(15,10))
data.hist(column='ApplicantIncome', bins=15, figsize=(15,10))
#use more bins for a more detailed way



data.hist(column='ApplicantIncome', bins=30, figsize=(15,10))
#group the histogram



data.hist(column='ApplicantIncome', by="Loan_Status", bins=15, figsize=(15,10))
data.describe()
##define the function custombin ##



def custombin(col, cuttingpoints,labels):

    min_val = col.min()

    max_val = col.max()

    

    breaking_points = [min_val] + cuttingpoints + [max_val]

    print(breaking_points)

    

## here we bin ## 

cuttingpoints = [90,150,190] # low range from 9.0-90, medium from 90-150,...

labels = ["low", "medium", "high", "very high"]

data["LoanAmountBinned"] = custombin(data["LoanAmount"], cuttingpoints, labels) 

##define the function custombin ##



def custombin(col, cuttingpoints,labels):

    min_val = col.min()

    max_val = col.max()

    

    breaking_points = [min_val] + cuttingpoints + [max_val]

    print(breaking_points)

    

    ##new columne

    colBinned = pd.cut(col, bins = breaking_points, labels = labels, include_lowest = True) #label is parameter & Value but it is different!

    return colBinned

    

## here we bin ## 

cuttingpoints = [90,150,190] # low range from 9.0-90, medium from 90-150,...

labels = ["low", "medium", "high", "very high"]

data["LoanAmountBinned"] = custombin(data["LoanAmount"], cuttingpoints, labels) 



data.head(10)

#show the new columne LoanAmountBinned
print(pd.value_counts(data["LoanAmountBinned"], sort=False))
pd.value_counts(data["Married"])
## replacing information ## 



def custom_coding(col, dictonary):

    col_coded = pd.Series(col, copy=True)

    for key, value in dictonary.items():

        col_coded.replace(key, value, inplace=True)

        

    return col_coded



## we want to code LoanStatus - Y --> 1, N --> 0 

print(pd.value_counts(data["Loan_Status"]))



data["Loan_Status_Coded"] = custom_coding(data["Loan_Status"], {"N":0, "Y":1, "No":0, "Yes":1, "no":0, "yes":1})



data.head(10)

data.dtypes


