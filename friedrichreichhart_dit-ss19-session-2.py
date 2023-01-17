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
# load csv file

#data = pd.read_csv("../input/dit-loan-train.txt")

## add an index column to our data file

data = pd.read_csv("../input/dit-loan-train.txt", index_col="LOAN_ID")



#display header + first rows

data.head()
data.describe()
data.tail()
data.loc[(data["Gender"] == "Female")]
data.loc[(data["Gender"] == "Female"), ["Gender", "Education", "Loan_Status"]]
data.loc[(data["Gender"] == "Female") & (data["Education"] == "Not Graduate"), ["Gender", "Education", "Loan_Status"]]
# added >> & (data["Loan_Status"] == "Y")

data.loc[(data["Gender"] == "Female") & (data["Education"] == "Not Graduate") & (data["Loan_Status"] == "Y"), ["Gender", "Education", "Loan_Status"]]
# working with custom fuctions



def num_missing(x):

    return sum(x.isnull())



data.apply(num_missing, axis=0)
data.apply(num_missing, axis=1).head()
data["Gender"].value_counts()
data["Gender"].fillna("Male", inplace=True)
data["Gender"].value_counts()
data.apply(num_missing, axis=0)
print(data["Married"].value_counts())

print(data["Self_Employed"].value_counts())

##

data["Married"].fillna("Yes", inplace=True)

data["Self_Employed"].fillna("No", inplace=True)

##

print(data["Married"].value_counts())

print(data["Self_Employed"].value_counts())

data.apply(num_missing, axis=0)
## working with pivot tables



impute_grps = data.pivot_table(values=["LoanAmount"],

                               index=["Gender", "Married", "Self_Employed"],

                               aggfunc=np.mean)



print(impute_grps)
data.apply(num_missing, axis=0)
# iterate over rows with missing LoanAmount



#data.loc[data["LoanAmount"].isnull()]



for i,row in data.loc[data["LoanAmount"].isnull()].iterrows():

    ind = tuple([row["Gender"], row["Married"], row["Self_Employed"]])

    #print(impute_grps.loc[ind])

    data.loc[i,"LoanAmount"] = impute_grps.loc[ind].values[0]



data.apply(num_missing, axis=0)
pd.crosstab(data["Credit_History"], data["Loan_Status"], margins=True)

#pd.crosstab(data["Credit_History"], data["Loan_Status"])
def percConvert(ser):

  return ser/float(ser[-1])

pd.crosstab(data["Credit_History"],data["Loan_Status"],margins=True).apply(percConvert, axis=1)
82+378
#460/564

print("{:.1%}".format(460/564))
prop_rates = pd.DataFrame([1000, 5000, 12000], index=['Rural','Semiurban','Urban'],columns=['rates'])

prop_rates

data.head(5)
data_merged = data.merge(right=prop_rates, how="inner", left_on="Property_Area", right_index=True, sort=False)

data_merged.head(10)
data_merged.pivot_table(values='Credit_History',index=['Property_Area','rates'], aggfunc=len)
data_sort1 = data.sort_values(['ApplicantIncome', 'CoapplicantIncome'], ascending=False)

data_sort1[['ApplicantIncome', 'CoapplicantIncome']].head(10)
import matplotlib.pyplot as plt

%matplotlib inline



data.boxplot(column="ApplicantIncome", figsize=(15,10))
data.boxplot(column="ApplicantIncome",by="Loan_Status", figsize=(15,10))
data.hist(column="ApplicantIncome", bins=15, figsize=(15,10))
data.hist(column="ApplicantIncome", bins=30, figsize=(15,10))
data.hist(column="ApplicantIncome", by="Loan_Status", bins=15, figsize=(12,7))
data.describe()
## define function custombin ##

def custombin(col, cuttingpoints, cust_labels):

    min_val = col.min()

    max_val = col.max()

    

    breaking_points = [min_val] + cuttingpoints + [max_val]

    print(breaking_points)

    

    colBinned = pd.cut(col, bins=breaking_points, labels=cust_labels, include_lowest=True)

    return colBinned

    

    

## here we bin ##

cuttingpoints = [90,150,190]

cust_labels = ["low", "medium", "high", "very high"]

data["LoanAmountBinned"] = custombin(data["LoanAmount"], cuttingpoints, cust_labels)



data.head(10)
print(pd.value_counts(data["LoanAmountBinned"], sort=False))
pd.value_counts(data["Married"])
## replacing information ##

def custom_coding(col, dictonary):

    col_coded = pd.Series(col, copy=True)

    for key, value in dictonary.items():

        col_coded.replace(key, value, inplace=True)

        

    return col_coded



## we want to code LoanStatus - Y > 1, N > 0

print(pd.value_counts(data["Loan_Status"]))



data["Loan_Status_Coded"] = custom_coding(data["Loan_Status"], {"N":0, "Y":1, "No":0, "Yes": 1, "no": 0, "yes": 1})



data.head(10)

print(pd.value_counts(data["Loan_Status_Coded"]))
data.dtypes