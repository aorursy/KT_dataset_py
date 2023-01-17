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
# load csv file into kaggle notebook, store it in a variable

dataframe = pd.read_csv("/kaggle/input/ditloantest/dit-loan-test.txt")



#

dataframe.head(10)

#

dataframe.tail(10)
df = dataframe

df.head(5)
df.describe()
df = pd.read_csv("/kaggle/input/ditloantest/dit-loan-test.txt", index_col="LOAN_ID")

df.head(5)
df.loc[(df["Gender"] == "Female")]
df.loc[(df["Gender"] == "Female"), ["Gender", "Education", "Loan_Status"]]
df.loc[(df["Gender"] == "Female") & (df["Education"] == "Not Graduate"), ["Gender", "Education", "Loan_Status"]]
# working with custom functions

def num_missing(x):

    return sum(x.isnull())



df.apply(num_missing, axis=0)

df.apply(num_missing, axis=1)

df.head(5)
df["Gender"].value_counts()
df["Gender"].fillna("Male", inplace=True)

df["Gender"].value_counts()
## print old value counts

print(df["Married"].value_counts())

print(df["Self_Employed"].value_counts())



## fill nan values

df["Married"].fillna("Yes", inplace=True)

df["Self_Employed"].fillna("No", inplace=True)



## print new value counts

print(df["Married"].value_counts())

print(df["Self_Employed"].value_counts())
df.apply(num_missing, axis=0)
impute_grps = df.pivot_table(values=["LoanAmount"], index=["Gender", "Married", "Self_Employed"], aggfunc=np.mean)

print(impute_grps)
# iterate over rows with missing LoanAmount



for i,row in df.loc[df["LoanAmount"].isnull()].iterrows():

    #print(row)

    ind = tuple([row["Gender"], row["Married"], row["Self_Employed"]])

    print(impute_grps.loc[ind])

    df.loc[i, "LoanAmount"] = impute_grps.loc[ind].values[0]



df.apply(num_missing, axis=0)
pd.crosstab(df["Credit_History"], df["Property_Area"], margins=True)
def percConvert(ser):

    return ser/float(ser[-1])



pd.crosstab(df["Credit_History"], df["Property_Area"], margins=True).apply(percConvert, axis=1)
pd.crosstab(df["Credit_History"], df["Property_Area"], margins=True).apply(percConvert, axis=0)
import matplotlib.pyplot as plt



df.boxplot(column="ApplicantIncome", figsize=(15,7))
df.boxplot(column="ApplicantIncome", by="Married", figsize=(15,7))
df.hist(column="ApplicantIncome", bins=15, figsize=(15,7))
df.hist(column="ApplicantIncome", bins=30, figsize=(15,7))
df.hist(column="ApplicantIncome", by="Married", bins=15, figsize=(15,7))
df.hist(column="ApplicantIncome", by="Gender", bins=15, figsize=(15,7))
df.hist(column="ApplicantIncome", by="Gender", bins=15, figsize=(15,7))
df.hist(column="ApplicantIncome", by="Gender", bins=15, figsize=(15,7))
df.hist(column="ApplicantIncome", by="Gender", bins=15, figsize=(15,7))


fig, ax = plt.subplots()

df.hist(column="ApplicantIncome", by="Gender", bins=15, figsize=(15,7), ax=ax)

fig.savefig("test.png")


