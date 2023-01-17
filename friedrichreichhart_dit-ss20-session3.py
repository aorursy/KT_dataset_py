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
dataframe = pd.read_csv("/kaggle/input/ditloantrain/dit-loan-train.txt")

#
dataframe.head(10)
dataframe.hist(column="ApplicantIncome", by="Loan_Status", bins=15, figsize=(12,7))
dataframe.describe()
def customized_bin(column, cuttingpoints, custom_labels):
    min_val = column.min()
    max_val = column.max()
    
    breaking_points = [min_val] + cuttingpoints + [max_val]
    print(breaking_points)
    
    colBinned = pd.cut(column, bins=breaking_points, labels=custom_labels, include_lowest=True)
    return colBinned

## call the function ##
cuttingpoints = [90, 150, 190]
custom_labels = ["low", "medium", "high", "very high"]
dataframe["LoanAmountBinned"] = customized_bin(dataframe["LoanAmount"], cuttingpoints, custom_labels)

## see output ##
dataframe.head(10)

print(pd.value_counts(dataframe["LoanAmountBinned"], sort=False))
pd.value_counts(dataframe["Married"])
## replacing information ##
def custom_coding(column, dictionary):
    column_coded = pd.Series(column, copy=True)
    for key, value in dictionary.items():
        column_coded.replace(key, value, inplace=True)
    
    return column_coded

## code LoanStatus - Y > 1, N > 0, yes > 1, Yes > 1, ...
dataframe["Loan_Status_Coded"] = custom_coding(dataframe["Loan_Status"], {"N":0, "Y":1, "No":0, "Yes":1, "no":0, "yes":1})

dataframe.head(10)
    
dataframe.describe()
dataframe['Property_Area'].value_counts()
dataframe["ApplicantIncome"].hist(bins=10)
dataframe["ApplicantIncome"].hist(bins=50)
dataframe.boxplot(column="ApplicantIncome", figsize=(15,8))
dataframe.boxplot(column="ApplicantIncome", by="Education", figsize=(15,8))
dataframe["LoanAmount"].hist(bins=50, figsize=(12,8))
dataframe.boxplot(column="LoanAmount", figsize=(12,8))
dataframe["Credit_History"].value_counts(ascending=True)
dataframe["Credit_History"].value_counts(ascending=True, normalize=True)
dataframe["Property_Area"].value_counts()
dataframe["Loan_Status"].value_counts()
dataframe.pivot_table(values="Loan_Status", index=["Credit_History"], aggfunc=lambda x: x.map({"Y": 1, "N":0}).mean())
dataframe["Credit_History"].value_counts()
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15,6))

ax1 = fig.add_subplot(2,3,1)
ax1.set_xlabel("Credit_History")
ax1.set_ylabel("Count of Applicants")
ax1.set_title("Applicants by Credit_History")
dataframe["Credit_History"].value_counts().plot(kind="bar")


ax2 = fig.add_subplot(2,3,6)
ax2.set_xlabel("Credit_History")
ax2.set_ylabel("Probability of getting loan")
ax2.set_title("Probability of getting loan by credit history")
dataframe.pivot_table(values="Loan_Status", index=["Credit_History"], aggfunc=lambda x: x.map({"Y": 1, "N":0}).mean()).plot(kind="bar")
dataframe["Credit_History"].value_counts().plot(kind="bar")
dataframe.pivot_table(values="Loan_Status", index=["Credit_History"], aggfunc=lambda x: x.map({"Y": 1, "N":0}).mean()).plot(kind="bar")
temp = pd.crosstab(dataframe["Credit_History"], dataframe["Loan_Status"])
temp.plot(kind="bar", stacked=True, color=["red", "blue"])
temp1 = pd.crosstab([dataframe["Credit_History"], dataframe["Gender"]], dataframe["Loan_Status"])
print(temp1)
temp1.plot(kind="bar", stacked=True, color=["orange", "grey"], grid=True, figsize=(12,6))
temp1 = pd.crosstab(dataframe["Credit_History"], [dataframe["Gender"], dataframe["Loan_Status"]])
print(temp1)
dataframe.apply(lambda x: sum(x.isnull()), axis=0)
dataframe["LoanAmount"].fillna(dataframe["LoanAmount"].mean(), inplace=True)
dataframe.apply(lambda x: sum(x.isnull()), axis=0)
dataframe['Self_Employed'].value_counts()
dataframe['Self_Employed'].value_counts(normalize=True)
dataframe["Self_Employed"].fillna("No",inplace=True)
dataframe.apply(lambda x: sum(x.isnull()), axis=0)
dataframe["LoanAmount"].hist(bins=20)
dataframe["LoanAmount_log"] = np.log(dataframe["LoanAmount"])
dataframe["LoanAmount_log"].hist(bins=20)
dataframe.head(10)
dataframe["TotalIncome"] = dataframe["ApplicantIncome"] + dataframe["CoapplicantIncome"]
dataframe["TotalIncome"].hist(bins=20)
dataframe["TotalIncome_log"] = np.log(dataframe["TotalIncome"])
dataframe["TotalIncome_log"].hist(bins=20)
dataframe.apply(lambda x: sum(x.isnull()),axis=0) 
dataframe["Married"].value_counts(normalize=True)
dataframe["Married"].mode()
dataframe["Married"].fillna(dataframe["Married"].mode()[0], inplace=True)
dataframe.apply(lambda x: sum(x.isnull()),axis=0) 
dataframe["Gender"].fillna(dataframe["Gender"].mode()[0], inplace=True)
dataframe["Dependents"].fillna(dataframe["Dependents"].mode()[0], inplace=True)
dataframe["Loan_Amount_Term"].fillna(dataframe["Loan_Amount_Term"].mode()[0], inplace=True)
dataframe["Credit_History"].fillna(dataframe["Credit_History"].mode()[0], inplace=True)
dataframe.apply(lambda x: sum(x.isnull()),axis=0) 
dataframe.dtypes
dataframe.head(6)
dataframe["Dependents"].value_counts()
from sklearn.preprocessing import LabelEncoder
columns_2_encode = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area", "Loan_Status"]

labelEncoder = LabelEncoder()

for i in columns_2_encode:
    dataframe[i] = labelEncoder.fit_transform(dataframe[i])
dataframe.dtypes
dataframe.head(10)