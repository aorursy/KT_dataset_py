import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import scipy.stats as stat
loan_data = pd.read_csv("../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv")
loan_data.shape
# Getting the head of the data
loan_data.head()
# Type of each column
loan_data.info()
loan_data.isnull().sum()
sns.set(font_scale=1.1)
correlation_train = loan_data.corr()
mask = np.triu(correlation_train.corr())
plt.figure(figsize=(20, 7))
sns.heatmap(correlation_train,
            annot=True,
            fmt='.1f',
            cmap='coolwarm',
            square=True,
            mask=mask,
            linewidths=1,
            cbar=False)

plt.show()
# No duplicates present
len(loan_data["Loan_ID"].unique())
loan_data["Loan_Status"].unique()
loan_data["Loan_Status"].value_counts(normalize=True)
sns.countplot(loan_data["Loan_Status"])
loan_data["Gender"].unique()
loan_data["Gender"].value_counts(normalize=True)
sns.countplot(loan_data["Gender"])
# Relation between Gender and Loan Status
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
sns.countplot(loan_data["Loan_Status"], hue=loan_data["Gender"], ax=ax[0])
sns.countplot(loan_data["Gender"], hue=loan_data["Loan_Status"], ax=ax[1])
# Numerical form of Left Graph
loan_data.groupby(by="Loan_Status")["Gender"].value_counts(normalize=True)
# Numerical form of Right Graph
loan_data.groupby(by="Gender")["Loan_Status"].value_counts(normalize=True)
loan_data["Married"].unique()
loan_data["Married"].value_counts(normalize=True)
sns.countplot(loan_data["Married"])
# Relation between Married and Loan Status
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
sns.countplot(loan_data["Loan_Status"], hue=loan_data["Married"], ax=ax[0])
sns.countplot(loan_data["Married"], hue=loan_data["Loan_Status"], ax=ax[1])
# Numerical form of Left Graph
loan_data.groupby(by="Loan_Status")["Married"].value_counts(normalize=True)
# Numerical form of Right Graph
loan_data.groupby(by="Married")["Loan_Status"].value_counts(normalize=True)
loan_data["Dependents"].unique()
loan_data["Dependents"].value_counts()
sns.countplot(loan_data["Dependents"])
# Relation between Married and Loan Status
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
sns.countplot(loan_data["Loan_Status"], hue=loan_data["Dependents"], ax=ax[0])
sns.countplot(loan_data["Dependents"], hue=loan_data["Loan_Status"], ax=ax[1])
loan_data.groupby(by="Dependents")["Loan_Status"].value_counts(normalize=True)
loan_data["Education"].unique()
loan_data["Education"].value_counts()
sns.countplot(loan_data["Education"])
# Relation between Married and Loan Status
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
sns.countplot(loan_data["Loan_Status"], hue=loan_data["Education"], ax=ax[0])
sns.countplot(loan_data["Education"], hue=loan_data["Loan_Status"], ax=ax[1])
loan_data.groupby(by="Education")["Loan_Status"].value_counts(normalize=True)
loan_data.groupby(by="Loan_Status")["Education"].value_counts(normalize=True)
loan_data["Self_Employed"].unique()
loan_data["Self_Employed"].value_counts()
sns.countplot(loan_data["Self_Employed"])
# Relation between Married and Loan Status
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
sns.countplot(loan_data["Loan_Status"], hue=loan_data["Self_Employed"], ax=ax[0])
sns.countplot(loan_data["Self_Employed"], hue=loan_data["Loan_Status"], ax=ax[1])
loan_data.groupby(by="Self_Employed")["Loan_Status"].value_counts(normalize=True)
loan_data.groupby(by="Loan_Status")["Self_Employed"].value_counts(normalize=True)
loan_data["ApplicantIncome"].describe()
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
sns.distplot(loan_data["ApplicantIncome"], bins=50, ax=ax[0])
sns.boxplot(loan_data["ApplicantIncome"], ax=ax[1])
# Kurtosis is a statistical measure that defines how heavily the tails of a distribution differ from the tails of a 
# normal distribution. In other words, kurtosis identifies whether the tails of a given distribution contain extreme 
# values.
print("Skewness :  ", stat.skew(loan_data["ApplicantIncome"]))
print("Kurtosis :  ", stat.kurtosis(loan_data["ApplicantIncome"]))
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
sns.kdeplot(loan_data[loan_data["Loan_Status"] == "Y"]["ApplicantIncome"], shade=True,label="Loan Accepeted",ax=ax[0])
sns.kdeplot(loan_data[loan_data["Loan_Status"] == "N"]["ApplicantIncome"], shade=True,label="Loan Rejected",ax=ax[0])
sns.boxplot(y=loan_data["ApplicantIncome"], x=loan_data["Loan_Status"])
print("Loan Accepted")
print("Skewness :  ", stat.skew(loan_data[loan_data["Loan_Status"] == 'Y']["ApplicantIncome"]))
print("Kurtosis :  ", stat.kurtosis(loan_data[loan_data["Loan_Status"] == 'Y']["ApplicantIncome"]))
print("*"*60)
print("Loan Rejected")
print("Skewness :  ", stat.skew(loan_data[loan_data["Loan_Status"] == 'N']["ApplicantIncome"]))
print("Kurtosis :  ", stat.kurtosis(loan_data[loan_data["Loan_Status"] == 'N']["ApplicantIncome"]))
# Numerical Representation
print(loan_data.groupby("Loan_Status")["ApplicantIncome"].describe())
loan_data["CoapplicantIncome"].describe()
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
sns.distplot(loan_data["CoapplicantIncome"], bins=50, ax=ax[0])
sns.boxplot(loan_data["CoapplicantIncome"], ax=ax[1])
print("Skewness :  ", stat.skew(loan_data["CoapplicantIncome"]))
print("Kurtosis :  ", stat.kurtosis(loan_data["CoapplicantIncome"]))
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
sns.kdeplot(loan_data[loan_data["Loan_Status"] == "Y"]["CoapplicantIncome"], shade=True,label="Loan Accepeted",ax=ax[0])
sns.kdeplot(loan_data[loan_data["Loan_Status"] == "N"]["CoapplicantIncome"], shade=True,label="Loan Rejected",ax=ax[0])
sns.boxplot(y=loan_data["CoapplicantIncome"], x=loan_data["Loan_Status"])
# Numerical Representation
print(loan_data.groupby("Loan_Status")["CoapplicantIncome"].describe())
len(loan_data["CoapplicantIncome"].unique())
loan_data["LoanAmount"].describe()
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
sns.distplot(loan_data["LoanAmount"], bins=50, ax=ax[0])
sns.boxplot(loan_data["LoanAmount"], ax=ax[1])
print("Skewness :  ", stat.skew(loan_data.dropna()["LoanAmount"]))
print("Kurtosis :  ", stat.kurtosis(loan_data.dropna()["LoanAmount"]))
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
sns.kdeplot(loan_data[loan_data["Loan_Status"] == "Y"]["LoanAmount"], shade=True,label="Loan Accepeted",ax=ax[0])
sns.kdeplot(loan_data[loan_data["Loan_Status"] == "N"]["LoanAmount"], shade=True,label="Loan Rejected",ax=ax[0])
sns.boxplot(y=loan_data["LoanAmount"], x=loan_data["Loan_Status"])
# Numerical Representation
print(loan_data.groupby("Loan_Status")["LoanAmount"].describe())
loan_data["Loan_Amount_Term"].unique()
loan_data["Loan_Amount_Term"].value_counts()
sns.countplot(loan_data["Loan_Amount_Term"])
loan_data.groupby("Loan_Amount_Term")["Loan_Status"].value_counts(normalize=False)
loan_data["Credit_History"].unique()
loan_data["Credit_History"].value_counts()
sns.countplot(loan_data["Credit_History"])
# Relation between Married and Loan Status
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
sns.countplot(loan_data["Loan_Status"], hue=loan_data["Credit_History"], ax=ax[0])
sns.countplot(loan_data["Credit_History"], hue=loan_data["Loan_Status"], ax=ax[1])
loan_data.groupby("Loan_Status")["Credit_History"].value_counts(normalize=True)
loan_data.groupby("Credit_History")["Loan_Status"].value_counts(normalize=True)
loan_data["Property_Area"].unique()
loan_data["Property_Area"].value_counts()
sns.countplot(loan_data["Property_Area"])
# Relation between Married and Loan Status
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
sns.countplot(loan_data["Loan_Status"], hue=loan_data["Property_Area"], ax=ax[0])
sns.countplot(loan_data["Property_Area"], hue=loan_data["Loan_Status"], ax=ax[1])
loan_data.groupby("Property_Area")["Loan_Status"].value_counts(normalize=True)
sns.scatterplot(loan_data["LoanAmount"], loan_data["ApplicantIncome"])
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
sns.scatterplot(loan_data["LoanAmount"], loan_data["ApplicantIncome"], hue=encoder.fit_transform(loan_data["Loan_Status"]))
sns.scatterplot(loan_data["ApplicantIncome"], loan_data["CoapplicantIncome"], hue=encoder.fit_transform(loan_data["Loan_Status"]))

