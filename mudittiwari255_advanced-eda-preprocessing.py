# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #data visualization
import seaborn as sns #data visualization
from scipy import stats #Statistics
from sklearn.cluster import DBSCAN  #outlier detection
from collections import Counter
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from fancyimpute import KNN #KNN imputation

import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
test_df = pd.read_csv("../input/test_AV3.csv")
train_df = pd.read_csv("../input/train_AV3.csv")
train_df.head()
test_df.head()
train_df.shape
train_df['Property_Area'].hist(color = 'orange')
plt.xlabel('Property Area')
plt.ylabel('Frequency')
plt.show()
train_df.info()
train_df.dtypes
train_df.describe()
train_df.mode().iloc[0][1:]   
#Number of null values
train_df.isnull().sum()
#train_df.LoanAmount.value_counts(dropna=False)
fig,axes = plt.subplots(nrows=1, ncols=2)
mode = train_df.LoanAmount.mode().iloc[0]
fig1 = train_df.LoanAmount.fillna(mode).plot(kind = 'hist', bins = 30, ax = axes[1])
print(train_df.LoanAmount.fillna(mode).describe())
print(train_df.LoanAmount.describe())
train_df.LoanAmount.plot(kind = 'hist', bins = 30, ax = axes[0])
plt.show()
#Using mean to fill NAN values
fig,axes = plt.subplots(nrows=1, ncols=2)
mean = train_df.LoanAmount.mean()
train_df.LoanAmount.fillna(mean).plot(kind = 'hist', bins = 30, ax = axes[1])
print(train_df.LoanAmount.fillna(mean).describe())
print(train_df.LoanAmount.describe())
train_df.LoanAmount.plot(kind = 'hist', bins = 30, ax = axes[0])
#Using median to fill out NAN values
#Using mean to fill NAN values
fig,axes = plt.subplots(nrows=1, ncols=2)
median = train_df.LoanAmount.median()
train_df.LoanAmount.fillna(median).plot(kind = 'hist', bins = 30, ax = axes[1])
print(train_df.LoanAmount.fillna(median).describe())
print(train_df.LoanAmount.describe())
train_df.LoanAmount.plot(kind = 'hist', bins = 30, ax = axes[0])
#Using KNN to fill the values
train_df.ApplicantIncome = train_df.ApplicantIncome.astype('float')
train_df_numeric = train_df.select_dtypes('float')
df_filled = KNN(k=5).complete(train_df_numeric)
df_filled = pd.DataFrame(df_filled)
df_filled.shape
df_filled.index = train_df_numeric.index
df_filled.columns = train_df_numeric.columns
df_filled.info()
#Make a copy of train_df to fill up the values from df_filled
train_df_c = train_df.copy()
train_df_c.LoanAmount = df_filled.LoanAmount
fig, axes = plt.subplots(nrows=1,ncols=2)
train_df.LoanAmount.hist(bins = 30, ax = axes[0])
train_df_c.LoanAmount.hist(bins = 30, ax = axes[1])
print(train_df.LoanAmount.describe())
print(train_df_c.LoanAmount.describe())
train_df_c.ApplicantIncome = df_filled.ApplicantIncome
train_df_c.CoapplicantIncome = df_filled.CoapplicantIncome
train_df_c.Loan_Amount_Term = df_filled.Loan_Amount_Term
plt.hist(train_df_c.LoanAmount, bins = 30, alpha = 0.5, label = 'Imputed Loan Amount', color = 'orange', stacked = True)
plt.hist(train_df.LoanAmount.dropna(), bins = 30, alpha = 0.5, label = 'Original Loan Amount', color = 'green')
plt.legend()
plt.show()
plt.hist(train_df_c.Loan_Amount_Term, bins = 10, alpha = 0.5, label = 'Imputed Loan Amount Term', color = 'orange', stacked = True)
plt.hist(train_df.Loan_Amount_Term.dropna(), bins = 10, alpha = 0.5, label = 'Original Loan Amount Term', color = 'green')
plt.legend()
plt.show()
train_df.describe(include = ['O'])
train_df = train_df_c.copy()
#Transforming Loan_Status, Gender, Married, Education, & Property area from object to proper data type like bool, int, categoryetc.
d = {'Y':True, 'N':False}
train_df.Loan_Status = train_df.Loan_Status.map(d)
d = {'Male':False, 'Female':True, 'NaN':np.nan}
train_df.Gender = train_df.Gender.map(d)
#We want to find if there is relationship between gender & Applicants income so that we can find missing gender more clearly.
train_df_c = train_df[train_df.Gender.notnull()].copy()
train_df_c.Gender = train_df_c.Gender.astype('int64')
fig, ax = plt.subplots(figsize = (18,18))
sns.heatmap(train_df_c.corr(), ax = ax, annot = True)
train_df.Gender = train_df.Gender.fillna(True)
#Filling Nan using mode
train_df.Credit_History = train_df.Credit_History.fillna(train_df.Credit_History.mode()[0])
d = {'No':False, 'Yes':True, 'Nan':np.nan}
train_df.Self_Employed = train_df.Self_Employed.map(d)
train_df_c = train_df[train_df.Self_Employed.notnull()]
train_df_c.Self_Employed = train_df_c.Self_Employed.astype('bool')
fig, ax = plt.subplots(figsize = (10,10))
sns.heatmap(train_df_c.corr(), annot = True, ax = ax)
train_df.Self_Employed = train_df.Self_Employed.fillna(train_df.Self_Employed.mode()[0])
train_df.Dependents = train_df.Dependents.fillna(train_df.Dependents.mode()[0])
train_df.Married = train_df.Married.fillna(train_df.Married.mode()[0])
train_df.isnull().sum()
#Now we hav no null values in our data, also in this cell we delete our useless data frames.
del(train_df_c)
del(train_df_numeric)
train_df.head()
#Visualizing scatter plot of all continuous numerical data, 
#Selecting all numerical attributes with Loan Status
train_df_c = train_df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term','Loan_Status']]
color = ['orange' if i==True else 'green' for i in train_df_c.loc[:,'Loan_Status']]
pd.plotting.scatter_matrix(train_df_c.loc[:, train_df_c.columns != 'Loan_Status'], c = color, figsize=(15,15), diagonal='hist', alpha = 0.5,
                          s= 200)
plt.show()
fig = sns.FacetGrid(train_df, row = 'Gender', col='Education', size = 4)
fig = fig.map(plt.hist, 'LoanAmount', color = 'g', bins = 30)
fig = sns.FacetGrid(train_df, row = 'Gender', col='Education', size = 4)
fig = fig.map(plt.hist, 'ApplicantIncome', color = 'c', bins = 30)
fig = sns.FacetGrid(train_df, col='Credit_History', row = 'Loan_Status', size = 4)
fig = fig.map(plt.hist, 'LoanAmount', color = 'k', bins = 30)
train_df.boxplot(by = 'Loan_Status', column = 'LoanAmount', figsize = (7,5))
plt.show()
train_df.boxplot(by = 'Loan_Status', column = 'ApplicantIncome', figsize = (7,5))
plt.show()
train_df.boxplot(by = 'Loan_Status', column = 'CoapplicantIncome', figsize = (7,5))
plt.show()
train_df.boxplot(by = 'Education', column = 'ApplicantIncome', figsize = (7,5))
plt.show()
# Applicant inclome + Coapplicant income = Family income
train_df['Family_Income'] = train_df.ApplicantIncome + train_df.CoapplicantIncome
train_df.Family_Income.plot(kind = 'hist', bins = 50)
plt.show()

#Applicant's class 
# 0 - 1000 Lower class
# 1001 - 5000 Lower Middle class
# 5001 - 10000 Upper middle Class
# 10000+ Upper class
values = []
for i in train_df.loc[:,'ApplicantIncome']:
    if i <= 1000:
        values.append('Lower Class')
    elif i > 1000 and i <= 5000:
        values.append('Lower Middle Class')
    elif i > 5000 and i <= 10000:
        values.append('Upper Middle Class')
    else:
        values.append('Upper Class')

values = np.array(values)
train_df['ApplicantClass'] = values
fig, ax = plt.subplots(figsize = (10,10))
sns.countplot(x = 'ApplicantClass',data =  train_df, ax = ax)
fig = sns.FacetGrid(train_df, col = 'ApplicantClass')
fig.map(plt.hist, 'LoanAmount', bins = 50)
plt.show()
train_df.head()
train_df[['Family_Income','LoanAmount']].corr()
train_df.boxplot(column='Family_Income', by = 'Loan_Status')
train_df.Credit_History = train_df.Credit_History.astype('bool')
Cols = [i for i in train_df.columns if train_df[i].dtype == 'float64']
Cols
train_df_c = np.log(train_df[['ApplicantIncome', 'Family_Income','LoanAmount']]).copy()
model = DBSCAN(eps = 0.7, min_samples= 25).fit(train_df_c)
print(train_df_c[model.labels_ == -1])
print(Counter(model.labels_))























