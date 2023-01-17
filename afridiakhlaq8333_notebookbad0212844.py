# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt



from sklearn.preprocessing import Normalizer

from  sklearn.preprocessing import OrdinalEncoder

from sklearn.model_selection import train_test_split

from pylab import plot, show, subplot, specgram, imshow, savefig

from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score,f1_score, confusion_matrix

%matplotlib inline





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/loan-data-set/loan_data_set.csv")

df
df.describe(include = 'all')
df.mean()
df.median()
df.mode()
df1=df._get_numeric_data()

df1
numd = list(df._get_numeric_data().columns)

numd
ggs=sns.relplot(x="Loan_Amount_Term", y="LoanAmount", data=df,kind="line",hue="Education",ci=None)

ggs.fig.set_size_inches(15,7)

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

sns.heatmap(data=df.corr().round(2),annot=True,linewidths=0.5,cmap="Blues")

plt.show()
numer = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term","Credit_History"]
df[numer].hist(figsize=(12,10), bins=20)

plt.suptitle("Histograms ")

plt.show()
fig,ax=plt.subplots(figsize=(4,5))

sns.countplot(x = "Education", data=df, order = df["Education"].value_counts().index)

plt.show()
#compare coloumn

df1.T.plot(kind='line')
fig,ax=plt.subplots(figsize=(4,5))

sns.countplot(x = "Education", data=df, order = df["Education"].value_counts().index)

plt.show()
fig,ax=plt.subplots(figsize=(4,5))

sns.countplot(x = "Gender", data=df, order = df["Gender"].value_counts().index)

plt.show()
sns.factorplot('Dependents',kind='count',data=df,hue='Loan_Status')
print(pd.crosstab(df["Married"],df["Loan_Status"]))

Married=pd.crosstab(df["Married"],df["Loan_Status"])

Married.div(Married.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))

plt.xlabel("Married")

plt.ylabel("Percentage")

plt.show()
#corelation

df.corr() 
fig,ax=plt.subplots(figsize=(15,8))

sns.heatmap(data=df.corr().round(2),annot=True,linewidths=0.5,cmap="Blues")

plt.show()
comparison_column = np.where(df["ApplicantIncome"] == df["CoapplicantIncome"], True, False)

df["new comp"] = comparison_column

df
comparison_column = np.where(df["Married"] == df["Self_Employed"], True, False)

df["new comp"] = comparison_column

df
sns.factorplot('Dependents',kind='count',data=df,hue='Loan_Status')
print(pd.crosstab(df["Married"],df["Loan_Status"]))

Married=pd.crosstab(df["Married"],df["Loan_Status"])

Married.div(Married.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(5,5))

plt.xlabel("Married")

plt.ylabel("Percentage")

plt.show()
sns.relplot(x="ApplicantIncome", y="LoanAmount", data=df, col="Gender",color="Green",alpha=0.4)

plt.show()
ggs=sns.relplot(x="Loan_Amount_Term", y="LoanAmount", data=df,kind="line",hue="Education",ci=None)

ggs.fig.set_size_inches(15,7)

plt.show()