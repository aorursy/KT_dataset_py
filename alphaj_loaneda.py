# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import precision_score, recall_score, f1_score, log_loss, accuracy_score



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
link_train="/kaggle/input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv"

df=pd.read_csv(link_train)
df.columns
df.head()
df.describe()
df.groupby("Loan_Status").Loan_Status.count()
df.duplicated().any()
plt.figure(figsize=(10,6))

sns.countplot(df['Loan_Status'])

print("Yes: ", df.Loan_Status.value_counts()[0]/len(df))

print("No: ", df.Loan_Status.value_counts()[1]/len(df))
sns.swarmplot(x="Credit_History", y="Loan_Status", data=df)
plt.figure(figsize=(6,6))

sns.countplot(x='Married', hue='Loan_Status', data=df)

# if one is married, the person has a higher chance of getting the loan
df.groupby("Dependents").Dependents.count()
plt.figure(figsize=(6,6))

sns.countplot(x='Dependents', hue='Loan_Status', data=df)
sns.countplot(x='Education',hue='Loan_Status',data=df)
sns.countplot(x='Self_Employed',hue='Loan_Status',data=df)
sns.countplot(x='Gender',hue='Loan_Status',data=df)
df.groupby("Loan_Status").median()
df.isnull().sum().sort_values(ascending=False)
cat=[]

num=[]

for i,c in enumerate (df.dtypes):

    if c == object:

        cat.append(df.iloc[:,i])

    else:

        num.append(df.iloc[:,i])
cat= pd.DataFrame(cat).transpose()

num = pd.DataFrame(num).transpose()
cat.head()
num.head()