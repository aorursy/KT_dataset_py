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
!pip install pandas-profiling
df_train = pd.read_csv("/kaggle/input/loanprediction/train_ctrUa4K.csv")

df_test = pd.read_csv("/kaggle/input/loanprediction/test_lAUu6dG.csv")

df_train.head(5)
import pandas_profiling

df_train.profile_report(style={'full_width':True})
import math

import seaborn as sns

import matplotlib.pyplot as plt
df_train.columns
gender = pd.crosstab(df_train['Gender'], df_train['Loan_Status'])

print(gender)

gender.plot(kind="bar", stacked=True, figsize=(5,5))
gender = pd.crosstab(df_train['Gender'], df_train['Loan_Status'])

print(gender)

gender.plot(kind="bar", stacked=True, figsize=(5,5))
married = pd.crosstab(df_train['Married'], df_train['Loan_Status'])

print(married)

married.plot(kind="bar", stacked=True, figsize=(5,5))
dependents = pd.crosstab(df_train['Dependents'], df_train['Loan_Status'])

print(dependents)

dependents.plot(kind="bar", stacked=True, figsize=(5,5))
education = pd.crosstab(df_train['Education'], df_train['Loan_Status'])

print(education)

education.plot(kind="bar", stacked=True, figsize=(5,5))
self_employed = pd.crosstab(df_train['Self_Employed'], df_train['Loan_Status'])

print(self_employed)

self_employed.plot(kind="bar", stacked=True, figsize=(5,5))
credit_history = pd.crosstab(df_train['Credit_History'], df_train['Loan_Status'])

print(credit_history)

credit_history.plot(kind="bar", stacked=True, figsize=(5,5))
df_train[(df_train['Self_Employed'] == 'Yes') & (df_train['Credit_History'] == 0.0)]
df_nopass = df_train[df_train['Loan_Status'] == 'N']

df_nopass.head(10)
df_train[(df_train['Self_Employed'] == 'No') & (df_train['Credit_History'] == 1.0)]
property_area = pd.crosstab(df_train['Property_Area'], df_train['Loan_Status'])

print(property_area)

property_area.plot(kind="bar", stacked=True, figsize=(5,5))
df_new_train = df_train[df_train['Credit_History'].notnull()]

sns.heatmap(df_new_train.isnull(), cbar=False)