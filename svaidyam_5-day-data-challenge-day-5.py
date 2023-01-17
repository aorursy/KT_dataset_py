# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import scipy.stats as sp

from scipy.stats import chisquare 

df = pd.read_csv("../input/CAERS_ASCII_2004_2017Q2.csv",low_memory=False)

df.info()

df.head(20)

a = ["Not Available"]

df = df[~df['CI_Gender'].isin(a)]

df['COUNTER'] =1       #initially, set that counter to 1.

group_data = df.groupby(['CI_Gender','SYM_One Row Coded Symptoms'])['COUNTER'].sum() #sum function

print(group_data)

df["CI_Gender"] = df["CI_Gender"].astype('category')

#df.dtypes



df["CI_Gender_cat"] = df["CI_Gender"].cat.codes

#df.head()



df["SYM_One Row Coded Symptoms"] = df["SYM_One Row Coded Symptoms"].astype('category')

#df.dtypes



df["SYM_One Row Coded Symptoms_cat"] = df["SYM_One Row Coded Symptoms"].cat.codes

#df.head()

ct1=pd.crosstab(df["CI_Gender_cat"], df["SYM_One Row Coded Symptoms_cat"])

print("\n\t\t\tCross Table for the 2 columns selected:\n\n")

print (ct1)



colsum=ct1.sum(axis=0)

colpct=ct1/colsum

print("\n\n\n\t\t\tCross Table values in Percentages:\n\n")

print(colpct)



# chi-square

print("\tNull Hypothesis :\n")

print("There is no difference in the number of observations across different groups",\

      "but is just the result of random variation")



print ("\n\tResults - chi square value, p value, dof, expected counts: \n")

chi2, p, dof, ex = sp.chi2_contingency(ct1)

print ("chi square value : ", chi2)

print("p value : ", p)

print("degrees of freedom : ", dof)

print("expected counts :\n")

print(ex)



print("\n\n\tInference :\n")

print("Since the P-value(0.0) is less than the significance level (0.05),"\

      "we reject the null hypothesis.")
import matplotlib.pyplot as plt

# Create the lists with survival values for each gender

import seaborn as sns

N = 1000

gender = np.random.choice(df["CI_Gender_cat"],N)

symptoms = np.random.choice(df["SYM_One Row Coded Symptoms_cat"],N)



df1 = pd.DataFrame({'gender':gender,'symptom':symptoms})

ct = pd.crosstab(df1.gender, df1.symptom)

# now stack and reset

stacked = ct.stack().reset_index().rename(columns={0:'value'})



 # plot grouped bar chart

sns.barplot(x=stacked.gender, y=stacked.value, hue=stacked.symptom)