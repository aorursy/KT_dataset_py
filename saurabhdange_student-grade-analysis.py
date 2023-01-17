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
#importing csv file into pandas dataframe
import pandas as pd
df = pd.read_csv("../input/student-mat.csv")

df.info()
df.describe()
df.corr()
import seaborn as sns
corr = df.corr()
sns.heatmap(corr,xticklabels=corr.columns,
        yticklabels=corr.columns)
# bucketing students according to age
import matplotlib.pyplot as plt
bins = [15,16,17,18,19,20,21,22,23,24]
plt.hist(df["age"], bins, histtype='bar', rwidth=0.8)
#scatter plot of G1 marks vs age
plt.scatter(df["age"],df["G1"], color='k', s=25, marker="o")
plt.show()
#stackplot of age , G1 , G2, G3 
plt.stackplot( df.index,df["age"],df["G1"],df["G2"],df["G3"], colors=['m','c','r','k'])
#number of females in each school
print(df[(df['school'] == "GP") & (df['sex'] == "F")]["sex"].count())
print(df[(df['school'] == "MS") & (df['sex'] == "F")]["sex"].count())
#pie chart showing distribution of females in schools
slices = [df[(df['school'] == "GP") & (df['sex'] == "F")]["sex"].count(),df[(df['school'] == "MS") & (df['sex'] == "F")]["sex"].count()]
activities = ["GP","MS"]
cols = ['c','m']

plt.pie(slices,
        labels=activities,
        colors=cols,
        startangle=90,
        shadow= True,
        explode=(0,0.1),
        autopct='%1.1f%%')

plt.title('Female percentage in schools')
plt.show()
#pie chart showing distribution of females in schools
slices = [df[(df['school'] == "GP") & (df['sex'] == "M")]["sex"].count(),df[(df['school'] == "MS") & (df['sex'] == "M")]["sex"].count()]
activities = ["GP","MS"]
cols = ['c','m']

plt.pie(slices,
        labels=activities,
        colors=cols,
        startangle=90,
        shadow= True,
        explode=(0,0.1),
        autopct='%1.1f%%')

plt.title('Male percentage in schools')
plt.show()
#scatter plot of G1 marks vs G2 marks
plt.scatter(df["G1"],df["G2"], color='k', s=25, marker="o")
plt.xlabel('G1 marks')
plt.ylabel('G2 marks')
plt.legend()
plt.show()
#scatter plot of G1 marks vs G2 marks
plt.scatter(df["G1"],df["G3"], color='k', s=25, marker="o")
plt.xlabel('G1 marks')
plt.ylabel('G3 marks')
plt.legend()
plt.show()
df["G1G2"]=(df["G1"] + df["G2"])/2
#scatter plot of G1G2 marks vs G3 marks
plt.scatter(df["G1G2"],df["G3"], color='k', s=25, marker="o")
plt.xlabel('G1G2 marks')
plt.ylabel('G3 marks')
plt.legend()
plt.show()
bins = [-1,0,1,2,3,4]
plt.hist(df["failures"], bins, histtype='bar', rwidth=0.5)