# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from scipy.stats import ttest_ind

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory







#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/cereal.csv")

#print(df.head())

print(df.describe())

cal=df["calories"]

plt.hist(cal, bins=50)

plt.title("calories")

plt.show()

#print(a)
y=df["rating"]

x=df.drop('rating',1)
#correlation between different features using pearson method

a=df.corr(method='pearson')
a.columns

#checking which elements are related to each other using a threshhold



for i in a.columns:

    for j in a.columns:

        if (a[i][j]>0.6 or a[i][j]<-0.6) and i!=j:

            print(i,j,a[i][j])

            

        
#no idea why we are doing this

t,prob=ttest_ind(x['calories'],y,equal_var=False)
print(t)

print(prob)
plt.hist(x['calories'])



plt.show()
plt.hist(y)

plt.show()