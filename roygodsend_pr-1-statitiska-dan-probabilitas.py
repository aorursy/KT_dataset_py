# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/500-person-gender-height-weight-bodymassindex/500_Person_Gender_Height_Weight_Index.csv")
# Remove "Height" feature

df = df.drop(["Height"],axis=1)

# Mean

print("The mean of the weight is :",np.mean(df["Weight"]))
# Median



print("The median of the weight is :",np.median(df["Weight"]))
# Mode

data = df["Weight"]

maxValue = pd.Series.max(data.value_counts())



print("The modus of the weight are : ")

print(data.value_counts()[data.value_counts() == maxValue])
# Range



print(pd.Series.max(data) - pd.Series.min(data))
# Variance

mean = np.mean(data)



variance = (sum((data - mean)**2))/(len(data)-1)

print("Variance of Weight data is " ,variance)
# Standar Deviation

std = variance**0.5



print("Standar deviation of Weight data is" ,std)
# Percentile-15

print("Percentile-15 of the data is:", df.Weight.quantile(0.15))



# Percentile-90

print("Percentile-90 of the data is:", df.Weight.quantile(0.90))
IQR = df.Weight.quantile(0.75) - df.Weight.quantile(0.25)



print("IQR of the Weight data is:",IQR)



RLB = df.Weight.quantile(0.25) - 1.5*IQR

print("RLB of the Weight data is", RLB)



RUB = df.Weight.quantile(0.75) +1.5*IQR

print("RUB of the Weight data is", RUB)

print("Min of Weight data is " ,df["Weight"].min())

print("Max of Weight data is ", df["Weight"].max())
plt.boxplot(df["Weight"])
group_num = (max(df["Weight"]) - min(df["Weight"])) // 10



f, ax = plt.subplots(1,1, figsize=(8,4))

ax = sns.distplot(df['Weight'], bins = group_num, color = '#156DF3', kde = False)

plt.ylabel('Frequency', size = 15)

plt.xlabel('Weight', size = 15)

plt.title('Weight Histogram', size = 20)



plt.show()