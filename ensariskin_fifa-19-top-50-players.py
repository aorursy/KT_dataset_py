# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df  = pd.read_csv('../input/data.csv')

df_top = df.head(50)





df_top.drop(["ID"],axis=1,inplace = True)

df_top.drop(["Photo"],axis=1,inplace = True)

df_top.drop(["Flag"],axis=1,inplace = True)

df_top.drop(["Club Logo"],axis=1,inplace = True)

df_top.drop(["Club"],axis=1,inplace = True)



df_top_clrd = df_top.loc[:,"Name":"Wage"]



print(df_top_clrd)

i = 0

for each in df_top_clrd.Value:

    ln_data = len(each)

    df_top_clrd.Value[i] = each[1:ln_data-1]

    i = i+1



i = 0

for each in df_top_clrd.Wage:

    ln_data = len(each)

    df_top_clrd.Wage[i] = each[1:ln_data-1]

    i = i+1

    

print(df_top_clrd)

df_top_clrd.info()
df_top_clrd["Value"] = df_top_clrd.Value.astype(float)

df_top_clrd["Wage"] = df_top_clrd.Wage.astype(int)

df_top_clrd.corr()

f,ax = plt.subplots(figsize=(15, 15))

sns.heatmap(df_top_clrd.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
df_top_clrd.plot(kind="scatter", x = "Potential", y = "Value")

plt.show()
df_top_clrd.plot(kind="scatter", x = "Wage", y = "Age")

plt.show()
countries = df_top_clrd.Nationality.unique()

print(countries)



dic = pd.value_counts(df_top_clrd.Nationality)

print(dic)

df_top_clrd.boxplot(column = "Value")

plt.show()

print("Avarage    : ", "€"+str(df_top_clrd.Value.mean())+"M")

print("Max Values : ", "€"+str(np.sort(df_top_clrd.Value)[49])+"M","&","€"+str(np.sort(df_top_clrd.Value)[48])+"M")

print("Min Value  : " ,"€"+str(df_top_clrd.Value.min())+"M")

df_top_clrd.boxplot(column = "Wage")

plt.show()
print("Avarage  : ", "€"+str(df_top_clrd.Wage.mean())+"K")

print("Max Wage : " ,"€"+str(df_top_clrd.Wage.max())+"K")

print("Min Wage : " ,"€"+str(df_top_clrd.Wage.min())+"K")