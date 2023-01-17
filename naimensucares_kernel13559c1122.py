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
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



data=pd.read_csv("/kaggle/input/calcofi/bottle.csv")

data.head()
data.columns

data2=pd.read_csv("/kaggle/input/calcofi/cast.csv")

data2.head(3)
data2.info()


example=data[["Depthm",'T_degC','Salnty']]

# line=example[example["Depthm"]==0].index

line=example[example["Depthm"]==0].iloc[:40].index

line



example=example[:line[-1]]

example.info()
example=example.fillna(example.mean())

example.info()
sns.lmplot(x='T_degC',y='Depthm',data = example)

sns.lmplot(x='Salnty',y='T_degC',data = example)

sns.lmplot(x='Salnty',y='Depthm',data = example)
sns.heatmap(example.corr(),annot= True)
Q1 = example.T_degC.quantile(.25)

Q3 = example.T_degC.quantile(.75)

IQR = Q3-Q1

print("Q1=",Q1,"Q1=",Q3,"IQR=",IQR)

lower_bound = Q1 - 1.5*IQR

upper_bound = Q3 + 1.5*IQR

print(lower_bound)

print(upper_bound)





print("Outliers: ",example[(example.T_degC<lower_bound) | (example.T_degC > upper_bound)].T_degC.values)



melted_data = pd.melt(example,value_vars = ['T_degC'])



sns.boxplot(x = "variable", y = "value", data= melted_data)
example1=example[:line[1]]

example1.info()
sns.heatmap(example1.corr(),annot= True)
example1.describe()
print(example1.var(),"\n")

print(example1.mean(),"\n")

print(example1.std())
sns.lmplot(x='T_degC',y='Depthm',data = example1)

sns.lmplot(x='Salnty',y='T_degC',data = example1)

sns.lmplot(x='Salnty',y='Depthm',data = example1)


Q1 = example1.Salnty.quantile(.25)

Q3 = example1.Salnty.quantile(.75)

IQR = Q3-Q1

print(Q1,Q3,IQR)

lower_bound = Q1 - 1.5*IQR

upper_bound = Q3 + 1.5*IQR

print(lower_bound)

print(upper_bound)





print("Outliers: ",example1[(example1.Salnty<lower_bound) | (example1.Salnty > upper_bound)].Salnty.values)



melted_data = pd.melt(example1,id_vars = 'Depthm',value_vars = ['Salnty'])



sns.boxplot(x = "variable", y = "value", data= melted_data)

Q1 = example1.Depthm.quantile(.25)

Q3 = example1.Depthm.quantile(.75)

IQR = Q3-Q1

print(Q1,Q3,IQR)

lower_bound = Q1 - 1.5*IQR

upper_bound = Q3 + 1.5*IQR

print(lower_bound)

print(upper_bound)





print("Outliers: ",example1[(example1.Depthm<lower_bound) | (example1.Depthm > upper_bound)].Depthm.values)



melted_data = pd.melt(example1,value_vars = ['Depthm'])

melted_data



sns.boxplot(x="variable",y="value",data=melted_data)
Q1 = example1.T_degC.quantile(.25)

Q3 = example1.T_degC.quantile(.75)

IQR = Q3-Q1

print(Q1,Q3,IQR)

lower_bound = Q1 - 1.5*IQR

upper_bound = Q3 + 1.5*IQR

print(lower_bound)

print(upper_bound)





print("Outliers: ",example1[(example1.T_degC<lower_bound) | (example1.T_degC > upper_bound)].T_degC.values)



melted_data = pd.melt(example1,value_vars = ['T_degC'])

melted_data



sns.boxplot(x="variable",y="value",data=melted_data)
example2=example[line[-2]:line[-1]]

example2.info()
example2

sns.lmplot(x='T_degC',y='Depthm',data = example2)

sns.lmplot(x='Salnty',y='T_degC',data = example2)

sns.lmplot(x='Salnty',y='Depthm',data = example2)
Q1 = example2.T_degC.quantile(.25)

Q3 = example2.T_degC.quantile(.75)

IQR = Q3-Q1

print(Q1,Q3,IQR)

lower_bound = Q1 - 1.5*IQR

upper_bound = Q3 + 1.5*IQR

print(lower_bound)

print(upper_bound)





print("Outliers: ",example2[(example2.T_degC<lower_bound) | (example2.T_degC > upper_bound)].T_degC.values)



melted_data = pd.melt(example2,value_vars = ['T_degC'])

melted_data



sns.boxplot(x="variable",y="value",data=melted_data)