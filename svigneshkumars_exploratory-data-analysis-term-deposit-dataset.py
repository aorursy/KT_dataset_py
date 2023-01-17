# importing warnings

import warnings

warnings.filterwarnings("ignore")
# importing packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# importing dataset



df = pd.read_csv("../input/termdepositdataset/bank_marketing_updated_v1.csv")

df.head()
df = pd.read_csv("../input/termdepositdataset/bank_marketing_updated_v1.csv", skiprows = 2)

df.head()
df.drop("customerid", axis = 1, inplace = True)

df.head()
df['job'] = df.jobedu.apply(lambda x: x.split(",")[0])

df.head()
df['education'] = df.jobedu.apply(lambda x: x.split(",")[1])

df.head()
#### dropping "jobedu" column



df.drop("jobedu", axis = 1, inplace = True)

df.head()
#df['year'] = df.month.apply(lambda x: x.split(",")[1])

#df.head()



# this code is showing an error so its commented
df[df.month.apply(lambda x: isinstance(x,float) == True)]
df.isnull().sum()
df.shape
(20/45211)*100
## creating another data frame of non null values in age



df1 = df[~df.age.isnull()].copy()
type(df1.age.loc[0])
df1.age.describe()
df1.head()
df1.shape
df.shape[0]-df1.shape[0]
df1.head(15)
df1.month.isnull().sum()
type(df.month.loc[0])
df1.month.value_counts()
df1.month.value_counts(normalize = True)
month_mode = df1.month.mode()[0]

month_mode
df1.month.fillna(month_mode, inplace = True)
df1.isnull().sum()
(df1.response.isnull().sum()/df.shape[0])*100
df2 = df1[~df1.response.isnull()]
df1.shape[0] - df2.shape[0]
df2.response.isnull().sum()
df2.shape
df2.head(10)
df2.salary.describe()
df2.salary.isnull().sum()
plt.figure(figsize = [15,10])

sns.boxplot(df2.salary)

plt.show()
type(df2.duration.loc[0])
df2.duration.value_counts()
df2.marital.value_counts()
df2.marital.value_counts(normalize = True)
df2.marital.value_counts(normalize = True).plot.barh()

plt.show()
df2.head()
df2.job.value_counts(normalize = True)
plt.figure(figsize = (15,10))

df2.job.value_counts(normalize = True).plot.bar()
df2.head()
df2['response_flag'] = np.where(df2.response == "yes",1,0)
df2.head()
df2.groupby(['education'])['response_flag'].mean()