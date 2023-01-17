#Load libraries and the dataset

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

df = pd.read_csv('../input/insurance.csv')

df.head()
df.dtypes #check datatype
df.mean()
df.corr()
df.count()
df.max()
df.std()
df.isnull().sum(axis=0) #check null
#Crosstab - we can validate some basic hypothesis using PANDAS crosstab

pd.crosstab(df["sex"],df["region"],margins=True)
def percConvert(ser):

    return ser/float(ser[-1])

pd.crosstab(df["sex"],df["region"],margins=True).apply(percConvert, axis=1)
df.loc[(df["sex"]=="male") & (df["smoker"]=="yes") & (df["region"]=="southwest"), ["sex","smoker","region"]].head(10)
df_sorted = df.sort_values(['smoker','region'], ascending=False)

df_sorted[['smoker','region']].head(10)
# Pivot

impute_grps = df.pivot_table(values=["charges"], index=["sex","smoker"], aggfunc=np.mean)

print (impute_grps)
df.boxplot(column="charges",by="region", figsize=(18, 8))
df.boxplot(column="charges",by="smoker", figsize=(18, 8))
df.boxplot(column="charges",by="children", figsize=(18, 8))
df.hist(column="charges",by="sex",figsize=(18, 8), bins=30)
#Scatter 

df.plot.scatter(x='charges', y='bmi', figsize=(18, 8))
# Area Plot

df_new = df.drop(columns = 'charges') #dropping charges for the plot

df_new.plot.area(figsize=(18, 8))
# Kernel Density Estimation plot (KDE)

df['charges'].plot(kind='kde')
from IPython.display import YouTubeVideo

YouTubeVideo("B42n3Pc-N2A")