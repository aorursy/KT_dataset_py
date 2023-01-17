%matplotlib inline
import matplotlib.pyplot as plt 

import seaborn as sns

import numpy as np

import pandas as pd
degrees_that_pay_back = pd.read_csv("../input/degrees-that-pay-back.csv")
degrees_that_pay_back.head()
cols = list(degrees_that_pay_back)



cols.insert(1,cols.pop(cols.index('Percent change from Starting to Mid-Career Salary')))

cols.insert(5,cols.pop(cols.index('Mid-Career Median Salary')))



cols
degrees_that_pay_back=degrees_that_pay_back.loc[:,cols]

degrees_that_pay_back.columns = cols=["Major",'Percentchange_start_to_mid','Start','Mid_10th','Mid_25th','Mid_50th','Mid_75th','Mid_90th']

degrees_that_pay_back.head(3)
df = degrees_that_pay_back

df.info()
type(df.Start[0])
def convert(col):

#for col in columns:

    df[col] = df[col].map(lambda x : x.split('$')[1]).map(lambda x: float(x.split(',')[0]+x.split(',')[1]))

for col in df.columns[2:]:

    convert(col)
df.head(3)
df.describe()
df.sort_values(by="Start",ascending=False,inplace=True)

df.reset_index(inplace=True)

df.head(4)
f,ax = plt.subplots(figsize = (8,9))

df['index'] =sorted(df['index'],reverse=True) 

ax1 = df.plot(kind="scatter",x="Start",y="index",ax=ax,color='g',label="Start")

ax2 = df.plot(kind="scatter",x="Mid_10th",y="index",ax=ax,color='c',label="Mid_10th")

ax3 = df.plot(kind="scatter",x="Mid_25th",y="index",ax=ax,color='y',label="Mid_25th")

ax4 = df.plot(kind="scatter",x="Mid_50th",y="index",ax=ax,color='b',label="Mid_50th")

ax5 = df.plot(kind="scatter",x="Mid_75th",y="index",ax=ax,color='m',label="Mid_75th")

ax6 = df.plot(kind="scatter",x="Mid_90th",y="index",ax=ax,color='r',label="Mid_90th")



ax.set_xlabel("Salary")

ax.set_ylabel("Majors")
df1 = df.drop(["index","Percentchange_start_to_mid"],axis=1)
df1 = df1.melt(id_vars="Major",value_vars=['Start','Mid_10th','Mid_25th','Mid_50th','Mid_75th','Mid_90th'],var_name="Category",value_name="Salary")

df1.head()
f,ax = plt.subplots(figsize=(8,7))

sns.swarmplot( data=df1, y="Category", x="Salary", hue="Major",orient="h",ax=ax)

ax.legend_.remove()



# THE COLORS REPRESENT MAJORS
f,ax = plt.subplots(figsize=(9,6))

sns.distplot(df.Percentchange_start_to_mid,bins=10)