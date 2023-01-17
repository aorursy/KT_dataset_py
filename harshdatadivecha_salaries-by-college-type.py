%matplotlib inline



import seaborn as sns

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

dataset = pd.read_csv("../input/salaries-by-college-type.csv")
dataset.head(5)
cols = list(dataset)

cols.insert(5,cols.pop(cols.index('Mid-Career Median Salary')))



cols
dataset=dataset.loc[:,cols]

dataset.columns = cols=["School_name",'College_type','Start','Mid_10th','Mid_25th','Mid_50th','Mid_75th','Mid_90th']

dataset.head(3)
df = dataset
print ( type(df.Mid_10th[0]),type(df.Mid_10th[1]))


def convert(col):

#for col in columns:

    df[col] = df[col].map(

        lambda x : x.split('$')[1] if not pd.isnull(x) else None).map(

        lambda x: float(x.split(',')[0]+x.split(',')[1]) if not pd.isnull(x) else np.nan)
for col in df.columns[2:]:

    convert(col)
df.info()




def fill_nas(cols):

    for col in cols:

        df[col]=df.groupby("College_type")[col].apply(lambda x: x.fillna(x.mean()))
cols = df.columns[2:]

fill_nas(cols)
df.info()
df.head(4)
grouped_df = df.groupby("College_type").mean().sort_values(by="Start",ascending=False)

grouped_df
grouped_df=grouped_df.transpose()
f,ax = plt.subplots(figsize=(8,7))

grouped_df.plot(ax=ax)

ax.set_ylabel("Salary")

ax.set_xlabel("Career")
df_melt = df.drop("School_name",axis=1)

df_melt=df_melt.melt(id_vars="College_type",var_name="Career",value_name="Salary")
df_melt.head(6)
f,ax= plt.subplots(figsize=(8,6))



sns.boxplot(data=df_melt,x="Career",y="Salary",ax=ax)
f,ax = plt.subplots(figsize=(8,6))

sns.swarmplot(ax=ax,x=df_melt.College_type,y=df_melt.Salary)

ax.set_title("Salaries acc to college type")
f,ax = plt.subplots(figsize=(8,6))

sns.swarmplot(ax=ax,x=df_melt.Career,y=df_melt.Salary,hue=df_melt.College_type)

ax.set_title("Salaries acc to Career")
df_melt.College_type.value_counts()