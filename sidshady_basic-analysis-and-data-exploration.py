# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt 

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_schema = pd.read_csv("../input/schema.csv")
df_schema.head(10)
df_mcq = pd.read_csv("../input/multipleChoiceResponses.csv",encoding="ISO-8859-1")
df_mcq.head()
df_mcq.info()
plt.figure(figsize=(12,8))



sns.countplot(df_mcq["GenderSelect"])
plt.figure(figsize=(20,15))



sns.boxplot(x=df_mcq["GenderSelect"],y=df_mcq["Age"],hue=df_mcq["EmploymentStatus"])



plt.legend(loc="upper right")

plt.ylabel("Age")

plt.xlabel("Genders")
female_count = len(df_mcq.query("GenderSelect=='Female'"))

male_count = len(df_mcq.query("GenderSelect=='Male'"))
for idx,count in zip(df_mcq.query("GenderSelect=='Female'")["EmploymentStatus"].value_counts().index,df_mcq.query("GenderSelect=='Female'")["EmploymentStatus"].value_counts()) : 

    print(idx,":",format((count*100)/female_count,".2f"))
for idx,count in zip(df_mcq.query("GenderSelect=='Male'")["EmploymentStatus"].value_counts().index,df_mcq.query("GenderSelect=='Male'")["EmploymentStatus"].value_counts()) : 

    print(idx,":",format((count*100)/male_count,".2f"))
df_mcq['Country'].value_counts(20)
df_mcq_top20 = df_mcq[df_mcq['Country'].isin(df_mcq['Country'].value_counts().head(20).index)]
plt.figure(figsize=(20,15))



g = sns.countplot(df_mcq_top20['Country'])

locs, labels = plt.xticks()

g.set_xticklabels(labels,rotation=45)
plt.figure(figsize=(20,15))



g = sns.boxplot(x=df_mcq_top20["Country"],y=df_mcq_top20["Age"])



plt.legend(loc="upper right")

plt.ylabel("Age")

plt.xlabel("Countries")

locs, labels = plt.xticks()

g.set_xticklabels(labels,rotation=45)

plt.tight_layout()
plt.figure(figsize=(12,10))

g = sns.boxplot(x=df_mcq['CurrentJobTitleSelect'],y=df_mcq["Age"],hue=df_mcq['TitleFit'])

plt.legend(loc="upper right")

locs, labels = plt.xticks()

g.set_xticklabels(labels,rotation=90)

plt.tight_layout()
df_data_scientists = df_mcq.query("CurrentJobTitleSelect =='Data Scientist'")
df_data_scientists.head()
ds_count = df_data_scientists['Country'].value_counts().head(20) 

plt.figure(figsize=(15,12))

sns.barplot(y=ds_count.index,x=ds_count.values)



plt.title("Count of Data Scientists Across the top 20 Countries")