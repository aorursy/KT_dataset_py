# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('../input/crime-in-india/20_Victims_of_rape.csv')

df
df.isna().sum()
df.describe
df.dtypes
df.hist(figsize=(10,10),edgecolor="k")

plt.tight_layout()

plt.show()
Victims_above_50 = df['Victims_Above_50_Yrs'].sum()

Victims_30_to_50 = df['Victims_Between_30-50_Yrs'].sum()

Victims_18_to_38 = df['Victims_Between_18-30_Yrs'].sum()

Victims_14_to_18 = df['Victims_Between_14-18_Yrs'].sum()

Victims_10_to_14 = df['Victims_Between_10-14_Yrs'].sum()

Victims_upto_10 = df['Victims_Upto_10_Yrs'].sum()



Age=['Victims_Above_50_Yrs','Victims_Between_30-50_Yrs','Victims_Between_18-30_Yrs','Victims_Between_14-18_Yrs','Victims_Between_10-14_Yrs',

      'Victims_Upto_10_Yrs']

SUM=[Victims_above_50,Victims_30_to_50,Victims_18_to_38,Victims_14_to_18,Victims_10_to_14,Victims_upto_10]



fig1, ax1 = plt.subplots(figsize=(8,8))

ax1.pie(SUM, labels=Age, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
sns.barplot(x="Year", y="Rape_Cases_Reported", data=df)
sns.barplot(x="Subgroup", y="Victims_Above_50_Yrs", data=df)

plt.figure(figsize=(8,8))
sns.barplot(x="Subgroup", y="Victims_Between_30-50_Yrs", data=df)

plt.figure(figsize=(8,8))
sns.barplot(x="Subgroup", y="Victims_Between_18-30_Yrs", data=df)

plt.figure(figsize=(8,8))
sns.barplot(x="Subgroup", y="Victims_Between_14-18_Yrs", data=df)

plt.figure(figsize=(8,8))
sns.barplot(x="Subgroup", y="Victims_Between_10-14_Yrs", data=df)

plt.figure(figsize=(8,8))
sns.barplot(x="Subgroup", y="Victims_Upto_10_Yrs", data=df)

plt.figure(figsize=(8,8))
pd.value_counts(df['Subgroup']).plot.bar()
ab=df[df['Subgroup']=='Victims of Incest Rape']

bc=df[df['Subgroup']=='Victims of Other Rape']



fig = plt.figure()

ax = fig.add_axes([0,0,1,1])



Victims_above_50 = ab['Victims_Above_50_Yrs'].sum()

Victims_30_to_50 = ab['Victims_Between_30-50_Yrs'].sum()

Victims_18_to_38 = ab['Victims_Between_18-30_Yrs'].sum()

Victims_14_to_18 = ab['Victims_Between_14-18_Yrs'].sum()

Victims_10_to_14 = ab['Victims_Between_10-14_Yrs'].sum()

Victims_upto_10 = ab['Victims_Upto_10_Yrs'].sum()



Range=['Above_50','30-50','18-30','14-18','10-14','Upto_10']

Total=[Victims_above_50,Victims_30_to_50,Victims_18_to_38,Victims_14_to_18,Victims_10_to_14,Victims_upto_10]



ax.bar(Range,Total)

plt.show()
fig = plt.figure()

ax = fig.add_axes([0,0,1,1])



Victims_above_50 = bc['Victims_Above_50_Yrs'].sum()

Victims_30_to_50 = bc['Victims_Between_30-50_Yrs'].sum()

Victims_18_to_38 = bc['Victims_Between_18-30_Yrs'].sum()

Victims_14_to_18 = bc['Victims_Between_14-18_Yrs'].sum()

Victims_10_to_14 = bc['Victims_Between_10-14_Yrs'].sum()

Victims_upto_10 = bc['Victims_Upto_10_Yrs'].sum()



Range=['Above_50','30-50','18-30','14-18','10-14','Upto_10']

Total=[Victims_above_50,Victims_30_to_50,Victims_18_to_38,Victims_14_to_18,Victims_10_to_14,Victims_upto_10]



ax.bar(Range,Total)

plt.show()
df1=pd.read_csv('../input/crime-in-india/30_Auto_theft.csv')

df1.head()
df1.isna().sum()
df1['Auto_Theft_Coordinated/Traced'].fillna((df1['Auto_Theft_Coordinated/Traced'].mean()), inplace=True)

df1['Auto_Theft_Recovered'].fillna((df1['Auto_Theft_Recovered'].mean()),inplace=True)
df1.describe()
df1.hist(figsize=(10,10),edgecolor="k")

plt.tight_layout()

plt.show()
import pandas as pd



new = df1[['Auto_Theft_Stolen', 'Auto_Theft_Recovered', 'Auto_Theft_Coordinated/Traced']].copy()



stolen_df = new['Auto_Theft_Stolen'].sum()

theft_df = new['Auto_Theft_Recovered'].sum()

traced_df = new['Auto_Theft_Coordinated/Traced'].sum()



fig = plt.figure()

ax = fig.add_axes([0,0,1,1])





total = [stolen_df,theft_df,traced_df]

columns = ['Stolen','Recovered','Traced']



ax.bar(columns,total)

plt.show()

fig1, ax1 = plt.subplots()

ax1.pie(total, labels=columns, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
sns.barplot(x="Year", y="Auto_Theft_Stolen", data=df1)
sns.barplot(x="Year", y="Auto_Theft_Coordinated/Traced", data=df1)
sns.barplot(x="Year", y="Auto_Theft_Recovered", data=df1)