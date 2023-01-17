import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sn
df = pd.read_excel(io='/kaggle/input/bank-loan-modelling/Bank_Personal_Loan_Modelling.xlsx' ,sheet_name='Data')
for i in df.columns:

    print('Column name is' , i , 'and its location is' ,df.columns.get_loc(i))
df.info()
df.isnull().values.any()
df.duplicated().any()
df.head(10)
df.describe()
for col in df.columns:

    print(col , sum([n < 0 for n  in df[col].values.flatten()]))
df[df['Experience'] < 0]
df_cleaned = df[df['Experience'] > 0]
df_cleaned.describe()
melted_data = pd.melt(df_cleaned.iloc[:,9:])

melted_data.loc[melted_data['value'] == 0 , ['value']] = 'No'

melted_data.loc[melted_data['value'] == 1 , ['value']] = 'Yes'

plt.figure(figsize=(15,5))

sn.countplot(x="variable", hue="value", data=melted_data)

plt.show()

plt.close()
plt.figure(figsize=(10,5))

sn.relplot(x="Income", y="Mortgage" ,aspect = 2 ,data=df_cleaned)

plt.show()

plt.clf()
plt.figure(figsize=(10,5))

sn.relplot(x="Income", y="CCAvg" ,aspect = 2 ,data=df_cleaned)

plt.show()

plt.clf()
plt.figure(figsize=(10,5))

sn.relplot(x="Income", y="Mortgage", #hue="Personal Loan",

            col="Education", data=df_cleaned)

plt.show()

plt.clf()
plt.figure(figsize=(5,5))

sn.relplot(x="Experience", y="Income",col = "Education",

             data=df_cleaned)

plt.show()

plt.clf()
plt.figure(figsize=(5,5))

sn.relplot(x="Income", y="Mortgage",col = "Family",# hue="Education",

             data=df_cleaned)

plt.show()

plt.clf()
numerical = ['Age' , 'Experience' ,'Family' ,'Income' , 'CCAvg' , 'Mortgage']

fig, ax = plt.subplots(2, 3, figsize=(15, 10))

for var, subplot in zip(numerical, ax.flatten()):

    sn.boxplot(x='Personal Loan', y=var, data=df_cleaned, ax=subplot)

plt.show()

plt.clf()
plt.figure(figsize=(5,5))

sn.relplot(x="Income", y="Family",#col = "Family", 

           hue="Personal Loan",

             data=df_cleaned)

plt.show()

plt.clf()