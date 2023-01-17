import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/forest-fires-in-brazil/amazon.csv",encoding="ISO-8859-1")

df.head(10)
df.isnull().sum()
list1 = df['month'].unique()

list1
df.month.replace('Janeiro', 'January',inplace=True)

df.month.replace('Fevereiro','February',inplace=True)

df.month.replace('Março','March',inplace=True)

df.month.replace('Abril','April',inplace=True)

df.month.replace('Maio','May',inplace=True)

df.month.replace('Junho','June',inplace=True)

df.month.replace('Julho','July',inplace=True)

df.month.replace('Agosto','August',inplace=True)

df.month.replace('Setembro','September',inplace=True)

df.month.replace('Outubro','October',inplace=True)

df.month.replace('Novembro','November',inplace=True)

df.month.replace('Dezembro','December',inplace=True)

df['month'].unique()
Yearly_Sum=df[['year','number']].groupby('year', as_index=False).sum()

Yearly_Avg=df[['year','number']].groupby('year', as_index=False).mean()


plt.figure(figsize=(14,8))

sns.barplot(Yearly_Avg['year'],Yearly_Avg['number'])

plt.xlabel('Years')

plt.ylabel('Average number of fire incidents')

plt.title('Average Yearly Comparison')



plt.figure(figsize=(14,8))

sns.barplot(Yearly_Sum['year'],Yearly_Sum['number'])

plt.xlabel('Years')

plt.ylabel('Total number of fire incidents')

plt.title('Total Yearly Comparison')
df['month'] = pd.Categorical(df['month'], categories=df.month.unique(), ordered=True)

df.sort_values('month')

Yearly_Sum=df[['year','month','number']].groupby(['year','month'], as_index=False).sum()

Yearly_Avg=df[['year','month','number']].groupby(['year','month'], as_index=False).mean()
plt.figure(figsize=(18,12))

sns.pointplot(Yearly_Avg['month'],Yearly_Avg['number'],hue = Yearly_Avg['year'], dodge=True)

plt.xlabel('Month')

plt.ylabel('Average number of incidents')

plt.title('Average number of incidents VS Month')
plt.figure(figsize=(18,12))

sns.pointplot(Yearly_Sum['month'],Yearly_Sum['number'],hue = Yearly_Sum['year'], dodge=True)

plt.xlabel('Month')

plt.ylabel('Total number of incidents')

plt.title('Total number of incidents VS Month')
df['state'].unique()
plt.figure(figsize=(14,8))

plot = sns.swarmplot(x="state", y="number", data=df)

plt.xlabel('State')

plt.ylabel('Number of incidents')

plt.setp(plot.get_xticklabels(),rotation=70)

plt.title("State vs Number of incidents")

plt.show()
print(df.at[df["number"].idxmax(),'state'])

print(df.at[df["number"].idxmax(),'number'])
Yearly_State=df[['year','number','state']].groupby(['state','year'], as_index=False).sum()
plt.figure(figsize=(18,12))

sns.pointplot(Yearly_State['state'],Yearly_State['number'],hue = Yearly_State['year'], dodge=True)

plt.xticks(rotation=70)

plt.xlabel('State')

plt.ylabel('Year')

plt.title('State VS Month')