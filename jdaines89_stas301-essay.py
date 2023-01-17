import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_excel("../input/london-homicide-data-20082018/mmap.xlsx")
print(df.info())
plt.figure(figsize=(15,10))
plt.title("Homicides in London for the period 2008-2018", fontsize=25, y=1.02)
df.groupby(df.date.dt.year).size().plot(color='red', marker='o')
plt.ylabel("Number of Homicides",fontsize=20, labelpad=25)
plt.xlabel("Year", fontsize=20, labelpad=25)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig('Homicides.png')
plt.show()

df.Status.value_counts() * 100 / df.Status.value_counts().sum()
plt.figure(figsize=(10,4))
(df.Status.value_counts() * 100 / df.Status.value_counts().sum()).plot(kind='bar')
plt.title("% Solved")
df.groupby(df.date.dt.year).size()
s2= df.groupby([df.date.dt.year, df.date.dt.month]).size()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(s2)
plt.figure(figsize=(10,5))
y = df.groupby(df.date.dt.month).size()
y.plot(kind='bar', color = 'r')
plt.ylim(top=150)
plt.xlabel("Month", fontsize=14, labelpad=15)
plt.ylabel("Number of Homicides", fontsize=14, labelpad=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("Total Number of Homocides by Month in London (2008-2018)", y=1.01, fontsize=15)  


xlocs=[i+1 for i in range(-1,11)]
for i, v in enumerate(y):
    plt.text(xlocs[i] - 0.2, v + 1.5, str(v))
    

plt.savefig("Homicide_Monthly")
plt.show()
pd.crosstab(df.Status, df.vicsex, margins=True)
month = df.date.dt.month
df.groupby([month]).size().plot(kind='bar')
plt.title("Number of homocides by month")
df.groupby(df.date.dt.month).size()/10
df.groupby([df.date.dt.year]).catdom.value_counts()
df[df.vicsex=='M'].weapon.value_counts() * 100 / df[df.vicsex=='M'].weapon.value_counts().sum()
df[df.vicsex=='F'].weapon.value_counts() * 100 / df[df.vicsex=='F'].weapon.value_counts().sum()
df.weapon.value_counts() * 100 / df.weapon.value_counts().sum()
f, axes = plt.subplots(1,2, figsize=(12,5))

ax1 = (df[df.vicsex=='M'].weapon.value_counts() * 100 / df[df.vicsex=='M'].weapon.value_counts().sum()).plot(kind='bar', ax=axes[0], color = 'r')
ax1.set_ylabel("% of Total", labelpad=15)
ax1.set_xlabel("Weapon Used in Homicide", labelpad=15)
ax1.set_title("Male Victims By Weapon")


ax2 = (df[df.vicsex=='F'].weapon.value_counts() * 100 / df[df.vicsex=='F'].weapon.value_counts().sum()).plot(kind='bar', ax=axes[1], color = 'r')
ax2.set_ylabel("% of Total", labelpad=15)
ax2.set_xlabel("Weapon Used in Homicide", labelpad=15)
ax2.set_title("Female Victims by Weapon")
plt.tight_layout()
plt.savefig("VictimWeapon")
plt.show()
pd.crosstab(df[df.vicsex=='M'].weapon, df.Status)
pd.crosstab(df[df.vicsex=='F'].weapon, df.Status)
pd.crosstab(df.weapon, df.Status, normalize ='index')
pd.crosstab(df.weapon, df.Status, normalize ='columns')
dom = df['catdom']==1
df_dom = df[dom]

order = ['A. Child 0-6', 'B. Child 7-12', 'C. Teen 13-16','D. Teen 17-19','E. Adult 20-24','F. Adult 25-34','G. Adult 35-44', 'H. Adult 45-54','I. Adult 55-64','J. Adult 65 over']
df_dom.vicagegp.value_counts()

df_weekday = df.copy().groupby(df_dom.vicagegp).size().reindex(order)
df_weekday
df_weekday.plot(kind = 'bar')


plt.figure(figsize=(10,4))
(df.vicethnic.value_counts() * 100 / df.vicethnic.value_counts().sum()).plot(kind='bar')
plt.title("Ethnicity")
plt.figure(figsize=(10,4))
(df.vicsex.value_counts() * 100 / df.vicsex.value_counts().sum()).plot(kind='bar')
plt.title("Sex")
df.vicsex.value_counts() * 100 / df.vicsex.value_counts().sum()