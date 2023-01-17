# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
post = pd.read_csv('/kaggle/input/police-shootings/database.csv') #Importing the data
post.info()
post.head()
f = post['gender'] == 'F'
m = post['gender'] == 'M'
print("Killings Based on Gender")
print({'Female': sum(f), 'Male': sum(m)})
print(post.date.min())
print(post.date.max())
null_counts = post.isnull().sum()
print("The number of nulls in each column is: \n{}".format(null_counts))
post_clean = post.dropna(subset=['race', 'armed', 'age', 'flee'])
post_clean.info()
post_clean.race.value_counts().sum()
post_clean.race.value_counts()
post_clean.race.value_counts(normalize=True) #Proportion Based on Rest of Categories
pro = {'White':[1022/248410000], 'Black':[523/42970000], 'Hispanic':[355/60570000], 'Asian': [32/18280000], 'Native American': [26/4050000]}
prop = pd.DataFrame(data=pro)
prop
gen_data = [['White', 4, 1041], ['Black', 12, 2981], ['Hispanic', 6, 1490], ['Asian', 2, 497], ['Native American', 6, 1490]]
pd.DataFrame(gen_data, columns=["Race", "Number Killed Based Off Rates (For 1 Million People)", "Number Killed Based Off Rates (For 248,410,000 people)"])
rut = {'Race': ['White', 'Black', 'Hispanic', 'Asian', 'Native American'], 'OneMil' : [4,12,6,2,7], 'WhitePop' : [1041,2981,1490,497, 1739]}
rut1 = pd.DataFrame(rut, columns=['Race', 'OneMil', 'WhitePop'])
sns.set(style='whitegrid')
ax = sns.barplot(data=rut, y='Race', x='OneMil', palette="Blues_d", order= rut1.sort_values('WhitePop', ascending=False).Race)
plt.xlabel("Number of Projected Deaths")
plt.title("Killings Based Off Rates Per One Million People Over Two Years")
sns.set(style='whitegrid')
ax = sns.barplot(data=rut1, y='Race', x='WhitePop', palette="Blues_d", order= rut1.sort_values('WhitePop', ascending=False).Race)
plt.xlabel("Number of Projected Deaths")
plt.title("Killings Based Off Rates Per White Population Over Two Years")
post_clean['age'].describe()
x = post_clean.groupby('race')
print(x['age'].mean())
sns.boxplot(x='race', y='age', data=post_clean, palette='Blues_d')
plt.xlabel("Race")
plt.ylabel("Age")
plt.title("Killings Based on Age Separated by Race")
post_clean['unarmedrace'] = post_clean.armed.str.contains('unarmed', na=False)
postunarm = post_clean.groupby('race').unarmedrace.sum()
postunarm
orp = {'White':[60/248410000], 'Black':[56/42970000], 'Hispanic':[29/60570000], 'Asian': [0/18280000], 'Native American': [2/4050000]}
orpp = pd.DataFrame(data=orp)
orpp
tur = {'Race': ['White', 'Black', 'Hispanic', 'Asian', 'Native American'],'WhitePop' : [60, 248, 119, 0, 123]}
tur1 = pd.DataFrame(tur, columns=['Race','WhitePop'])
gen2_data = [['White', 60], ['Black', 248], ['Hispanic', 119], ['Asian', 0], ['Native American', 123]]
pd.DataFrame(gen2_data, columns=["Race", "Unarmed Killed Based Off Rates (For 248,410,000 people)"])
sns.set(style='whitegrid')
ax = sns.barplot(data=tur1, y='Race', x='WhitePop', palette="Blues_d", order= tur1.sort_values('WhitePop', ascending=False).Race)
plt.xlabel("Number of Projected Deaths")
plt.title("Rate of Killing of Unarmed Citizens by Police")
a = post_clean['flee'].describe()
b = post_clean['flee'].unique()
print(a, b)
print("Proportion of People Killed While Not Fleeing:", (1358/1986)*100,"%")
post_clean = post_clean.assign(fleeornot = post_clean['flee'])
post_clean.loc[post_clean['flee']== 'Not fleeing', 'fleeornot'] = 'Not Fleeing'
post_clean.loc[post_clean['flee']== 'Car', 'fleeornot'] = 'Flee'
post_clean.loc[post_clean['flee']== 'Foot', 'fleeornot'] = 'Flee'
post_clean.loc[post_clean['flee']== 'Other', 'fleeornot'] = 'Flee'
post_clean.fleeornot.unique()
post_clean['fleesornot'] = post_clean.fleeornot.str.contains('Not Fleeing', na=False)
postflee = post_clean.groupby('race').fleesornot.sum()
postflee
#Want a stacked bar chart of Not Fleeing vs. Fleeing Right Here
x = [26, 320, 235, 19, 737]
y = [32, 523, 355, 26, 1022]
index = ['Asian', 'Black', 'Hispanic', 'Nat Amer.', 'White']
df = pd.DataFrame({'Not Fleeing': x,
                   'Fleeing': y}, index=index)
ax = df.plot.bar(rot=0, color={"Not Fleeing": "lightskyblue", "Fleeing": "b"})
dfr = {'White':[737/248410000], 'Black':[320/42970000], 'Hispanic':[235/60570000], 'Asian': [26/18280000], 'Native American': [19/4050000]}
dfrr = pd.DataFrame(data=dfr)
dfrr
tur2 = {'Race': ['White', 'Black', 'Hispanic', 'Asian', 'Native American'],'WhitePop' : [737, 1739, 994, 248, 1242]}
tur3 = pd.DataFrame(tur2, columns=['Race','WhitePop'])
gen3_data = [['White', 737], ['Black', 1739], ['Hispanic', 994], ['Asian', 248], ['Native American', 1242]]
pd.DataFrame(gen3_data, columns=["Race", "Unarmed Killed Based Off Rates (For 248,410,000 people)"])
sns.set(style='whitegrid')
ax = sns.barplot(data=tur3, y='Race', x='WhitePop', palette="Blues_d", order= tur3.sort_values('WhitePop', ascending=False).Race)
plt.xlabel("Number of Projected Deaths")
plt.title("Rate of Killing of Non-Fleeing Citizens (Based on White Pop.)")
post_clean.head()
twogether = post_clean[(post_clean['flee'] == 'Not fleeing') & (post_clean['armed'] == 'unarmed')]
unarmflee = twogether.groupby('race').sum()
unarmflee
wbh = {'White':[39/248410000], 'Black':[29/42970000], 'Hispanic':[19/60570000]}
wbhh = pd.DataFrame(data=wbh)
wbhh
tur4 = {'Race': ['White', 'Black', 'Hispanic'],'WhitePop' : [39, 168, 78]}
tur5 = pd.DataFrame(tur4, columns=['Race','WhitePop'])
gen4_data = [['White', 39], ['Black', 168], ['Hispanic', 78]]
pd.DataFrame(gen4_data, columns=["Race", "Unarmed Killed Based Off Rates (For 248,410,000 people)"])
sns.set(style='whitegrid')
ax = sns.barplot(data=tur5, y='Race', x='WhitePop', palette="Blues_d", order= tur5.sort_values('WhitePop', ascending=False).Race)
plt.xlabel("Number of Projected Deaths")
plt.title("Rate of Killing of Unarmed Non-Fleeing Citizens (Based on White Pop.)")