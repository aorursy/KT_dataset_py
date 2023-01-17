import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
%matplotlib inline

df = pd.read_csv('../input/heart-disease-uci/heart.csv')
df.head()
df.tail()
df.shape
df.columns
df.info()
df.describe()
df.isnull().sum()
plt.figure(figsize=(18,10))
sns.heatmap(df.corr(), annot = True, cmap='cool')
plt.show()
print(len(df.sex))
df.sex.value_counts()
sns.countplot(df.sex)
plt.show()
male = len(df[df['sex'] == 1])
female = len(df[df['sex'] == 0])
plt.figure(figsize=(7, 6))

labels = 'Male', 'Female'
sizes = [male, female]
colors = ['orange', 'gold']
explode = (0, 0)

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=90)

plt.axis('equal')
plt.show()
male_1 = len(df[(df.sex == 1) & (df['target'] == 1)])
male_0 = len(df[(df.sex == 1) & (df['target'] == 0)])
sns.barplot(x = ['Male Target On', 'Male Target Off'], y = [male_1, male_0])
plt.xlabel('Male and Target')
plt.ylabel('Count')
plt.title('State of the Male gender')
plt.show()
female_1 = len(df[(df.sex == 0) & (df['target'] == 1)])
female_0 = len(df[(df.sex == 0) & (df['target'] == 0)])

sns.barplot(x = ['Female Target On', 'Female Target Off'], y = [female_1, female_0])
plt.xlabel('Female and Target')
plt.ylabel('Count')
plt.title('State of the Female gender')
plt.show()
sns.countplot(df.sex, hue = df.target)
plt.title('Male & Femele Heart health condition')
plt.show()
male = ((len(df[(df.sex == 1) & (df['target'] == 1)])) / len(df[df['sex'] == 1])) * 100
female = ((len(df[(df.sex == 0) & (df['target'] == 1)])) / len(df[df['sex'] == 0])) * 100
plt.figure(figsize=(8, 6))

labels = 'Male', 'Female'
sizes = [male, female]
colors = ['orange', 'yellow']
explode = (0, 0)

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.2f%%', shadow=True, startangle=90)

plt.axis('equal')
plt.show()
df['count'] = 1
df.groupby('age').count()['count']
sns.barplot(x=df.age.value_counts()[:10].index,y=df.age.value_counts()[:10].values)
plt.xlabel('Age')
plt.ylabel('Age counter')
plt.title('Age Analysis')
plt.show()
print('Min age:', min(df.age))
print('Max age:', max(df.age))
print('Mean age: ', df.age.mean())
print('Total adult people: ', len(df[(df.age >= 29) & (df.age <= 59)]))
print('Total senior adult people: ', len(df[(df.age > 59)]))
adult0 = ((len(df[(df.age >= 29) & (df.age <= 59)])) / len(df['age'])) * 100
senior0 = ((len(df[(df.age > 59)])) / len(df['age'])) * 100
plt.figure(figsize=(8, 6))

labels = 'Adult(19-59)', 'Senior Adult(60 or above)'
sizes = [adult0, senior0]
colors = ['orange', 'yellow']
explode = (0, 0)

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.2f%%', shadow=True, startangle=90)

plt.axis('equal')
plt.show()
adult1 = ((len(df[(df.age >= 29) & (df.age <= 59) & (df['target'] == 1)])) / len(df['age'])) * 100
senior1 = ((len(df[(df.age > 59) & (df['target'] == 1)])) / len(df['age'])) * 100
plt.figure(figsize=(8, 6))

labels = 'Adult(29-59)', 'Senior Adult(60 or above)'
sizes = [adult1, senior1]
colors = ['orange', 'yellow']
explode = (0, 0)

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.2f%%', shadow=True, startangle=90)

plt.axis('equal')
plt.show()
adult2 = ((len(df[(df.sex == 1) & (df.age >= 29) & (df.age <= 59) & (df['target'] == 1)])) / len(df['age'])) * 100
senior2 = ((len(df[(df.sex == 1) & (df.age > 59) & (df['target'] == 1)])) / len(df['age'])) * 100
plt.figure(figsize=(8, 6))

labels = 'Adult Male(29-59)', 'Senior Adult Male(60 or above)'
sizes = [adult2, senior2]
colors = ['orange', 'yellow']
explode = (0, 0)

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.2f%%', shadow=True, startangle=90)

plt.axis('equal')
plt.show()
adult3 = ((len(df[(df.sex == 0) & (df.age >= 29) & (df.age <= 59) & (df['target'] == 1)])) / len(df['age'])) * 100
senior3 = ((len(df[(df.sex == 0) & (df.age > 59) & (df['target'] == 1)])) / len(df['age'])) * 100
plt.figure(figsize=(8, 6))

labels = 'Adult Feale(29-59)', 'Senior Adult Female(60 or above)'
sizes = [adult3, senior3]
colors = ['orange', 'yellow']
explode = (0, 0)

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.2f%%', shadow=True, startangle=90)

plt.axis('equal')
plt.show()
df['fbs']
print('Total people with diabetes: ', len(df[(df.fbs == 1)]))
print('Total people without diabetes: ', len(df[(df.fbs == 0)]))
with_diabetes = (len(df[(df.fbs == 1)]) / len(df['fbs'])) * 100
without_diabetes =  (len(df[(df.fbs == 0)]) / len(df['fbs'])) * 100
plt.figure(figsize=(8, 6))

labels = 'Diabetes', 'No Diabetes'
sizes = [with_diabetes, without_diabetes]
colors = ['orange', 'yellow']
explode = (0, 0)

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.2f%%', shadow=True, startangle=90)

plt.axis('equal')
plt.show()
with_diabetes_on1 = (len(df[(df.fbs == 1) & (df['target'] == 1)]) / len(df['fbs'])) * 100
with_diabetes_off1 = (len(df[(df.fbs == 1) & (df['target'] == 0)]) / len(df['fbs'])) * 100
plt.figure(figsize=(8, 6))

labels = 'Have diabetes and heart problem', 'Have diabetes but heart problem'
sizes = [with_diabetes_on1, with_diabetes_off1]
colors = ['orange', 'yellow']
explode = (0, 0)

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.2f%%', shadow=True, startangle=90)

plt.axis('equal')
plt.show()
adult4 = ((len(df[(df.sex == 1) & (df.fbs == 1) & (df.age >= 29) & (df.age <= 59) & (df['target'] == 1)])) / len(df['age'])) * 100
senior4 = ((len(df[(df.sex == 1)  & (df.fbs == 1) & (df.age > 59) & (df['target'] == 1)])) / len(df['age'])) * 100
plt.figure(figsize=(8, 6))

labels = 'Adult Male(29-59) have both diabetes and heart problem', 'Senior Adult Male(60 or above) have both diabetes and heart problem'
sizes = [adult4, senior4]
colors = ['orange', 'yellow']
explode = (0, 0)

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.2f%%', shadow=True, startangle=90)

plt.axis('equal')
plt.show()
adult5 = ((len(df[(df.sex == 0) & (df.fbs == 1) & (df.age >= 29) & (df.age <= 59) & (df['target'] == 1)])) / len(df['age'])) * 100
senior5 = ((len(df[(df.sex == 0)  & (df.fbs == 1) & (df.age > 59) & (df['target'] == 1)])) / len(df['age'])) * 100
plt.figure(figsize=(8, 6))

labels = 'Adult female(29-59) have both diabetes and heart problem', 'Senior adult female(60 or above) have both diabetes and heart problem'
sizes = [adult5, senior5]
colors = ['orange', 'yellow']
explode = (0, 0)

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.2f%%', shadow=True, startangle=90)

plt.axis('equal')
plt.show()
adult6 = ((len(df[(df.sex == 1) & (df.fbs == 1) & (df.age >= 29) & (df.age <= 59) & (df['target'] == 1)])) / len(df['age'])) * 100
senior6 = ((len(df[(df.sex == 1)  & (df.fbs == 1) & (df.age > 59) & (df['target'] == 1)])) / len(df['age'])) * 100
plt.figure(figsize=(8, 6))

labels = 'Adult male(29-59) have both diabetes and heart problem', 'Senior adult male(60 or above) have both diabetes and heart problem'
sizes = [adult6, senior6]
colors = ['orange', 'yellow']
explode = (0, 0)

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.2f%%', shadow=True, startangle=90)

plt.axis('equal')
plt.show()