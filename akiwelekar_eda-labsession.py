import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

 

df=pd.read_csv('../input/titanicdataset-traincsv/train.csv')

df.head()

print(df.shape)
df.columns
print(df.info())
df.describe()
df.describe(include='object')
df['Sex'].value_counts()
df['Embarked'].value_counts()
df['Sex'].value_counts(normalize=True)
df['Name'].head()
df.loc[0:15,'Name':'Age']
df.iloc[0:15,3:6]
df[df['Age']== df[df['Sex'] =='male']['Age'].max()]['Name']
df.sort_values(by='Name').head()
df.sort_values(by=['Name','Age'],ascending=[True,False]).head()
d={'male':0, 'female':1}

df['Sex'] = df['Sex'].map(d)

df.head()
o={0: 'male', 1: 'female'}

df= df.replace({'Sex':o})

df.head()

df.groupby(by='Survived')['Age'].describe()
df.groupby(by='Embarked')['Age'].agg(np.mean)
pd.crosstab(df['Sex'],df['Survived'], normalize=True)
df.pivot_table(['Age', 'Fare'],['Survived'], aggfunc='mean')
sns.lineplot(x='Age',y='Fare',data=df)
sns.factorplot(x='Survived', data=df, kind='count')
sns.factorplot(x='Survived', data=df, kind='count', hue='Sex')
df['Age'].hist(bins=8)
as_fig = sns.FacetGrid(df,hue='Sex',aspect=5)

as_fig.map(sns.kdeplot,'Age',shade=True)

oldest = df['Age'].max()

as_fig.set(xlim=(0,oldest))

as_fig.add_legend()
df.plot.scatter(x='Age',y='Fare')
df.plot.scatter(x='Age',y='Survived')
import matplotlib.pyplot as plt

sizes= df['Survived'].value_counts()

fig1,ax1 = plt.subplots()

ax1.pie(sizes,labels=['Not Survived', 'Survived'],autopct='%1.1f%%',shadow=True)

plt.show()