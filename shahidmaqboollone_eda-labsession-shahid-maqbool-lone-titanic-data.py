import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
data=pd.read_csv('../input/titanicdataset-traincsv/train.csv')
data.head()

print(data.shape)
data.columns
print(data.info())
data.describe()
data.describe(include='object')
data['Sex'].value_counts()
data['Embarked'].value_counts()
data['Sex'].value_counts(normalize=True)
data['Name'].head()
data.loc[0:15,'Name':'Age']
data.iloc[0:15,3:6]
data[data['Age']== data[data['Sex'] =='male']['Age'].max()]['Name']
data.sort_values(by='Name').head()
data.sort_values(by=['Name','Age'],ascending=[True,False]).head()
d={'male':0, 'female':1}
data['Sex'] = data['Sex'].map(d)
data.head()
o={0: 'male', 1: 'female'}
data= data.replace({'Sex':o})
data.head()

data.groupby(by='Survived')['Age'].describe()
data.groupby(by='Embarked')['Age'].agg(np.mean)
pd.crosstab(data['Sex'],data['Survived'], normalize=True)
data.pivot_table(['Age', 'Fare'],['Survived'], aggfunc='mean')
sns.lineplot(x='Age',y='Fare',data=data)
sns.factorplot(x='Survived', data=data, kind='count')
sns.factorplot(x='Survived', data=data, kind='count', hue='Sex')
data['Age'].hist(bins=8)
as_fig = sns.FacetGrid(data,hue='Sex',aspect=5)
as_fig.map(sns.kdeplot,'Age',shade=True)
oldest = data['Age'].max()
as_fig.set(xlim=(0,oldest))
as_fig.add_legend()
data.plot.scatter(x='Age',y='Fare')
data.plot.scatter(x='Age',y='Survived')
import matplotlib.pyplot as plt
sizes= data['Survived'].value_counts()
fig1,ax1 = plt.subplots()
ax1.pie(sizes,labels=['Not Survived', 'Survived'],autopct='%1.1f%%',shadow=True)
plt.show()