import pandas as pd

import numpy as np



df = pd.read_csv('../input/machine-learning-on-titanic-data-set/train.csv')
df.head()
df['Fare'].mean()
df.loc[0:15,'Name':'Embarked']
df.iloc[0:15,0:4]
df[df['Age']==df['Age'].max()]
df[df['Age']==df['Age'].max()]['Name']
df.sort_values(by='Age',ascending=True).head()
df.sort_values(by=['Name','Age'], ascending=[True,False]).head()
new={'male':0,'female':1}

df['Sex']=df['Sex'].map(new)

df.head()
old={0:'male',1:'female'}

df=df.replace({'Sex':old})

df.head()
df.groupby(by='Survived')['Age'].describe()
pd.crosstab(df['Sex'],df['Survived'], normalize=True)