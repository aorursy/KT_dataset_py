import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('../input/quering-data-framecsv/train.csv')
df.head()
df['Fare'].mean()
df.loc[0:15,'Name']
df.loc[0:15,'Name':'Embarked']
df.iloc[0:15,0:4]
df[df['Age']==df['Age'].max]['Name']
df.sort_values(by='Age',ascending=True)
df.sort_values(by='Age',ascending=True).head()
df.sort_values(by=['Name', 'Age'],ascending=[True, False]).head()
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