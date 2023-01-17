import numpy as np
import pandas as pd
df = pd.read_csv('../input/titanicdataset/train.csv')
df.head()
df.shape
df.columns
df.info()
df.describe()
df.describe(include='object')
df['Embarked'].value_counts()
df['Sex'].value_counts(normalize='True')
df['Fare'].head()
df['Fare'].mean()
df.loc[5:15,'Name':'Embarked']
df['Age'].head()
df.iloc[0:15,0:4]
df[df['Age']==df['Age'].max()]['Name']
df.iloc[0]
df.sort_values(by=['Age','Name'],ascending=True).head()
new={'male':0,'female':1}
df['Sex']=df['Sex'].map(new)
df.head()
old={0:'male',1:'female'}
df['Sex']=df['Sex'].map(old)
df.head()
df.groupby(by='Survived')['Age'].describe()
pd.crosstab(df['Sex'],df['Survived'],normalize=True)