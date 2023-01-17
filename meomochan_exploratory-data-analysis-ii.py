import numpy as np

import pandas as pd



df = pd.read_csv('/kaggle/input/traincsv/train-200907-141856.csv')
df['Name']
df['Cabin'].value_counts()
df['Age'].mean()
df.loc[ 0:10,'Name':'Embarked']
df.iloc[1:10,1:4]
df[df['Age']==df['Age'].max()]
df[df['Age']==df['Age'].max()]['Name']
df.sort_values(by='Age',ascending='True').head()
df.sort_values(by=['Name','Age'], ascending=[ True, False ] ).head()
import numpy as np

import pandas as pd



df = pd.read_csv('/kaggle/input/traincsv/train-200907-141856.csv')



new={'male': 0,'female': 1}

df['Sex'] = df['Sex'].map(new)

df.head()
o={0:'male',1:'female'}

df= df.replace({'Sex':o})

df.head()
df.groupby(by='Survived')['Age'].describe()
pd.crosstab(df['Sex'],df['Survived'], normalize= True)
df.pivot_table(['Age','Fare'],['Survived'] , aggfunc='mean')