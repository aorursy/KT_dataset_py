import numpy as np

import pandas as pd
df = pd.read_csv('../input/titanicdataset/train.csv')
df.head()
df.describe()
import matplotlib.pyplot as plt

import seaborn as sns
sns.countplot(x='Survived',data=df,hue='Sex')
sns.lineplot(x='Age',y='Fare',data=df)
df.plot.scatter('Age','Fare')
sns.scatterplot(x='Age',y='Survived',data=df)
df['Age'].hist(bins=20)
sizes = df['Survived'].value_counts()

fig1,ax1 = plt.subplots()

ax1.pie(sizes,labels=['Not Survived','Survived'],autopct='%1.1f%%',shadow=True)

plt.show()