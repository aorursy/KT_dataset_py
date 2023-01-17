import pandas as pd
import numpy as np
import numpy as np
import seaborn as sns
df=pd.read_csv('../input/data-visualizationcsv/train.csv')
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