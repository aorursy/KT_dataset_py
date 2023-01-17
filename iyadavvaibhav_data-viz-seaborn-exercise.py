import seaborn as sns
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
titanic = pd.read_csv('../input/titanic.csv')
titanic.head()
titanic.count()
titanic.hist('fare')
sns.jointplot(x='fare',y='age',data=titanic)
sns.distplot(titanic['fare'],bins=30,kde=False,color='red')
sns.boxplot(x='class',y='age',data=titanic)
sns.swarmplot(x='class',y='age',data=titanic)
sns.violinplot(x='class',y='age',data=titanic)
sns.countplot(x='sex',data=titanic)
titanic.corr()
sns.heatmap(titanic.corr(),cmap='coolwarm')
plt.title('titanic.corr()')
g = sns.FacetGrid(data=titanic,col='sex')
g,map(plt.hist,'age')
