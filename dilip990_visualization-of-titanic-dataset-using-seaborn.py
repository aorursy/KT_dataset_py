import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
sns.set_style('whitegrid')
titanic = pd.read_csv('../input/train.csv')
titanic.head()
sns.jointplot(x='Fare',y='Age',data=titanic)
sns.distplot(titanic['Fare'],bins=30,kde=False,color='red')
sns.boxplot(x='Pclass',y='Age',data=titanic,palette='rainbow')
sns.swarmplot(x='Pclass',y='Age',data=titanic,palette='Set2')
sns.countplot(x='Sex',data=titanic)
sns.heatmap(titanic.corr(),cmap='coolwarm')
plt.title('titanic.corr()')
g = sns.FacetGrid(data=titanic,col='Sex')
g.map(plt.hist,'Age')