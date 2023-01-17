import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set_style('whitegrid')
# titanic = sns.load_dataset('titanic')
# sns.load_dataset('titanic')

titanic = pd.read_csv('../input/train.csv')
titanic.head()
sns.jointplot(x='Fare',y='Age',data=titanic)
sns.distplot(titanic["Fare"],kde=False) #without the kde
sns.boxplot(x='Pclass',y='Age',data=titanic)
sns.swarmplot(x='Pclass',y='Age',data=titanic)
sns.countplot(x='Sex',data=titanic)
# for heat maps, indexing / correaltions needs to be established
tc = titanic.corr()
sns.heatmap(tc,cmap='coolwarm')
plt.title('titanic.corr()')
g = sns.FacetGrid(data=titanic, col='Sex')
g.map(sns.distplot, 'Age',kde=False)
