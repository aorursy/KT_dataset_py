import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns





train = pd.read_csv('../input/titanic/train.csv')
plt.hist(train.loc[train['Survived'] == 0, 'Fare'].dropna(),

         range=(0, 250), bins=25, alpha=0.5, label='0')

plt.hist(train.loc[train['Survived'] == 1, 'Fare'].dropna(),

         range=(0, 250), bins=25, alpha=0.5, label='1')

plt.xlabel('Fare')

plt.ylabel('count')

plt.legend(title='Survived')

plt.xlim(-5, 250)

plt.show()
plt.hist(train.loc[train['Survived'] == 0, 'Fare'].dropna(),

         range=(0, 250), bins=25, alpha=0.5, label='0')

plt.hist(train.loc[train['Survived'] == 1, 'Fare'].dropna(),

         range=(0, 250), bins=25, alpha=0.5, label='1')

plt.xlabel('Fare')

plt.ylabel('count')

plt.legend(title='Survived')

plt.xlim(0, 250)

plt.show()