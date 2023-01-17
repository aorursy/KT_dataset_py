# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
data=pd.read_csv('/kaggle/input/titanic/train.csv')

data.head()
print('Oldest Passenger was of:',data['Age'].max(),'Years')

print('Youngest Passenger was of:',data['Age'].min(),'Years')

print('Average Age on the ship:',data['Age'].mean(),'Years')
df=pd.read_csv('/kaggle/input/titanic/train.csv')

Cherbourg_df=df.query('Embarked == "C"')

import seaborn as sns

import matplotlib.pyplot as plt

Cherbourg_df["Age"] = Cherbourg_df["Age"].fillna(-0.5)

# test["Age"] = test["Age"].fillna(-0.5)

bins = [-1, 0, 14, 25, 35, 60, np.inf]

labels = ['Unknown', 'Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']

Cherbourg_df['AgeGroup'] = pd.cut(Cherbourg_df["Age"], bins, labels = labels)

# test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)

#draw a bar plot of Age vs. survival

sns.barplot(x="AgeGroup", y="Age", data=Cherbourg_df)

plt.show()

g = sns.FacetGrid(Cherbourg_df, col='Survived')

g = g.map(sns.distplot, "Age")

g = sns.kdeplot(Cherbourg_df["Age"][(Cherbourg_df["Survived"] == 0) & (Cherbourg_df["Age"].notnull())], color="Red", shade = True)

g = sns.kdeplot(Cherbourg_df["Age"][(Cherbourg_df["Survived"] == 1) & (Cherbourg_df["Age"].notnull())], ax =g, color="Blue", shade= True)

g.set_xlabel("Age")

g.set_ylabel("Frequency")

g = g.legend(["Not Survived","Survived"])
