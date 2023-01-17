# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import seaborn as sns
train = pd.read_csv("../input/train.csv")



# Displaying sample data

train.tail()
train.describe()
train.describe(include=['O'])
sns.countplot(train['Sex'])

sns.countplot(train['Embarked'])
sns.countplot(train['Pclass'])

sns.countplot(train['Survived'])
import matplotlib.pyplot as plt
# Plotting visualization for Pclass and Survival Status



plt.hist([train[train['Survived'] == 1]['Pclass'].dropna(), train[train['Survived'] == 0]['Pclass'].dropna()], 

         histtype='bar', 

         stacked=False,

         label=['Survived', 'Not Survived'])

plt.title('Pclass & Survival Status')

plt.xlabel('Pclass')

plt.ylabel('No. of Passengers')

plt.legend()

plt.show()
# Plotting visualization for Embarked and Survival Status

train = train.dropna()

train['Embarked'] = train['Embarked'].map({'S': 1, 'C': 2,'Q': 3}).astype(int)

plt.hist([train[train['Survived'] == 1]['Embarked'], train[train['Survived'] == 0]['Embarked']], 

         histtype='bar', 

         stacked=False,

         label=['Survived', 'Not Survived'])

plt.title('Embarked & Survival Status')

plt.xlabel('Embarked')

plt.ylabel('No. of Passengers')

plt.legend()

plt.show()
# Plotting visualization for Embarked and Survival Status

train = train.dropna()

train['Sex'] = train['Sex'].map({'male': 1, 'female': 2}).astype(int)

plt.hist([train[train['Survived'] == 1]['Sex'], train[train['Survived'] == 0]['Sex']], 

         histtype='bar', 

         stacked=False,

         label=['Survived', 'Not Survived'])

plt.title('Sex & Survival Status')

plt.xlabel('Sex')

plt.ylabel('No. of Passengers')

plt.legend()

plt.show()