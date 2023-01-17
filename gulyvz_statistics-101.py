import numpy as np

import pandas as pd

import seaborn as sns

import scipy.stats as stats

import matplotlib.pyplot as plt

sns.set()

%matplotlib inline

train = pd.read_csv("../input/titanic/train.csv")

test= pd.read_csv("../input/titanic/test.csv")

train.head()
test.head()
survived = train.loc[train['Survived'] == 1]

death =  train.loc[train['Survived'] == 0]
survived.head()
death.head()
print('Shape of survived:',survived.shape)

print('Type of survived :',type(survived))

print('Shape of death:',death.shape)

print('Type of death :',type(death))
survived.describe()
death.describe()
survived_age=survived['Age']

death_age=death['Age']
plt.figure(figsize=(8,5))

plt.hist(survived_age, bins=20, color='g', label='Age')

plt.title('Survived Passenger Ages')

plt.xlabel('Age') ; plt.legend(loc='upper right')

plt.show()
plt.figure(figsize=(8,5))

plt.hist(death_age, bins=20, color='b', label='Age')

plt.title('Death Passenger Ages')

plt.xlabel('Age') ; plt.legend(loc='upper right')

plt.show()
plt.figure(figsize=(8,5))

plt.hist(survived_age, bins=20, density=True, color='g', label='Age') # remove density=True, it becomes freq dist

plt.title('Survived Passenger Ages')

plt.xlabel('Age') ; plt.ylabel('Density') ; plt.legend(loc='upper right')

plt.show()
plt.figure(figsize=(8,5))

plt.hist(death_age, bins=20, density=True, color='b', label='Age') # remove density=True, it becomes freq dist

plt.title('Death Passenger Ages')

plt.xlabel('Age') ; plt.ylabel('Density') ; plt.legend(loc='upper right')

plt.show()
plt.figure(figsize=(8,5))

plt.hist(survived_age, bins=20, density=True, color='g',cumulative=1, label='Age') # remove density=True, it becomes freq dist

plt.title('Survived Passenger Ages')

plt.xlabel('Age') ; plt.ylabel('Density') ; plt.legend(loc='upper right')

plt.show()
plt.figure(figsize=(8,5))

plt.hist(death_age, bins=20, density=True, color='b',cumulative=1, label='Age') 

plt.title('Death Passenger Ages')

plt.xlabel('Age') ; plt.ylabel('Density') ; plt.legend(loc='upper right')

plt.show()
survived_class=survived['Pclass']

death_class=death['Pclass']
plt.figure(figsize=(8,5))

plt.hist(survived_class, bins=20, color='g', label='Pclass')

plt.title('Survived Passenger Pclass')

plt.xlabel('Pclass') ; plt.legend(loc='upper right')

plt.show()
plt.figure(figsize=(8,5))

plt.hist(death_class, bins=20, color='g', label='Pclass')

plt.title('Death Passenger Pclass')

plt.xlabel('Pclass') ; plt.legend(loc='upper right')

plt.show()
print('Mean    :', survived_age.mean())

print('Median  :', survived_age.median())

print('Std dev :', survived_age.std() )
print('Mean    :', death_age.mean())

print('Median  :', death_age.median())

print('Std dev :', death_age.std() )
plt.figure(figsize=(8, 2))

sns.boxplot(survived_age, showmeans=True, orient='h', color='g');
plt.figure(figsize=(8, 2))

sns.boxplot(death_age, showmeans=True, orient='h',color='b');
stats.probplot(survived_age, dist="norm", plot=plt)

plt.title("Normal Q-Q plot")

plt.show()
stats.probplot(death_age, dist="norm", plot=plt)

plt.title("Normal Q-Q plot")

plt.show()
plt.axes([0.1,0.01,0.8,0.8])

plt.hist(survived_age, bins=50, density=True, facecolor="green", alpha=0.7)



plt.axes([1.05,0.01,0.8,0.8])

stats.probplot(survived_age, dist="norm", plot=plt)

plt.title("Normal Q-Q plot")

plt.show()
plt.axes([0.1,0.01,0.8,0.8])

plt.hist(death_age, bins=50, density=True, facecolor="blue", alpha=0.7)



plt.axes([1.05,0.01,0.8,0.8])

stats.probplot(death_age, dist="norm", plot=plt)

plt.title("Normal Q-Q plot")

plt.show()