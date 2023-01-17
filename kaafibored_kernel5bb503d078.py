# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/titanic_data.csv")
data
data.describe()
mean_age = data["Age"].mean()

std_age = data["Age"].std()

#len(data[data["Age"].isnull()])

ep = np.random.uniform(-1,1,(1,177))

newcol = (mean_age + std_age*ep).astype('int8')

data['Age'][data["Age"].isnull()] = newcol
data.describe()
print(data['Survived'].value_counts())

print(data['Pclass'].value_counts())

print(data['Sex'].value_counts())

print(data['SibSp'].value_counts())

print(data['Parch'].value_counts())

print(data['Embarked'].value_counts())

import seaborn as sns

import matplotlib.pyplot as plt
plt.hist(data["Age"],bins = 20,alpha = 0.8)

plt.show()
plt.hist(data["Fare"],bins = 20,alpha = 0.8)

plt.show()
sns.distplot(data['Age'],bins = 50)

plt.show()
sns.kdeplot(data['Fare'])

plt.show()
fig, ax = plt.subplots(6, 1, sharey=True, figsize=(10,40))

sns.swarmplot(y = data['Age'],x = data['Survived'], ax = ax[0])

sns.swarmplot(y = data['Age'],x = data['Pclass'], ax = ax[1])

sns.swarmplot(y = data['Age'],x = data['Sex'], ax = ax[2])

sns.swarmplot(y = data['Age'],x = data['SibSp'], ax = ax[3])

sns.swarmplot(y = data['Age'],x = data['Parch'], ax = ax[4])

sns.swarmplot(y = data['Age'],x = data['Embarked'], ax = ax[5])

plt.show()
fig, ax = plt.subplots(6, 1, sharey=True, figsize=(10, 40))

sns.swarmplot(y = data['Fare'],x = data['Survived'], ax = ax[0])

sns.swarmplot(y = data['Fare'],x = data['Pclass'], ax = ax[1])

sns.swarmplot(y = data['Fare'],x = data['Sex'], ax = ax[2])

sns.swarmplot(y = data['Fare'],x = data['SibSp'], ax = ax[3])

sns.swarmplot(y = data['Fare'],x = data['Parch'], ax = ax[4])

sns.swarmplot(y = data['Fare'],x = data['Embarked'], ax = ax[5])

plt.show()
plt.scatter(data['Age'],data['Fare'])

plt.show()
sns.scatterplot(data['Age'],data['Fare'], alpha = 0.5)
fig, ax = plt.subplots(6, 1, sharey=True, figsize=(10, 30))

sns.scatterplot(data['Age'],data['Fare'], alpha = 0.8, hue = data['Survived'], ax = ax[0])

sns.scatterplot(data['Age'],data['Fare'], alpha = 0.8, hue = data['Pclass'], ax = ax[1])

sns.scatterplot(data['Age'],data['Fare'], alpha = 0.8, hue = data['Sex'], ax = ax[2])

sns.scatterplot(data['Age'],data['Fare'], alpha = 0.8, hue = data['SibSp'], ax = ax[3])

sns.scatterplot(data['Age'],data['Fare'], alpha = 0.8, hue = data['Parch'], ax = ax[4])

sns.scatterplot(data['Age'],data['Fare'], alpha = 0.8, hue = data['Embarked'], ax = ax[5])