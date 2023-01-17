# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('/kaggle/input/titanic/train.csv',usecols=['Age','Fare','Survived'])
data.head()
plt.figure(figsize=(10,7))

sns.distplot(data['Age'])

plt.show()
# with the help of histogram

figure = data['Age'].hist(bins=50)

figure.set_title('dist of Age')

figure.set_xlabel('Age')

figure.set_ylabel('Age')

plt.show()
# with the help of boxplot

figure = data.boxplot(column='Age')

plt.show()
data[data['Age']>70]
df = data.copy()



IQR = df['Age'].quantile(0.75) - df['Age'].quantile(0.25)



lower = df['Age'].quantile(0.25) - 1.5* IQR

upper = df['Age'].quantile(0.75) + 1.5* IQR



outliers = np.where(df['Age']>upper,True, np.where(df['Age']<lower,True,False))



df = df.loc[~(outliers)]
# or you can write in this way too..

df = df[~((df['Age']<lower) & (df['Age']>upper))]

#df
## assuming Age is normally distributed

# calculate the lower and upper boundary using IQR

upper_boundary = data['Age'].mean() + 3*data['Age'].std()

lower_boundary = data['Age'].mean() - 3*data['Age'].std()

print(upper_boundary)

print(lower_boundary)
# here age cannot be negative so it's simple that we will impute the outliers with the upper boundary.

df = data.copy()

df['Age'] = np.where(df['Age']>upper_boundary, upper_boundary,data['Age'])
figure = df['Age'].hist(bins=50)

figure.set_title('dist of Age')

figure.set_xlabel('Age')

figure.set_ylabel('Age')

plt.show()
# thsi is also similar as above, here we use 1.5

IQR = data['Age'].quantile(0.75) - data['Age'].quantile(0.25)

IQR
lower = data['Age'].quantile(0.25) - IQR * 1.5

upper = data['Age'].quantile(0.75) + IQR * 1.5
df = data.copy()

df['Age'] = np.where(df['Age']>upper,upper, np.where(df['Age']<lower,lower,data['Age']))

figure = df['Age'].hist(bins=50)

figure.set_title('dist of Age')

figure.set_xlabel('Age')

figure.set_ylabel('Age')

plt.show()
sns.distplot(df['Age'])
# for Extreme Outliers use this boundaries.

lower_bridge = data['Age'].quantile(0.25) -  3 * IQR

upper_bridge = data['Age'].quantile(0.75) +  3 * IQR

print(lower_bridge)

print(upper_bridge)
lower = 1

upper = 71

df = data.copy()

df['Age'] = np.where(df['Age']>upper,upper, np.where(df['Age']<lower,lower,data['Age'])) 
df = data.copy()

lower = df['Age'].quantile(0.10)

upper = df['Age'].quantile(0.90)

print(lower)

print(upper)