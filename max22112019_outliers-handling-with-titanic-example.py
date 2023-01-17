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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
train=pd.read_csv('../input/titanic/train.csv')

test=pd.read_csv('../input/titanic/test.csv')
df=train.copy()
sns.distplot(df['Age'].dropna())
figure=df.Age.hist(bins=50)

figure.set_title('Age')

figure.set_xlabel('Age')

figure.set_ylabel('No. of Passengers')
figure=df.boxplot(column="Age", figsize=(20,20))
df['Age'].describe()
uppper_boundary=df['Age'].mean() + 1.5* df['Age'].std()

lower_boundary=df['Age'].mean() - 1.5* df['Age'].std()

print(lower_boundary), print(uppper_boundary),print(df['Age'].mean())
##### Assuming Age follows a Gaussian Distribution, we will calculate the boundaries which differentiates the outliers



uppper_boundary=df['Age'].mean() + 3* df['Age'].std()

lower_boundary=df['Age'].mean() - 3* df['Age'].std()

print(lower_boundary), print(uppper_boundary),print(df['Age'].mean())
df.loc[df['Age']>=73,'Age']=73

figure=df.boxplot(column="Age",figsize=(20,20))
figure=df.Age.hist(bins=50)

figure.set_title('Age')

figure.set_xlabel('Age')

figure.set_ylabel('No. of Passengers')
sns.distplot(df['Fare'].dropna())
figure=df.Fare.hist(bins=50)

figure.set_title('Fare')

figure.set_xlabel('Fare')

figure.set_ylabel('No. of Passengers')
df.boxplot(column="Fare",figsize=(10,10))
df['Fare'].describe()
IQR=df.Fare.quantile(0.75)-df.Fare.quantile(0.25)

lower_bridge=df['Fare'].quantile(0.25)-(IQR*1.5)

upper_bridge=df['Fare'].quantile(0.75)+(IQR*1.5)

print(lower_bridge), print(upper_bridge)
lower_bridge=df['Fare'].quantile(0.25)-(IQR*3)

upper_bridge=df['Fare'].quantile(0.75)+(IQR*3)

print(lower_bridge), print(upper_bridge)
df.loc[df['Fare']>=66,'Fare']=66
df.boxplot(column="Fare",figsize=(10,10))
figure=df.Fare.hist(bins=50)

figure.set_title('Fare')

figure.set_xlabel('Fare')

figure.set_ylabel('No. of Passengers')