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
df = pd.read_csv("../input/titanicdata/titanic_data.csv")
df.head()
df.shape
df.info
df.describe
X = df.iloc[:,:1].values

y = df.iloc[:, 1].values
import missingno as msno

m = msno.bar(df)
p = sns.barplot(x=df['Sex'], y=df['Survived'])
p = sns.scatterplot(x=df['Age'], y=df['Fare'], hue=df['Survived'])
p = sns.scatterplot(x=df['Age'], y=df['Sex'], hue=df['Survived'])
p = sns.scatterplot(x=df['Pclass'], y=df['Fare'], hue=df['Survived'])
p = sns.swarmplot(x=df['Sex'], y=df['Fare'], hue=df['Survived'])
p = sns.lineplot(x =df['Pclass'], y =df['Fare'], data = df)
p = sns.lineplot(x =df['Age'], y =df['Survived'], data = df)
p = sns.lineplot(x =df['Pclass'], y =df['Survived'], data = df)
p = sns.lineplot(x =df['Sex'], y =df['Survived'], data = df)
p = sns.lineplot(x =df['Parch'], y =df['Survived'], data = df)
p = sns.lineplot(x =df['SibSp'], y =df['Survived'], data = df)
p = sns.lineplot(x =df['Embarked'], y =df['Survived'], data = df)
p = sns.boxplot(x=df["Age"], y=df["Survived"])
p = sns.boxplot(x=df["Sex"], y=df["Survived"])
p = sns.boxplot(x=df["Parch"], y=df["Survived"])
p = sns.pairplot(data = df)
p = df[["Survived", "Age", "Sex", "Parch", "SibSp"]].plot()

plt.show()