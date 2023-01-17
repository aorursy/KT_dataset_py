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
df=pd.read_csv("/kaggle/input/titanic/train.csv")
df.head()
df.info()
df.describe()
df['Age'].mean()
df['Age'].fillna(df['Age'].mean(),inplace=True)    #filled all null values of age
df.info()
# dropping uncessary column
del df['Cabin']
del df['PassengerId']
del df['Name']
plt.hist(df.Age)
import seaborn as sns
sns.countplot(df.Survived)  #one for survived and 0 for not survived
sns.countplot(df.Survived,hue=df['Sex'])     #high chances of female may survived

df.hist()
df.corr()
df.Fare.hist()
sns.heatmap(df.corr())
sns.catplot(x="Pclass", y="Survived", kind="bar", data=df,label='1-class1,2-class2,3-class3')
plt.legend()
# Children had more chances to survived
# Female also had more chances
# Higher Fare also made chances to survived