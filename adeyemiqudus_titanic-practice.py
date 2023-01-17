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
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
train_df.head()
train_df = train_df.set_index('PassengerId')
test_df = test_df.set_index('PassengerId')
train_df.head()
train_df['Survived'].value_counts()
import seaborn as sns

sns.set()
train_df.hist(column='Survived',sharey=True, by=['Sex'])
pd.crosstab(train_df['Survived'], train_df['Sex'], margins=True)
train_df.describe()
train_df.info()
test_df.info()
train_df['Embarked'].unique()
train_df['Embarked'].value_counts()