# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv('../input/test.csv', header=0)
survivedIds = df[(df['Age'] < 60) & (df['Sex'] == 'female') & (df['Pclass'] != 3) & (df['Fare'] > np.mean(df['Fare']))]['PassengerId']
df['Survived'] = df['PassengerId'].map(lambda x: 1 if x in set(df['PassengerId']).intersection(survivedIds) else 0)
df[['PassengerId','Survived']]