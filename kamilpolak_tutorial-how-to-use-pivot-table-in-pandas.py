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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

plt.style.use('ggplot')
df = pd.read_csv('../input/titanic/train.csv')

df.head()
df.drop(['PassengerId','Ticket','Name'],inplace=True,axis=1)
grouped = pd.pivot_table(data=df,index=['Sex'])

grouped
grouped.plot(kind='bar')
grouped = pd.pivot_table(df,index=['Sex','Pclass'])

grouped
grouped = pd.pivot_table(df,index=['Sex','Pclass'],aggfunc={'Age':np.mean,'Survived':np.sum})

grouped
grouped = pd.pivot_table(df,index=['Sex','Pclass'],values=['Survived'], aggfunc=np.sum)

grouped
grouped.plot(kind='bar');
grouped = pd.pivot_table(df,index=['Sex'],columns=['Pclass'],values=['Survived'],aggfunc=np.sum)

grouped
grouped.plot(kind='bar');
grouped = pd.pivot_table(df,index=['Sex','Survived','Pclass'],columns=['Embarked'],values=['Age'],aggfunc=np.mean)

grouped
grouped = pd.pivot_table(df,index=['Sex','Survived','Pclass'],columns=['Embarked'],values=['Age'],aggfunc=np.mean,fill_value=np.mean(df['Age']))

grouped