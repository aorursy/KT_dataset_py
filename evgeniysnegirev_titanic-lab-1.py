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
data = pd.read_csv("/kaggle/input/titanic/train.csv")
data.head()
print(data['Sex'].value_counts()['male'], data['Sex'].value_counts()['female'])
print(np.round(data['Survived'].value_counts()[1] * 100 / data.shape[0], 2))
print(np.round(data['Pclass'].value_counts()[1] * 100 / data.shape[0], 2))
print(data['Age'].mean(), data['Age'].median())
print(data['SibSp'].corr(data['Parch']))
def filter_names(name):

    if 'Miss.' in name:

        lst = name.split()

        return lst[lst.index('Miss.') + 1]

    if '(' in name:

        return name[name.find('(') + 1: -1].split()[0]

names = data[data['Sex'] == 'female']['Name']

f_names = names.apply(filter_names)

print(max(f_names.to_list(),key=f_names.to_list().count))