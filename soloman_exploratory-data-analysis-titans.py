# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')



data.head()
data.info()
# setup ploting lib

import matplotlib.pyplot as plt

import seaborn as sn



%matplotlib inline

sn.set_style('whitegrid')
# count graphs



fig, ax = plt.subplots(2, 2)

for idx, dimension in enumerate(['Sex', 'Pclass', 'Survived', 'Embarked']):

    sn.countplot(x=dimension, data=data, ax=ax[idx // 2][idx % 2])
# plot of different attribute against surived field



fig, ax = plt.subplots(2, 2)



for idx, dimension in enumerate(['Sex', 'Pclass', 'Embarked']):

    sn.countplot(x='Survived', hue=dimension, data=data, ax=ax[idx // 2][idx % 2])
survived = data[data['Survived']==0]

not_survived = data[data['Survived']==1]



plt.hist([survived['Age'].fillna(0), not_survived['Age'].fillna(0)], color=['r','b'], alpha=0.5)

sn.swarmplot(x="Survived", y="Age", data=data)