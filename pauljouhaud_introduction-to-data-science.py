# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



# Display Matplotlib diagrams inside the notebook

%matplotlib inline
data = pd.read_csv('../input/train.csv')
data.info()
data.head()
data.describe()
f, ax = plt.subplots(figsize=(20, 20))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
diag = sns.FacetGrid(data, hue='Sex', aspect=4, row = 'Survived', col=None)

diag.map(sns.kdeplot, 'Age', shade=True)

diag.set( xlim=(0, data['Age'].max()))

diag.add_legend()
data.Age = data.Age.fillna(data.Age.mean())

data.info()
facet = sns.FacetGrid(data)

facet.map(sns.barplot, 'Embarked', 'Survived', order=None)
dummy_sex = pd.Series(np.where(data.Sex == 'female', 1, 0) , name='Sex')
dummy_embarked = pd.get_dummies(data.Embarked , prefix='Embarked')

dummy_embarked.head()
dummy_class = pd.get_dummies(data.Pclass, prefix='Class')

dummy_class.head()
cleaned_data = pd.concat([dummy_sex, dummy_embarked, dummy_class, data.Age, data.Fare, data.SibSp, data.Parch], axis=1)
cleaned_data.head()