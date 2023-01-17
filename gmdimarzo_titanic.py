# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/train.csv', header= 0, sep=',', )
dataset.head()
dataset['Survived'].value_counts(normalize=True)
dataset['Pclass'].value_counts()
dataset['Survived'].groupby(dataset['Pclass']).count()
sns.countplot(dataset['Pclass'], hue=dataset['Survived'])
dataset['NameTitle']=dataset['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])

dataset['NameTitle'].value_counts()
dataset['Sex'].value_counts()
dataset['Age'].value_counts()