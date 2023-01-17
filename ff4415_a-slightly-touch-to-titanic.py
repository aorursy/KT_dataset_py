# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/train.csv");df.columns
df.count()
df = df.drop(['Ticket', 'Cabin'], axis=1).dropna()

df.count()
sns.distplot(df['Survived'], kde=False);
df.count()
sns.jointplot(x='Survived', y='Age',data=df[['Survived', 'Age']])
sns.countplot(y='Survived', hue='Pclass',data=df[['Survived', 'Pclass']])
sns.kdeplot(df['Age'][df.Pclass == 1], label='Pclass 1')

sns.kdeplot(df['Age'][df.Pclass == 2], label='Pclass 2')

sns.kdeplot(df['Age'][df.Pclass == 3], label='Pclass 3')
sns.countplot(x='Embarked', hue="Survived", data=df)
g = sns.FacetGrid(df, row='Pclass', col='Embarked', margin_titles=True)

g.map(sns.countplot, "Survived")