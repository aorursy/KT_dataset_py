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
import seaborn as sns
import matplotlib as plt
#Coletando os dados para o estudo e inserindo no dataframe
df = pd.read_csv('../input/titanic/train.csv')
df.head()
#Tipo de dados de cada coluna
df.info()
sns.set_style('whitegrid')
sns.countplot(x='Survived', data=df)
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=df)
sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Pclass', data=df)
sns.set_style('whitegrid')
sns.countplot(x='Pclass', hue='Sex', data=df)
sns.distplot(df['Age'].dropna(), kde=False, color='Darkred', bins=40)
sns.set_style('whitegrid')
sns.countplot(x='Embarked', hue='Sex', data=df)