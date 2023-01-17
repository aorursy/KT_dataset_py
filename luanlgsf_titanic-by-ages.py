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

import matplotlib.pyplot as plt
df = pd.read_csv("../input/titanic/train.csv")

df_test = pd.read_csv('../input/titanic/test.csv')



df.head()
df_titanic_with_ages = df[pd.notnull(df['Age'])]
df_titanic_with_ages['Age'].isnull().any()
df_titanic_with_ages.hist(column = 'Age')

plt.xlabel('Idade')

plt.ylabel('Frequência')

plt.title('Histograma de Idade dos Sobreviventes');
bins = [0,18, 60,max(df_titanic_with_ages['Age'])]



age_labels = ['minor', 'adult', 'elder']

column_agegroup = pd.cut(df_titanic_with_ages['Age'], bins, labels = age_labels)

df_titanic_with_ages = df_titanic_with_ages.assign(age_group = column_agegroup)
df_titanic_with_ages.head()
df_titanic_with_ages['age_group'].isnull().any()

survived_by_agegroups = df_titanic_with_ages.groupby('age_group').agg('count')

survived_by_agegroups["Survived"]
fracs = 100.* np.true_divide(survived_by_agegroups['Survived'], int(survived_by_agegroups['Survived'].sum()))
plt.pie(fracs, autopct='%1.1f%%', labels = age_labels)

plt.title('Sobreviventes por Categoria de Idade');

plt.axis('square');