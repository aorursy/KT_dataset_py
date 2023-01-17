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
train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.info()
totAge = train.Age.notnull().count() - train.Age.count()
# Não entendi pq o notnull.count retorna todos os itens, incluindo os não nulos
# enquanto o Age.count retorna apenas os não nulos (??)
print("\n\nAge tem %d itens faltantes" % totAge)

totEmbarked = train.Embarked.notnull().count() - train.Embarked.count()
print("Embarked tem %d itens faltantes" % totEmbarked)

totCabin = train.Cabin.notnull().count() - train.Cabin.count()
print("Cabin tem %d itens faltantes" % totCabin)
train.hist(column="Age", by="Survived")
train.hist(column="Sex", by="Survived")
#train['Cabin'].sort_values()
train.hist(column="Cabin", by="Survived")
train.hist(column="Embarked", by="Survived")
