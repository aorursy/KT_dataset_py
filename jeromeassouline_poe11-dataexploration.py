# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#Chargement du fichier
df = pd.read_csv('../input/wine-reviews/winemag-data-130k-v2.csv')
df.head()
#Colonnes et types de variables
df.info()
#Infos sur les valeurs NA
print('Nombre de valeurs NA :','\n',df.isnull().sum(), '\n')
print('Pourcentage de valeurs NA :', '\n',df.isnull().mean())
#Distribution et étendue des valeurs
print('Valeurs uniques pour chaque colonne')
for x in df:
    print(x,':',df[x].unique().size)
#Répartition des pays (en fréquence)
df.country.value_counts(normalize=True)
#Répartition des points
sns.distplot(df.points)

#Répartition des prix
sns.distplot(df.price)