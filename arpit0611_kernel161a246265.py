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
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import scipy
data_digimon = pd.read_csv('/kaggle/input/digidb/DigiDB_digimonlist.csv')
data_move = pd.read_csv('/kaggle/input/digidb/DigiDB_movelist.csv')
data_support = pd.read_csv('/kaggle/input/digidb/DigiDB_supportlist.csv')
sns.pairplot(data=data_digimon[['Lv 50 HP', 'Lv50 SP', 'Lv50 Atk', 'Lv50 Def', 'Lv50 Int', 'Lv50 Spd']])
plt.show()
sns.countplot(x=data_digimon['Attribute'])
plt.show()
sns.countplot(x=data_digimon['Type'])
plt.show()
data_move['ratio_attack_sp'] = data_move['Power']/data_move['SP Cost']
data_move.head(5)
data_move['ratio_attack_sp'].idxmax()
print(data_move.iloc[80])
data_digimon.sort_values(by=['Lv50 Atk'], ascending=False).head(3)
data_digimon.sort_values(by=['Lv50 Def'], ascending=False).head(3)
sns.countplot(x=data_digimon['Type'])
plt.show()
plt.figure(figsize=(12,4))
sns.countplot(hue=data_digimon['Type'], x=data_digimon['Stage'])
plt.show()
plt.figure(figsize=(15,8))
sns.countplot(hue=data_digimon['Attribute'], x=data_digimon['Stage'])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()
