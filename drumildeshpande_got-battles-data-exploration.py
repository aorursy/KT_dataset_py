# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt
import seaborn as sns
sns.set(style="darkgrid")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
battles = pd.read_csv('/kaggle/input/game-of-thrones/battles.csv');
battles.head()
battles.info()
sns.relplot(y='attacker_size', x='defender_size', hue='attacker_outcome', style='attacker_outcome', data=battles);
sns.catplot(y='attacker_king', hue='attacker_outcome', kind='count', data=battles);
sns.catplot(y='region', hue='attacker_outcome', kind='count', data=battles);
sns.catplot(y='attacker_king', hue='region', kind='count', data=battles);
sns.catplot(y='attacker_1', hue='attacker_outcome', kind='count', data=battles);
sns.catplot(y='defender_1', hue='attacker_outcome', kind='count', data=battles);
sns.catplot(y='battle_type', hue='attacker_outcome', kind='count', data=battles);
sns.pairplot(battles, vars=['attacker_size', 'defender_size'], hue='attacker_outcome');
