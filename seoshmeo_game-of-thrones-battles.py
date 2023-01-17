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
battle = pd.read_csv("/kaggle/input/game-of-thrones/battles.csv")

deaths = pd.read_csv("/kaggle/input/game-of-thrones/character_deaths.csv")

pose = pd.read_csv("/kaggle/input/game-of-thrones/character_predictions_pose.csv")

battle.columns
battle.head(25)
attac  = battle.attacker_king.value_counts()

attac

defs = battle.defender_king.value_counts()

defs
new_battle = battle.loc[:,['name',"year", "attacker_size", "defender_size"]]

new_battle['sum'] = new_battle.loc[:, 'attacker_size':'defender_size'].sum(axis=1)

new_battle.groupby('sum')['name', 'year'].max().sort_index(ascending=False)

pose.head(3)
pose.columns
pose.isalive.value_counts()