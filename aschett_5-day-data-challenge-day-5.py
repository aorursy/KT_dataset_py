# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # Visualization

import scipy.stats # X^2



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/DigiDB_digimonlist.csv')

df.describe()
# Perform X^2 test

print(scipy.stats.chisquare(df['Lv50 SP']))

print(scipy.stats.chisquare(df['Lv50 Atk']))

print(scipy.stats.chisquare(df['Lv50 Def']))

print(scipy.stats.chisquare(df['Lv50 Int']))

print(scipy.stats.chisquare(df['Lv50 Spd']))
# Revisit visualization

sns.distplot(df['Lv50 SP'], kde=False).set_title('Lv50 SP')