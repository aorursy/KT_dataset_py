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
PATH = '../input/'
PATH+'Estadisticas.csv'
df_set = pd.read_csv(PATH+'Estadisticas.csv', error_bad_lines=False, encoding='utf-8')
df_set['Hora'].sample(n=4)
df_set.sample(n=4)
newHora= df_set['Hora'].str.split(' - ',n=1, expand = True)
df_set['Hora Inicio'] = newHora[0]

df_set['Hora Final'] = newHora[1]
df_set.sample(5)
#df_set['SubVictima'].apply(lambda st: st[st.find("[")+1:st.find("]")])

df_set['SubVictima'].replace(regex=True,inplace=True,to_replace=r'\[.*?\]',value=r'')
df_set.sample(5)