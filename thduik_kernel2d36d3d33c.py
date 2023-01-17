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
dfraw = pd.read_csv('/kaggle/input/ufcdata/raw_total_fight_data.csv')

dfraw = dfraw.reset_index()
dfraw.head(1)
saved_col_names = dfraw.columns[3]
dfraw.columns = ['col1','col2','col3','col4']
dfraw['col4'] = dfraw['col4'].fillna("")
listlol = dfraw['col1'] + dfraw['col2'] + dfraw['col3'] + dfraw['col4']
listlol
listlol = [i.split(';') for i in listlol]
from collections import Counter

countlol = Counter()

for i in listlol:

    countlol[len(i)] += 1
countlol.most_common(10)
col_names1 = col_names.split(';')
len(col_names1)
print(listlol[1])
dfnew = pd.DataFrame(listlol, columns = col_names1)
dfnew.tail()
#len of 1st list is 39