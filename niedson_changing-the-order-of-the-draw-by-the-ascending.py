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
df = pd.read_excel('/kaggle/input/lottery-br/megas.xls')

df
df.info()
df2 = df.drop(['date_occured','lottery'], axis=1)

df2 = df2.to_numpy()
type(df2)
df2
df2.sort(axis=1)

df2
df2 = pd.DataFrame({'Draw1': df2[:, 0], 'Draw2': df2[:, 1], 'Draw3': df2[:, 2], 'Draw4': df2[:, 3], 'Draw5': df2[:, 4], 'Draw6': df2[:, 5]})

df2
df3 = pd.concat([df, df2], axis=1, sort=False)

df3
df3 = df3.drop(['ball_01','ball_02','ball_03','ball_04','ball_05','ball_06'], axis=1)

df3