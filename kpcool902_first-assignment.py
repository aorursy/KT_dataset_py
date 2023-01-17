# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import pandas as pd
import numpy as np
PATH = '../input/athlete_events.csv'
df = pd.read_csv(PATH)
df.Age[(df['Sex'] == 'F') & (df['Year'] == 1996 )].min()

np.mean(df.Height[(df['Sex'] == 'F') & (df['Sport'] == 'Basketball') & (df['Year'] == 2000)])
np.std(df.Height[(df['Sex'] == 'F') & (df['Sport'] == 'Basketball') & (df['Year'] == 2000)])
df.Sport[df.Weight[df['Year'] == 2002].max()]
df.Year[df['Name'] == 'Pawe Abratkiewicz'].unique()
df.Name[(df['Team'] == 'Australia') & (df['Medal'] == 'Silver') & (df['Sport'] == 'Tennis')]
df.Medal[(df['Team'] == 'Switzerland') & (df['Year'] == 2016)].count() - df.Medal[(df['Team'] == 'Serbia')&(df['Year'] == 2016)].count()
df.Name[(df['Age'] >44) & (df['Age'] <56) ].count()
df[(df['Season'] == 'Summer') & (df['City'] == 'Lake Placid')].count()
df.Sport[(df['Season'] == 'Winter') & (df['City'] == 'Sankt Moritz')].unique()
df.Sport[df['Year'] == 1995].unique() 
df.Sport[df['Year'] == 2016].unique()