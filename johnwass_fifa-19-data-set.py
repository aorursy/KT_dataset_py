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
fifa = pd.read_csv("../input/data.csv")
print (fifa.dtypes)
fifa.info()
fifa.describe()
fifa.head()
fifa.tail()
fifa['Value'] = fifa.Value.apply(lambda x: x.strip ('M'))

fifa['Value'] = fifa.Value.apply(lambda x: x.strip ('€') )

fifa['Value'] = fifa.Value.apply(lambda x: x.strip ('K'))
fifa['Value']=fifa['Value'].astype(str).astype(float)
fifa['PlayerValue']=fifa['Value']/fifa['Overall']
print(fifa.Name,fifa.PlayerValue)