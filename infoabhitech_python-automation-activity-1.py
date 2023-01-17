# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings("ignore")



# Any results you write to the current directory are saved as output.
import re
L = []

f = open("../input/tablename_column.txt", "r")

for x in f:

    print(x)

f.close()
# Finding 2 patterns - string ending and string starting with ~

L = []

f = open("../input/tablename_column.txt", "r")

for x in f:

    pattern1 = "(\w+)[/~]"

    pattern2 = "[/~](.*)"

    str1 = ','.join(re.findall(pattern1, x))

    str2 = ','.join(re.findall(pattern2, x))

    sqlstr = "create table "  + str1 + "  (" + str2 + ");"

    L.append(sqlstr)

f.close()
L
df =  pd.DataFrame(L,columns=['sqltext'])
df
df.to_csv('sql.txt', index=False)