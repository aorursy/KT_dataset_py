# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np



data1 = {'Name':['Tom', 'Jack',],'Age':[28,34]}

data2 = {'Name':['Bob', 'Tom', 'Reddy', 'Arun'],'Age':[28,34,29,56]}

df1 = pd.DataFrame(data1)

df2 = pd.DataFrame(data2)

df2["clicked"]=0



df2[df2['Age'].apply(lambda v: v in df1['Age'])]['clicked'] = 1

df2