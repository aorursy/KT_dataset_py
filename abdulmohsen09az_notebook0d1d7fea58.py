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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline 
df = pd.read_csv('../input/HR_comma_sep.csv')
df['salary']=df['salary'].replace(['low','medium','high'],[1,2,3])

df.loc[:,'salary'] = 3
df
atisfaction = df['satisfaction_level']

salary = df['salary']

plt.plot(satisfaction,salary)
plt.plot(df['satisfaction_level'],df['salary'])