# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print("41")

df_pgen1=pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')
df_pgen1

df_pgen1['DC_POWER'].max()

df_pgen1.info()
df_pgen1.describe()
%whos
df_pgen1['DAILY_YIELD'].max()
df_pgen1['DAILY_YIELD'].unique()
from matplotlib import pyplot as plt
gr1=df_pgen1['DAILY_YIELD']

gr2=df_pgen1['AC_POWER']

gr3=df_pgen1['DC_POWER']

#gr4=df_pgen1['TOTAL_POWER']

#gr5=df_pgen1['PLANT_ID']

plt.plot(gr1,gr3)
plt.plot(gr1,gr2)


plt.plot(gr2,gr3)
t=df_pgen1.sort_values(by='AC_POWER')
gr0=t['AC_POWER']
plt.plot(gr1,gr0)
plt.plot(gr0,gr1)
t
type(t['PLANT_ID'])
gr01=t["TOTAL_YIELD"]
plt.plot(gr01,gr1)
plt.plot(gr0,gr2)
gr0dc=t['DC_POWER']
plt.plot(gr0dc,gr0)
s=df_pgen1.sort_values(by='DC_POWER')
gr01s=s['DC_POWER']
plt.plot(gr0dc,gr01s)
type(t)

t.loc[0:100,'AC_POWER']
t1=t.AC_POWER

t1
t3=t.groupby('AC_POWER').size()

t3
a=np.array(10)

a
b=np.array([[1,2,3],[5,6,7]])

b
