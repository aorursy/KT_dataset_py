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
df1=pd.read_table('../input/Airplane_Crashes_and_Fatalities_Since_1908.csv',sep=',',index_col=0)

df1.head()
df1.count()

for i in df1.columns:

     k=i.replace("#","")

     print(k.replace("/",""))


df2= df1.head().fillna(0).groupby('Operator',axis=1)

df2
import matplotlib.pyplot as plt

df3=pd.Series(range(10))

df3.plot()
