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
import pandas as pd
df=pd.read_csv("/kaggle/input/corona-virus-report/covid_19_clean_complete.csv")
df=df[(df['Country/Region']=="India") & (df['Confirmed']>0)]

df.set_index("Date",inplace=True)
df=df.drop(["Province/State","Country/Region","Lat","Long"],axis=1)
df.head(2)
df=df.pct_change()
df=df.apply(lambda x:x*100)
df=df.drop(['3/4/20'])
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
plt.figure(figsize=(50,25))
df=df.iloc[:,0:1]
df.plot(figsize=(20,6),markevery=20)
plt.show()