# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



df=pd.read_csv("../input/crime-in-ireland/IRELAND_CRIME_GARDA_DIVISION_wise_2003-2019.csv")



#df.head()



#df.columns



total_crime=dict()



for i in df.columns[5:]:

    total_crime[i]=df[i].sum()

    



plt.bar(range(len(total_crime)), total_crime.values(), align='center')

plt.xticks(range(len(total_crime)), list(total_crime.keys()))

plt.show()

# Any results you write to the current directory are saved as output.