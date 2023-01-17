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
bbdata1 = pd.read_csv("/kaggle/input/college-basketball-dataset/cbb.csv")

bbdata2 = pd.read_csv("/kaggle/input/college-basketball-dataset/cbb15.csv")

bbdata3 = pd.read_csv("/kaggle/input/college-basketball-dataset/cbb16.csv")

bbdata4 = pd.read_csv("/kaggle/input/college-basketball-dataset/cbb17.csv")

bbdata5 = pd.read_csv("/kaggle/input/college-basketball-dataset/cbb18.csv")

bbdata6 = pd.read_csv("/kaggle/input/college-basketball-dataset/cbb19.csv")

bbdata7 = pd.read_csv("/kaggle/input/college-basketball-dataset/cbb20.csv")



alldata = pd.merge(bbdata1,bbdata2,how='outer')

alldata = pd.merge(alldata,bbdata3,how='outer')

alldata = pd.merge(alldata,bbdata4,how='outer')

alldata = pd.merge(alldata,bbdata5,how='outer')

alldata = pd.merge(alldata,bbdata6,how='outer')

alldata = pd.merge(alldata,bbdata7,how='outer')



oedata = alldata.groupby('TEAM',as_index=False)['ADJDE'].max()



plottingdata = oedata.sort_values('ADJDE',ascending=False).head(10)



print(plottingdata.plot(kind='line',x='TEAM',y='ADJDE',figsize=(15,5)))