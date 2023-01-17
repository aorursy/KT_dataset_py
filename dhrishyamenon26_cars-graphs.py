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
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("../input/Cars.csv")
x=df.head(10)
x
x.plot.bar()
x.plot.bar(stacked=True)
x.plot.bar(stacked=False)
x.plot.barh();
x.plot.barh(stacked=True);
x.plot.area()
x.plot.area(stacked=True)
x.plot.scatter(x='HP',y='MPG')
x.plot.pie(subplots=True);
x.plot.pie(subplots=True,figsize=(8,8));
series = pd.Series([0.1] * 4, index=['a', 'b', 'c', 'd'], name='series2')
series
series.plot.pie(figsize=(6,6))
df.plot(kind='scatter',x='HP',y='MPG',color='DarkGreen',label='Label1')
df.plot(kind='hexbin',x='HP',y='MPG',gridsize=25)
dates=pd.date_range('1/1/2000',periods=8)
dates
x=pd.DataFrame(dates)
x