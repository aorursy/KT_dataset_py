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
import numpy as np

import matplotlib as plt

import sklearn

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb

import sklearn.preprocessing as skp

data=pd.read_csv('../input/cardataset/data.csv')

print(data)
import seaborn as sb

import sklearn.preprocessing as skp

size=(20,9)

plt.subplots(figsize=size)

plt.xticks(rotation=90)

sb.stripplot(x='Make',y='MSRP',data=data,size=10)

plt.show()
print(data.iloc[0])
size=(11,9)

plt.subplots(figsize=size)

plt.xticks(rotation=90)

sb.stripplot(x='Year',y='MSRP',data=data,size=10)
data.plot()
data.plot(kind='scatter',x='Engine HP',y='highway MPG')
ax=plt.hist(data['MSRP'],bins=50)

plt.xlim(0,150000)

plt.show()

%pylab inline

Make=data.groupby(['Make'])['MSRP'].median()

Make.plot(kind='bar',stacked=True)

pylab.ylabel('Median MSRP')

pylab.title('Chart: Median MSRP by Make')

plt.show()