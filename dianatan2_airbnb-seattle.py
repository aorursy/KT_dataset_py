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
import pandas as pd

df = pd.read_csv("/kaggle/input/reviews (1).csv")

print (df)



df.head()

df = df.rename(index=str, columns={"listing_id": "Listing", 'date' : 'Date'})
ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2008', periods=1000))

cum = ts.cumsum()

cum.plot()
import numpy as np

import pandas_datareader.data as web

import matplotlib.pyplot as plt

import datetime as dt



df.plot (y= "Listing")



#plt.figure(figsize=(9, 3))

#plt.subplot(131)

#plt.bar(Date, Listing)

#plt.subplot(132)

#plt.scatter(Date, Listing)

#plt.subplot(133)

#plt.plot(date, listing)

#plt.suptitle('Categorical Plotting')

#plt.show()
plt.style.use('ggplot')

df[["Listing","Date"]].plot()
df['Listing'].value_counts().plot(kind='bar')
#x=df[['Date']]

#y=df[['Listing']]

#plt.scatter(x,y, alpha='0.5',c='Blue')

df[['Date','Listing']].plot(kind='scatter', x='Date', y='Listing')