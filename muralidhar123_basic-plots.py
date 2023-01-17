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
import numpy as np

import matplotlib.pyplot as plt
x = np.linspace(np.pi , -np.pi , 256 , endpoint = True)

x

s , c , z = np.sin(x) , np.cos(x) , np.tan(x)
plt.plot(x , s)
plt.plot(x , c)



plt.plot(x , z)
plt.plot(x , s)

plt.plot(x , z)

plt.show()



plt.plot(x , c)

plt.plot(x , z)

plt.show()



plt.plot(x , s)

plt.plot(x , c)

plt.plot(x , z)

plt.show()

plt.hist(x)


import pandas as pd

salary = {'sno':[0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9],'Name':['satya','mohan','teja','ram','pavan','vijay','sai','lakshman','bhanu','subbu'],'Income':[20000 , 30000 , 40000 , 10000 , 50000 , 10000 , 30000 , 20000 , 70000 , 20000]}

df= pd.DataFrame(salary)

print(df)



df.info()



df.query('Income > 10000').plot(kind = 'scatter' , x = 'sno' , y = 'Income')



df.mean(0)



df.mean(1)



df.sum(0 , skipna = False)



df.sum(0 , skipna = 'True')



df.sum(1 , skipna = False)



df.sum(1 , skipna = 'True')



df.describe()



df.describe(percentiles = [.25 , .85 , .99])
df.query('Income > 10000').plot(kind = 'bar' , x = 'sno' , y = 'Income')

df.query('Income > 10000').plot(kind = 'hist' , x = 'sno' , y = 'Income')
