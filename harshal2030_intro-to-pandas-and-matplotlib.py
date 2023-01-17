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
import matplotlib.pyplot as plt
dictr={'number': [i for i in range(0, 50)], 'numbers_2': [i for i in range(50, 100)]}

df=pd.DataFrame(dictr)

#df=pd.read_csv(file_path)

#df=pd.read_excel(file_path)
df.head(5)#returns first 5 rows
df.tail(5) #returns last 5 rows
%matplotlib inline

plt.plot(df['number'], df['numbers_2'])

#plt.scatter(df['number'], df['numbers_2'])

plt.xlabel('numbers')#label the axis

plt.ylabel('numbers_2')
df.info()#returns df information
df.loc[[1], 'number']#1 is the index and the second argument is col name
df.loc[[2, 3, 4], 'numbers_2']#this are multiple index as input
df.loc[(df['number']==43), 'numbers_2']#this returns position where number==2
df.iloc[[1, 4, 5], [0, 1]]#this returns cloumn 1 and 2 of index 1, 5, 4