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
df = pd.read_csv('../input/autos.csv', sep=',', encoding='Latin1')

df.info()
df.describe()
df.sort_values('yearOfRegistration')
#we're going to remove all time-travelled and prehistoric cars for now :)

df = df[(df.yearOfRegistration>1850) & (df.yearOfRegistration<2017)]

df.describe()
df = df.drop('nrOfPictures', axis=1) #removing 0 value column

df.info()
df = df.drop('abtest', axis=1) #no use

df.sort_values('price')
df = df[df['price'] < 100000]
df.boxplot(column='price')
df1 = pd.cut(df['price'], [i for i in range(0,100000,10000)])
df1.plot(kind='hist')
df = df[(df.price>0)&(df.price<15000)]
df.boxplot(column='price')
df.dtypes
df.vehicleType.value_counts()