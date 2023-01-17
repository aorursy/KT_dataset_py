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
data=pd.read_csv('/kaggle/input/iris/Iris.csv')

data.head()
del data['Id']

df=data.groupby('Species')

np.unique(data['Species'])
df.get_group('Iris-setosa').head()
df_1=df.get_group('Iris-setosa')

df_1.name='Iris-setosa'

df_2=df.get_group('Iris-versicolor')

df_2.name='Iris-versicolor'

df_3=df.get_group('Iris-virginica')

df_3.name='Iris-virginica'
x_input=[4.8,3.1,1.4,0.3]

x_input=[5.1,3.5,1.4,0.2]
def get_probability(x,df):

    cols=df.columns

    for i in range(len(x)):

        count=0.0

        for j in df[cols[i]]:

            if j==x[i]:

                count+=1

        prob=count/len(df['SepalLengthCm'])

        x[i]=prob

    z=1.0

    for i in x:

        z=z*i

    return z
y_1=get_probability(x_input,df_1)

y_2=get_probability(x_input,df_2)

y_3=get_probability(x_input,df_3)



print(y_1,y_2,y_3)

# y_1 is max therefore y_1 is the reqd grp

# Iris-setosa