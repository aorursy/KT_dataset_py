# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import math as ma



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df= pd.read_csv('/kaggle/input/iris/Iris.csv')
df
df.info()
sns.distplot(df['SepalLengthCm'])
sns.distplot(df['SepalWidthCm'])
sns.distplot(df['PetalLengthCm'])

sns.distplot(df['PetalWidthCm'])
df['Species'].value_counts()
p_Species=50/150
std_Ise=df[df['Species']=='Iris-setosa'].std() 

std_Ive=df[df['Species']=='Iris-versicolor'].std()

std_Ivi=df[df['Species']=='Iris-virginica'].std() 

print('Iris-setosa\n',std_Ise)

print('\nIris-versicolor\n',std_Ive)

print('\nIris-virginica\n',std_Ivi)
m_Ise=df[df['Species']=='Iris-setosa'].mean() 

m_Ive=df[df['Species']=='Iris-versicolor'].mean()

m_Ivi=df[df['Species']=='Iris-virginica'].mean() 

print('Iris-setosa\n',m_Ise)

print('\nIris-versicolor\n',m_Ive)

print('\nIris-virginica\n',m_Ivi)
def fun(x,mean,std):

    num= ma.exp(-0.5*np.square((x-mean)/std))

    den= std*ma.sqrt(2*ma.pi)

    return(num/den)
P_Se=fun(4.7,5.006,0.352490)*fun(3.7,3.418,0.381024)*fun(2,1.464,1.173511)*fun(0.3,0.244,0.107210)*p_Species

P_Ve=fun(4.7,5.936,0.516171)*fun(3.7,2.770,0.313798)*fun(2,4.260,0.469911)*fun(0.3,1.326, 0.197753)*p_Species

P_Vi=fun(4.7,6.588,0.635880)*fun(3.7,2.974,0.322497)*fun(2,5.552,0.551895)*fun(0.3,2.026,0.274650)*p_Species



if (P_Se>P_Ve) & (P_Se>P_Vi):

    print('Iris-setosa')

elif (P_Ve>P_Se) & (P_Ve>P_Vi):

    print('Iris-versicolor')

elif (P_Vi>P_Se) & (P_Vi>P_Ve):

    print('Iris-virginica')