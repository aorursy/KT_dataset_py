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
iris=pd.read_csv('/kaggle/input/iris/Iris.csv')
iris

import seaborn as sns
import math
sns.distplot(iris['SepalLengthCm'])
sns.distplot(iris['SepalWidthCm'])
sns.distplot(iris['PetalLengthCm'])
sns.distplot(iris['PetalWidthCm'])
sns.distplot(iris['SepalWidthCm'])
sns.distplot(iris['PetalLengthCm'])
sns.distplot(iris['PetalWidthCm'])
data1=iris[iris['Species']=='Iris-setosa']
data2=iris[iris['Species']=='Iris-virginica']
data3=iris[iris['Species']=='Iris-versicolor']
def gaussiancorr(num,mean,sd):
    m=(1/(sd*math.sqrt(2*math.pi)))*(math.pow(math.e,(-0.5*(((num-mean)/sd)**2))))
    return m
                                   
p_sertosa=len(data1)/len(iris)
p_virginica=len(data2)/len(iris)
p_versicolor=len(data3)/len(iris)
p_versicolor
mean1=data1['SepalLengthCm'].mean()
sd1=data1['SepalLengthCm'].std()
d1=gaussiancorr(4.7,mean1,sd1)
mean2=data1['SepalWidthCm'].mean()
sd2=data1['SepalWidthCm'].std()
c1=gaussiancorr(3.7,mean2,sd2)
mean3=data1['PetalLengthCm'].mean()
sd3=data1['PetalLengthCm'].std()
b1=gaussiancorr(2,mean3,sd3)
mean4=data1['PetalWidthCm'].mean()
sd4=data1['PetalWidthCm'].std()
a1=gaussiancorr(0.3,mean4,sd4)
p_sertosa_=a1*b1*c1*d1*p_sertosa
mean5=data2['SepalLengthCm'].mean()
sd5=data2['SepalLengthCm'].std()
d2=gaussiancorr(4.7,mean5,sd5)
mean6=data2['SepalWidthCm'].mean()
sd6=data2['SepalWidthCm'].std()
c2=gaussiancorr(3.7,mean6,sd6)
mean7=data2['PetalLengthCm'].mean()
sd7=data2['PetalLengthCm'].std()
b2=gaussiancorr(2,mean7,sd7)
mean8=data2['PetalWidthCm'].mean()
sd8=data2['PetalWidthCm'].std()
a2=gaussiancorr(0.3,mean8,sd8)
p_virginica_=a2*b2*c2*d2*p_virginica
mean9=data3['SepalLengthCm'].mean()
sd9=data3['SepalLengthCm'].std()
d3=gaussiancorr(4.7,mean9,sd9)
mean10=data3['SepalWidthCm'].mean()
sd10=data3['SepalWidthCm'].std()
c3=gaussiancorr(3.7,mean10,sd10)
mean11=data3['PetalLengthCm'].mean()
sd11=data3['PetalLengthCm'].std()
b3=gaussiancorr(2,mean11,sd11)
mean12=data3['PetalWidthCm'].mean()
sd12=data3['PetalWidthCm'].std()
a3=gaussiancorr(0.3,mean12,sd12)
p_versicolor_=a3*b3*c3*d3*p_versicolor
p_versicolor_>p_sertosa_
p_versicolor_>p_virginica_
p_virginica_>p_sertosa
there