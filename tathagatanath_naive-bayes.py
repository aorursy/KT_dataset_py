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
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv('/kaggle/input/iris/Iris.csv')
data.info()
data.head()
data['Species'].value_counts()
data_setosa=data[data['Species']=='Iris-setosa']
sns.distplot(data_setosa['SepalLengthCm'])
data_setosa['SepalLengthCm'].skew()
sns.distplot(data_setosa['SepalWidthCm'])
data_setosa['SepalWidthCm'].skew()
sns.distplot(data_setosa['PetalLengthCm'])
data_setosa['PetalLengthCm'].skew()
sns.distplot(data_setosa['PetalWidthCm'])
data_setosa['PetalWidthCm'].skew()
data_virginica=data[data['Species']=='Iris-virginica']
sns.distplot(data_virginica['SepalLengthCm'])
data_virginica['SepalLengthCm'].skew()
sns.distplot(data_virginica['SepalWidthCm'])
data_virginica['SepalWidthCm'].skew()
sns.distplot(data_virginica['PetalLengthCm'])
data_virginica['PetalLengthCm'].skew()
sns.distplot(data_virginica['PetalWidthCm'])
data_virginica['PetalWidthCm'].skew()
data_versicolor=data[data['Species']=='Iris-versicolor']
sns.distplot(data_versicolor['SepalLengthCm'])
data_versicolor['SepalLengthCm'].skew()
sns.distplot(data_versicolor['SepalWidthCm'])
data_versicolor['SepalWidthCm'].skew()
sns.distplot(data_versicolor['PetalLengthCm'])
data_versicolor['PetalLengthCm'].skew()
sns.distplot(data_versicolor['PetalWidthCm'])
data_versicolor['PetalWidthCm'].skew()
from math import pi, e, sqrt
def calculate_probability(series, x) :
    
    m=series.mean()
    sd=series.std()
    
    probability =  (1/(sd * sqrt(2 * pi))) * pow(e, ((-1/2) * pow(((x-m)/sd), 2)))
    
    return probability
def find_probability(df) :
    
    probability = df.shape[0] / data.shape[0]
    
    for i in criteria :
        
        probability *= calculate_probability(df[i], criteria[i])
        
    return probability
criteria={}
print('Enter Sepal Length')
criteria['SepalLengthCm']=float(input())
print('Enter Sepal Width')
criteria['SepalWidthCm']=float(input())
print('Enter Petal Length')
criteria['PetalLengthCm']=float(input())
print('Enter Petal Width')
criteria['PetalWidthCm']=float(input())
criteria
p_setosa = find_probability(data_setosa)
p_versicolor = find_probability(data_versicolor)
p_virginica = find_probability(data_virginica)

print("SETOSA :", p_setosa)
print("VERSICOLOR :", p_versicolor)
print("VIRGINICA :", p_virginica)
max(p_setosa, p_versicolor, p_virginica)
