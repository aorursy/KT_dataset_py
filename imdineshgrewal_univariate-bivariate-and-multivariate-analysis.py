# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #charts

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

Iris = pd.read_csv("../input/iris/Iris.csv")

Iris.head()
data = Iris.copy()

data.shape
#List unique values in the data['Species'] column

data['Species'].unique()
# Data sepration

df_setosa = data.loc[data['Species']=='Iris-setosa']

df_versicolor = data.loc[data['Species']=='Iris-versicolor']

df_virginica = data.loc[data['Species']=='Iris-virginica']
plt.plot(df_setosa['PetalLengthCm'], np.zeros_like(df_setosa['PetalLengthCm']),'o')

plt.plot(df_versicolor['PetalLengthCm'], np.zeros_like(df_versicolor['PetalLengthCm']),'o')

plt.plot(df_virginica['PetalLengthCm'], np.zeros_like(df_virginica['PetalLengthCm']),'o')

plt.xlabel('Petal Length')

plt.show()
sns.FacetGrid(data,hue='Species',size=5).map(plt.scatter,'PetalLengthCm','SepalWidthCm').add_legend();

plt.show()
sns.pairplot(data,hue='Species',size=2)