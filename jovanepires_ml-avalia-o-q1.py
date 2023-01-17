# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns; sns.set()

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler



from warnings import filterwarnings

filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/ex1data1.txt', header=None, names=['X', 'y'], index_col=False)
# amostra dos dados

data.head()
# verificar nulos

nullable = data.isnull().sum().sort_values(ascending=False)

nullable = nullable[nullable > 0]

nullable
# pré-processamento

scaler = MinMaxScaler()

data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
# plotar gráficos

plt.scatter(x=data['X'], y=data['y'])

plt.show()