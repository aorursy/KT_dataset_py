import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

print(os.listdir("../input"))
import matplotlib.pyplot as plt

import seaborn as sns



data = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")
data.groupby('species').size()
data.head()
data.describe()
print(data)
import missingno as msno

msno.matrix(data)
len(data)
length = len(data)

num = int(0.15*length)

idx_replace = np.random.randint(0, length-1, num)



data.loc[idx_replace, 'species'] = np.nan

#data.loc[idx_replace, 'sepal_width'] = np.nan



#data.loc[idx_replace, 'iris-setosa'] = np.nan

#print(titanic_train)
import missingno as msno

msno.matrix(data)
data.to_csv('results_continuous.csv ')