# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from pathlib import Path

import matplotlib.pyplot as plt

import matplotlib.image as mpimg
#Importação do dataset e listgem da pasta de classificação

path_img = Path('/kaggle/input/br-coins/classification_dataset/all')

fnames = list(path_img.glob('*.jpg'))

fnames[:5]
fnames[0].name
img=mpimg.imread(fnames[0])

imgplot = plt.imshow(img)

plt.title(fnames[0].name)

plt.show()
import pandas as pd

df = pd.read_csv("../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")
df.info()
df.index = df['CustomerID']

df.drop(columns='CustomerID', inplace=True)
df.head()
from pandas_profiling import ProfileReport
ProfileReport(df)
df.describe()
_ =plt.hist(df['Age'], bins=25)

_ =plt.xlabel('Idade')

_ =plt.ylabel('Número de consumidores')



plt.show
_ =plt.hist(df['Annual Income (k$)'], bins=25)

_ =plt.xlabel('Salário')

_ =plt.ylabel('Número de consumidores')



plt.show
_ =plt.hist(df['Spending Score (1-100)'], bins=25)

_ =plt.xlabel('Índice de gasto')

_ =plt.ylabel('Número de consumidores')



plt.show