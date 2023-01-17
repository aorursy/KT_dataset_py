# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Esse trecho de código bloqueia os avisos de importação de mensagens de aviso 

import warnings 

warnings.filterwarnings('ignore')



# Importe bibliotecas e verifique as versões

import sys

import missingno as msno

import matplotlib.pyplot as plt

import matplotlib

import pandas_profiling

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



print('Python version ' + sys.version)

print('Numpy version ' + np.__version__)

print('Pandas version ' + pd.__version__)

print('Matplotlib version ' + matplotlib.__version__ )

print('Missingno version ' + msno.__version__)

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
!pip install pandas_profiling
data = {'name': ['Michael', 'Jessica', 'Sue', 'Jake', 'Amy', 'Tye'],

        'gender':[None,'F',np.NaN,'F',np.NaN, 'M'],

        'height': [123, 145, 100 , np.NaN, None, 150],

        'weight': [10, np.NaN , 30, np.NaN, None, 20],

        'age': [14, None, 29 , np.NaN, 52, 45],

        }

df = pd.DataFrame(data, columns = ['name','gender', 'height', 'weight', 'age'])

df
df.info()
df.isnull().sum()
df.notnull().sum()
msno.matrix(df.sample(6))
pandas_profiling.ProfileReport(df)
msno.bar(df.sample(6), log=True)
msno.heatmap(df)