# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
g_data=pd.read_csv('../input/groceries-dataset/Groceries_dataset.csv')
g_data.shape
g_data.head()
total_g=[x for x in set(g_data['itemDescription'])]
len(total_g)
c=0

a=[]

for i in total_g:

    for j in g_data['itemDescription']:

        if i==j:

            c+=1

    a.append(c)

    c=0
temp=pd.DataFrame({'itemDescription':total_g,'frequency':a})
temp
temp['frequency'].max()
temp[:][temp['frequency']==2502]
temp['frequency'].min()
temp[:][temp['frequency']==1]
total_g
temp[:][temp['frequency']>=800]
total_mem=g_data['Member_number']
mem=[x for x in set(total_mem)]
len(mem)
c=0

a=[]

for i in mem:

    for j in g_data['Member_number']:

        if i==j:

            c+=1

    a.append(c)

    c=0
mem_freq=pd.DataFrame({'Member':mem,'Frequency':a})
mem_freq
mem_freq['Frequency'].max()
mem_freq[:][mem_freq['Frequency']==36]