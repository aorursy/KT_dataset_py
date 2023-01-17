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
#QUESTION 1



import pandas as pd

import numpy as np





cricket_data = {'name': ['Sourav', 'Dhoni', 'Kohli', 'Raina', 'Rohit', 'Yuvraj', 'Ashwin', 'Bhuvi', 'Dhawan', 'Unmesh'],

'score': [78, 102, 118, 22, 34, 20, np.nan, np.nan, 8, np.nan],

'matches': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1]}



labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']



df = pd.DataFrame(cricket_data , index=labels)

new = df.iloc[[1,2,3],[1,2]]

print(new)

print('\n\n')
#QUESTION 2



import random 

import matplotlib.pyplot as plt



height = []

   

for i in range(50):

    height.append(random.randint(60, 80))

height[10] = 172

height[20] = 172

height[30] = 12

height[40] = 12

   

plt.boxplot(height)

plt.show()
#QUESTION 3



import pandas as pd

import numpy as np 



#custom dictionary 

dict = {'A':[1, 4, 6, 9], 

        'B':[np.NaN, 5, 8, np.NaN], 

        'C':[7, 3, np.NaN, 2], 

        'D':[1, np.NaN, np.NaN, np.NaN]} 



data = pd.DataFrame(dict) 

print(data.info())

print("Shape of the data:")

print(data.shape)

print("\nData Type:")

print(type(data))

labels = ['1', '2', '3', '4', '5', '6']

df = pd.DataFrame(data,index=labels)

print("Number of observation:",len(df.index))

print("Number of missing datas:",data.isnull().sum().sum())

print("Number of NaN datas:",data.isnull().sum().sum())