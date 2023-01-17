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
import pandas as pd

import numpy as np 



#custom dictionary 

Data = {'A':[1, 4, 6, 9], 

        'B':[np.NaN, 5, 8, np.NaN], 

        'C':[7, 3, np.NaN, 2], 

        'D':[1, np.NaN, np.NaN, np.NaN]} 

labels = ['1', '2', '3', '4']

data = pd.DataFrame(Data,index = labels) 



print("Number of observation:",len(data.index))

print("Number of NaN datas:",data.isnull().sum().sum())