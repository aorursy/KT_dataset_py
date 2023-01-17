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
# Problem 1
from matplotlib import pyplot as plt

Bloodgroups=['O+','A+','B+','AB+','O-','A-','B-','AB-']
Patients= [12,8,10,6,5,3,2,4]
  
label = ['O+','A+','B+','AB+','O-','A-','B-','AB-'] 
  
plt.bar(Bloodgroups , Patients , label=label, width = 0.5, color = ['g', 'g','g','g','r','g','g','g']) 
plt.xlabel('Bloodgroup') 
plt.ylabel('Patient')  
plt.title('Blood Group Graph')
#Problem 2

import matplotlib.pyplot as plt
slices = ['English','Bengali','Hindi','Maths','History','Geography']
marks = [96,90,85,92,88,94]
plt.title('Marks')
plt.pie(marks,labels=slices,startangle=90,shadow=True,explode=(0,0,0.3,0,0,0),autopct='%1.1f%%')
# Problem 3

import numpy as np
import random 
import matplotlib.pyplot as plt

height = np.array([])
for i in range(50):
    height = np.append(height , [random.randint(52,90)])
height[10] = 172
height[48] = 172
height[4] = 12
height[27] = 12
plt.boxplot(height)
plt.show()