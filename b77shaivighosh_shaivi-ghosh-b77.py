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
from matplotlib import pyplot as plt



BLOODGROUPS=['O+','A+','B+','AB+','O-','A-','B-','AB-']



 

PATIENTS = [10,8,7,9,6,6,1,3]

  

tick_label = ['O+','A+','B+','AB+','O-','A-','B-','AB-'] 

  

plt.box(BLOODGROUPS, PATIENTS, tick_label = tick_label, width = 0.8, color = ['green', 'green','green','green','red','green','green','green']) 

  

plt.xlabel('BLOODGROUP') 

plt.ylabel('PATIENTS')  

plt.title('Blood Group Graph')
import matplotlib.pyplot as plt

slices = ['English','Bengali','Hindi','Maths','History','Geography']

marks = [90,50,91,99,88,87]

plt.pie(marks,labels=slices,startangle=90,shadow=True,explode=(0,0.1,0,0,0,0),autopct='%1.1f%%')
import numpy as np

import random as rn

import matplotlib.pyplot as plt



height = np.array([])

    # creating sample data

for i in range(50):

    height = np.append(height , [rn.randint(62,75)])

height[21] = 172

height[13] = 172

height[10] = 12

height[43] = 12

    # sample data created

plt.boxplot(height)

plt.show()