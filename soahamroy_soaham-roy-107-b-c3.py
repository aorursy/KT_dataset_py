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
import matplotlib.pyplot as plt

height=[46,20,22,24,46,28,26,44,40,56,24,35,67,43,32,34,25,36,35,45,172,12,37,12,72,172,29,39,42,33,49,58,100,102,25,58,12,102,20,43,24,32,53,54,56,76,34,54,78,23]

plt.boxplot(height)

plt.show()
import matplotlib.pyplot as plt

slices=[ 88,82,93,94,15,99 ]

subj=['English','Bengali','Hindi','Maths','History','Geography']

cols=['r','b','g','m','c']

plt.pie(slices,labels=subj,colors=cols,startangle=90,shadow= True,explode=(0,0,0,0,0.4,0),autopct='%1.1f%%')

plt.title('MARKS IN SUBJECT')

plt.show()
import matplotlib.pyplot as plt

blood_grp=['O+','A+','B+','AB+','O-','A-','B-','AB-','O+','0-','O-','O+','A+','O+','A+','A-','O+',

'B-','AB+','AB-','O+','0-','O+','A+','A-','O+','B-','AB+','AB-','O+','0-','O-','O+','A+','B+','O+','A+',

'A-,O+','B-','AB+','AB-','O+','O+','0-','O-','O+']

bins = [5,10,15,20]

plt.hist(blood_grp,bins, histtype='bar',rwidth=0.2, color = 'red'  ) 

plt.xlabel('blood_grp') 

plt.ylabel('patients')  

plt.title('BLOOD GROUP OF PATIENTS')

plt.show()