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
slices=[88,78,85,89,98,90]
subj=['eng','bengali','history','hindi','geography','maths']
cols=['c','m','r','b','y','g']
plt.pie(slices,labels=subj,colors=cols,startangle=90,shadow= True,explode=(0,0.2,0,0,0,0),autopct='%1.1f%%')
plt.title('MARKS')
plt.show()
import matplotlib.pyplot as plt
data=[50,66,60,172,45,69,58,50,60,72,172,12,45,66,65,61,63,12,60,61,58,59,64,67,63,61,69,68,49,56,58,57,71,66,64,61,62,58,45,47,39,46,48,38,44,65,55,51,53,50]
plt.boxplot(data)
plt.show()
import matplotlib.pyplot as plt
blood_grps=['O+','A+','A-','O+','B-','AB+','AB-','O+','0-','O-','O+','A+','B+','O+','A+','A-','O+','B-','AB+','AB-','O+','0-','O-','O+','A+','O+','A+','A-','O+','B-','AB+','AB-','O+','0-','O-','O+','A+','B+','O+','A+','A-','O+','B-','AB+','AB-','O+','0-','O-','O+','A+']
bins=[5,10,15,20]
plt.hist(blood_grps,bins,histtype='bar',rwidth=0.5,colour='red')
plt.xlabel('number of patients')
plt.ylabel('blood groups')
plt.title('blood samples')
plt.show()

