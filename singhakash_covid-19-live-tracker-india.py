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
df = pd.read_json('https://www.mohfw.gov.in/data/datanew.json')
df
df=df.iloc[:-1,1:]
df.head()
df.tail()
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
total_states = np.arange(len(df['state_name']))

total_states
df['active']
max(df['positive'])

from matplotlib.pyplot import figure
figure(num=None, figsize=(11, 8), dpi=80, facecolor='w', edgecolor='k')
plt.barh(total_states,df['positive'], align='center', alpha=0.5,  
                 color=(1,0,0),  
                 edgecolor=(0.5,0.2,0.8))
    
plt.yticks(total_states, df['state_name'])  
plt.xlim(1,max(df['positive'])+100) 
plt.xlabel('Positive Number of Cases')  
plt.title('Corona Virus Cases')  
plt.show()
from matplotlib.pyplot import figure
figure(num=None, figsize=(11, 8), dpi=80, facecolor='w', edgecolor='k')
plt.barh(total_states,df['new_active'], align='center', alpha=0.5,  
                 color=(1,0.5,0),  
                 edgecolor=(0.5,0.4,0.8)  )
    
plt.yticks(total_states, df['state_name'])  
plt.xlim(1,max(df['new_active'])+10) 
plt.xlabel('Active Number of Cases')  
plt.title('Corona Virus Cases')  
plt.show()
from matplotlib.pyplot import figure
figure(num=None, figsize=(11, 9), dpi=80, facecolor='w', edgecolor='k')
plt.barh(total_states,df['death'], align='center', alpha=0.5,  
                 color=(0,0,1),  
                 edgecolor=(0.5,0.4,0.8)  )
    
plt.yticks(total_states, df['state_name'])  
plt.xlim(1,max(df['death'])+10) 
plt.xlabel('Death Number of Cases')  
plt.title('Corona Virus Cases')  
plt.show()
df.columns
df=df.set_index('state_name',drop=True)
df.head()

### Plotting Based on Stacking
df.plot.barh(stacked=True,figsize=(10,10))
df1=df.iloc[:,1:3]
df1.plot.barh(color={"cured": "red", "positive": "green"},figsize=(10,10))
