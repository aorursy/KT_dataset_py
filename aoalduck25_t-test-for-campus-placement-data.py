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

import scipy.stats as st

from scipy.stats import ttest_1samp
df = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

degree_p=df[["degree_p"]]

degree_p.info() #there is no null value
#popluation histogram

result = plt.hist(degree_p['degree_p'], edgecolor='k', alpha=0.65)

plt.axvline(60, color='k', linestyle='dashed', linewidth=1)
#get 100 samples from popluation and its histogram

sample = degree_p.sample(n=100, random_state=25)

smaple_result = plt.hist(sample["degree_p"], edgecolor='k', alpha=0.65)

plt.axvline(60, color='k', linestyle='dashed', linewidth=1)
result=ttest_1samp(sample['degree_p'],60) #t-test result 
print("t-test statistics : ", result[0])

print("p-value : ", result[1])

print("df : ", len(sample)-1)

print("95 percent confidence interval : ",st.t.interval(0.95,len(sample)-1, loc=np.mean(sample['degree_p']),scale=st.sem(sample['degree_p'])) )

print("mean of x : ", np.mean(sample['degree_p']))