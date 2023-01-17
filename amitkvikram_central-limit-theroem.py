# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Distribution f, takes value 1 and 0 with equal probability
def f():
    if(np.random.rand() > 0.5):
        return 1
    else:
        return 0
    
true_mu = 0.5
true_var = 0.25
sampleMuList = []
    
n = 100
    
for it in range(1, 1001):       #repeating the experiment 1000 times
    sample_mu = 0
    for i in range(1, n + 1):    #calculating sample mean of 1000 samples taken indpendently from distribution 
        sample_mu += f()
    sample_mu/=n
    sampleMuList.append(sample_mu)
print("mean = ", np.mean(sampleMuList))
print("variance = ", np.var(sampleMuList))
sns.distplot(sampleMuList)
n = 2000
diff = []
for it in range(1000):    #Repeat experiment 1000 times
    sample_mu = 0
    for i in range(1, n + 1):   #Take n samples from ditribution and calculate mean
        sample_mu += f()
    sample_mu/=n
    diff.append((np.sqrt(n)*(sample_mu - true_mu)) / np.sqrt(true_var))
print("mean = ", np.mean(diff))
print("variance = ", np.var(diff))
sns.distplot(diff)