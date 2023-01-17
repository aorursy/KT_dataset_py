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
import numpy as np

import pylab

import scipy.stats as stats
#N(0,1)



std_norm= np.random.normal(loc= 0, scale= 1, size= 1000)



for i in range(0,101):

    print(i, np.percentile(std_norm, i))
# genrate 100 samples from N(20,5)



measurements= np.random.normal(loc= 20, scale= 5, size= 1000)



stats.probplot(measurements, dist= "norm", plot= pylab)
# genrate 100 samples from N(20,5)



measurements= np.random.normal(loc= 20, scale= 5, size= 50000)



stats.probplot(measurements, dist= "norm", plot= pylab)
# generate 100 samples from N(20,5)

measurements= np.random.uniform(low= -1, high= 1, size= 10000)

stats.probplot(measurements, dist= "norm", plot= pylab)
# generate 100 samples from N(20,5)

measurements= np.random.uniform(low= -1, high= 1, size= 10000)

stats.probplot(measurements, dist= "uniform", plot= pylab)