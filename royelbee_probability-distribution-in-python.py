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
# Create 10000 samples from Normal distributions
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
samples_std1 = np.random.normal(20, 1, size=100000)
samples_std2 = np.random.normal(20, 3, size=100000)
samples_std3 = np.random.normal(20, 10, size=100000)
plt.hist(samples_std1, bins=100, histtype='step')
plt.hist(samples_std2, bins=100, histtype='step')
plt.hist(samples_std3, bins=100, histtype='step')

#plt.ylim(0, .42)
plt.legend(('std=1', 'std=2', 'std =3'))
plt.show()