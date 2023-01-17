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
data=pd.read_csv('../input/health-care-data-set-on-heart-attack-possibility/heart.csv')
data.describe
data.head
target1=data[data.target==1]
target0=data[data.target==0]
len(target1)
len(target0)
len(data)
data.columns
import matplotlib.pyplot as plt
fig,axes=plt.subplots(4,figsize=(10,10))

cols=['age','chol','thalach','trestbps']

for i,col in enumerate(['age','chol','thalach','trestbps']):

    _,bins=np.histogram(data[cols[i]],bins=30)

    axes[i].hist(target1[cols[i]],bins=bins,color='red')

    axes[i].hist(target0[cols[i]],bins=bins,color='blue')

    axes[i].set_title(cols[i])

axes[0].legend(['1','0'])

fig.tight_layout()