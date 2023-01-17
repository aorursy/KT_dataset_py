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

import seaborn as sns

%matplotlib inline
df=pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')
df.info()
df.head()
df.describe()
plt.figure(dpi=100)

plt.title('Correlation Analysis')

sns.heatmap(df.corr(),annot=True,lw=1,linecolor='white',cmap='plasma')

plt.xticks(rotation=60)

plt.yticks(rotation = 60)

plt.show()
corr = df.corr()

mask = np.triu(np.ones_like(corr,dtype = bool))



plt.figure(dpi=100)

plt.title('Correlation Analysis')

sns.heatmap(df.corr(),mask=mask,annot=True,lw=1,linecolor='white',cmap='plasma')

plt.xticks(rotation=60)

plt.yticks(rotation = 60)

plt.show()