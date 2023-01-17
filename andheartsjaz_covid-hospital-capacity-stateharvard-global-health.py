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
df = pd.read_csv('/kaggle/input/uncover/harvard_global_health_institute/hospital-capacity-by-state-20-population-contracted.csv')
df.info()
df.head()
df.isna().sum()
df = df.dropna()
df.head(20)
cols = df.columns.to_list()

print(cols[0:5])

# df.columns
top_cols = cols[0:5]
df.groupby(['state']).sum()[['total_hospital_beds', 'total_icu_beds', 'hospital_bed_occupancy_rate', 'icu_bed_occupancy_rate']].head()
!pip install matplotlib -U
import matplotlib.plyplot as plt
%matplotlib inline
df.groupby(['state']).sum()[['total_hospital_beds', 'total_icu_beds']].plot(kind='bar')
plt.xticks(rotation=40)
plt.show()
df.head()