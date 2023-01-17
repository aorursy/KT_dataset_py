import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%matplotlib inline
df = pd.read_csv('/kaggle/input/avocado-prices/avocado.csv')
#Check for NaNs
df.isnull().values.any()
df.columns
df.head()
df.dtypes
sns.heatmap(df.corr())
plt.figure(figsize=(10,5))
sns.clustermap(df.corr(), cmap='Blues', robust=True)