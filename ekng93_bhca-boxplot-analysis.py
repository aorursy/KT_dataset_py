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
#Libraries

from matplotlib import pyplot as plt

import pandas as pd
data_url = '../input/mgcf-bhca/mgcf.csv'

df = pd.read_csv(data_url, index_col=0)   #index_col=0 means first column will become index, otherwise specify with column name 'example name'



print (df)

print (df.dtypes)

print (df.shape)

print (df.columns)
print (df.describe())
first_series = df.loc[:, ('BHCA License Utilization (%)')]



labels = ['BHCA License Utilization (%)']



plt.figure(figsize=(12, 6))

plt.boxplot(first_series, notch=True, vert=True, patch_artist=True, labels=labels)  

plt.title('MGCF')

    

#Plot   

plt.grid(axis="x", color="green", alpha=.3, linewidth=2, linestyle=":")

plt.grid(axis="y", color="black", alpha=.5, linewidth=.5)

plt.ylabel('Utilization Percentage (%)'); plt.title('MGCF');

plt.show()


