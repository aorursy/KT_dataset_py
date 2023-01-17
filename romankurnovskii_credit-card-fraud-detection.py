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
import matplotlib.pyplot as plt
urlData = '/kaggle/input/creditcardfraud/creditcard.csv'
data = pd.read_csv(urlData)

data.describe()
data["Class"].value_counts()
column_class = data["Class"].value_counts()

column_class
column_class.plot()
column_class.plot(kind="bar") 
column_class.plot(kind="bar", logy=True) 
V1_class_1 = data.loc[data.Class == 1, 'V1']

V1_class_0 = data.loc[data.Class == 0, 'V1']
fig, ax = plt.subplots(nrows=1,ncols=2)

ax1,ax0 = ax.flatten()



ax1.hist(V1_class_1)

ax1.set_title("Class 1")

ax1.legend()



ax0.hist(V1_class_0)

ax0.set_title("Class 0")

ax0.legend()



fig.set_size_inches(20,7)
plt.hist(V1_class_0, density=True, bins=20, alpha=0.5, label='Class 0', color='grey')

plt.hist(V1_class_1, density=True, bins=20, alpha=0.5, label='Class 1', color='red')

plt.xlabel("Class")

plt.legend()

plt.show()