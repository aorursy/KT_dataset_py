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
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")
filepath = "/kaggle/input/heart-disease-uci/heart.csv"
heart_data = pd.read_csv(filepath , index_col="age" )
print("Loaded Data")
heart_data.head()
heart_data.tail()
plt.figure(figsize=(16, 6))
sns.lineplot(data = heart_data['chol'])
plt.figure(figsize=(16, 6))
sns.barplot(x=heart_data.index , y=heart_data['chol'])
plt.figure(figsize=(16, 6))
sns.lineplot(data = heart_data['cp'])
plt.figure(figsize=(16, 6))
sns.barplot(x=heart_data.index , y=heart_data['cp'])
plt.ylabel("Chest Pain")
plt.figure(figsize=(16, 6))
sns.barplot(x=heart_data.index , y=heart_data['trestbps'])
plt.figure(figsize=(16, 6))
sns.heatmap(data=heart_data , annot = True)
plt.figure(figsize=(16, 6))
sns.barplot(x=heart_data.sex , y=heart_data['cp'])
plt.ylabel("Chest Pain")