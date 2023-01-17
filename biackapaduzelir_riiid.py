# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/riiid-test-answer-prediction/lectures.csv")
df.head()
df.info()
df.corr()
f,ax = plt.subplots(figsize=(10,10))
sns.heatmap(df.corr(),annot=True, linewidths=.5,fmt=".1f",ax=ax)
plt.show()
df.columns
df.lecture_id.plot(kind='line', color='r', label='lecture_id', linewidth=1, alpha=1, grid=True, linestyle=':')
df.part.plot(color='b', label='part', linewidth=1, alpha=1, grid=True, linestyle='--')
plt.legend(loc='best')
plt.xlabel('x axis') #lecture_id
plt.ylabel('y axis') #part
plt.show()
df.lecture_id.plot(kind='hist',bins=50,figsize=(15,15))
df.describe()