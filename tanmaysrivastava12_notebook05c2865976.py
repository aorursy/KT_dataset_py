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
df = pd.read_csv("/kaggle/input/heart-disease-dataset-from-uci/HeartDisease.csv")
df
df['Age'].median()
df.median()
df.mean()
df.mode()
n = list(df._get_numeric_data().columns)
n
categorical = list(set(df.columns) - set(df._get_numeric_data().columns))
categorical
# Visual on basis of gender and age of students
T = df.groupby(['Sex', 'Age']).size().unstack()
T.plot(kind='bar', figsize=(15,10))
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt
#correlation
correlation=df.corr()
df.plot.bar()
df.plot.hist()
df.plot.line()
import seaborn as sns
plt.figure(figsize=(8,8))
sns.heatmap(correlation,annot=True, cmap="Reds")
plt.title('Correlation Heatmap',fontsize=10)
df.where(Age.values==Sex.values).notna()
