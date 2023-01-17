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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/titanic/train_and_test2.csv')

df.head(10)
r,c = df.shape

print("Rows    = ",r)

print("Columns = ",c)
df.info()
print("Null values ?",df.isnull().values.any())
df = df.dropna()
df
print(df.isnull().values.any())
plt.figure(figsize=(20,7))

plt.xlabel("Age")

plt.ylabel("Emarked")

plt.xticks(rotation=90)

sns.barplot(df['Age'],df['Embarked'])
plt.figure(figsize=(20,7))

plt.bar(df['Age'],df['2urvived'])