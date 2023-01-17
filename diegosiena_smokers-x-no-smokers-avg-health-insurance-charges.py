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


from sklearn.model_selection import train_test_split
df = pd.read_csv("/kaggle/input/insurance/insurance.csv")
print(df.head())
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
print(df_train.head())
print(df_train['age'].mean())
from matplotlib import pyplot as plt
plt.scatter(df_train[df_train['smoker'] == 'no']['age'], df_train[df_train['smoker'] == 'no']['charges'], color='blue')
plt.scatter(df_train[df_train['smoker'] == 'yes']['age'], df_train[df_train['smoker'] == 'yes']['charges'], color='red')
# plt.scatter(df[df['sex'] == 'male']['age'], df[df['sex'] == 'male']['charges'], color='purple')
# plt.scatter(df[df['sex'] == 'female']['age'], df[df['sex'] == 'female']['charges'], color='yellow')
plt.plot(df_train[df_train['smoker'] == 'no']['age'], [df_train[df_train['smoker'] == 'no']['charges'].mean()] * len(df_train[df_train['smoker'] == 'no']), '-r', color='blue')
plt.plot(df_train[df_train['smoker'] == 'yes']['age'], [df_train[df_train['smoker'] == 'yes']['charges'].mean()] * len(df_train[df_train['smoker'] == 'yes']), '-r', color='red')
plt.show()
