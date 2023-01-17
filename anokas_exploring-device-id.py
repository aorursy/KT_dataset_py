import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

df_train = pd.read_csv('../input/gender_age_train.csv')
print(df_train.head())
plt.figure(figsize=(13,8))
plt.hist2d(df_train['device_id'].values, df_train['age'].values, bins=80)
plt.xlabel('Device ID', fontsize=15)
plt.ylabel('Age of user', fontsize=15)
plt.title('Distribution of ages based on device ID - 80 bins', fontsize=20)
plt.show()
gender_bin = (df_train['gender'] == 'M').tolist()

plt.figure(figsize=(13,5))
plt.hist2d(df_train['device_id'].values, gender_bin, bins=(50, 2))
plt.xlabel('Device ID', fontsize=15)
plt.ylabel('Female                   Male', fontsize=15)
plt.title('Distribution of gender based on device ID - 100 bins', fontsize=20)
plt.show()