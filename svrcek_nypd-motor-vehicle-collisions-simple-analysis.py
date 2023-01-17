import os
# print(os.listdir)
print(os.listdir("../input"))
%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files 
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.
nypd_crash = pd.read_csv("../input/nypd-motor-vehicle-collisions.csv", low_memory=False)
nypd_crash.columns
len(nypd_crash)
nypd_crash.info()
nypd_crash.head(10)
nypd_crash.isnull().sum()
nypd_crash.head(5)
print("TOTAL NUMBER OF KILLED BY CAR:")
killedStats = nypd_crash['NUMBER OF PERSONS KILLED'].value_counts(dropna = False)
killedStats.plot(kind='bar', figsize=(8,8))
plt.ylabel("Total No. of Killed")
plt.xlabel("QUANTITY")
plt.show();
# CROSS STREET NAMES
plt.figure(figsize=(12,6))
sns.countplot(nypd_crash['CROSS STREET NAME'])
plt.title('CROSS STREET Distribution')
plt.show()
# count of INJURED PEDESTRIANS
PEDESTRIANS= nypd_crash['NUMBER OF PEDESTRIANS INJURED'].value_counts() 

plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
PEDESTRIANS[:10].plot.bar()
plt.title('Count for INJURED PEDESTRIANS')
plt.ylabel('Quantity')      
plt.show();
fig, ax = plt.subplots(figsize=(14, 10))
(nypd_crash[['UNIQUE KEY', 'NUMBER OF PERSONS INJURED']].groupby(['NUMBER OF PERSONS INJURED'])['UNIQUE KEY'].nunique().sort_values()).plot.barh(ax=ax)
plt.title('Events by INJURED PERSONS')
plt.show()
