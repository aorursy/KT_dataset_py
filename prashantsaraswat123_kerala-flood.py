# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
img = np.array(Image.open('../input/kerala-map/kerala_map.png'))
fig=plt.figure(figsize=(25,25))
plt.imshow(img,interpolation='bilinear')
plt.axis('off')
plt.ioff()
plt.show()
df = pd.read_csv("../input/rainfall-in-india/rainfall in india 1901-2015.csv")
df.head(10)
df.info()
df.isnull().sum()
df['JAN'].fillna((df['JAN'].mean()), inplace=True)
df['FEB'].fillna((df['FEB'].mean()), inplace=True)
df['MAR'].fillna((df['MAR'].mean()), inplace=True)
df['APR'].fillna((df['APR'].mean()), inplace=True)
df['MAY'].fillna((df['MAY'].mean()), inplace=True)
df['JUN'].fillna((df['JUN'].mean()), inplace=True)
df['JUL'].fillna((df['JUL'].mean()), inplace=True)
df['AUG'].fillna((df['AUG'].mean()), inplace=True)
df['SEP'].fillna((df['SEP'].mean()), inplace=True)
df['OCT'].fillna((df['OCT'].mean()), inplace=True)
df['NOV'].fillna((df['NOV'].mean()), inplace=True)
df['DEC'].fillna((df['DEC'].mean()), inplace=True)
df['ANNUAL'].fillna((df['ANNUAL'].mean()), inplace=True)
df['Jan-Feb'].fillna((df['Jan-Feb'].mean()), inplace=True)
df['Mar-May'].fillna((df['Mar-May'].mean()), inplace=True)
df['Jun-Sep'].fillna((df['Jun-Sep'].mean()), inplace=True)
df['Oct-Dec'].fillna((df['Oct-Dec'].mean()), inplace=True)
ax = df.groupby('YEAR').mean()['ANNUAL'].plot(ylim=(600, 2200), color='g', marker='o', linestyle='--', linewidth=2, figsize=(12, 9))
df['RAIN'] = df.groupby('YEAR').mean()['ANNUAL'].rolling(10).mean()
df.RAIN.plot(color='r', linewidth=4)
plt.xlabel('Year', fontsize=18)
plt.ylabel('Annual Rainfall (in mm)', fontsize=18)
plt.title('Annual Rainfall in India from Year 1901 to 2015', fontsize=23)
ax.tick_params(labelsize=15)
plt.grid()
plt.ioff()
df[['YEAR', 'Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']].groupby('YEAR').mean().plot(figsize=(12, 9))
plt.xlabel('YEAR', fontsize=18)
plt.ylabel('Seasonal Rainfall (in mm)', fontsize=18)
plt.title('Seasonal Rainfall from Year 1901 to 2015', fontsize=23)
ax.tick_params(labelsize=15)
plt.grid()
plt.ioff()
df[['SUBDIVISION', 'Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']].groupby('SUBDIVISION').mean().plot.bar(width=0.5, edgecolor='k',
                                                                                       align='center', stacked=True, figsize=(16, 9))
plt.xlabel('Subdivision', fontsize=22)
plt.ylabel('Rainfall (in mm)', fontsize=22)
plt.title('Rainfall in Subdivisions of India', fontsize=27)
ax.tick_params(labelsize=15)
plt.grid()
plt.ioff()
kerala = df[df['SUBDIVISION'] == 'KERALA']
kerala
ax = kerala.groupby("YEAR").mean()['ANNUAL'].plot(ylim=(1000,5000),color='g',marker='o',linestyle='-',linewidth=2,figsize=(12,9));
plt.xlabel('Year',fontsize=18)
plt.ylabel('Kerala Annual Rainfall (in mm)',fontsize=18)
plt.title('Kerala Annual Rainfall from Year 1901 to 2015',fontsize=23)
ax.tick_params(labelsize=15)
plt.grid()
plt.ioff()
