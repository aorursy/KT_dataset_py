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
df=pd.read_csv(r'/kaggle/input/indian-food-101/indian_food.csv')
df.head()
df.info()
df.region.value_counts()
temp1=df.pivot_table(values='name',index=['region'],columns='diet', aggfunc = 'count')
temp1=temp1[1:]
fig=plt.figure(1)
# ax1=fig.add_subplot()
ax1.set_xlabel('region')
ax1.set_ylabel('no of dishes')
temp1.plot(kind='bar',stacked=True, color=['red','green'],grid=False)
temp2=df.pivot_table(values='name',index=['region'],columns='course', aggfunc = 'count')
temp2=temp2[1:]
fig=plt.figure(1)
# ax1=fig.add_subplot()
ax1.set_xlabel('region')
ax1.set_ylabel('no of dishes')
temp2.plot(kind='bar',stacked=True, grid=False)
temp3=df.pivot_table(values='name',index=['region'],columns='flavor_profile', aggfunc = 'count')
temp3=temp3.iloc[1:,1:]
fig=plt.figure(1)
# ax1=fig.add_subplot()
ax1.set_xlabel('region')
ax1.set_ylabel('no of dishes')
temp3.plot(kind='bar',stacked=True, grid=False)
df.head()
temp4=df.pivot_table(values=['prep_time','cook_time'],index='course',aggfunc=np.mean)
temp4.plot(kind='bar',stacked=True)
temp5=df.pivot_table(values=['prep_time','cook_time'],index='flavor_profile',aggfunc=np.mean)
temp5=temp5[1:]
temp5.plot(kind='bar',stacked=True)
# df1=df[df.flavor_profile!=-1]
# df1=df1[df1.prep_time!=-1]
# df1[df1.prep_tim==-1]
temp6=df.pivot_table(values=['prep_time','cook_time'],index=['course','flavor_profile'],aggfunc=np.mean)
# temp6=temp6[temp6.flavor_profile!=-1]
temp6.plot(kind='bar',stacked=True)
df.head()
temp7=df.pivot_table(values=['prep_time','cook_time'],index=['region','diet'],aggfunc=np.mean)
temp7=temp7[1:]
temp7.plot(kind='bar',stacked=True)