# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('../input/arrivlas/ARRIVAL_REPORT_1979-2020.xlsx'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_excel('../input/arrivlas/ARRIVAL_REPORT_1979-2020.xlsx')
df.shape

df.head()

import matplotlib.pyplot as plt
fig=plt.figure(figsize=(15,7))

fig.suptitle('GROWTH CHART comparison FRUITS',fontsize=20)



ax1=fig.add_subplot(231)

ax1.set_title('APPLE')

ax1.plot(df['YEAR'],df['APPLE'],color='green')



ax2=fig.add_subplot(232)

ax2.set_title('BANANA')

ax2.plot(df['YEAR'],df['BANANA'],color='purple')



ax3=fig.add_subplot(233)

ax3.set_title('MANGO')

ax3.plot(df['YEAR'],df['MANGO'],color='magenta')



ax4=fig.add_subplot(234)

ax4.set_title('ORANGE')

ax4.plot(df['YEAR'],df['ORANGE'],color='orange')



ax5=fig.add_subplot(235)

ax5.set_title('FRUIT')

ax5.plot(df['YEAR'],df['FRUIT'],color='teal')



ax6=fig.add_subplot(236)

ax6.set_title('OTHER')

ax6.plot(df['YEAR'],df['OTHER'],color='chocolate')
fig=plt.figure(figsize=(15,7))

fig.suptitle('GROWTH CHART for comparison Vegetables',fontsize=20)



ax1=fig.add_subplot(231)

ax1.set_title('POTATO')

ax1.plot(df['YEAR'],df['POTATO'],color='green')



ax2=fig.add_subplot(232)

ax2.set_title('ONION')

ax2.plot(df['YEAR'],df['ONION'],color='purple')



ax3=fig.add_subplot(233)

ax3.set_title('TOMATO')

ax3.plot(df['YEAR'],df['TOMATO'],color='magenta')



ax4=fig.add_subplot(234)

ax4.set_title('PEAS')

ax4.plot(df['YEAR'],df['PEAS'],color='orange')



ax5=fig.add_subplot(235)

ax5.set_title('VEG')

ax5.plot(df['YEAR'],df['VEG'],color='teal')



ax6=fig.add_subplot(236)

ax6.set_title('OTHER.1')

ax6.plot(df['YEAR'],df['OTHER'],color='chocolate')
df.head()
df_fruits=df[['YEAR','APPLE','BANANA','MANGO','ORANGE']].head(30)
df_fruits
fig, ax_tempF=plt.subplots()

fig.set_figwidth(12)

fig.set_figheight(6)



ax_tempF.set_xlabel('COMPARISON BETWEEN BANANA AND APPLE')

ax_tempF.tick_params(axis='x',bottom=False,labelbottom=False)

ax_tempF.set_ylabel('ARRIVAL IN TONS FOR APPLE', color='red', size='x-large')

ax_tempF.tick_params(axis='y',labelcolor='red',labelsize='large')

ax_tempF.plot(df_fruits['YEAR'],df_fruits['APPLE'],color='red')



ax_precip=ax_tempF.twinx()

ax_precip.set_ylabel('BANANA ARRIVALS IN TONS',color='blue',size='x-large')



ax_precip.tick_params(axis='y',labelcolor='orange',labelsize='large')

ax_precip.plot(df_fruits['YEAR'],df_fruits['BANANA'],color='blue')



df_fruits.head()
x=df_fruits['YEAR']
y=np.vstack([df_fruits['APPLE'],

            df_fruits['BANANA'],

            df_fruits['MANGO'],

            df_fruits['ORANGE']])
labels=['Apple','Banana','Mango','Orange']

colors=['sandybrown','tomato','skyblue','blue']

plt.stackplot(x,y,labels=labels,colors=colors,edgecolor='black')

plt.legend(loc=2)

plt.show()
from mpl_toolkits.mplot3d import Axes3D

fig=plt.figure(figsize=(10,6))

ax=fig.add_subplot(111,projection='3d')

ax.scatter(df_fruits['YEAR'],df_fruits['APPLE'],df_fruits['BANANA'],s=50)

plt.show()