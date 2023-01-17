# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
sns.set_style('whitegrid')
plt.figure(figsize=(6,5))

x=range(1,8)

y= [2,3,4,1,0,7,6]

plt.plot(x,y)

plt.title('Lineplot')

plt.xlabel("Numbers")

plt.ylabel("Frequency")

plt.show()
cars=pd.read_csv("../input/praactice-data/mt1cars.csv")

cars.head()
cars.rename(columns={'Unnamed: 0': 'cars_name'}, inplace=True)
mpg=cars['mpg']

mpg.plot()
df= cars[['cyl','wt','mpg']]

df.plot()
plt.bar(x,y)
plt.figure(figsize=(8,7))

mpg.plot(kind='bar', color = 'g')
plt.figure(figsize=(8,7))

mpg.plot(kind='barh', color = 'r')
x=[1,2,3,4,5]

plt.pie(x)

plt.show()
plt.savefig('pie_chart.jpeg')

plt.show()
%pwd
x=range(1,10)

y= [1,2,8,9,3,4,6,3,8]

fig, ax = plt.subplots(figsize=(6,6))

ax.plot(x,y)
fig, ax = plt.subplots(figsize=(6,6))

ax.set_xlim([1,9])

ax.set_ylim([0,5])

ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10])

ax.set_yticks([0,2,4,6,8,10])

ax.plot(x,y)
fig=plt.figure(figsize=(10,6))

fig, (ax1, ax2) = plt.subplots(1,2)

ax1.plot(x)

ax2.plot(x,y)
plt.bar(x,y)
wide=[0.5, 0.5, 0.5, 0.7, 0.7, 0.5, 0.5, 0.7, 0.8]

color=['salmon']

plt.bar(x,y, width=wide, color=color, align = 'center')
df= cars[['cyl','wt','mpg']]

color_t=['darkgray','lightsalmon', 'powderblue']

df.plot(color=color_t)
z = [2,4,6,8,10]

color_theme = ['#A9A9A9', '#FFA07A','#B0E0E6','#FFE4C4','#BDB76B']

plt.pie(z, colors=color_theme)

plt.show()
x1=range(0,10)

y1=[10,9,8,7,6,5,4,3,2,1]

plt.plot(x,y, ls='steps', lw=10)

plt.plot(x1,y1, ls='--', lw=5)
plt.plot(x,y, marker='1', mew=20)

plt.plot(x1,y1, marker='+', mew=15)
x2=range(1,10)

y2=(1,2,3,4,5,7,8,9,10)

plt.bar(x,y)

plt.xlabel('x_axis')

plt.ylabel('y_axis')

plt.show()
z2 = [2,3,4,5,6]

fruit=['fig', 'mango','apple', 'coco', 'tamarind']

plt.pie(z2, labels=fruit)


fig, ax = plt.subplots(1,1, figsize=(15,10))

ax.set_xticks(range(32))

mpg.plot()

ax.set_xticklabels(cars.cars_name , rotation = 60, fontsize= 'medium')

ax.set_title("mpg of cars")

ax.set_xlabel('car names')

ax.set_ylabel('m-p-g')

ax.legend(loc='best')

plt.show()
plt.pie(z2)

plt.legend(fruit, loc='best')

plt.show()
mpg.max()
fig, ax = plt.subplots(1,1, figsize=(15,10))

mpg.plot()

ax.set_ylim([0,45])

ax.set_title("mpg of cars")

ax.set_xlabel('car names')

ax.set_ylabel('m-p-g')

ax.annotate('Toyota Corolla', xy=(19,33.9), xytext=(21,35), arrowprops=dict(facecolor='black', shrink = 0.05))
mpg.plot(kind='hist')
plt.hist(mpg)

plt.show()
sns.distplot(mpg)
cars.plot(kind='scatter', x= 'hp', y ='mpg', color ='g', s=150 )
sns.regplot(x='hp', y = 'mpg', data=cars)
sns.pairplot(cars)
cars_df=cars[['mpg', 'disp', 'hp', 'wt']]

cars_df.values
cars_target= cars[['am']]

cars_target.values

target_names=[0,1]
from pandas import Series
cars_df['group']=pd.Series(cars_target, dtype='category')

sns.pairplot(cars_df, hue='group', palette='hls')
cars.boxplot(column='mpg', by='am')

cars.boxplot(column='wt', by='am')
sns.boxplot(x='am', y='mpg', data=cars)