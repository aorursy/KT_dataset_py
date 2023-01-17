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
import matplotlib.pyplot as plt 

%matplotlib inline    

#only for jupyter notebook  
data = pd.Series([0,1,2,3,4,5])

plt.plot(data,data**2)
plt.plot(data,data**2)

plt.xlabel('X axis ')

plt.ylabel('Y axis ')
plt.plot(data,data**2)

plt.xlabel('X axis')

plt.ylabel('Y axis')

plt.title('Square of numbers')
plt.plot(data,data**3)

plt.show()
plt.scatter(data,data**2)
plt.scatter(data,data**2,alpha = 0.5,c ='Red')

plt.scatter(data,data**3,alpha = 0.3,c = 'Blue' )
#subplot(nrows,ncols,plot_number)

plt.subplot(1,2,1)

plt.plot(data,data**2)

plt.subplot(1,2,2)

plt.plot(data,data**3, color = 'Red')

fig = plt.figure()

ax = fig.add_axes([0.8,0.5,0.5,0.5])

ax.plot(data,data**2)  # left, bottom ,width,height

ax.set_xlabel('X')

ax.set_ylabel('Y')

ax.set_title('Square')
fig = plt.figure()

axes1 = fig.add_axes([0.8,0.8,0.8,0.8])

axes2 = fig.add_axes([0.9,1.2,0.3,0.3])



axes1.plot(data,data**2,color = 'Green')

axes1.set_xlabel('X')

axes1.set_ylabel('Y')

axes1.set_title('square')



axes2.plot(data,data**3,color = 'Red')

axes2.set_xlabel('X')

axes2.set_ylabel('Y')

axes2.set_title('Cube')



axes2.yaxis.labelpad = 2
fig , ax = plt.subplots(1,2, figsize=(12,4))  # here 1 is refer to row and 2 is refer to columns so 1 row and two columns for subplots

ax[0].plot(data,data**2)

ax[0].set_xlabel('X axis label for first subplot ')

ax[0].set_ylabel('Y axis label for first subplot')

ax[0].set_title('Square is the title for first subplot')



ax[1].plot(data,data**3)

ax[1].set_xlabel('X axis label for second subplot ')

ax[1].set_ylabel('Y axis label for second subplot')

ax[1].set_title('Cube is the title for second subplot')
# dpi --- dots per inch or pixel per inch

fig,ax = plt.subplots(figsize = (10,3),dpi = 100) 

# figsize = (width,height) and dpi = 100 dots per inch

ax.plot(data,data**2)
fig.savefig('figure1.png') # you can save figures in any extention for images like jpeg,jpg,.. etc.
x = pd.Series(np.arange(10))

y = x*3

fig, ax  = plt.subplots(2,2,figsize = (12,6))



for i in range(len(ax)):

    for j in range(len(ax)):

        if i == 0  and j == 0:

            ax[i][j].plot(x,y,color = 'green',label = 'x,y')

            ax[i][j].plot(x*2,y*3,color = 'Red',label = 'x*2,y*3')

            ax[i][j].set_xlabel('X')

            ax[i][j].set_ylabel('Y')

            ax[i][j].set_title('x,y ')

            ax[i][j].legend(['x,y','x*2,y*3'])

        elif i == 0 and j == 1:

            ax[i][j].plot(x**2,y*2,color = 'Blue',label = 'x**2,y*2')

            ax[i][j].plot(x**3,y*3,color = 'Orange',label = 'x**3,y*3')

            ax[i][j].set_xlabel('X')

            ax[i][j].set_ylabel('Y')

            ax[i][j].set_title('x,y ')

            ax[i][j].legend(['x**2,y*2','x**3,y*3'])

        elif i == 1 and j == 0:

            ax[i][j].plot(x*0.8,y*0.9,color = 'Purple',label = 'x*.8,y*.9')

            ax[i][j].plot(x*0.4,y*0.6,color = 'Yellow',label = 'x*.4,y*.6')

            ax[i][j].set_xlabel('X')

            ax[i][j].set_ylabel('Y')

            ax[i][j].set_title('x,y ')

            ax[i][j].legend(['x*.8,y*.9','x*.4,y*.6'])

        else:

            ax[i][j].plot(x*2.5,y*1.5,color = 'Purple',label = 'x*2.5,y*1.5')

            ax[i][j].plot(x*1.5,y*2.5,color = 'Yellow',label = 'x*1.5,y*2.5')

            ax[i][j].set_xlabel('X')

            ax[i][j].set_ylabel('Y')

            ax[i][j].set_title('x,y ')

            ax[i][j].legend(['x*2.5,y*1.5','x*1.5,y*2.5'])

fig.tight_layout()
x = pd.Series(np.arange(0,10))

y = x*2

z = x**2



fig, ax = plt.subplots(1,2,figsize = (12,6))

ax[0].plot(x,y, linewidth = 3,linestyle = '--',marker = 'o',markersize = 8,markeredgewidth = 4,

           markerfacecolor = 'yellow', markeredgecolor = 'green',color = 'blue')

ax[1].plot(x,z,lw = 3,ls = '-',color = 'Red',marker = 's',markersize = 8,markerfacecolor = 'yellow',

           markeredgewidth = 3,markeredgecolor = 'purple')
fig , ax = plt.subplots()

ax.plot(x,z,'r.-')



ax.set_xlim([0,20])

ax.set_ylim([0,100])
fig, ax = plt.subplots(1,3, figsize = (12,6))



ax[0].plot(x,z)

ax[0].plot(x,y)

ax[0].set_title('Default axis ranges')



ax[1].plot(x,z)

ax[1].plot(x,y)

ax[1].axis('tight')

ax[1].set_title('tight axis ')



ax[2].plot(x,z)

ax[2].plot(x,y)

ax[2].set_xlim([0,20])

ax[2].set_ylim([0,120])

ax[2].set_title('custom axis ranges')
plt.scatter(x,y)
plt.hist(x,z)
plt.boxplot(x)