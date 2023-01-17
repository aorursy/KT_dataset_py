# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import seaborn as sb

sb.set()
# this is a cell
#this is another cell



Advertising = pd.read_csv("../input/ISLR-Auto/Advertising.csv")

Auto = pd.read_csv("../input/ISLR-Auto/Auto.csv")

Ch10Ex11 = pd.read_csv("../input/ISLR-Auto/Ch10Ex11.csv")

College = pd.read_csv("../input/ISLR-Auto/College.csv")

Credit = pd.read_csv("../input/ISLR-Auto/Credit.csv")

Heart = pd.read_csv("../input/ISLR-Auto/Heart.csv")

Income1 = pd.read_csv("../input/ISLR-Auto/Income1.csv")

Income2 = pd.read_csv("../input/ISLR-Auto/Income2.csv")


# James Cell

trimmed = Income2.drop('Unnamed: 0',1)

print(trimmed.head())

# incomePlot = sb.load_dataset("../input/ISLR-Auto/Income1.csv")

sb.lmplot(x="Income", y="Education", data=trimmed)

fig = plt.figure()

# ax = plt.axes(projection='3d')

# ax.scatter3D(trimmed["Education"],trimmed["Seniority"], trimmed["Income"],)
#james 

#Gonna try and make a bunch of sub plots

trimcred = Credit.drop(['Unnamed: 0','Student','Married'],1)

# print(trimcred[trimcred.columns[0]])

fig, ax = plt.subplots(9,9)

# fig.axis('off')

fig.set_figwidth(20)

fig.set_figheight(20)

for i in range(9):

    for j in range(9):

        if(i==j):

            ax[i,j].text(0.5,0.5,trimcred.columns[i], ha='center')

            ax[i,j].set_xticks([])

            ax[i,j].set_yticks([])

        else:

            ax[i,j].scatter(trimcred[trimcred.columns[i]],trimcred[trimcred.columns[j]],data=trimcred)

            ax[i,j].set_xticks([])

            ax[i,j].set_yticks([])

###Garrett

###I'm just here so I don't get fined



###Calculate the acceptance rate and add it to the data frame

rate=College['Accept']/College['Apps']

College['A-Rate']=rate



###Make sure all Private answers are exactly yes and no

if np.all(np.logical_or(College['Private']=='Yes', College['Private']=='No')):

    print('We good!')

else:

    print('Noooooooo')



###I feel like there is probably a much better way to go about doing this, but it's what I know at the moment

priv_ind=np.where(College['Private']=='Yes')[0]

print(priv_ind[:20])

pub_ind=np.where(College['Private']=='No')[0]

print(pub_ind[:20])

sb.distplot(College['A-Rate'][priv_ind], label='Private')

sb.distplot(College['A-Rate'][pub_ind], label='Public')

plt.legend()

plt.show()