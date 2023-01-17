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
#libraries to import 

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from scipy.integrate import odeint

from pylab import *

from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

ff = pd.read_csv("../input/fifa19/data.csv")
ff
pp = ff.Potential

pa = ff.Age



plt.hist(pa, bins=30,color=list(plt.rcParams['axes.prop_cycle'])[2]['color'] )

plt.ylabel('Frequency')

plt.xlabel('Age')

plt.title('Fifa age Ranges')

plt.show()
plt.hist(pp, bins=30,color=list(plt.rcParams['axes.prop_cycle'])[2]['color'] )

plt.ylabel('Frequency')

plt.xlabel('Potential')

plt.title('Fifa Potential Frequencies')

plt.show()
left = ff.loc[ff['Preferred Foot'] == 'Left'].count()[0]

right = ff.loc[ff['Preferred Foot'] == 'Right'].count()[0]



explode = (.2, .1)

labels = ['Left', 'Right']

plt.title('Preferred Foot')

plt.pie([left, right], labels = labels, autopct='%.1f%%', explode =explode)



plt.show()
plt.hist2d(pa, pp, bins=30, cmap=plt.cm.BuPu)

cb = plt.colorbar()

plt.ylabel('Potential')#'Overall Performance')

plt.xlabel('Age')#'Age')

cb.set_label('counts in bin')
plt.title('Nationality Age Comparisons')



#plt.style.use('default')



eng =  ff.loc[ff.Nationality == 'England']['Age']

bra = ff.loc[ff.Nationality == 'Brazil']['Age']



labels = ['England' ,'Brazil']



plt.boxplot([eng, bra], labels = labels)

plt.ylabel('Age')

plt.xlabel('Nationality')



plt.show()