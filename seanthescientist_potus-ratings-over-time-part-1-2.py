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
import matplotlib as mpl

import matplotlib.pyplot as plt

import pandas as pd

import pylab as pl

import numpy as np

%matplotlib inline
pi=pd.read_csv('/kaggle/input/pollnumbers/Party_ID.csv')

pi.head()
japp=pd.read_csv('/kaggle/input/pollnumbers/Job_approv.csv')

japp.head()
pi.shape
japp.shape
pi.head()
pi.describe()
#pi.set_index('Years', inplace=True) need first to move from column to row index

pi.head()

pi.set_index('Years', inplace=True)
pi.plot(kind='line')



plt.title('POTUS Job Approval by Party Identification')

plt.ylabel('Approval Percentage')

plt.xlabel('Years')



plt.show()
fig_size = plt.rcParams["figure.figsize"]

 

print ("Current size:", fig_size)
fig_size[0] = 12

fig_size[1] = 9

plt.rcParams["figure.figsize"] = fig_size
pi.plot(kind='line')



plt.title('POTUS Trump Political Approval')

plt.ylabel('Approval Percentage')

plt.xlabel('Years')



plt.show()
japp.head()
japp_t=japp.set_index('Years', inplace=True)
japp.plot(kind='line')



plt.title('POTUS Trump General Approval')

plt.ylabel('Approval Percentage')

plt.xlabel('Years')



plt.show()
bogen=pd.read_csv('../input/pastpotusratings/ObamaGeneralRatings.csv')

bogen.head()
bogen.describe()
bogen.plot()
bogen.set_index('Years', inplace=True)
bogen.plot(kind='line')



plt.title('POTUS Obama General Approval')

plt.ylabel('Approval Percentage')

plt.xlabel('Years')



plt.show()
obpart=pd.read_csv('../input/pastpotusratings/ObamaRatingParty.csv')

obpart.head()
obpart.describe()
obpart.shape
obpart.plot()
obpart.set_index('Years', inplace=True)
obpart.plot(kind='line')



plt.title('POTUS Obama Political Party Approval')

plt.ylabel('Approval Percentage')

plt.xlabel('Years')

plt.show()
wbushgen=pd.read_csv('../input/pastpotusratings/WBushGenRatings.csv')

wbushgen.head()
wbushgen.describe()
wbushgen.shape
wbushgen.plot()
wbushgen.set_index('Years', inplace=True)
wbushgen.plot(kind='line')



plt.title('POTUS W. Bush General Approval')

plt.ylabel('Approval Percentage')

plt.xlabel('Years')



plt.show()
wbushrat=pd.read_csv('../input/pastpotusratings/WBushRatingParty.csv')

wbushrat.head()
wbushrat.describe()
wbushrat.plot()
wbushrat.set_index('Years', inplace=True)
wbushrat.plot(kind='line')



plt.title('POTUS W Bush Political Party Approval')

plt.ylabel('Approval Percentage')

plt.xlabel('Years')



plt.show()
bclingen=pd.read_csv('../input/pastpotusratings/BClintonGenRatings.csv')

bclingen.head()
bclingen.describe()
bclingen.shape
bclingen.plot()
bclingen.set_index('Years', inplace=True)
bclingen.plot(kind='line')



plt.title('POTUS Clinton General Approval')

plt.ylabel('Approval Percentage')

plt.xlabel('Years')



plt.show()
bclinpol=pd.read_csv('../input/pastpotusratings/BClintonRatingParty.csv')

bclinpol.head()
bclinpol.shape
bclinpol.describe()
bclinpol.plot()
bclinpol.set_index('Years', inplace=True)
bclinpol.plot(kind='line')



plt.title('POTUS Clinton Political Party Approval')

plt.ylabel('Approval Percentage')

plt.xlabel('Years')



plt.show()