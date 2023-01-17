# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import math

# Any results you write to the current directory are saved as output.



import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import matplotlib.ticker as ticker



# Rounding the integer to the next hundredth value plus an offset of 100

def roundup(x):

    return 100 + int(math.ceil(x / 100.0)) * 100 



# Some random data

dfWIM = pd.DataFrame({'AXLES': np.random.normal(8, 2, 5000).astype(int)})

ncount = len(dfWIM)



plt.figure(figsize=(12,8))

ax = sns.countplot(x="AXLES", data=dfWIM, order=[3,4,5,6,7,8,9,10,11,12])

plt.title('Distribution of Truck Configurations')

plt.xlabel('Number of Axles')



# Make twin axis



ax.yaxis.tick_right()





ax.set_ylabel('Frequency')



ax.set_ylim(0,ncount)



    # Iterate through the list of axes' patches

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), 

            fontsize=12, color='red', ha='center', va='bottom')



plt.show()
