# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
Train=pd.read_csv('../input/train.csv')
Train.head()
DF=Train['LotFrontage']

DF1=Train['LotArea']

plt.plot(DF, color='purple')

plt.xlabel('Lotfrontage')

plt.ylabel('Frequencies')

plt.title('Bar plot of houses Lot Frontage Data')

plt.show()
plt.hist(DF1, color='green')

plt.xlabel('LotArea')

plt.ylabel('Frequencies')

plt.title('Histogram of houses LotArea Data')

plt.show()