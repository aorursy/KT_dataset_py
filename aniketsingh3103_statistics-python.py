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
import numpy as np

a = [29.5,49.3,30.6,28.2,28.0,26.3,33.9,29.4,23.5,31.6]



print(np.std(a))

print(np.var(a))
x = [116.4,115.9,114.6,115.2,115.8]

x= [i+100 for i in x]

mean = 0



for i in x:

    mean=mean+i

mean=mean/len(x)

print("Mean="+str(mean))



stddev = 0

for i in x:

        stddev = stddev+(np.square(i-mean))

stddev = stddev/(len(x)-1)

print("Standerd Devation="+str(np.absolute(stddev)))

print(np.std(x))

print(np.mean(x))

print(np.var(x))