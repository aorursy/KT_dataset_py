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
import pandas as pd

Data_Cortex_Nuclear = pd.read_csv("../input/mice-protein-expression/Data_Cortex_Nuclear.csv")

sample = Data_Cortex_Nuclear.head(50)
def euclidean_distance(p1,p2):

    dist = 0

    for i in range(p1):

        dist += (p1[i]-p2[i])**2

    return dist**(1/2)
# Create the function that generates a gaussian with mean and standard deviation as input

def gaussian_generation_2d(nb_points, mean, std, start=-20, stop=20):

    interval = stop - start

    step = interval/nb_points

    x = np.array(np.arange(start,stop,step))

    y = np.zeros(nb_points)

    for i in range(nb_points):

        y[i] = (1/(std*np.sqrt(2*np.pi)))*np.exp((-1/2)*((x[i]-mean)/std)**2)

    return x,y
x,y = gaussian_generation_2d(100,[5,2],0.2)