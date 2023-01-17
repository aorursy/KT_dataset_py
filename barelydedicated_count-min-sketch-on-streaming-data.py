# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



%matplotlib inline

np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def misha_gries(file, k):    

    with open(file, 'r') as data:

        data = data.read().replace('\n', '')

        currently_used = 0

        locations = {}

        for i in range(len(data)):

            if data[i] in locations:

                locations[data[i]] = locations[data[i]]+1

            else:

                if 0 in locations.values() or len(locations) < k-1: 

                    for key,value in locations.items():

                        if value is 0:

                            locations.pop(key)

                            break

                    locations[data[i]] = 1

                elif data[i] not in locations:

                    for key, value in locations.items():

                        locations[key] = value - 1

    print(file, ': ', locations)
misha_gries("../input/S1.txt",10)
misha_gries("../input/S2.txt",10)
def count_min_sketch(file,k,t, query= ''):

    with open(file, 'r') as data:

        data = data.read().replace('\n', '')

        table = np.zeros((t,k))

        for i in range(len(data)):

            for j in range(t):

                table[j][(hash(str(j)+data[i])%k)] = table[j][(hash(str(j)+data[i])%k)]+1

        if query is not '':        

            min = math.inf

            for j in range(t):

                if table[j][(hash(str(j)+query)%k)] < min:

                    min = table[j][(hash(str(j)+query)%k)]

            return min
count_min_sketch("../input/S1.txt",10,5,'a')
count_min_sketch("../input/S1.txt",10,5,'b')
count_min_sketch("../input/S1.txt",10,5,'c')
count_min_sketch("../input/S2.txt",10,5,'a')
count_min_sketch("../input/S2.txt",10,5,'b')
count_min_sketch("../input/S2.txt",10,5,'c')