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
train = pd.read_csv("/kaggle/input/tmdb-box-office-prediction/train.csv")
example = train.cast[1]

print(example)

print("hej")



#for i in range(train.cast.shape[0])

#    example = train.cast[i]

#    str = (example.split('\'id\':'))[1].split(',')[0]

#    int(str)

##print((s.split(start))[1].split(end)[0])



print('')

print(int((train.cast[1].split('\'id\':'))[1].split(',')[0]))

print((train.cast[1].split('\'name\': \''))[1].split('\',')[0])
print("Hello")
print("version 7? from carol")
print("version 8")
print("version 10")