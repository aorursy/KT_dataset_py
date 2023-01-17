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

import pandas as pd

customerrecord = pd.read_csv("../input/customerrecord.csv")

customerrecord2 = pd.read_csv("../input/customerrecord2.csv")
#print(customerrecord)

print(customerrecord2)
print(customerrecord.info())
print(customerrecord2.info())


print(customerrecord.shape)

print(customerrecord2.shape)
set(customerrecord.columns).intersection(set(customerrecord2.columns))
print(customerrecord.isnull().sum().sum())

print(customerrecord2.isnull().sum().sum())

print(customerrecord['Rehire Date'].isnull().sum())
#s1 = pd.merge(customerrecord, customerrecord2, how='outer', on=['First Name','Last Name','Country','DOB'])

#print(s1.where['DOB'].isnull)



result = pd.merge(customerrecord,

                 customerrecord2[['First Name', 'Last Name', 'DOB']],

                 on='DOB', 

                 how='outer', 

                 indicator=True)

print(result)