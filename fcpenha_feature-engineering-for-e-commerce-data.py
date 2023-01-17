# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Create a dictionary that contains a Pandas Dataframe for each label in the index

data = {}

for k in os.listdir("../input"):
    
    address = ('../input/' + k)
    
    data[k] = pd.read_csv(address, sep=',')
for k in os.listdir("../input"):
    
    print('')
    
    print(k)
    
    display(
        data[k].\
        sample(5, random_state=43)
    )
