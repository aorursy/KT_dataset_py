# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/data 4"))



# Any results you write to the current directory are saved as output.


import pandas as pd



data = pd.read_csv('../input/data 4/hbp.txt', sep=">",header=None)



sequences=data[0].dropna()

labels=data[1].dropna()





sequences.reset_index(drop=True, inplace=True)

labels.reset_index(drop=True, inplace=True)

list_of_series=[sequences.rename("sequences"),labels.rename("labels")]

df = pd.concat(list_of_series, axis=1)

df.head()