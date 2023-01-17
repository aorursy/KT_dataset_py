# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful apackages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



data=pd.read_table('../input/auto.txt',delim_whitespace=True,header=None,names=['mpg','cylinders','displacement','horsepower','weight','acceleration','modelYear','origin','carName'])





# Any results you write to the current directory are saved as output.
data