# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np 

import pandas as pd 

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



groundhog=pd.read_csv('../input/Groundhog.csv')







# Any results you write to the current directory are saved as output.
groundhog.describe()
groundhog.head()
cols = groundhog.columns

cols = cols.map(lambda x: x.replace(' ', '_') if isinstance(x, (str, unicode)) else x)

groundhog.columns = cols