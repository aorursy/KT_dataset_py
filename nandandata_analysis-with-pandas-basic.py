# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

sns.set();
autos = pd.read_csv('../input/autos.csv', encoding='ISO--8859-1')
autos.head()
autos.head(2)   #to get first 2 rows
autos.tail(2)   #to get last 2 rows
type(autos)   #check type of input file 
autos.shape    # (a,b)  here a give rows and b gives columns count 
autos.columns   #to check out columns name 
autos.dtypes   #to check column name with data type
autos.describe     #describe the dataframe in full details
autos.info   # same as Describe commands , give informations about columns , rows ,