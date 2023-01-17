# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

# Load dataset

# url = "https://goo.gl/sXleFv"

# names = ['CRIM' , 'ZN' , 'INDUS' , 'CHAS' , 'NOX' , 'RM' , 'AGE' , 'DIS' , 'RAD' , 'TAX' , 'PTRATIO' ,

# 'B' , 'LSTAT' , 'MEDV' ]

# dataset = pd.read_csv(url, delim_whitespace=True, names=names)

# dataset.shape



import pandas as pd

from pandas import Series,DataFrame

import numpy as np



# For Visualization

import matplotlib.pyplot as plt

import matplotlib

#%matplotlib inline



matplotlib.style.use('ggplot')

df=pd.read_csv('../input/indicators_by_company.csv')
df.head()
df.describe()
df.info()