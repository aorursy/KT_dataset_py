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
import os

import csv

import sys



#Read arg to check directory

#if len(sys.argv) > 1:

#   path= sys.argv[1]

#else:



path = '../input'   

trpath= os.path.join(path,"properties_2016.csv")







#load

kgl_housing = pd.read_csv(trpath)

#Read top 5 rows

kgl_housing.head()
kgl_housing.info()
kgl_housing.describe()
#only in Jupyter notebook

%matplotlib inline    

import matplotlib.pyplot as plt

kgl_housing.hist(bins=50,figsize=(20,15))

plt.show()