from sklearn.neighbors import KNeighborsClassifier

from sklearn.cross_validation import train_test_split

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

%matplotlib inline
from subprocess import check_output

print(check_output["ls","../input/properties_2016.csv"]).decode("utf8")
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#import numpy as np # linear algebra

#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
mydata = pd.read_csv("../input/properties_2016.csv.zip")

mydata.head()