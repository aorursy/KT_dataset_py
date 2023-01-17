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
#Load Libraries

import warnings

warnings.filterwarnings('ignore')



%matplotlib inline



import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn import linear_model

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor



from sklearn import metrics

from sklearn import pipeline
#Load the dataset

data_math = pd.read_csv('../input/student-mat.csv')

#data_por = pd.read_csv('../input/student-por.csv')
#Load at the peek dataset

data_math.head()
#dimensions of the dataset

data_math.shape
data_math.dtypes
#statstical summary

from pandas import set_option

set_option('precision', 3)

data_math.describe()
#Pairwise pearson correlation between dataset

data_math.corr(method='pearson')
#skweness 

data_math.skew()
# kurtosis

data_math.kurt()