# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#load packages

import sys #access to system parameters https://docs.python.org/3/library/sys.html

print("Python version: {}". format(sys.version))



import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features

print("pandas version: {}". format(pd.__version__))



import matplotlib #collection of functions for scientific and publication-ready visualization

print("matplotlib version: {}". format(matplotlib.__version__))



import numpy as np #foundational package for scientific computing

print("NumPy version: {}". format(np.__version__))





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
pdf_train = pd.read_json("/kaggle/input/whats-cooking-kernels-only/train.json")

pdf_test = pd.read_json("/kaggle/input/whats-cooking-kernels-only/test.json")
pdf_train.head()
pdf_test.head()
total = pdf_train.isnull().sum().sort_values(ascending = False)

percent = (pdf_train.isnull().sum()/pdf_train.isnull().count()*100).sort_values(ascending = False)

missing_train_data  = pd.concat([total, percent], axis=1, keys=['Total missing', 'Percent missing'])

missing_train_data.head(20)
type(pdf_train.ingredients[0])
def print_iterator(it):

    for x in it:

        print(x, end=' ')

    print('')  # for new line
print_iterator(map(lambda x : x*2, [1, 2, 3, 4]))
feature_set  = set()
set(map(lambda x : x*2, [1, 1, 3, 1])).union(feature_set)
pdf_train.ingredients[0:3]
#pdf_train.ingredients[0:3].apply(lambda x: map(lambda x: x.replace('[','').replace(']','').split(','), x)).apply(lambda y: print_iterator(y))

# no need to do this as it's a list and not a string
# the above code cannot work with Series

pdf_train.ingredients[0:3].apply(lambda x: print(x))
# adding all the ingredients to a singular set to find unique ingredients in the data set

pdf_train.ingredients.apply(lambda x: feature_set.update(set(x)))
len(feature_set)
# creating columns for each feature_set

for val in feature_set: 

    pdf_train[val] = 0
def assign_onehot_encoding(x):

    for val in x:

        pdf_train[x] = 1
# setting appropriate values in each row to 1

pdf_train.ingredients.apply(lambda x: assign_onehot_encoding(x))
pdf_train.head(3)