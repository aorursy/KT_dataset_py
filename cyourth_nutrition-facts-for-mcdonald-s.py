# 

# Python environment check

# 



%matplotlib inline



import sys

import numpy as np

import pandas as pd

import scipy as sp

import matplotlib as mpl

import statsmodels

import sklearn as skl

import nltk

import gensim



print("python:", sys.version)

print("numpy:", np.__version__)

print("pandas:", pd.__version__)

print("scipy:", sp.__version__)

print("matplotlib:", mpl.__version__)

print("statsModels:", statsmodels.__version__)

print("scikit-learn:", skl.__version__)

print("nltk:", nltk.__version__)

print("gensim:", gensim.__version__)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.