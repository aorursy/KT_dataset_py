# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from fuzzywuzzy import fuzz, process
mtchStr = "Black & Decker Inc."

testStr = "Black + Decker incorporation"
print(fuzz.ratio(testStr.lower(), mtchStr.lower()))             # Exact match

print(fuzz.partial_ratio(testStr.lower(), mtchStr.lower()))     # Partial match

print(fuzz.token_sort_ratio(testStr.lower(), mtchStr.lower()))  # Tokenized + Sorted

print(fuzz.token_set_ratio(testStr.lower(), mtchStr.lower()))   # Tokenized sets