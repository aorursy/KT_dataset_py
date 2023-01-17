# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.linear_model import LogisticRegression
import pandas as pd
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")
#for y in range(1, 200):
 #   for x in range(1, 200):

%%load_ext cython