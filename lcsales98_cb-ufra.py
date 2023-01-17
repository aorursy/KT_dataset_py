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
import pandas as pd

train_bc = pd.read_csv("../input/mydataset/data.csv")
train_bc = train_bc.drop(['compactness_mean', 'concavity_mean', 'concavity_mean',

                          'radius_se','perimeter_se','compactness_se','concavity_se',

                          'symmetry_se','fractal_dimension_se','radius_worst', 'symmetry_worst', 'Unnamed: 32'], axis=1)



train_bc.head()