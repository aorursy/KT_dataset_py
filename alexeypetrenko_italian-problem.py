# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import random

import itertools
#import pandas as pd
data = [

    [350, 400],

    [450, 550],

    [550, 550],

    [650, 700],

    [300, 300],

    [450, 550],

    [600, 650],

    [750, 800],

    [350, 400],

    [550, 500],

    [350, 350],

    [650, 700]

]
data
random.sample(data,4)
#t = [2,2,2,2,4]

#c = list(itertools.combinations(t, 4))
c = list(itertools.combinations(data, 4))
len(c)
c