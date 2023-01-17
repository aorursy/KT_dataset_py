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

import pandas as pd 

import bz2

import gc

import re

import os

print(os.listdir("../input"))
! ls ../input/
#train_file = bz2.BZ2File('../input/train.ft.txt.bz2')

# test_file = bz2.BZ2File('reviews from AWS/test.ft.txt.bz2')

test_file = open('../input/test.ft.txt').read()
lines = test_file.split('\n')
lines[0]
label, text = lines[0][:10],  lines[0][11:]
label
text