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

import matplotlib.pyplot as plt
%matplotlib inline

pd.set_option("display.max_columns",20)

# We'll also import seaborn, a Python graphing library
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns

import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

import sklearn as skl
# Next, let's load in the data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# train.head(20)
# Scann through the data, we find that:
# var15 is interesting --> range from 23 - 55
# imp_op_var41_comer_ult3 --> two decimal number, is it about money?
# imp_op_var41_ult1, imp_op_var39_ult1 & lots of others
#   --> same as above; what does extreme large number mean?
# var36 --> range from 1,2,3 & 99; 99 is probably default value
# var38 --> lots of different numbers with decimals
# TARGET == 1 --> the use is unsatified ≈ 4% of users 
# TARGET == 0 --> the use is NOT unsatified (happy)
a = train.groupby('TARGET').size().reset_index()
a.columns = ["TARGET", "Count"]
a['percent'] = a.Count/a.Count.sum()
a
# Check for nan --> No nan
train.isnull().sum().sum()
