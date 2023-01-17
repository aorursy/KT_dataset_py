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



df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
df_train.time.describe()
df_test.time.describe()
range_train = df_train.time.max() - df_train.time.min()
print(range_train)

range_test = df_train.test.max() - df_train.test.min()
print(range_test)
# number of days of data if time is given in seconds
range_train  / (60 * 60 * 24)
# number of days of data if time is given in minutes - This is over a year of data?
range_train  / (60 * 24)