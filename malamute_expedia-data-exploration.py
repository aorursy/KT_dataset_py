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
# counting the number of rows in each file
print(check_output(["wc", "-l", "../input/train.csv"]).decode("utf8"))
print(check_output(["wc", "-l", "../input/test.csv"]).decode("utf8"))
print(check_output(["wc", "-l", "../input/destinations.csv"]).decode("utf8"))
# read in a sample of the train data and analyze
train= pd.read_csv("../input/train.csv", 
                   parse_dates=['date_time', 'srch_ci', 'srch_co'], 
                   nrows= 100000)
train.info()
train.shape
train.head()
train.describe()
