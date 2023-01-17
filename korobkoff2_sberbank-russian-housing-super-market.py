# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



train = pd.read_csv('../input/train.csv', header='infer', parse_dates=True, keep_date_col=True)

print("Train: " + str(train.shape))

#train.head(10)

test = pd.read_csv('../input/test.csv', header='infer', parse_dates=True, keep_date_col=True)

print("Test: " + str(test.shape))

test.head(10)



macro = pd.read_csv('../input/macro.csv', header='infer', parse_dates=True, keep_date_col=True)

macro.head(10)
