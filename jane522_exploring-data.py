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


test = pd.read_csv('../input/train.csv')
test.head()
test.describe()
test.info()
len(test['place_id'].unique())
len(test['row_id'].unique())
# row id is unique identifier of event

test['place_id'].hist()
import matplotlib.pyplot as plt
plt.hist(test['accuracy'], bins = 10)
plt.xlim(0, 400)
# I want to summarize place stats for each x, y location
grouped = pd.groupby(test, by = ['x', 'y']).len

