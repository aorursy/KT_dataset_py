# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%matplotlib inline



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.plotly as py



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data_train = pd.read_csv("../input/train_2016.csv", parse_dates = ['transactiondate'])

data_train.shape
data_train.head()
plt.figure(figsize = (12, 8))

plt.scatter(range(data_train.shape[0]), np.sort(data_train.logerror.values))

plt.show()
help(plt.hist)