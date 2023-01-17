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
# Import libraries necessary

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
#load the train dataset

data = pd.read_csv("../input/train.csv")

#display dataset info

data.info()
data.head(10)
data.describe()
#Display all columns with -1

Missing_Values_col = data.loc[:].columns[(data.loc[:] == -1).any()]

print ("There are {} columns with missing values".format(len(Missing_Values_col)))

print (data[data == -1][Missing_Values_col].count())
Missing_data = data[Missing_Values_col]

Missing_data = Missing_data.replace(-1,np.NaN)

import missingno as msno

msno.matrix(Missing_data)
plt.title("Target")

plt.xticks([0,1])

plt.bar([0,1],data["target"].value_counts().values)
plt.title("ps_ind_06_bin")

plt.xticks([0,1])

plt.bar([0,1],data["ps_ind_06_bin"].value_counts().values)
plt.title("ps_ind_07_bin")

plt.xticks([0,1])

plt.bar([0,1],data["ps_ind_07_bin"].value_counts().values)
plt.title("ps_ind_08_bin")

plt.xticks([0,1])

plt.bar([0,1],data["ps_ind_08_bin"].value_counts().values)
plt.title("ps_ind_09_bin")

plt.xticks([0,1])

plt.bar([0,1],data["ps_ind_09_bin"].value_counts().values)