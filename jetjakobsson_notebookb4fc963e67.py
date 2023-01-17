# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv", index_col=0)

test = pd.read_csv("../input/test.csv", index_col=0)

train["Age"] = train["Age"].fillna(train["Age"].mean())

train.info()

bins = np.linspace(0, 80, 10)

plt.hist(train[train.Sex == "male"]["Age"], alpha = 0.5, label = "Males", bins = bins)

plt.hist(train[train.Sex == "female"]["Age"], alpha = 0.5, label = "Females", bins = bins)

plt.legend(loc="upper right")

plt.show()
males = train[train.Sex == "male"]["Age"]

females = train[train.Sex == "female"]["Age"]

plt.boxplot(males)
train.values