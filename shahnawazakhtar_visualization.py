# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("Loaded")
train["var38"].describe()
onesv38 = train.loc[train["TARGET"] == 1]
zerosv38 = train.loc[train["TARGET"] == 0]
plt.scatter(zerosv38["var15"], zerosv38["var38"])
plt.show()

plt.scatter(onesv38["var15"], onesv38["var38"])
plt.show()
plt.scatter(train["TARGET"], train["var38"])
plt.show()
plt.scatter(train["TARGET"], train["var15"])
plt.show()
op = train["var15"]*train["var38"]
plt.scatter(train["TARGET"], op)
plt.show()
op = train["var15"]-train["var38"]
plt.scatter(train["TARGET"], op)
plt.show()
op = train["var15"]+train["var38"]
plt.scatter(train["TARGET"], op)
plt.show()
op = train["var15"]/train["var38"]
plt.scatter(train["TARGET"], op)
plt.show()
