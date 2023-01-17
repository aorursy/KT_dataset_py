# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
sample = pd.read_csv("../input/sample_submission.csv")
test = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")
train.head()
plt.plot(train["v1"], train["target"], linestyle='none', marker='o')

#plt.plot(train["v3"], train["target"], linestyle='none', marker='o')



plt.plot(train["v2"], train["target"], linestyle='none', marker='o')
plt.plot(train["v4"], train["target"], linestyle='none', marker='o')
plt.plot(train["v5"], train["target"], linestyle='none', marker='o')

plt.plot(train["v6"], train["target"], linestyle='none', marker='o')

plt.plot(train["v7"], train["target"], linestyle='none', marker='o')

plt.plot(train["v8"], train["target"], linestyle='none', marker='o')

plt.plot(train["v8"] ** train["v130"], train["target"], linestyle='none', marker='o')

