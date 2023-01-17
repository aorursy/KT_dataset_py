# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

museum_df = pd.read_csv("../input/museums.csv")
museum_summary = museum_df.describe()
#print(museum_summary)

income = museum_df["Income"]
# Too much disparity in data, need to normalize it somehow
# However, can't do zero, negative values w/logs
income = list(filter(lambda n: n>0, income))
income = list(map(lambda n: math.log(n, 10), income))
n, bins, patches = plt.hist(income)
plt.show()



# Any results you write to the current directory are saved as output.