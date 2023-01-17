# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# First look at the data
data = pd.read_csv("../input/train.csv")
print(data[:5])
# data[:200].plot(kind="scatter",x='x', y='y', marker='.')
sns.FacetGrid(data[:200], hue="place_id", size=5) \
   .map(plt.scatter, "x", "x") \



data['place_id'].count()