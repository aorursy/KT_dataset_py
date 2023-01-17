# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("../input/tmdb_5000_movies.csv") 
data.describe()
data_frame1 = data[["budget"]]
data_frame2 = data[["revenue"]]
data_frame1.head()
data_frame2.head()
data.budget.plot(kind="line",color = "red",label = "Bütçe ")
data.revenue.plot(kind="line", color = "green",label = "Gelir ",alpha = 0.5)
plt.grid(True)
plt.xlabel("Bütçe")
plt.ylabel("Hasılat")
plt.legend()
plt.show()
