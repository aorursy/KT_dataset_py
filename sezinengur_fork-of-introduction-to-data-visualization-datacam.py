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
data = pd.read_csv("../input/percent-bachelors-degrees-women-usa.csv")
data.head()
data.keys()
year = np.array(data["Year"])
year
physical_sciences = np.array(data["Physical Sciences"])
physical_sciences
computer_science = np.array(data["Computer Science"])
computer_science
import matplotlib.pyplot as plt
%matplotlib inline 

plt.plot(year,physical_sciences, color='blue')
plt.plot(year,computer_science, color='red')
plt.show()
