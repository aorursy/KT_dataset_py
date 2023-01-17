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
data = pd.read_csv('../input/presidential_polls.csv')

data.head()
import matplotlib.pyplot as plt

x = data['enddate']

plt.plot(data.index, data['adjpoll_clinton'], color="blue")

plt.plot(data.index, data['adjpoll_trump'], color="red")
plt.plot(data.index[9000:], data['adjpoll_clinton'][9000:], color="blue")

plt.plot(data.index[9000:], data['adjpoll_trump'][9000:], color="red")