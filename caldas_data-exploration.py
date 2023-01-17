import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv("../input/train.csv")

print(train.columns)
data = train.as_matrix()
uniqueIds = np.unique(data[:,5])
print("Number of distinct places: " + str(uniqueIds.shape[0]))
#
print(data)
import matplotlib.pyplot as plt
dataX = data[:1500,:]
plt.figure()
plt.scatter(dataX[:, 1], dataX[:, 2])
plt.show()
print(dataX)