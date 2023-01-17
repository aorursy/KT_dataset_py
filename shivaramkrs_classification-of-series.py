# Importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# read train data
data = pd.read_csv("../input/train.csv")

# Convert sequence into list of integer lists (convenient for accessing)
seqs = data['Sequence'].tolist()
seqsL = [list(map(int, x.split(","))) for x in seqs]
series = seqsL[0]
print(series)
divSeries = [float(n)/m for n, m in zip(series[1:], series[:-1])]
plt.plot(divSeries)
plt.show()
series = seqsL[1]
print(series)
plt.plot(series)
plt.show()
diffSeries = [n - m for n, m in zip(series[1:], series[:-1])]
divSeries = [float(n)/m for n, m in zip(series[1:], series[:-1])]
