import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import os
os.listdir("../input")
my_data = pd.read_csv('../input/youtube-vocabulary/vocabulary.csv')
my_data.shape
my_data.describe()
my_data.info()
my_data.head()
plt.figure(figsize=(10,8))
my_data.groupby('Vertical1').TrainVideoCount.sum().plot(kind='bar')
plt.title('Average TrainVideoCount per vartical1')
plt.show()

plt .figure(figsize=(10,8))
my_data.groupby('Vertical1').Index.count().plot(kind='bar')

plt.title('Average Number of Video per vartical1')

plt.show()

plt .figure(figsize=(10,8))
my_data.groupby('Vertical2').TrainVideoCount.count().plot(kind='bar')

plt.title('Average Number of Video per vartical2')

plt.show()