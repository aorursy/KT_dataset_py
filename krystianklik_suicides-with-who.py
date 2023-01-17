import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

source = '../input/who_suicide_statistics.csv'
data = pd.read_csv(source)
data.head()
data.sample(20)
print('Columns with null values:\n', data.isnull().sum())
data.describe(include = 'all')
modified_data = data.dropna()
print('Columns with null values:\n', modified_data.isnull().sum())
plt.figure(figsize=[30,50])

plt.subplot(231)
plt.boxplot(x=modified_data['suicides_no'], showmeans = True, meanline = True)
plt.title('Suicide numbers')
plt.ylabel('Numbers of suicide')

