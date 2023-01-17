# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data2017 = pd.read_csv('../input/2017.csv')
data2017.info()
data2017.columns
data2017.corr()
data2017.describe()
data2017.rename(columns=lambda x: x[0:11], inplace=True)

data2017.columns = [ each.split('.')[0]+ "_" + each.split('.')[1] if ("." in each) else each for each in data2017.columns]
print(data2017.columns)

data2017.Happiness_S.plot(kind='line', color='green', label='Happiness_Score')
plt.xlabel('Index')
plt.ylabel('Frequency')
plt.title('Happiness_Score - Frequency')
plt.show()
data2017.plot(kind='scatter', x='Health_', y='Happiness_S', color='red', label='Happiness_Score')
plt.xlabel('Health_Life')
plt.ylabel('Happiness_Score')
plt.title(' Health_Life - Hapiness Score')
plt.show()
data_frame = pd.DataFrame(data2017)
data_frame.head(10)
data_frame.plot(kind='scatter', x='Happiness_S', y = 'Health_' , color='green', figsize= (9,9))
plt.xlabel('Happiness_Score')
plt.ylabel('Health_Life')
plt.title('Hapiness Score - Health_Life')
plt.show()
dataframe = pd.DataFrame(data2017.head(30))
dataframe.plot(kind='bar', x='Country', y = 'Health_' , color='green', figsize= (9,9))
plt.xlabel('Country')
plt.ylabel('Health_Life')
plt.title('Country - Health_Life')
plt.show()
dataframe.columns
dataframe = pd.DataFrame(data2017.head(30))
#dataframe.Happiness_Score.plot(kind='bar', color='r', figsize= (12,12), alpha=0.5)
dataframe.Health_.plot(kind='bar', color='g', figsize= (12,12), alpha=0.5)
dataframe.Freedom.plot(kind='bar', color='b', figsize= (12,12), alpha=0.5)
dataframe.Economy_.plot(kind='bar', color='y', figsize= (12,12), alpha=0.5)
plt.xlabel('Country')
plt.ylabel('Frequency')
plt.title('First 30 Country')
plt.show()
data2017.head(5).plot(figsize= (15,15), grid=True)
plt.xlabel('Country Index')
plt.ylabel('Frequency')
plt.title('Comparison First 5 Country')
plt.show()
data2017.head(5).plot(figsize= (15,15), subplots =True, grid=True)
plt.xlabel('Country Index')
plt.suptitle('Comparison First 5 Country')
plt.show()



