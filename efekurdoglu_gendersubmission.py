# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/gender-submission/gender_submission1.csv") #read csv file
data.info()#Column, count, Dtype
data.describe()#Observing data in detail
data.head(10)#Showing first ten data values
data.plot(kind='scatter', x='PassengerId', y='Survived', alpha = 1,color = 'blue')

plt.xlabel('Open')              # label = name of label

plt.ylabel('High')

plt.title('PassengerId Survived Scatter Plot')  

plt.show()
# Histogram

# bins = number of bar in figure

data.Survived.plot(kind = 'hist',figsize = (12,12))

plt.show()